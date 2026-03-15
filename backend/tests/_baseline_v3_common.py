from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency guard
    psutil = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.spread.feature_contracts import DEFAULT_FEATURE_CONTRACT_VERSION, FEATURE_NAMES
from src.spread.dataset_catalog import catalog_snapshot_manifest, register_sqlite_dataset
from src.spread.runtime_audit import create_sqlite_snapshot
from src.spread.train_model import certify_data_for_training, run_clean_training_cycle, run_threshold_preflight
from src.spread.training_certification import collect_sqlite_integrity

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_FRESH_HOURS = 8.0
DEFAULT_CHECKPOINT_MINUTES = 30
DEFAULT_PREDICTION_HORIZON_SEC = 14_400
DEFAULT_SEQUENCE_LENGTH = 15
DEFAULT_CERT_TIMEOUT_SEC = 1_800
DEFAULT_POSITIVE_EPISODE_THRESHOLD = 0.50
STATE_DB_PATH = ROOT_DIR / "out" / "config" / "tracker_history.sqlite"
SERVER_LOCK_PATH = ROOT_DIR / "out" / "config" / "server.instance.lock"
TRAINING_RUNS_DIR = ROOT_DIR / "out" / "config" / "training_runs"
NON_BLOCKING_T0_FAILURE_REASONS = {"runtime_audit_stale"}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def utc_now_ts() -> float:
    return float(time.time())


def utc_now_iso(ts: float | None = None) -> str:
    resolved = float(utc_now_ts() if ts is None else ts)
    return datetime.fromtimestamp(resolved, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_state_path(state_path_arg: str = "") -> Path:
    if not state_path_arg:
        return STATE_DB_PATH
    state_path = Path(state_path_arg)
    if not state_path.is_absolute():
        state_path = (ROOT_DIR / state_path).resolve()
    return state_path


def resolve_run_dir(run_dir_arg: str = "", *, label: str = "baseline_fresh") -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
        if not run_dir.is_absolute():
            run_dir = (Path.cwd() / run_dir).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = (TRAINING_RUNS_DIR / f"{stamp}_{label}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def context_path(run_dir: Path) -> Path:
    return Path(run_dir) / "run_context.json"


def load_run_context(run_dir: Path) -> dict[str, Any]:
    path = context_path(run_dir)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return path


def append_ndjson(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default, ensure_ascii=True))
        handle.write("\n")
    return path


def write_markdown(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return path


def update_run_context(run_dir: Path, **fields: Any) -> dict[str, Any]:
    payload = load_run_context(run_dir)
    payload.update(fields)
    payload.setdefault("run_dir", str(Path(run_dir).resolve()))
    payload.setdefault("updated_at_utc", utc_now_iso())
    payload["updated_at_utc"] = utc_now_iso()
    write_json(context_path(run_dir), payload)
    return payload


def resolve_restart_ts(run_dir: Path, restart_ts_arg: str = "") -> float:
    if str(restart_ts_arg).strip():
        return float(restart_ts_arg)
    context = load_run_context(run_dir)
    restart_ts = float(context.get("restart_ts", 0.0) or 0.0)
    if restart_ts <= 0.0:
        raise ValueError("restart_ts is required via --restart-ts or existing run_context.json")
    return restart_ts


def markdown_sections(title: str, sections: list[tuple[str, list[str]]]) -> str:
    lines = [f"# {title}", ""]
    for heading, items in sections:
        lines.append(f"## {heading}")
        if items:
            lines.extend(items)
        else:
            lines.append("- n/a")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def read_server_lock_status(lock_path: Path = SERVER_LOCK_PATH) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if not lock_path.is_file():
        return {
            "lock_exists": False,
            "lock_path": str(lock_path),
            "pid": 0,
            "pid_exists": False,
            "stale": False,
            "payload": {},
        }
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    pid = int(payload.get("pid") or 0)
    pid_exists = bool(pid and _pid_exists(pid))
    return {
        "lock_exists": True,
        "lock_path": str(lock_path),
        "pid": pid,
        "pid_exists": pid_exists,
        "stale": bool(pid and not pid_exists),
        "payload": payload,
    }


def _pid_exists(pid: int) -> bool:
    numeric = int(pid or 0)
    if numeric <= 0:
        return False
    if psutil is not None:
        try:
            return bool(psutil.pid_exists(numeric))
        except Exception:
            return False
    if os.name != "nt":  # pragma: no cover - Windows path exercised in CI
        try:
            os.kill(numeric, 0)
            return True
        except OSError:
            return False
    return False


def _nested_value(payload: dict[str, Any], *path: str, default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)


def fetch_json(base_url: str, path: str, *, timeout_sec: int = 10) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    with urllib.request.urlopen(url, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def collect_runtime_probe(base_url: str) -> dict[str, Any]:
    health_payload: dict[str, Any] = {}
    perf_payload: dict[str, Any] = {}
    health_ok = False
    perf_ok = False
    health_error = ""
    perf_error = ""
    try:
        health_payload = fetch_json(base_url, "/api/v1/system/health", timeout_sec=5)
        health_ok = True
    except Exception as exc:
        health_error = str(exc)
    try:
        perf_payload = fetch_json(base_url, "/api/debug/perf", timeout_sec=5)
        perf_ok = True
    except Exception as exc:
        perf_error = str(exc)
    return {
        "base_url": base_url.rstrip("/"),
        "health_ok": bool(health_ok),
        "perf_ok": bool(perf_ok),
        "health_error": health_error,
        "perf_error": perf_error,
        "health_payload": health_payload,
        "perf_payload": perf_payload,
        "rejection_rate_pct": _safe_float(health_payload.get("rejection_rate_pct"), 0.0),
        "rss_mb": _safe_float(_nested_value(perf_payload, "runtime", "process_rss_mb"), 0.0),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (max(0.0, min(100.0, float(pct))) / 100.0) * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float((ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight))


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def run_pragma_integrity_check(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"ok": False, "result": "missing_db"}
    with _connect(db_path) as conn:
        row = conn.execute("PRAGMA integrity_check").fetchone()
    result = str(row[0] if row else "").strip().lower()
    return {"ok": bool(result == "ok"), "result": result or "unknown"}


def collect_db_overview(db_path: Path) -> dict[str, Any]:
    with _connect(db_path) as conn:
        table_counts = {
            "tracker_records": int(conn.execute("SELECT COUNT(*) FROM tracker_records").fetchone()[0]),
            "tracker_pair_episodes": int(conn.execute("SELECT COUNT(*) FROM tracker_pair_episodes").fetchone()[0]),
            "tracker_pairs": int(conn.execute("SELECT COUNT(*) FROM tracker_pairs").fetchone()[0]),
        }
        max_ts = _safe_float(conn.execute("SELECT MAX(ts) FROM tracker_records").fetchone()[0], 0.0)
        min_ts = _safe_float(conn.execute("SELECT MIN(ts) FROM tracker_records").fetchone()[0], 0.0)
        max_session_id = int(conn.execute("SELECT COALESCE(MAX(session_id), 0) FROM tracker_records").fetchone()[0])
        recent_session_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM tracker_records WHERE session_id = (SELECT COALESCE(MAX(session_id), 0) FROM tracker_records)"
            ).fetchone()[0]
        )
    return {
        "db_path": str(db_path),
        "db_exists": db_path.is_file(),
        "db_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
        "table_counts": table_counts,
        "max_ts": max_ts,
        "min_ts": min_ts,
        "max_ts_utc": utc_now_iso(max_ts) if max_ts > 0 else "",
        "min_ts_utc": utc_now_iso(min_ts) if min_ts > 0 else "",
        "max_session_id": max_session_id,
        "recent_session_count": recent_session_count,
    }


def collect_episode_v3_completeness(db_path: Path, *, since_ts: float | None = None) -> dict[str, Any]:
    filters = ["is_closed = 1"]
    params: list[Any] = []
    if since_ts is not None:
        filters.append("start_ts >= ?")
        params.append(float(since_ts))
    where_sql = f"WHERE {' AND '.join(filters)}"
    with _connect(db_path) as conn:
        closed_total = int(conn.execute(f"SELECT COUNT(*) FROM tracker_pair_episodes {where_sql}", params).fetchone()[0])
        complete_total = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM tracker_pair_episodes
                {where_sql}
                  AND exit_spread_at_close IS NOT NULL
                  AND peak_entry_spread IS NOT NULL
                  AND duration_sec IS NOT NULL
                  AND duration_sec >= 0
                  AND peak_entry_spread = peak_entry_spread
                  AND exit_spread_at_close = exit_spread_at_close
                  AND duration_sec = duration_sec
                """,
                params,
            ).fetchone()[0]
        )
        peak_positive = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM tracker_pair_episodes
                {where_sql}
                  AND peak_entry_spread > 0
                """,
                params,
            ).fetchone()[0]
        )
        exit_zero = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM tracker_pair_episodes
                {where_sql}
                  AND exit_spread_at_close = 0
                """,
                params,
            ).fetchone()[0]
        )
    completeness_rate = float(complete_total / max(closed_total, 1)) if closed_total > 0 else 0.0
    return {
        "closed_episode_count": closed_total,
        "complete_gate06_episode_count": complete_total,
        "gate06_completeness_rate": completeness_rate,
        "peak_positive_episode_count": peak_positive,
        "peak_positive_rate": float(peak_positive / max(closed_total, 1)) if closed_total > 0 else 0.0,
        "peak_non_positive_episode_count": int(max(closed_total - peak_positive, 0)),
        "peak_non_positive_rate": float(max(closed_total - peak_positive, 0) / max(closed_total, 1)) if closed_total > 0 else 0.0,
        "exit_zero_episode_count": exit_zero,
        "exit_zero_rate": float(exit_zero / max(closed_total, 1)) if closed_total > 0 else 0.0,
        "complete_v3_episode_count": complete_total,
        "v3_completeness_rate": completeness_rate,
        "since_ts": None if since_ts is None else float(since_ts),
        "since_ts_utc": "" if since_ts is None else utc_now_iso(float(since_ts)),
    }


def collect_positive_episode_estimate(
    db_path: Path,
    *,
    threshold_pct: float = DEFAULT_POSITIVE_EPISODE_THRESHOLD,
    since_ts: float | None = None,
) -> dict[str, Any]:
    filters = ["is_closed = 1"]
    params: list[Any] = []
    if since_ts is not None:
        filters.append("start_ts >= ?")
        params.append(float(since_ts))
    where_sql = f"WHERE {' AND '.join(filters)}"
    with _connect(db_path) as conn:
        total = int(conn.execute(f"SELECT COUNT(*) FROM tracker_pair_episodes {where_sql}", params).fetchone()[0])
        positive = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM tracker_pair_episodes
                {where_sql}
                  AND (COALESCE(peak_entry_spread, 0.0) + COALESCE(exit_spread_at_close, 0.0)) >= ?
                """,
                [*params, float(threshold_pct)],
            ).fetchone()[0]
        )
    return {
        "threshold_pct": float(threshold_pct),
        "closed_episode_count": total,
        "positive_episode_count": positive,
        "positive_rate": float(positive / max(total, 1)) if total > 0 else 0.0,
        "since_ts": None if since_ts is None else float(since_ts),
        "since_ts_utc": "" if since_ts is None else utc_now_iso(float(since_ts)),
    }


def collect_recent_interval_regularity(db_path: Path, *, sample_size: int = 1_000) -> dict[str, Any]:
    with _connect(db_path) as conn:
        rows = list(
            conn.execute(
                """
                SELECT block_id, ts
                FROM tracker_records
                WHERE block_id IS NOT NULL
                ORDER BY ts DESC
                LIMIT ?
                """,
                (int(sample_size),),
            )
        )
    by_block: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_block[int(row["block_id"] or 0)].append(float(row["ts"] or 0.0))
    deltas: list[float] = []
    for timestamps in by_block.values():
        ordered = sorted(ts for ts in timestamps if ts > 0.0)
        deltas.extend(curr - prev for prev, curr in zip(ordered, ordered[1:]) if curr > prev)
    return {
        "sample_size": int(sample_size),
        "delta_count": len(deltas),
        "interval_p50_sec": _percentile(deltas, 50.0),
        "interval_p90_sec": _percentile(deltas, 90.0),
    }


def validate_feature_contract_smoke() -> dict[str, Any]:
    return {
        "default_feature_contract_version": str(DEFAULT_FEATURE_CONTRACT_VERSION),
        "feature_count": int(len(FEATURE_NAMES)),
        "ok": bool(DEFAULT_FEATURE_CONTRACT_VERSION == "v3_exit_aware_40" and len(FEATURE_NAMES) == 40),
    }


def normalize_cert_verdict(raw_value: Any) -> str:
    normalized = str(raw_value or "").strip().upper()
    if normalized in {"CERTIFIED", "PASS"}:
        return "PASS"
    if normalized in {"CERTIFIED_WITH_WARNINGS", "WARN", "WARNING"}:
        return "WARN"
    if normalized in {"FAILED", "FAIL"}:
        return "FAIL"
    return normalized or "UNKNOWN"


def summarize_certification(certification: dict[str, Any]) -> dict[str, Any]:
    failure_reasons = [str(value) for value in (certification.get("failure_reasons") or [])]
    warnings = [str(value) for value in (certification.get("warnings") or [])]
    return {
        "verdict": normalize_cert_verdict(certification.get("verdict") or certification.get("certification_verdict")),
        "failure_reasons": failure_reasons,
        "warnings": warnings,
    }


def run_quick_certification(
    *,
    state_file: Path,
    artifact_dir: Path,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    prediction_horizon_sec: int = DEFAULT_PREDICTION_HORIZON_SEC,
    max_certification_duration_sec: int = DEFAULT_CERT_TIMEOUT_SEC,
) -> dict[str, Any]:
    return certify_data_for_training(
        state_file=Path(state_file),
        artifact_dir=Path(artifact_dir),
        sequence_length=int(sequence_length),
        prediction_horizon_sec=int(prediction_horizon_sec),
        certification_mode="quick",
        max_certification_duration_sec=int(max_certification_duration_sec),
    )


def run_preflight_check(
    *,
    state_file: Path,
    output_path: Path,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    prediction_horizon_sec: int = DEFAULT_PREDICTION_HORIZON_SEC,
    window_stride: int = 5,
) -> dict[str, Any]:
    return run_threshold_preflight(
        state_file=Path(state_file),
        output_path=Path(output_path),
        sequence_length=int(sequence_length),
        prediction_horizon_sec=int(prediction_horizon_sec),
        window_stride=int(window_stride),
    )


def run_full_training(
    *,
    state_file: Path,
    artifact_dir: Path,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    prediction_horizon_sec: int = DEFAULT_PREDICTION_HORIZON_SEC,
    window_stride: int = 5,
    max_certification_duration_sec: int = DEFAULT_CERT_TIMEOUT_SEC,
) -> dict[str, Any]:
    return run_clean_training_cycle(
        state_file=Path(state_file),
        artifact_dir=Path(artifact_dir),
        sequence_length=int(sequence_length),
        prediction_horizon_sec=int(prediction_horizon_sec),
        window_stride=int(window_stride),
        batch_size=1024,
        hidden_size=128,
        num_layers=2,
        dropout=0.35,
        max_epochs=80,
        patience=5,
        learning_rate=0.001,
        weight_decay=1e-4,
        focal_gamma=2.0,
        min_train_positive_samples=500,
        min_val_positive_samples=50,
        min_test_positive_samples=50,
        certification_mode="full",
        max_certification_duration_sec=int(max_certification_duration_sec),
    )


def create_fresh_snapshot(*, source_db: Path, run_dir: Path, name: str = "snapshot.sqlite") -> Path:
    return create_sqlite_snapshot(Path(source_db), Path(run_dir) / str(name))


def register_snapshot_dataset(
    *,
    snapshot_path: Path,
    certification: dict[str, Any] | None = None,
    label: str | None = None,
    role: str = "baseline_snapshot",
    tags: list[str] | None = None,
    bless: bool = False,
    related_files: list[Path] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return register_sqlite_dataset(
        Path(snapshot_path),
        base_path=ROOT_DIR / "out",
        certification=certification,
        label=label,
        role=role,
        tags=tags,
        bless=bless,
        related_files=related_files,
        extra_metadata=extra_metadata,
    )


def sync_existing_snapshot_catalog() -> dict[str, Any]:
    return catalog_snapshot_manifest(base_path=ROOT_DIR / "out")


def count_records_since(db_path: Path, *, since_ts: float) -> int:
    with _connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM tracker_records WHERE ts >= ?", (float(since_ts),)).fetchone()[0])


def count_closed_episodes_since(db_path: Path, *, since_ts: float) -> int:
    with _connect(db_path) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM tracker_pair_episodes WHERE is_closed = 1 AND start_ts >= ?",
                (float(since_ts),),
            ).fetchone()[0]
        )


def current_server_pid(lock_status: dict[str, Any]) -> int:
    pid = int(lock_status.get("pid") or 0)
    return pid if pid and bool(lock_status.get("pid_exists")) else 0


def get_process_rss_mb(pid: int) -> float:
    numeric = int(pid or 0)
    if numeric <= 0 or psutil is None:
        return 0.0
    try:
        proc = psutil.Process(numeric)
        return round(float(proc.memory_info().rss) / (1024 * 1024), 3)
    except Exception:
        return 0.0
