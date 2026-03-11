from __future__ import annotations

import json
import math
import sqlite3
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .auto_retrain import (
    backend_out_root,
    latest_snapshot_entry,
    latest_training_run,
    load_snapshot_manifest,
    load_training_manifest,
    should_retrain,
    snapshot_manifest_path,
    state_file_path,
    training_manifest_path,
)
from .feature_contracts import DEFAULT_FEATURE_CONTRACT_VERSION, V2_MULTISCALE_FEATURE_NAMES
from .ml_dataset import (
    _build_pair_segments,
    _load_blocks_from_sqlite,
    build_dataset_bundle,
    compute_label_threshold,
)
from .runtime_audit import _iter_ndjson, _write_json, build_runtime_summary
from .spread_tracker import SpreadTracker

_DEFAULT_STAGE_DURATIONS_SEC = {
    "stage1": 3 * 60 * 60,
    "stage2": 24 * 60 * 60,
}
_DEFAULT_STAGE_INTERVALS_SEC = {
    "stage1": 5 * 60,
    "stage2": 15 * 60,
}
_DEFAULT_MEMORY_WINDOW_SEC = 12 * 60 * 60
_DEFAULT_STORAGE_WINDOW_SEC = 8 * 24 * 60 * 60


@dataclass(frozen=True)
class SoakPaths:
    root_dir: Path
    artifact_root: Path
    output_dir: Path
    db_path: Path
    runtime_audit_root: Path
    snapshot_manifest_path: Path
    training_manifest_path: Path
    auto_retrain_state_path: Path


def _resolve_path(raw_value: str | Path | None, base_dir: Path) -> Path | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path.resolve()


def resolve_soak_paths(
    root_dir: Path,
    *,
    output_dir_arg: str = "",
    db_path_arg: str = "",
    artifact_root_arg: str = "",
    runtime_audit_dir_arg: str = "",
) -> SoakPaths:
    resolved_root = Path(root_dir).resolve()
    artifact_root = _resolve_path(artifact_root_arg, resolved_root) or backend_out_root(resolved_root / "out")
    runtime_audit_root = _resolve_path(runtime_audit_dir_arg, resolved_root) or (artifact_root / "runtime_audit")
    db_path = _resolve_path(db_path_arg, resolved_root) or (artifact_root / "config" / "tracker_history.sqlite")
    if output_dir_arg:
        output_dir = _resolve_path(output_dir_arg, resolved_root) or (artifact_root / "soak_reports")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (artifact_root / "soak_reports" / f"arbml_v2_soak_{stamp}").resolve()
    return SoakPaths(
        root_dir=resolved_root,
        artifact_root=artifact_root.resolve(),
        output_dir=output_dir.resolve(),
        db_path=db_path.resolve(),
        runtime_audit_root=runtime_audit_root.resolve(),
        snapshot_manifest_path=snapshot_manifest_path(artifact_root).resolve(),
        training_manifest_path=training_manifest_path(artifact_root).resolve(),
        auto_retrain_state_path=state_file_path(artifact_root).resolve(),
    )


def default_duration_for_stage(stage: str) -> int:
    return int(_DEFAULT_STAGE_DURATIONS_SEC.get(str(stage), _DEFAULT_STAGE_DURATIONS_SEC["stage1"]))


def default_interval_for_stage(stage: str) -> int:
    return int(_DEFAULT_STAGE_INTERVALS_SEC.get(str(stage), _DEFAULT_STAGE_INTERVALS_SEC["stage1"]))


def append_ndjson(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True))
        handle.write("\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _numeric_summary(values: Iterable[float]) -> dict[str, float]:
    data: list[float] = []
    for value in values:
        parsed = _safe_float(value, float("nan"))
        if math.isfinite(parsed):
            data.append(float(parsed))
    if not data:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(data)

    def _pct(percentile: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        rank = (len(ordered) - 1) * (float(percentile) / 100.0)
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        if lower == upper:
            return float(ordered[lower])
        weight = rank - lower
        return float(ordered[lower] + ((ordered[upper] - ordered[lower]) * weight))

    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
        "p50": _pct(50.0),
        "p95": _pct(95.0),
        "p99": _pct(99.0),
    }


def _fetch_json(base_url: str, path: str, *, timeout_sec: int = 15) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    with urllib.request.urlopen(url, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
        return {
            "ok": True,
            "status": int(response.status),
            "payload": payload if isinstance(payload, dict) else {"data": payload},
            "error": "",
            "url": url,
        }


def _safe_fetch_json(base_url: str, path: str, *, timeout_sec: int = 15) -> dict[str, Any]:
    try:
        return _fetch_json(base_url, path, timeout_sec=timeout_sec)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        payload: dict[str, Any] = {}
        if body:
            try:
                decoded = json.loads(body)
                if isinstance(decoded, dict):
                    payload = decoded
            except json.JSONDecodeError:
                payload = {"raw": body}
        return {
            "ok": False,
            "status": int(exc.code),
            "payload": payload,
            "error": str(exc),
            "url": f"{base_url.rstrip('/')}{path}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": 0,
            "payload": {},
            "error": str(exc),
            "url": f"{base_url.rstrip('/')}{path}",
        }


def _nested_value(payload: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return float(default)
        current = current.get(key)
    return _safe_float(current, default)


def _bool_gate(name: str, ok: bool, *, value: Any = None, expected: str = "", detail: str = "") -> dict[str, Any]:
    return {
        "name": str(name),
        "ok": bool(ok),
        "value": value,
        "expected": str(expected),
        "detail": str(detail),
    }


def collect_http_checkpoint(base_url: str) -> dict[str, Any]:
    debug_perf = _safe_fetch_json(base_url, "/api/debug/perf")
    system_health = _safe_fetch_json(base_url, "/api/v1/system/health")
    pipeline_status = _safe_fetch_json(base_url, "/api/v1/ml/pipeline/status")
    scanner_lite = _safe_fetch_json(base_url, "/api/spread/opportunities-lite")
    latest_run = _safe_fetch_json(base_url, "/api/v1/ml/training/runs/latest")

    debug_payload = dict(debug_perf.get("payload") or {})
    perf_payload = dict(debug_payload.get("perf") or {})
    runtime_payload = dict(debug_payload.get("runtime") or {})
    scanner_payload = dict(scanner_lite.get("payload") or {})
    scanner_rows = list(scanner_payload.get("data") or []) if isinstance(scanner_payload.get("data"), list) else []

    dashboard_low_spread_count = 0
    success_probabilities: list[float] = []
    invalid_probability_count = 0
    signal_action_counts: Counter[str] = Counter()
    model_status_counts: Counter[str] = Counter()
    for row in scanner_rows:
        if not isinstance(row, dict):
            continue
        entry_spread = _safe_float(row.get("entrySpread"), 0.0)
        if entry_spread < 0.2:
            dashboard_low_spread_count += 1
        probability = row.get("successProbability")
        if probability is not None:
            parsed_probability = _safe_float(probability, float("nan"))
            if math.isfinite(parsed_probability) and 0.0 <= parsed_probability <= 1.0:
                success_probabilities.append(parsed_probability)
            else:
                invalid_probability_count += 1
        signal_action = str(row.get("signalAction") or "WAIT")
        model_status = str(row.get("modelStatus") or "unknown")
        signal_action_counts[signal_action] += 1
        model_status_counts[model_status] += 1

    return {
        "ts": time.time(),
        "base_url": base_url.rstrip("/"),
        "http": {
            "debug_perf": debug_perf,
            "system_health": system_health,
            "pipeline_status": pipeline_status,
            "scanner_lite": scanner_lite,
            "latest_training_run": latest_run,
        },
        "metrics": {
            "rss_mb": _nested_value(debug_payload, "runtime", "process_rss_mb"),
            "calculate_ms_p95": _nested_value(perf_payload, "scanner_cycle", "calculate_ms", "p95"),
            "event_loop_lag_ms_p95": _nested_value(perf_payload, "event_loop_lag_ms", "p95"),
            "event_loop_lag_ms_p99": _nested_value(perf_payload, "event_loop_lag_ms", "p99"),
            "tracker_records_enqueued_p50": _nested_value(perf_payload, "scanner_cycle", "tracker_records_enqueued", "p50"),
            "tracker_records_enqueued_p95": _nested_value(perf_payload, "scanner_cycle", "tracker_records_enqueued", "p95"),
            "tracker_rejections_enqueued_p50": _nested_value(perf_payload, "scanner_cycle", "tracker_rejections_enqueued", "p50"),
            "tracker_rejections_enqueued_p95": _nested_value(perf_payload, "scanner_cycle", "tracker_rejections_enqueued", "p95"),
            "tracker_cycle_every": _nested_value(runtime_payload, "tracker_cycle_every"),
            "dashboard_low_spread_count": int(dashboard_low_spread_count),
            "scanner_row_count": len(scanner_rows),
            "invalid_probability_count": int(invalid_probability_count),
            "success_probability_summary": _numeric_summary(success_probabilities),
            "signal_action_counts": dict(signal_action_counts),
            "model_status_counts": dict(model_status_counts),
            "hour_verdict": str((system_health.get("payload") or {}).get("hour_verdict") or ""),
            "rejection_rate_pct": _safe_float((system_health.get("payload") or {}).get("rejection_rate_pct"), 0.0),
            "next_snapshot_sec": _safe_float((system_health.get("payload") or {}).get("next_snapshot_sec"), 0.0),
        },
    }


def _sqlite_connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def _latest_tracker_ts(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT MAX(ts) AS max_ts FROM tracker_records").fetchone()
    if row is None:
        return 0.0
    return _safe_float(row["max_ts"], 0.0)


def _compute_recent_pair_gaps(rows: list[sqlite3.Row]) -> dict[str, dict[str, float]]:
    by_pair: dict[str, list[float]] = defaultdict(list)
    last_ts_by_pair: dict[str, float] = {}
    for row in rows:
        pair_key = str(row["pair_key"])
        ts = _safe_float(row["ts"], 0.0)
        previous = last_ts_by_pair.get(pair_key)
        if previous is not None and ts > previous:
            by_pair[pair_key].append(ts - previous)
        last_ts_by_pair[pair_key] = ts
    payload: dict[str, dict[str, float]] = {}
    for pair_key, gaps in by_pair.items():
        if not gaps:
            continue
        ordered = sorted(gaps)
        payload[pair_key] = {
            "median_gap_sec": float(statistics.median(ordered)),
            "max_gap_sec": float(max(ordered)),
            "gap_count": float(len(ordered)),
        }
    return payload


def collect_tracker_sql_checks(db_path: Path, *, lookback_sec: int = 3600, pair_sample_limit: int = 12) -> dict[str, Any]:
    path = Path(db_path)
    if not path.is_file():
        return {"ok": False, "error": f"tracker db not found: {path}"}
    with _sqlite_connect(path) as conn:
        newest_ts = _latest_tracker_ts(conn)
        reference_ts = newest_ts if newest_ts > 0.0 else time.time()
        window_start = float(reference_ts) - float(max(int(lookback_sec), 1))
        low_spread_records = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM tracker_records
                WHERE entry_spread_pct >= 0.05
                  AND entry_spread_pct < 0.20
                  AND ts >= ?
                """,
                (window_start,),
            ).fetchone()[0]
        )
        low_spread_pairs = int(
            conn.execute(
                """
                SELECT COUNT(DISTINCT r.pair_id)
                FROM tracker_records r
                JOIN tracker_pairs p ON p.id = r.pair_id
                WHERE p.history_enabled = 1
                  AND r.entry_spread_pct >= 0.05
                  AND r.entry_spread_pct < 0.20
                  AND r.ts >= ?
                """,
                (window_start,),
            ).fetchone()[0]
        )
        min_entry_spread = _safe_float(
            conn.execute(
                "SELECT MIN(entry_spread_pct) AS min_entry_spread FROM tracker_records WHERE ts >= ?",
                (window_start,),
            ).fetchone()["min_entry_spread"],
            0.0,
        )
        stale_rows = list(
            conn.execute(
                """
                SELECT
                    p.symbol || '|' || p.buy_ex || '|' || p.buy_mt || '|' || p.sell_ex || '|' || p.sell_mt AS pair_key,
                    COUNT(*) AS record_count,
                    COUNT(DISTINCT ROUND(r.entry_spread_pct, 6)) AS distinct_spreads
                FROM tracker_records r
                JOIN tracker_pairs p ON p.id = r.pair_id
                WHERE r.ts >= ?
                GROUP BY r.pair_id
                HAVING COUNT(*) >= 20
                   AND COUNT(DISTINCT ROUND(r.entry_spread_pct, 6)) = 1
                ORDER BY record_count DESC, pair_key ASC
                """,
                (window_start,),
            )
        )
        pair_mean_rows = list(
            conn.execute(
                """
                SELECT
                    p.symbol || '|' || p.buy_ex || '|' || p.buy_mt || '|' || p.sell_ex || '|' || p.sell_mt AS pair_key,
                    COUNT(*) AS record_count,
                    AVG(r.entry_spread_pct) AS avg_entry_spread_pct,
                    MIN(r.entry_spread_pct) AS min_entry_spread_pct,
                    MAX(r.entry_spread_pct) AS max_entry_spread_pct
                FROM tracker_records r
                JOIN tracker_pairs p ON p.id = r.pair_id
                WHERE r.ts >= ?
                GROUP BY r.pair_id
                ORDER BY record_count DESC, pair_key ASC
                LIMIT ?
                """,
                (window_start, max(int(pair_sample_limit), 1)),
            )
        )
        hourly_health_rows = list(
            conn.execute(
                """
                SELECT
                    hour_start_ts,
                    hour_end_ts,
                    quality_verdict,
                    records_total,
                    records_rejected,
                    rejection_rate_pct,
                    cross_exchange_flags
                FROM tracker_hourly_health
                ORDER BY hour_start_ts DESC
                LIMIT 24
                """
            )
        )
        retention_row = conn.execute(
            """
            SELECT MIN(ts) AS oldest_ts, MAX(ts) AS newest_ts, COUNT(*) AS record_count
            FROM tracker_records
            """
        ).fetchone()
        gap_rows = list(
            conn.execute(
                """
                SELECT
                    p.symbol || '|' || p.buy_ex || '|' || p.buy_mt || '|' || p.sell_ex || '|' || p.sell_mt AS pair_key,
                    r.ts
                FROM tracker_records r
                JOIN tracker_pairs p ON p.id = r.pair_id
                WHERE r.ts >= ?
                ORDER BY pair_key ASC, r.ts ASC
                """,
                (window_start,),
            )
        )
    pair_means = [
        {
            "pair_key": str(row["pair_key"]),
            "record_count": int(row["record_count"]),
            "avg_entry_spread_pct": _safe_float(row["avg_entry_spread_pct"], 0.0),
            "min_entry_spread_pct": _safe_float(row["min_entry_spread_pct"], 0.0),
            "max_entry_spread_pct": _safe_float(row["max_entry_spread_pct"], 0.0),
        }
        for row in pair_mean_rows
    ]
    hourly_health = [
        {
            "hour_start_ts": _safe_float(row["hour_start_ts"], 0.0),
            "hour_end_ts": _safe_float(row["hour_end_ts"], 0.0),
            "quality_verdict": str(row["quality_verdict"] or "healthy"),
            "records_total": int(row["records_total"]),
            "records_rejected": int(row["records_rejected"]),
            "rejection_rate_pct": _safe_float(row["rejection_rate_pct"], 0.0),
            "cross_exchange_flags": int(row["cross_exchange_flags"]),
        }
        for row in hourly_health_rows
    ]
    return {
        "ok": True,
        "db_path": str(path),
        "reference_ts": float(reference_ts),
        "lookback_sec": int(lookback_sec),
        "low_spread_records": int(low_spread_records),
        "low_spread_pairs_history_enabled": int(low_spread_pairs),
        "min_entry_spread_pct_recent": float(min_entry_spread),
        "stale_pairs": [
            {
                "pair_key": str(row["pair_key"]),
                "record_count": int(row["record_count"]),
                "distinct_spreads": int(row["distinct_spreads"]),
            }
            for row in stale_rows
        ],
        "pair_means": pair_means,
        "recent_pair_gaps": _compute_recent_pair_gaps(gap_rows),
        "hourly_health": hourly_health,
        "latest_hourly_health": hourly_health[0] if hourly_health else None,
        "retention": {
            "oldest_ts": _safe_float(retention_row["oldest_ts"], 0.0) if retention_row else 0.0,
            "newest_ts": _safe_float(retention_row["newest_ts"], 0.0) if retention_row else 0.0,
            "record_count": int(retention_row["record_count"]) if retention_row else 0,
        },
    }


def _tracker_read_only(
    db_path: Path,
    *,
    storage_window_sec: int = _DEFAULT_STORAGE_WINDOW_SEC,
    memory_window_sec: int = _DEFAULT_MEMORY_WINDOW_SEC,
    record_interval_sec: float = 15.0,
) -> SpreadTracker:
    tracker = SpreadTracker(
        window_sec=int(storage_window_sec),
        memory_window_sec=int(memory_window_sec),
        record_interval_sec=float(record_interval_sec),
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=None,
    )
    tracker.db_path = Path(db_path)
    return tracker


def _pick_tracker_pair(tracker: SpreadTracker, *, min_records: int) -> tuple[str, str, str, str, str] | None:
    with tracker._lock:
        candidates: list[tuple[int, tuple[str, str, str, str, str]]] = []
        for pair_key, pair_stats in tracker._pairs.items():
            if len(pair_stats.records) >= int(min_records):
                candidates.append((len(pair_stats.records), pair_key))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def run_feature_history_harness(db_path: Path, *, limit: int = 15) -> dict[str, Any]:
    path = Path(db_path)
    if not path.is_file():
        return {"ok": False, "error": f"tracker db not found: {path}"}
    with _sqlite_connect(path) as conn:
        newest_ts = _latest_tracker_ts(conn)
    tracker = _tracker_read_only(path)
    tracker.load_from_storage(now_ts=newest_ts if newest_ts > 0.0 else time.time())
    pair = _pick_tracker_pair(tracker, min_records=max(int(limit), 8))
    if pair is None:
        return {"ok": False, "error": "no tracker pair has enough records for feature-history harness"}
    rows = tracker.get_feature_history(*pair, limit=int(limit), feature_names=list(V2_MULTISCALE_FEATURE_NAMES))
    if not rows:
        return {"ok": False, "error": "get_feature_history returned no rows", "selected_pair": "|".join(pair)}
    multiscale_indices = {
        name: index
        for index, name in enumerate(V2_MULTISCALE_FEATURE_NAMES)
        if any(window in name for window in ("30m", "2h", "8h"))
    }
    variances: dict[str, float] = {}
    nonzero_multiscale_features: list[str] = []
    for name, index in multiscale_indices.items():
        values = [_safe_float(row[index], 0.0) for row in rows if len(row) > index]
        variance = float(statistics.pvariance(values)) if len(values) >= 2 else 0.0
        variances[name] = variance
        if variance > 0.0:
            nonzero_multiscale_features.append(name)
    with tracker._lock:
        pair_stats = tracker._pairs.get(pair)
        if pair_stats is None or not pair_stats.records:
            return {"ok": False, "error": "selected pair disappeared during harness", "selected_pair": "|".join(pair)}
        last_record = pair_stats.records[-1]
        last_ts = _safe_float(last_record.timestamp, 0.0)
        last_entry = _safe_float(last_record.entry_spread_pct, 0.0)
        last_exit = _safe_float(last_record.exit_spread_pct, 0.0)
    tracker.record_spread(
        *pair,
        last_entry + 0.031,
        last_exit - 0.017,
        now_ts=last_ts + float(max(tracker.record_interval_sec, 1.0)),
    )
    updated_rows = tracker.get_feature_history(*pair, limit=int(limit), feature_names=list(V2_MULTISCALE_FEATURE_NAMES))
    cache_invalidated = bool(updated_rows and rows and updated_rows[-1] != rows[-1])
    return {
        "ok": True,
        "selected_pair": "|".join(pair),
        "row_count": len(rows),
        "feature_count": len(rows[0]) if rows else 0,
        "feature_contract_version": DEFAULT_FEATURE_CONTRACT_VERSION,
        "nonzero_multiscale_features": nonzero_multiscale_features,
        "multiscale_variance": variances,
        "cache_invalidated": cache_invalidated,
    }


def run_memory_window_harness(
    db_path: Path,
    *,
    storage_window_sec: int = _DEFAULT_STORAGE_WINDOW_SEC,
    memory_window_sec: int = _DEFAULT_MEMORY_WINDOW_SEC,
) -> dict[str, Any]:
    path = Path(db_path)
    if not path.is_file():
        return {"ok": False, "error": f"tracker db not found: {path}"}
    with _sqlite_connect(path) as conn:
        newest_ts = _latest_tracker_ts(conn)
        retention_row = conn.execute(
            "SELECT MIN(ts) AS oldest_ts, MAX(ts) AS newest_ts, COUNT(*) AS record_count FROM tracker_records"
        ).fetchone()
    reference_ts = newest_ts if newest_ts > 0.0 else time.time()
    tracker = _tracker_read_only(
        path,
        storage_window_sec=int(storage_window_sec),
        memory_window_sec=int(memory_window_sec),
    )
    tracker.load_from_storage(now_ts=reference_ts)
    memory_oldest_ts = reference_ts
    memory_newest_ts = 0.0
    memory_record_count = 0
    with tracker._lock:
        for pair_stats in tracker._pairs.values():
            records = list(pair_stats.records)
            if not records:
                continue
            memory_record_count += len(records)
            memory_oldest_ts = min(memory_oldest_ts, min(_safe_float(item.timestamp, reference_ts) for item in records))
            memory_newest_ts = max(memory_newest_ts, max(_safe_float(item.timestamp, 0.0) for item in records))
    disk_oldest_ts = _safe_float(retention_row["oldest_ts"], 0.0) if retention_row else 0.0
    return {
        "ok": True,
        "db_path": str(path),
        "reference_ts": float(reference_ts),
        "storage_window_sec": int(storage_window_sec),
        "tracker_memory_window_sec": int(memory_window_sec),
        "disk_oldest_ts": float(disk_oldest_ts),
        "disk_newest_ts": _safe_float(retention_row["newest_ts"], 0.0) if retention_row else 0.0,
        "disk_record_count": int(retention_row["record_count"]) if retention_row else 0,
        "memory_oldest_ts": float(memory_oldest_ts) if memory_record_count > 0 else 0.0,
        "memory_newest_ts": float(memory_newest_ts),
        "memory_record_count": int(memory_record_count),
        "has_disk_records_older_than_8h": bool(disk_oldest_ts > 0.0 and (reference_ts - disk_oldest_ts) > (8 * 60 * 60)),
        "has_disk_records_older_than_13h": bool(disk_oldest_ts > 0.0 and (reference_ts - disk_oldest_ts) > (13 * 60 * 60)),
        "memory_kept_older_than_13h": bool(memory_record_count > 0 and (reference_ts - memory_oldest_ts) > (13 * 60 * 60)),
    }


def _runtime_audit_package_dirs(runtime_audit_root: Path) -> list[Path]:
    root = Path(runtime_audit_root)
    if not root.exists():
        return []
    candidates: list[Path] = []
    if (root / "events.ndjson").is_file():
        candidates.append(root)
    for child in root.iterdir():
        if child.is_dir() and (child / "events.ndjson").is_file():
            candidates.append(child)
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates


def collect_runtime_audit_checks(runtime_audit_root: Path) -> dict[str, Any]:
    packages = _runtime_audit_package_dirs(runtime_audit_root)
    if not packages:
        return {"ok": False, "error": f"no runtime audit package found under {runtime_audit_root}"}
    package_dir = packages[0]
    events_path = package_dir / "events.ndjson"
    alerts_path = package_dir / "alerts.ndjson"
    ws_latency_path = package_dir / "ws_latency_summary.json"
    summary_path = package_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            summary = build_runtime_summary(package_dir)
    else:
        summary = build_runtime_summary(package_dir)
    alerts = list(_iter_ndjson(alerts_path))
    disconnect_alerts = [alert for alert in alerts if "disconnect" in str(alert.get("code") or "").lower()]
    reconnect_alerts = [alert for alert in alerts if "reconnect" in str(alert.get("code") or "").lower()]
    duration_sec = max(_safe_float(summary.get("duration_sec"), 0.0), 0.0)
    duration_hours = duration_sec / 3600.0 if duration_sec > 0.0 else 0.0
    events_size_mb = round(events_path.stat().st_size / (1024.0 * 1024.0), 6) if events_path.is_file() else 0.0
    alerts_size_mb = round(alerts_path.stat().st_size / (1024.0 * 1024.0), 6) if alerts_path.is_file() else 0.0
    book_age_summary = dict((summary.get("dashboard_validation") or {}).get("book_age_p95_sec") or {})
    return {
        "ok": True,
        "package_dir": str(package_dir),
        "events_size_mb": events_size_mb,
        "alerts_size_mb": alerts_size_mb,
        "events_size_mb_per_hour": (events_size_mb / duration_hours) if duration_hours > 0.0 else events_size_mb,
        "duration_sec": duration_sec,
        "ws_latency_summary_exists": ws_latency_path.is_file(),
        "disconnect_alert_count": len(disconnect_alerts),
        "reconnect_alert_count": len(reconnect_alerts),
        "book_age_p95_sec": _safe_float(book_age_summary.get("p95"), 0.0),
        "summary": summary,
    }


def collect_snapshot_checks(artifact_root: Path) -> dict[str, Any]:
    root = Path(artifact_root)
    manifest = load_snapshot_manifest(root)
    training_manifest = load_training_manifest(root)
    snapshots = list(manifest.get("snapshots") or [])
    pass_snapshots = [item for item in snapshots if str(item.get("certification_verdict") or "") == "PASS"]
    duplicate_ranges: list[dict[str, Any]] = []
    seen_ranges: dict[tuple[Any, Any], dict[str, Any]] = {}
    for snapshot in snapshots:
        record_time_range = snapshot.get("record_time_range") or {}
        key = (
            _safe_float(record_time_range.get("oldest_ts"), 0.0),
            _safe_float(record_time_range.get("newest_ts"), 0.0),
        )
        if key in seen_ranges:
            duplicate_ranges.append(
                {
                    "current": str(snapshot.get("filename") or ""),
                    "previous": str(seen_ranges[key].get("filename") or ""),
                    "range": {"oldest_ts": key[0], "newest_ts": key[1]},
                }
            )
        else:
            seen_ranges[key] = snapshot
    latest_state: dict[str, Any] = {}
    state_path = state_file_path(root)
    if state_path.is_file():
        try:
            latest_state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            latest_state = {}
    running_duration_sec = 0.0
    if str(latest_state.get("status") or "").lower() == "running":
        started_at_ts = _safe_float(latest_state.get("started_at_ts"), 0.0)
        if started_at_ts > 0.0:
            running_duration_sec = max(0.0, time.time() - started_at_ts)
    return {
        "ok": True,
        "artifact_root": str(root),
        "snapshot_manifest": manifest,
        "training_manifest": training_manifest,
        "total_snapshots": len(snapshots),
        "pass_snapshots": len(pass_snapshots),
        "latest_snapshot": latest_snapshot_entry(root),
        "latest_training_run_manifest": latest_training_run(root),
        "retrain_trigger": should_retrain(base_path=root),
        "duplicates": duplicate_ranges,
        "snapshots_missing_certification_or_sha": [
            str(item.get("filename") or "")
            for item in snapshots
            if not str(item.get("certification_verdict") or "") or not str(item.get("sha256") or "")
        ],
        "auto_retrain_state": latest_state,
        "auto_retrain_running_duration_sec": float(running_duration_sec),
    }


def audit_snapshot_labeling(
    snapshot_path: Path,
    *,
    sequence_length: int = 15,
    prediction_horizon_sec: int = 14_400,
    label_cost_floor_pct: float = 1.0,
    label_percentile: int = 70,
    label_episode_window_days: int = 5,
    selected_only: bool = False,
    closed_only: bool = True,
) -> dict[str, Any]:
    path = Path(snapshot_path)
    if not path.is_file():
        return {"ok": False, "error": f"snapshot db not found: {path}"}
    bundle = build_dataset_bundle(
        state_path=path,
        sequence_length=int(sequence_length),
        prediction_horizon_sec=int(prediction_horizon_sec),
        min_total_spread_pct=float(label_cost_floor_pct),
        label_cost_floor_pct=float(label_cost_floor_pct),
        label_percentile=int(label_percentile),
        label_episode_window_days=int(label_episode_window_days),
        selected_only=bool(selected_only),
        closed_only=bool(closed_only),
    )
    blocks, _, _ = _load_blocks_from_sqlite(
        path,
        selected_only=bool(selected_only),
        closed_only=bool(closed_only),
    )
    pair_segments, _ = _build_pair_segments(
        blocks,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=None,
    )
    segments_by_pair_session: dict[tuple[str, int], dict[str, Any]] = {}
    for segment in pair_segments:
        pair_id = str(segment.get("pair_id") or "")
        for session_id in list(segment.get("session_ids") or []):
            segments_by_pair_session[(pair_id, int(session_id))] = segment
    zero_leakage_violations: list[dict[str, Any]] = []
    fallback_samples = 0
    for pair_key, session_id, current_ts, threshold in zip(
        bundle.pair_ids,
        bundle.session_ids,
        bundle.timestamps,
        bundle.label_thresholds,
    ):
        segment = segments_by_pair_session.get((str(pair_key), int(session_id)))
        if segment is None:
            zero_leakage_violations.append(
                {
                    "pair_id": str(pair_key),
                    "session_id": int(session_id),
                    "timestamp": float(current_ts),
                    "expected_threshold": None,
                    "observed_threshold": float(threshold),
                    "prior_episode_support": None,
                    "error": "segment_not_found",
                }
            )
            if len(zero_leakage_violations) >= 10:
                break
            continue
        episodes = list(segment.get("episodes") or [])
        window_start = float(current_ts) - (int(label_episode_window_days) * 24 * 60 * 60)
        prior_episodes = [
            episode
            for episode in episodes
            if _safe_float(getattr(episode, "end_ts", 0.0), 0.0) <= float(current_ts)
            and _safe_float(getattr(episode, "end_ts", 0.0), 0.0) >= window_start
        ]
        if not prior_episodes:
            fallback_samples += 1
        expected_threshold = compute_label_threshold(
            pair_key,
            prior_episodes,
            cost_floor_pct=float(label_cost_floor_pct),
            percentile=float(label_percentile),
        )
        if abs(float(threshold) - float(expected_threshold)) > 1e-9:
            zero_leakage_violations.append(
                {
                    "pair_id": str(pair_key),
                    "session_id": int(session_id),
                    "timestamp": float(current_ts),
                    "expected_threshold": float(expected_threshold),
                    "observed_threshold": float(threshold),
                    "prior_episode_support": len(prior_episodes),
                }
            )
            if len(zero_leakage_violations) >= 10:
                break
    label_threshold_summary = dict(bundle.summary.get("label_thresholds") or {})
    elevated_pairs = {
        pair_id: payload
        for pair_id, payload in label_threshold_summary.items()
        if _safe_float((payload or {}).get("latest"), 0.0) > float(label_cost_floor_pct)
        or _safe_float((payload or {}).get("p50"), 0.0) > float(label_cost_floor_pct)
    }
    return {
        "ok": True,
        "snapshot_path": str(path),
        "summary": dict(bundle.summary),
        "positive_rate": _safe_float(bundle.summary.get("positive_rate"), 0.0),
        "label_threshold_mode": str(bundle.summary.get("label_threshold_mode") or ""),
        "label_thresholds": label_threshold_summary,
        "elevated_threshold_pairs": elevated_pairs,
        "fallback_cost_floor_samples": int(fallback_samples),
        "zero_leakage_ok": not zero_leakage_violations,
        "zero_leakage_violations": zero_leakage_violations,
        "feature_contract_version": DEFAULT_FEATURE_CONTRACT_VERSION if len(bundle.feature_names) == len(V2_MULTISCALE_FEATURE_NAMES) else "",
        "sample_count": int(bundle.summary.get("num_samples", 0)),
    }


def _extract_auc(result: dict[str, Any]) -> float:
    candidates = [
        result.get("roc_auc"),
        (result.get("metrics") or {}).get("test", {}).get("roc_auc"),
        (result.get("metrics") or {}).get("val_threshold", {}).get("roc_auc"),
        (result.get("artifact_metadata") or {}).get("test_metrics", {}).get("roc_auc"),
        (result.get("training") or {}).get("last_auc"),
    ]
    for candidate in candidates:
        value = _safe_float(candidate, float("nan"))
        if math.isfinite(value) and value > 0.0:
            return float(value)
    return 0.0


def validate_latest_run_payload(latest_run_payload: dict[str, Any], *, scanner_checkpoint: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(latest_run_payload or {})
    result = dict(payload.get("result") or {})
    artifact_metadata = dict(result.get("artifact_metadata") or {})
    training_config = dict(result.get("training_config") or artifact_metadata.get("training_config") or {})
    scanner_metrics = dict((scanner_checkpoint or {}).get("metrics") or {})
    platt_scale = _safe_float(
        (result.get("training") or {}).get("platt_scale", artifact_metadata.get("platt_scale", result.get("platt_scale"))),
        float("nan"),
    )
    platt_bias = _safe_float(
        (result.get("training") or {}).get("platt_bias", artifact_metadata.get("platt_bias", result.get("platt_bias"))),
        float("nan"),
    )
    feature_contract_version = str(
        artifact_metadata.get("feature_contract_version")
        or training_config.get("feature_contract_version")
        or result.get("feature_contract_version")
        or ""
    )
    label_threshold_mode = str(
        training_config.get("label_threshold_mode")
        or (result.get("dataset_summary") or {}).get("label_threshold_mode")
        or ""
    )
    return {
        "ok": bool(payload),
        "status": str(payload.get("status") or ""),
        "run_id": int(payload.get("id") or 0),
        "artifact_dir": str(payload.get("artifact_dir") or ""),
        "started_at": _safe_float(payload.get("started_at"), 0.0),
        "finished_at": _safe_float(payload.get("finished_at"), 0.0),
        "model_status": str(result.get("model_status") or ""),
        "auc": _extract_auc(result),
        "platt_scale": platt_scale,
        "platt_bias": platt_bias,
        "platt_finite": math.isfinite(platt_scale) and math.isfinite(platt_bias),
        "feature_contract_version": feature_contract_version,
        "label_threshold_mode": label_threshold_mode,
        "selected_threshold": _safe_float(training_config.get("selected_threshold", result.get("selected_threshold")), 0.0),
        "selected_label_config": dict(training_config.get("selected_label_config") or result.get("selected_label_config") or {}),
        "scanner_invalid_probability_count": int(scanner_metrics.get("invalid_probability_count", 0)),
        "scanner_probability_summary": dict(scanner_metrics.get("success_probability_summary") or {}),
        "signal_action_counts": dict(scanner_metrics.get("signal_action_counts") or {}),
    }


def _execute_signal_count(point: dict[str, Any]) -> int:
    counts = dict(point.get("signal_action_counts") or {})
    return int(counts.get("EXECUTE", 0) or 0) + int(counts.get("STRONG_EXECUTE", 0) or 0)


def _signal_anti_spike_gate(
    checkpoints: list[dict[str, Any]],
    *,
    deploy_ts: float,
    baseline_window_sec: int = 30 * 60,
    post_window_sec: int = 30 * 60,
) -> dict[str, Any]:
    if deploy_ts <= 0.0:
        return _bool_gate(
            "signal_gate_anti_spike",
            True,
            value={"applies": False, "reason": "deploy timestamp unavailable"},
            expected="no spike > 2x baseline over 30m",
            detail="No deploy timestamp was available in the latest training payload, so the anti-spike gate is informational only.",
        )
    series = [
        {
            "ts": _safe_float(item.get("ts"), 0.0),
            "execute_signals": _execute_signal_count(dict(item.get("metrics") or {})),
        }
        for item in checkpoints
        if isinstance(item, dict)
    ]
    baseline_points = [
        point
        for point in series
        if (deploy_ts - float(baseline_window_sec)) <= float(point["ts"]) < deploy_ts
    ]
    post_points = [
        point
        for point in series
        if deploy_ts <= float(point["ts"]) <= (deploy_ts + float(post_window_sec))
    ]
    if not baseline_points or not post_points:
        return _bool_gate(
            "signal_gate_anti_spike",
            True,
            value={
                "applies": False,
                "baseline_points": len(baseline_points),
                "post_points": len(post_points),
                "deploy_ts": deploy_ts,
            },
            expected="no spike > 2x baseline over 30m",
            detail="The checkpoint series does not bracket the deploy event with enough pre/post samples; review manually if needed.",
        )
    baseline_avg = float(statistics.fmean(float(point["execute_signals"]) for point in baseline_points))
    post_avg = float(statistics.fmean(float(point["execute_signals"]) for point in post_points))
    allowed = max(1.0, baseline_avg * 2.0)
    return _bool_gate(
        "signal_gate_anti_spike",
        post_avg <= allowed,
        value={
            "deploy_ts": deploy_ts,
            "baseline_avg_execute_signals": baseline_avg,
            "post_avg_execute_signals": post_avg,
            "baseline_points": len(baseline_points),
            "post_points": len(post_points),
            "allowed_post_avg": allowed,
        },
        expected="post-deploy average execute/strong_execute <= max(1, 2x baseline)",
    )


def _series_growth_pct_per_hour(points: list[dict[str, Any]], metric_key: str) -> float:
    if len(points) < 2:
        return 0.0
    first = _safe_float(points[0].get(metric_key), 0.0)
    last = _safe_float(points[-1].get(metric_key), 0.0)
    elapsed_sec = max(_safe_float(points[-1].get("ts"), 0.0) - _safe_float(points[0].get("ts"), 0.0), 1.0)
    hours = elapsed_sec / 3600.0
    if hours <= 0.0:
        return 0.0
    if abs(first) < 1e-9:
        return 0.0 if abs(last) < 1e-9 else float("inf")
    return ((last - first) / first) / hours * 100.0


def _slice_relative_window(points: list[dict[str, Any]], start_hour: float, end_hour: float) -> list[dict[str, Any]]:
    if not points:
        return []
    first_ts = _safe_float(points[0].get("ts"), 0.0)
    selected: list[dict[str, Any]] = []
    for point in points:
        offset_hours = (_safe_float(point.get("ts"), 0.0) - first_ts) / 3600.0
        if start_hour <= offset_hours < end_hour:
            selected.append(point)
    return selected


def _window_average(points: list[dict[str, Any]], metric_key: str) -> float:
    values = [_safe_float(point.get(metric_key), float("nan")) for point in points]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return 0.0
    return float(statistics.fmean(finite))


def evaluate_stage1(
    checkpoints: list[dict[str, Any]],
    *,
    sql_checks: dict[str, Any],
    feature_harness: dict[str, Any],
    runtime_audit_checks: dict[str, Any],
    warmup_sec: int = 15 * 60,
) -> dict[str, Any]:
    stage_points = [
        {"ts": float(item.get("ts", 0.0)), **dict(item.get("metrics") or {})}
        for item in checkpoints
        if isinstance(item, dict)
    ]
    http_points = [dict(item.get("http") or {}) for item in checkpoints if isinstance(item, dict)]
    stage_points.sort(key=lambda item: float(item.get("ts", 0.0)))
    if stage_points:
        warmup_cutoff = float(stage_points[0]["ts"]) + float(max(int(warmup_sec), 0))
        steady_state_points = [point for point in stage_points if float(point["ts"]) >= warmup_cutoff]
    else:
        steady_state_points = []
    latest_metrics = stage_points[-1] if stage_points else {}
    latest_hour = dict(sql_checks.get("latest_hourly_health") or {})
    runtime_summary = dict(runtime_audit_checks.get("summary") or {})
    latest_http = http_points[-1] if http_points else {}
    book_age_p95_sec = _safe_float(
        ((runtime_summary.get("dashboard_validation") or {}).get("book_age_p95_sec") or {}).get("p95"),
        0.0,
    )
    gates = [
        _bool_gate(
            "debug_perf_api",
            bool((latest_http.get("debug_perf") or {}).get("ok")),
            value=(latest_http.get("debug_perf") or {}).get("status"),
            expected="HTTP 200",
        ),
        _bool_gate(
            "system_health_api",
            bool((latest_http.get("system_health") or {}).get("ok")),
            value=(latest_http.get("system_health") or {}).get("status"),
            expected="HTTP 200",
        ),
        _bool_gate(
            "pipeline_status_api",
            bool((latest_http.get("pipeline_status") or {}).get("ok")),
            value=(latest_http.get("pipeline_status") or {}).get("status"),
            expected="HTTP 200",
        ),
        _bool_gate(
            "scanner_lite_api",
            bool((latest_http.get("scanner_lite") or {}).get("ok")),
            value=(latest_http.get("scanner_lite") or {}).get("status"),
            expected="HTTP 200",
        ),
        _bool_gate(
            "rss_growth_pct_per_hour_after_warmup",
            _series_growth_pct_per_hour(steady_state_points, "rss_mb") <= 5.0 if steady_state_points else False,
            value=round(_series_growth_pct_per_hour(steady_state_points, "rss_mb"), 6) if steady_state_points else None,
            expected="<= 5.0",
        ),
        _bool_gate(
            "calculate_ms_p95",
            _safe_float(latest_metrics.get("calculate_ms_p95"), float("inf")) < 50.0,
            value=latest_metrics.get("calculate_ms_p95"),
            expected="< 50",
        ),
        _bool_gate(
            "event_loop_lag_ms_p95",
            _safe_float(latest_metrics.get("event_loop_lag_ms_p95"), float("inf")) < 200.0,
            value=latest_metrics.get("event_loop_lag_ms_p95"),
            expected="< 200",
        ),
        _bool_gate(
            "event_loop_lag_ms_p99",
            _safe_float(latest_metrics.get("event_loop_lag_ms_p99"), float("inf")) < 500.0,
            value=latest_metrics.get("event_loop_lag_ms_p99"),
            expected="< 500",
        ),
        _bool_gate(
            "record_sink_idle_cycles",
            _safe_float(latest_metrics.get("tracker_records_enqueued_p50"), float("inf")) == 0.0
            and _safe_float(latest_metrics.get("tracker_rejections_enqueued_p50"), float("inf")) == 0.0,
            value={
                "tracker_records_enqueued_p50": latest_metrics.get("tracker_records_enqueued_p50"),
                "tracker_rejections_enqueued_p50": latest_metrics.get("tracker_rejections_enqueued_p50"),
                "tracker_records_enqueued_p95": latest_metrics.get("tracker_records_enqueued_p95"),
                "tracker_rejections_enqueued_p95": latest_metrics.get("tracker_rejections_enqueued_p95"),
            },
            expected="p50 values == 0",
        ),
        _bool_gate(
            "tracker_low_spread_capture",
            int(sql_checks.get("low_spread_records", 0)) > 0 and int(sql_checks.get("low_spread_pairs_history_enabled", 0)) > 0,
            value={
                "low_spread_records": sql_checks.get("low_spread_records"),
                "low_spread_pairs_history_enabled": sql_checks.get("low_spread_pairs_history_enabled"),
            },
            expected="records > 0 and pairs > 0",
        ),
        _bool_gate(
            "dashboard_filters_sub_0_2_spread",
            int(latest_metrics.get("dashboard_low_spread_count", 0)) == 0,
            value=latest_metrics.get("dashboard_low_spread_count"),
            expected="== 0",
        ),
        _bool_gate(
            "data_distribution_reaches_below_0_2",
            _safe_float(sql_checks.get("min_entry_spread_pct_recent"), 1.0) < 0.2,
            value=sql_checks.get("min_entry_spread_pct_recent"),
            expected="< 0.2",
        ),
        _bool_gate(
            "stale_pairs_absent",
            not list(sql_checks.get("stale_pairs") or []),
            value=len(list(sql_checks.get("stale_pairs") or [])),
            expected="0 stale pairs",
        ),
        _bool_gate(
            "cross_exchange_rejection_rate",
            _safe_float(latest_hour.get("rejection_rate_pct"), 0.0) < 10.0 if latest_hour else False,
            value=latest_hour.get("rejection_rate_pct") if latest_hour else None,
            expected="< 10%",
        ),
        _bool_gate(
            "feature_history_contract",
            bool(feature_harness.get("ok"))
            and int(feature_harness.get("feature_count", 0)) == len(V2_MULTISCALE_FEATURE_NAMES)
            and bool(feature_harness.get("cache_invalidated"))
            and bool(feature_harness.get("nonzero_multiscale_features")),
            value={
                "feature_count": feature_harness.get("feature_count"),
                "cache_invalidated": feature_harness.get("cache_invalidated"),
                "nonzero_multiscale_features": len(list(feature_harness.get("nonzero_multiscale_features") or [])),
            },
            expected="25 features, cache invalidates, multiscale variance > 0",
        ),
        _bool_gate(
            "hourly_digest_present",
            bool(latest_hour) and str(latest_hour.get("quality_verdict") or "") in {"healthy", "degraded", "unhealthy"},
            value=latest_hour.get("quality_verdict") if latest_hour else None,
            expected="tracker_hourly_health row present",
        ),
        _bool_gate(
            "runtime_audit_events_budget",
            bool(runtime_audit_checks.get("ok")) and _safe_float(runtime_audit_checks.get("events_size_mb_per_hour"), float("inf")) < 5.0,
            value=runtime_audit_checks.get("events_size_mb_per_hour"),
            expected="< 5 MB/hour",
        ),
        _bool_gate(
            "runtime_audit_ws_latency_summary",
            bool(runtime_audit_checks.get("ok")) and bool(runtime_audit_checks.get("ws_latency_summary_exists")),
            value=runtime_audit_checks.get("ws_latency_summary_exists"),
            expected="true",
        ),
        _bool_gate(
            "books_p95_under_5s",
            book_age_p95_sec > 0.0 and book_age_p95_sec < 5.0,
            value=book_age_p95_sec,
            expected="< 5.0 sec",
        ),
        _bool_gate(
            "disconnects_absent",
            int(runtime_audit_checks.get("disconnect_alert_count", 0)) == 0 and int(runtime_audit_checks.get("reconnect_alert_count", 0)) == 0,
            value={
                "disconnect_alert_count": runtime_audit_checks.get("disconnect_alert_count"),
                "reconnect_alert_count": runtime_audit_checks.get("reconnect_alert_count"),
            },
            expected="both == 0",
        ),
    ]
    return {
        "stage": "stage1",
        "ok": all(bool(gate["ok"]) for gate in gates),
        "gates": gates,
        "sample_count": len(stage_points),
        "steady_state_sample_count": len(steady_state_points),
        "latest_metrics": latest_metrics,
        "sql_checks": sql_checks,
        "feature_harness": feature_harness,
        "runtime_audit_checks": runtime_audit_checks,
    }


def evaluate_stage2(
    checkpoints: list[dict[str, Any]],
    *,
    snapshot_checks: dict[str, Any],
    memory_harness: dict[str, Any],
    labeling_audit: dict[str, Any] | None = None,
    post_retrain_checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_points = [
        {"ts": float(item.get("ts", 0.0)), **dict(item.get("metrics") or {})}
        for item in checkpoints
        if isinstance(item, dict)
    ]
    stage_points.sort(key=lambda item: float(item.get("ts", 0.0)))
    early_points = _slice_relative_window(stage_points, 1.0, 4.0)
    late_points = _slice_relative_window(stage_points, 20.0, 24.0)
    latest_metrics = stage_points[-1] if stage_points else {}
    latest_state = dict(snapshot_checks.get("auto_retrain_state") or {})
    retrain_status = str(latest_state.get("status") or "idle").lower()
    label_audit = dict(labeling_audit or {})
    retrain_checks = dict(post_retrain_checks or {})
    gates = [
        _bool_gate(
            "pass_snapshots",
            int(snapshot_checks.get("pass_snapshots", 0)) >= 3,
            value=snapshot_checks.get("pass_snapshots"),
            expected=">= 3",
        ),
        _bool_gate(
            "snapshot_certification_and_sha",
            not list(snapshot_checks.get("snapshots_missing_certification_or_sha") or []),
            value=list(snapshot_checks.get("snapshots_missing_certification_or_sha") or []),
            expected="no missing certification/sha",
        ),
        _bool_gate(
            "snapshot_idempotency",
            not list(snapshot_checks.get("duplicates") or []),
            value=len(list(snapshot_checks.get("duplicates") or [])),
            expected="0 duplicate time ranges",
        ),
        _bool_gate(
            "sqlite_retains_gt_8h",
            bool(memory_harness.get("has_disk_records_older_than_8h")),
            value={
                "disk_oldest_ts": memory_harness.get("disk_oldest_ts"),
                "reference_ts": memory_harness.get("reference_ts"),
            },
            expected="true",
        ),
        _bool_gate(
            "memory_drops_gt_13h",
            bool(memory_harness.get("ok")) and not bool(memory_harness.get("memory_kept_older_than_13h")),
            value=memory_harness.get("memory_kept_older_than_13h"),
            expected="false",
        ),
        _bool_gate(
            "rss_24h_under_2x_warmup",
            (_window_average(late_points, "rss_mb") / max(_window_average(early_points, "rss_mb"), 1.0)) < 2.0
            if early_points and late_points
            else False,
            value=(
                _window_average(late_points, "rss_mb") / max(_window_average(early_points, "rss_mb"), 1.0)
                if early_points and late_points
                else None
            ),
            expected="< 2.0x",
        ),
        _bool_gate(
            "window_sla_calculate_ms",
            (_window_average(late_points, "calculate_ms_p95") - _window_average(early_points, "calculate_ms_p95")) < 100.0
            if early_points and late_points
            else False,
            value={
                "early_h1_h4": _window_average(early_points, "calculate_ms_p95"),
                "late_h20_h24": _window_average(late_points, "calculate_ms_p95"),
            },
            expected="late window not materially degraded",
        ),
        _bool_gate(
            "window_sla_event_loop",
            (_window_average(late_points, "event_loop_lag_ms_p95") - _window_average(early_points, "event_loop_lag_ms_p95")) < 150.0
            if early_points and late_points
            else False,
            value={
                "early_h1_h4": _window_average(early_points, "event_loop_lag_ms_p95"),
                "late_h20_h24": _window_average(late_points, "event_loop_lag_ms_p95"),
            },
            expected="late window not materially degraded",
        ),
        _bool_gate(
            "auto_retrain_not_stuck",
            retrain_status != "running" or _safe_float(snapshot_checks.get("auto_retrain_running_duration_sec"), 0.0) <= (2 * 60 * 60),
            value={
                "status": retrain_status,
                "running_duration_sec": snapshot_checks.get("auto_retrain_running_duration_sec"),
            },
            expected="not running > 2h",
        ),
    ]
    if label_audit:
        gates.extend(
            [
                _bool_gate(
                    "label_threshold_mode",
                    str(label_audit.get("label_threshold_mode") or "") == "rolling_pair_percentile",
                    value=label_audit.get("label_threshold_mode"),
                    expected="rolling_pair_percentile",
                ),
                _bool_gate(
                    "positive_rate_band",
                    0.05 <= _safe_float(label_audit.get("positive_rate"), -1.0) <= 0.20,
                    value=label_audit.get("positive_rate"),
                    expected="between 0.05 and 0.20",
                ),
                _bool_gate(
                    "label_zero_leakage",
                    bool(label_audit.get("zero_leakage_ok")),
                    value=len(list(label_audit.get("zero_leakage_violations") or [])),
                    expected="no leakage violations",
                ),
            ]
        )
    if retrain_checks:
        deploy_ts = max(
            _safe_float(retrain_checks.get("finished_at"), 0.0),
            _safe_float(retrain_checks.get("started_at"), 0.0),
        )
        gates.extend(
            [
                _bool_gate("trained_model_auc", _safe_float(retrain_checks.get("auc"), 0.0) > 0.55, value=retrain_checks.get("auc"), expected="> 0.55"),
                _bool_gate("platt_scaling_finite", bool(retrain_checks.get("platt_finite")), value={"platt_scale": retrain_checks.get("platt_scale"), "platt_bias": retrain_checks.get("platt_bias")}, expected="finite"),
                _bool_gate(
                    "inference_probabilities_bounded",
                    int(retrain_checks.get("scanner_invalid_probability_count", 0)) == 0,
                    value=retrain_checks.get("scanner_invalid_probability_count"),
                    expected="== 0 invalid probabilities",
                ),
                _bool_gate(
                    "feature_contract_version",
                    str(retrain_checks.get("feature_contract_version") or "") in {DEFAULT_FEATURE_CONTRACT_VERSION, "v1_micro_10"},
                    value=retrain_checks.get("feature_contract_version"),
                    expected=f"{DEFAULT_FEATURE_CONTRACT_VERSION} or v1_micro_10",
                ),
                _bool_gate(
                    "training_metadata_threshold_mode",
                    str(retrain_checks.get("label_threshold_mode") or "") == "rolling_pair_percentile",
                    value=retrain_checks.get("label_threshold_mode"),
                    expected="rolling_pair_percentile",
                ),
                _signal_anti_spike_gate(checkpoints, deploy_ts=deploy_ts),
            ]
        )
    return {
        "stage": "stage2",
        "ok": all(bool(gate["ok"]) for gate in gates),
        "gates": gates,
        "sample_count": len(stage_points),
        "early_window_count": len(early_points),
        "late_window_count": len(late_points),
        "latest_metrics": latest_metrics,
        "snapshot_checks": snapshot_checks,
        "memory_harness": memory_harness,
        "labeling_audit": label_audit,
        "post_retrain_checks": retrain_checks,
    }


def build_soak_markdown_report(*, stage_result: dict[str, Any], checkpoints: list[dict[str, Any]]) -> str:
    stage_name = str(stage_result.get("stage") or "stage")
    lines = [
        f"# ArbML v2 Soak Report ({stage_name})",
        "",
        f"- Verdict: {'PASS' if stage_result.get('ok') else 'FAIL'}",
        f"- Samples: {int(stage_result.get('sample_count', 0))}",
        "",
        "## Gates",
    ]
    for gate in list(stage_result.get("gates") or []):
        verdict = "PASS" if gate.get("ok") else "FAIL"
        lines.append(
            f"- [{verdict}] {gate.get('name')}: value={json.dumps(gate.get('value'), ensure_ascii=True)} expected={gate.get('expected')}"
        )
        detail = str(gate.get("detail") or "").strip()
        if detail:
            lines.append(f"  detail: {detail}")
    if checkpoints:
        lines.extend(
            [
                "",
                "## Last Checkpoint",
                "```json",
                json.dumps(checkpoints[-1], indent=2, sort_keys=True),
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def write_soak_outputs(
    *,
    output_dir: Path,
    stage_result: dict[str, Any],
    checkpoints: list[dict[str, Any]],
) -> dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_path = target_dir / "summary.json"
    report_path = target_dir / "report.md"
    checkpoints_path = target_dir / "checkpoints.ndjson"
    _write_json(summary_path, stage_result)
    report_path.write_text(
        build_soak_markdown_report(stage_result=stage_result, checkpoints=checkpoints),
        encoding="utf-8",
    )
    checkpoints_path.unlink(missing_ok=True)
    for checkpoint in checkpoints:
        append_ndjson(checkpoints_path, checkpoint)
    return {
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "checkpoints_path": str(checkpoints_path),
    }
