from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .runtime_audit import _build_snapshot_psi_summary, create_sqlite_snapshot
from .train_model import run_clean_training_cycle

SNAPSHOT_SLOT_HOURS = (0, 8, 16)
STATE_FILE_NAME = "auto_retrain.state.json"
TRAINING_MANIFEST_NAME = "manifest.json"


def backend_out_root(base_path: Path | None = None) -> Path:
    return Path(base_path) if base_path is not None else Path(__file__).resolve().parent.parent.parent / "out"


def snapshots_dir(base_path: Path | None = None) -> Path:
    return backend_out_root(base_path) / "snapshots"


def training_dir(base_path: Path | None = None) -> Path:
    return backend_out_root(base_path) / "training"


def archive_dir(base_path: Path | None = None) -> Path:
    return backend_out_root(base_path) / "archive"


def config_dir(base_path: Path | None = None) -> Path:
    return backend_out_root(base_path) / "config"


def state_file_path(base_path: Path | None = None) -> Path:
    return config_dir(base_path) / STATE_FILE_NAME


def training_manifest_path(base_path: Path | None = None) -> Path:
    return training_dir(base_path) / TRAINING_MANIFEST_NAME


def snapshot_manifest_path(base_path: Path | None = None) -> Path:
    return snapshots_dir(base_path) / "manifest.json"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def next_snapshot_slot(now: datetime | None = None) -> datetime:
    current = (now or utc_now()).astimezone(timezone.utc)
    candidates = [
        current.replace(hour=hour, minute=0, second=0, microsecond=0)
        for hour in SNAPSHOT_SLOT_HOURS
        if current < current.replace(hour=hour, minute=0, second=0, microsecond=0)
    ]
    if candidates:
        return min(candidates)
    next_day = current + timedelta(days=1)
    return next_day.replace(hour=SNAPSHOT_SLOT_HOURS[0], minute=0, second=0, microsecond=0)


def current_snapshot_slot(now: datetime | None = None) -> datetime:
    current = (now or utc_now()).astimezone(timezone.utc)
    same_day_slots = [
        current.replace(hour=hour, minute=0, second=0, microsecond=0)
        for hour in SNAPSHOT_SLOT_HOURS
        if current >= current.replace(hour=hour, minute=0, second=0, microsecond=0)
    ]
    if same_day_slots:
        return max(same_day_slots)
    previous_day = current - timedelta(days=1)
    return previous_day.replace(hour=SNAPSHOT_SLOT_HOURS[-1], minute=0, second=0, microsecond=0)


def seconds_until_next_snapshot_slot(now: datetime | None = None) -> float:
    current = (now or utc_now()).astimezone(timezone.utc)
    return max((next_snapshot_slot(current) - current).total_seconds(), 0.0)


def certification_verdict_view(certification: dict[str, Any]) -> str:
    verdict = str(certification.get("verdict") or "").upper()
    if verdict == "CERTIFIED":
        return "PASS"
    if verdict == "CERTIFIED_WITH_WARNINGS":
        return "WARN"
    return "FAIL"


def is_trainable_snapshot(snapshot: dict[str, Any]) -> bool:
    return str(snapshot.get("certification_verdict") or "").upper() in {"PASS", "WARN"}


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sqlite_table_counts(sqlite_path: Path) -> dict[str, int]:
    with sqlite3.connect(sqlite_path, timeout=30.0) as conn:
        return {
            "tracker_records": int(conn.execute("SELECT COUNT(*) FROM tracker_records").fetchone()[0]),
            "tracker_pairs": int(conn.execute("SELECT COUNT(*) FROM tracker_pairs").fetchone()[0]),
            "tracker_pair_blocks": int(conn.execute("SELECT COUNT(*) FROM tracker_pair_blocks").fetchone()[0]),
            "tracker_pair_episodes": int(conn.execute("SELECT COUNT(*) FROM tracker_pair_episodes").fetchone()[0]),
        }


def snapshot_time_range(sqlite_path: Path) -> tuple[float, float]:
    with sqlite3.connect(sqlite_path, timeout=30.0) as conn:
        row = conn.execute("SELECT MIN(ts), MAX(ts) FROM tracker_records").fetchone()
    return float(row[0] or 0.0), float(row[1] or 0.0)


def snapshot_session_ids(sqlite_path: Path) -> list[int]:
    with sqlite3.connect(sqlite_path, timeout=30.0) as conn:
        rows = conn.execute("SELECT DISTINCT id FROM tracker_capture_sessions ORDER BY id ASC").fetchall()
    return [int(row[0]) for row in rows if int(row[0] or 0) > 0]


def load_snapshot_manifest(base_path: Path | None = None) -> dict[str, Any]:
    return read_json(snapshot_manifest_path(base_path), {"snapshots": [], "last_updated_utc": ""})


def load_training_manifest(base_path: Path | None = None) -> dict[str, Any]:
    return read_json(training_manifest_path(base_path), {"runs": [], "last_updated_utc": ""})


def latest_snapshot_entry(base_path: Path | None = None) -> dict[str, Any] | None:
    manifest = load_snapshot_manifest(base_path)
    snapshots = list(manifest.get("snapshots") or [])
    if not snapshots:
        return None
    return max(snapshots, key=lambda item: float(item.get("created_at_utc_ts", 0.0) or 0.0))


def latest_training_run(base_path: Path | None = None) -> dict[str, Any] | None:
    manifest = load_training_manifest(base_path)
    runs = list(manifest.get("runs") or [])
    if not runs:
        return None
    return max(runs, key=lambda item: float(item.get("created_at_ts", 0.0) or 0.0))


def update_snapshot_manifest(
    snapshot_path: Path,
    certification: dict[str, Any],
    *,
    base_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_snapshot_manifest(base_path)
    time_range = snapshot_time_range(snapshot_path)
    table_counts = sqlite_table_counts(snapshot_path)
    created_at = utc_now()
    entry = {
        "filename": snapshot_path.name,
        "path": str(snapshot_path),
        "created_at_utc": created_at.isoformat().replace("+00:00", "Z"),
        "created_at_utc_ts": created_at.timestamp(),
        "record_time_range": [float(time_range[0]), float(time_range[1])],
        "record_count": int(table_counts.get("tracker_records", 0)),
        "pair_count": int(table_counts.get("tracker_pairs", 0)),
        "block_count": int(table_counts.get("tracker_pair_blocks", 0)),
        "episode_count": int(table_counts.get("tracker_pair_episodes", 0)),
        "session_ids": snapshot_session_ids(snapshot_path),
        "certification_verdict": certification_verdict_view(certification),
        "certification_raw_verdict": str(certification.get("verdict") or ""),
        "certification_id": str(certification.get("certification_id") or ""),
        "size_bytes": int(snapshot_path.stat().st_size if snapshot_path.exists() else 0),
        "sha256": sha256_file(snapshot_path),
    }
    snapshots = [item for item in list(manifest.get("snapshots") or []) if str(item.get("filename") or "") != snapshot_path.name]
    snapshots.append(entry)
    snapshots.sort(key=lambda item: float(item.get("created_at_utc_ts", 0.0) or 0.0))
    manifest["snapshots"] = snapshots
    manifest["last_updated_utc"] = created_at.isoformat().replace("+00:00", "Z")
    write_json(snapshot_manifest_path(base_path), manifest)
    return manifest


def update_training_manifest(run_summary: dict[str, Any], *, base_path: Path | None = None) -> dict[str, Any]:
    manifest = load_training_manifest(base_path)
    runs = [item for item in list(manifest.get("runs") or []) if str(item.get("run_id") or "") != str(run_summary.get("run_id") or "")]
    runs.append(dict(run_summary))
    runs.sort(key=lambda item: float(item.get("created_at_ts", 0.0) or 0.0))
    manifest["runs"] = runs
    manifest["last_updated_utc"] = utc_now().isoformat().replace("+00:00", "Z")
    write_json(training_manifest_path(base_path), manifest)
    return manifest


def current_model_auc(model_dir: Path) -> float | None:
    metadata_path = Path(model_dir) / "best_lstm_model.meta.json"
    if not metadata_path.is_file():
        return None
    payload = read_json(metadata_path, {})
    metrics = dict(payload.get("test_metrics") or {})
    value = metrics.get("roc_auc")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def current_model_version(model_dir: Path) -> str:
    metadata_path = Path(model_dir) / "best_lstm_model.meta.json"
    payload = read_json(metadata_path, {})
    return str(payload.get("version") or "")


def current_psi_values(model_dir: Path, *, base_path: Path | None = None) -> dict[str, float]:
    metadata_path = Path(model_dir) / "best_lstm_model.meta.json"
    payload = read_json(metadata_path, {})
    latest_snapshot = latest_snapshot_entry(base_path)
    if latest_snapshot is None:
        return {}
    psi_summary = _build_snapshot_psi_summary(
        Path(str(latest_snapshot.get("path") or "")),
        artifact_dir=Path(model_dir),
        sequence_length=int(payload.get("sequence_length", 15) or 15),
        prediction_horizon_sec=int(payload.get("training_config", {}).get("prediction_horizon_sec", 14_400) or 14_400),
    )
    return {
        str(name): float(metrics.get("psi", 0.0) or 0.0)
        for name, metrics in dict(psi_summary.get("features") or {}).items()
        if isinstance(metrics, dict)
    }


def should_retrain(
    *,
    base_path: Path | None = None,
    model_dir: Path | None = None,
) -> str | None:
    manifest = load_snapshot_manifest(base_path)
    certified = [
        item
        for item in list(manifest.get("snapshots") or [])
        if is_trainable_snapshot(item)
    ]
    if not certified:
        return None
    last_run = latest_training_run(base_path)
    last_train_ts = float(last_run.get("finished_at_ts", 0.0) or last_run.get("created_at_ts", 0.0) or 0.0) if last_run else 0.0
    now_ts = time.time()
    if last_train_ts <= 0.0 and len(certified) >= 3:
        return "first_training"
    if last_train_ts > 0.0 and (now_ts - last_train_ts) > (7 * 86400):
        return "scheduled_weekly"
    if last_train_ts > 0.0:
        new_certified = [item for item in certified if float(item.get("created_at_utc_ts", 0.0) or 0.0) > last_train_ts]
        if len(new_certified) >= 3:
            return "new_data_available"
    if model_dir is not None:
        psi_values = current_psi_values(model_dir, base_path=base_path)
        if any(float(value) > 0.20 for value in psi_values.values()):
            return "drift_detected"
    return None


def _copy_with_sequence(
    target: sqlite3.Connection,
    table: str,
    columns: list[str],
    rows: list[tuple[Any, ...]],
) -> None:
    if not rows:
        return
    placeholders = ",".join("?" for _ in columns)
    target.executemany(
        f"INSERT OR IGNORE INTO {table}({','.join(columns)}) VALUES({placeholders})",
        rows,
    )


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (str(table_name),),
    ).fetchone()
    return row is not None


def merge_snapshots(selected_snapshots: list[dict[str, Any]], *, target: Path) -> Path:
    if not selected_snapshots:
        raise ValueError("No snapshots selected for merge")
    ordered = sorted(
        selected_snapshots,
        key=lambda item: float(item.get("created_at_utc_ts", 0.0) or 0.0),
    )
    newest = Path(ordered[-1]["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    create_sqlite_snapshot(newest, target)
    with sqlite3.connect(target, timeout=30.0) as merged:
        merged.execute("PRAGMA journal_mode=WAL")
        merged.execute("PRAGMA foreign_keys=OFF")
        merged.row_factory = sqlite3.Row
        for source_entry in reversed(ordered[:-1]):
            source_min_ts = float((source_entry.get("record_time_range") or [0.0, 0.0])[0] or 0.0)
            merged_min_ts = float(merged.execute("SELECT MIN(ts) FROM tracker_records").fetchone()[0] or 0.0)
            if merged_min_ts > 0.0 and source_min_ts >= merged_min_ts:
                continue
            with sqlite3.connect(Path(source_entry["path"]), timeout=30.0) as source:
                source.row_factory = sqlite3.Row
                merged.execute("BEGIN")
                pair_mapping: dict[int, int] = {}
                for row in source.execute(
                    """
                    SELECT id, symbol, buy_ex, buy_mt, sell_ex, sell_mt, last_state, last_seen_ts, last_crossover_ts, history_enabled
                    FROM tracker_pairs
                    ORDER BY id ASC
                    """
                ):
                    existing = merged.execute(
                        """
                        SELECT id FROM tracker_pairs
                        WHERE symbol = ? AND buy_ex = ? AND buy_mt = ? AND sell_ex = ? AND sell_mt = ?
                        """,
                        (row["symbol"], row["buy_ex"], row["buy_mt"], row["sell_ex"], row["sell_mt"]),
                    ).fetchone()
                    if existing is None:
                        merged.execute(
                            """
                            INSERT INTO tracker_pairs(symbol, buy_ex, buy_mt, sell_ex, sell_mt, last_state, last_seen_ts, last_crossover_ts, history_enabled)
                            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                row["symbol"],
                                row["buy_ex"],
                                row["buy_mt"],
                                row["sell_ex"],
                                row["sell_mt"],
                                int(row["last_state"]),
                                float(row["last_seen_ts"]),
                                float(row["last_crossover_ts"]),
                                int(row["history_enabled"]),
                            ),
                        )
                        existing = merged.execute(
                            """
                            SELECT id FROM tracker_pairs
                            WHERE symbol = ? AND buy_ex = ? AND buy_mt = ? AND sell_ex = ? AND sell_mt = ?
                            """,
                            (row["symbol"], row["buy_ex"], row["buy_mt"], row["sell_ex"], row["sell_mt"]),
                        ).fetchone()
                    pair_mapping[int(row["id"])] = int(existing["id"])
                _copy_with_sequence(
                    merged,
                    "tracker_capture_sessions",
                    [
                        "id",
                        "started_at",
                        "ended_at",
                        "status",
                        "record_interval_sec",
                        "tracking_window_sec",
                        "gap_threshold_sec",
                        "created_by",
                        "approved_for_training",
                        "excluded_reason",
                        "manual_override",
                        "notes",
                        "created_at",
                        "updated_at",
                    ],
                    [
                        tuple(row)
                        for row in source.execute(
                            """
                            SELECT id, started_at, ended_at, status, record_interval_sec, tracking_window_sec,
                                   gap_threshold_sec, created_by, approved_for_training, excluded_reason,
                                   manual_override, notes, created_at, updated_at
                            FROM tracker_capture_sessions
                            ORDER BY id ASC
                            """
                        )
                    ],
                )
                _copy_with_sequence(
                    merged,
                    "tracker_pair_blocks",
                    [
                        "id",
                        "pair_id",
                        "session_id",
                        "start_ts",
                        "end_ts",
                        "record_count",
                        "max_gap_sec",
                        "boundary_reason",
                        "selected_for_training",
                        "disabled_reason",
                        "manual_override",
                        "notes",
                        "is_open",
                        "created_at",
                        "updated_at",
                    ],
                    [
                        (
                            int(row["id"]),
                            int(pair_mapping.get(int(row["pair_id"]), 0) or 0),
                            int(row["session_id"]),
                            float(row["start_ts"]),
                            float(row["end_ts"]),
                            int(row["record_count"]),
                            float(row["max_gap_sec"]),
                            str(row["boundary_reason"]),
                            int(row["selected_for_training"]),
                            str(row["disabled_reason"] or ""),
                            int(row["manual_override"]),
                            str(row["notes"] or ""),
                            int(row["is_open"]),
                            float(row["created_at"]),
                            float(row["updated_at"]),
                        )
                        for row in source.execute(
                            """
                            SELECT id, pair_id, session_id, start_ts, end_ts, record_count, max_gap_sec, boundary_reason,
                                   selected_for_training, disabled_reason, manual_override, notes, is_open, created_at, updated_at
                            FROM tracker_pair_blocks
                            ORDER BY id ASC
                            """
                        )
                        if int(pair_mapping.get(int(row["pair_id"]), 0) or 0) > 0
                    ],
                )
                merged.executemany(
                    """
                    INSERT OR IGNORE INTO tracker_records(pair_id, ts, entry_spread_pct, exit_spread_pct, session_id, block_id)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            int(pair_mapping.get(int(row["pair_id"]), 0) or 0),
                            float(row["ts"]),
                            float(row["entry_spread_pct"]),
                            float(row["exit_spread_pct"]),
                            int(row["session_id"]) if row["session_id"] is not None else None,
                            int(row["block_id"]) if row["block_id"] is not None else None,
                        )
                        for row in source.execute(
                            """
                            SELECT pair_id, ts, entry_spread_pct, exit_spread_pct, session_id, block_id
                            FROM tracker_records
                            ORDER BY pair_id ASC, ts ASC
                            """
                        )
                        if int(pair_mapping.get(int(row["pair_id"]), 0) or 0) > 0
                    ],
                )
                merged.executemany(
                    """
                    INSERT OR IGNORE INTO tracker_events(pair_id, event_type, ts, session_id, block_id)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            int(pair_mapping.get(int(row["pair_id"]), 0) or 0),
                            str(row["event_type"]),
                            float(row["ts"]),
                            int(row["session_id"]) if row["session_id"] is not None else None,
                            int(row["block_id"]) if row["block_id"] is not None else None,
                        )
                        for row in source.execute(
                            """
                            SELECT pair_id, event_type, ts, session_id, block_id
                            FROM tracker_events
                            ORDER BY pair_id ASC, ts ASC
                            """
                        )
                        if int(pair_mapping.get(int(row["pair_id"]), 0) or 0) > 0
                    ],
                )
                merged.executemany(
                    """
                    INSERT OR IGNORE INTO tracker_pair_episodes(
                        pair_id, session_id, block_id, start_ts, peak_ts, end_ts, duration_sec,
                        peak_entry_spread, exit_spread_at_close, baseline_median, baseline_mad,
                        activation_threshold, release_threshold, source_version, is_closed
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            int(pair_mapping.get(int(row["pair_id"]), 0) or 0),
                            int(row["session_id"]),
                            int(row["block_id"]),
                            float(row["start_ts"]),
                            float(row["peak_ts"]),
                            float(row["end_ts"]),
                            float(row["duration_sec"]),
                            float(row["peak_entry_spread"]),
                            float(row["exit_spread_at_close"]),
                            float(row["baseline_median"]),
                            float(row["baseline_mad"]),
                            float(row["activation_threshold"]),
                            float(row["release_threshold"]),
                            str(row["source_version"]),
                            int(row["is_closed"]),
                        )
                        for row in source.execute(
                            """
                            SELECT pair_id, session_id, block_id, start_ts, peak_ts, end_ts, duration_sec,
                                   peak_entry_spread, exit_spread_at_close, baseline_median, baseline_mad,
                                   activation_threshold, release_threshold, source_version, is_closed
                            FROM tracker_pair_episodes
                            ORDER BY pair_id ASC, end_ts ASC
                            """
                        )
                        if int(pair_mapping.get(int(row["pair_id"]), 0) or 0) > 0
                    ],
                )
                if _table_exists(source, "tracker_hourly_health") and _table_exists(merged, "tracker_hourly_health"):
                    _copy_with_sequence(
                        merged,
                        "tracker_hourly_health",
                        [
                            "hour_start_ts",
                            "hour_end_ts",
                            "records_total",
                            "records_rejected",
                            "rejection_rate_pct",
                            "exchanges_active",
                            "exchanges_circuit_open_json",
                            "pairs_with_records",
                            "pairs_with_gaps",
                            "cross_exchange_flags",
                            "avg_book_age_json",
                            "episode_count",
                            "quality_verdict",
                            "created_at",
                        ],
                        [
                            tuple(row)
                            for row in source.execute(
                                """
                                SELECT hour_start_ts, hour_end_ts, records_total, records_rejected, rejection_rate_pct,
                                       exchanges_active, exchanges_circuit_open_json, pairs_with_records, pairs_with_gaps,
                                       cross_exchange_flags, avg_book_age_json, episode_count, quality_verdict, created_at
                                FROM tracker_hourly_health
                                ORDER BY hour_start_ts ASC
                                """
                            )
                        ],
                    )
                merged.commit()
        merged.execute("PRAGMA foreign_keys=ON")
    return target


def latest_snapshot_timestamp(*, base_path: Path | None = None) -> float | None:
    entry = latest_snapshot_entry(base_path)
    if entry is None:
        return None
    return float(entry.get("created_at_utc_ts", 0.0) or 0.0)


def archive_expired_snapshots(*, base_path: Path | None = None, retention_days: int = 14) -> list[str]:
    manifest = load_snapshot_manifest(base_path)
    cutoff = time.time() - (max(int(retention_days), 1) * 86400)
    moved: list[str] = []
    archive_root = archive_dir(base_path)
    for item in list(manifest.get("snapshots") or []):
        created_at_ts = float(item.get("created_at_utc_ts", 0.0) or 0.0)
        snapshot_path = Path(str(item.get("path") or ""))
        cert_path = snapshot_path.with_suffix(".cert.json")
        if created_at_ts <= 0.0 or created_at_ts >= cutoff or not snapshot_path.is_file():
            continue
        week_dir = archive_root / utc_now().strftime("%Y-W%W")
        week_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(snapshot_path), str(week_dir / snapshot_path.name))
        if cert_path.is_file():
            shutil.move(str(cert_path), str(week_dir / cert_path.name))
        item["path"] = str(week_dir / snapshot_path.name)
        moved.append(snapshot_path.name)
    if moved:
        write_json(snapshot_manifest_path(base_path), manifest)
    return moved


def _write_state(status: str, payload: dict[str, Any], *, base_path: Path | None = None) -> dict[str, Any]:
    data = dict(payload)
    data["status"] = str(status)
    data["updated_at_utc"] = utc_now().isoformat().replace("+00:00", "Z")
    data["updated_at_ts"] = time.time()
    write_json(state_file_path(base_path), data)
    return data


def _load_test_auc(result: dict[str, Any]) -> float:
    for path in (
        ("test_metrics", "roc_auc"),
        ("artifact_metadata", "test_metrics", "roc_auc"),
        ("training", "test_metrics", "roc_auc"),
    ):
        current: Any = result
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        try:
            if current is not None:
                return float(current)
        except (TypeError, ValueError):
            continue
    return 0.0


def launch_auto_retrain_worker(
    *,
    reason: str,
    tracker_db_path: Path,
    base_path: Path | None = None,
    window_days: int = 5,
) -> subprocess.Popen[str]:
    payload = {
        "status": "queued",
        "reason": str(reason),
        "created_at_utc": utc_now().isoformat().replace("+00:00", "Z"),
        "created_at_ts": time.time(),
        "tracker_db_path": str(tracker_db_path),
        "base_path": str(backend_out_root(base_path)),
    }
    write_json(state_file_path(base_path), payload)
    command = [
        sys.executable,
        "-m",
        "spread.auto_retrain",
        "--reason",
        str(reason),
        "--tracker-db",
        str(tracker_db_path),
        "--base-path",
        str(backend_out_root(base_path)),
        "--window-days",
        str(int(window_days)),
    ]
    return subprocess.Popen(command, cwd=str(Path(__file__).resolve().parent.parent))


def run_auto_retrain_worker(
    *,
    reason: str,
    tracker_db_path: Path,
    base_path: Path | None = None,
    window_days: int = 5,
) -> dict[str, Any]:
    snapshot_manifest = load_snapshot_manifest(base_path)
    cutoff = time.time() - (max(int(window_days), 1) * 86400)
    selected = [
        item
        for item in list(snapshot_manifest.get("snapshots") or [])
        if is_trainable_snapshot(item)
        and float(item.get("created_at_utc_ts", 0.0) or 0.0) >= cutoff
    ]
    state = _write_state(
        "running",
        {
            "reason": str(reason),
            "created_at_utc": utc_now().isoformat().replace("+00:00", "Z"),
            "created_at_ts": time.time(),
            "tracker_db_path": str(tracker_db_path),
            "base_path": str(backend_out_root(base_path)),
        },
        base_path=base_path,
    )
    if len(selected) < 3:
        result = {"status": "insufficient_snapshots", "reason": str(reason), "selected_snapshot_count": len(selected)}
        _write_state("idle", result, base_path=base_path)
        return result
    run_id = utc_now().strftime("run_%Y%m%d_%H%M%S")
    artifact_dir = training_dir(base_path) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    merged_db = merge_snapshots(selected, target=artifact_dir / "merged.sqlite")
    deploy_root = config_dir(base_path)
    champion_auc = current_model_auc(deploy_root)
    champion_version = current_model_version(deploy_root)
    try:
        report = run_clean_training_cycle(
            state_file=merged_db,
            artifact_dir=artifact_dir,
            sequence_length=15,
            prediction_horizon_sec=14_400,
            certification_mode="full",
        )
        challenger_auc = _load_test_auc(report)
        status = "deployed"
        deployed = False
        if champion_auc is not None and challenger_auc < (champion_auc - 0.05):
            status = "challenger_rejected"
        else:
            for suffix in (".pth", ".meta.json", ".report.json"):
                candidate = artifact_dir / f"best_lstm_model{suffix}"
                if candidate.is_file():
                    shutil.copy2(candidate, deploy_root / candidate.name)
            deployed = True
        summary = {
            "run_id": run_id,
            "reason": str(reason),
            "status": status,
            "created_at_ts": float(state.get("created_at_ts", time.time()) or time.time()),
            "finished_at_ts": time.time(),
            "artifact_dir": str(artifact_dir),
            "merged_db_path": str(merged_db),
            "selected_snapshot_count": len(selected),
            "selected_snapshots": [str(item.get("filename") or "") for item in selected],
            "challenger_auc": float(challenger_auc),
            "champion_auc": champion_auc,
            "champion_version": champion_version,
            "deployed": bool(deployed),
            "result": report,
        }
        update_training_manifest(summary, base_path=base_path)
        _write_state("idle", summary, base_path=base_path)
        return summary
    except Exception as exc:
        result = {
            "run_id": run_id,
            "reason": str(reason),
            "status": "failed",
            "created_at_ts": float(state.get("created_at_ts", time.time()) or time.time()),
            "finished_at_ts": time.time(),
            "artifact_dir": str(artifact_dir),
            "merged_db_path": str(merged_db),
            "selected_snapshot_count": len(selected),
            "error": str(exc),
        }
        update_training_manifest(result, base_path=base_path)
        _write_state("failed", result, base_path=base_path)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", required=True)
    parser.add_argument("--tracker-db", required=True)
    parser.add_argument("--base-path", default="")
    parser.add_argument("--window-days", type=int, default=5)
    args = parser.parse_args(argv)
    base_path = Path(args.base_path).resolve() if str(args.base_path or "").strip() else None
    run_auto_retrain_worker(
        reason=str(args.reason),
        tracker_db_path=Path(args.tracker_db).resolve(),
        base_path=base_path,
        window_days=int(args.window_days),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
