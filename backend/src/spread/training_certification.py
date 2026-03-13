from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .ml_dataset import (
    FEATURE_NAMES,
    _block_threshold_counts,
    _build_block_diagnostics,
    _load_blocks_from_sqlite,
    _quantile_summary,
    _sqlite_has_blocks,
    build_dataset_bundle,
)
from .spread_tracker import SpreadRecord, compute_closed_episodes

CHECKPOINT_WINDOW_SEC = 30 * 60
DEFAULT_CERTIFICATION_THRESHOLDS = [0.8, 1.0, 1.2]
DEFAULT_CERTIFICATION_LABEL_PERCENTILES = [60, 70, 80]
DEFAULT_DUAL_PREFLIGHT_CONFIGS = [
    {"sequence_length": 4, "prediction_horizon_sec": 240},
    {"sequence_length": 8, "prediction_horizon_sec": 600},
    {"sequence_length": 15, "prediction_horizon_sec": 14_400},
]
DEFAULT_RUNTIME_AUDIT_STALENESS_SEC = 24 * 60 * 60
GATE_02_MIN_INTERVAL_SEC = 10.0
GATE_02_MAX_INTERVAL_SEC = 40.0
GATE_03_MIN_RECORDS_FOR_COMPLETENESS = 10
GATE_03_DISAPPEARED_WARN_RATE = 0.20
GATE_03_DISAPPEARED_FAIL_RATE = 0.95
GATE_03_INTERMITTENT_WARN_RATE = 0.15
GATE_03_INTERMITTENT_FAIL_RATE = 0.60
GATE_06_MIN_EPISODE_DURATION_SEC = 30.0
GATE_12_MIN_RECORDS_PER_ACTIVE_HOUR = 10.0
GATE_12_OUTLIER_HOUR_RATE_THRESHOLD = 0.15
GATE_12_PAIR_QUALITY_FAIL_RATE = 0.15
GATE_12_EPISODE_OUTLIER_FAIL_RATE = 0.02


def _normalize_threshold_values(thresholds: list[float] | None) -> list[float]:
    values = [float(value) for value in (thresholds or DEFAULT_CERTIFICATION_THRESHOLDS)]
    normalized: list[float] = []
    seen: set[float] = set()
    for value in values:
        if not math.isfinite(value):
            continue
        numeric = float(value)
        if numeric in seen:
            continue
        seen.add(numeric)
        normalized.append(numeric)
    return normalized or [float(value) for value in DEFAULT_CERTIFICATION_THRESHOLDS]


def _normalize_label_percentiles(label_percentiles: list[int] | None) -> list[int]:
    values = [int(value) for value in (label_percentiles or DEFAULT_CERTIFICATION_LABEL_PERCENTILES)]
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        numeric = max(0, min(100, int(value)))
        if numeric in seen:
            continue
        seen.add(numeric)
        normalized.append(numeric)
    return normalized or [int(value) for value in DEFAULT_CERTIFICATION_LABEL_PERCENTILES]


def _adaptive_label_requested(
    *,
    thresholds: list[float] | None,
    label_percentiles: list[int] | None,
) -> bool:
    return label_percentiles is not None or thresholds is None


def _label_config_payload(label_config: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(label_config or {})
    percentile_raw = config.get("label_percentile")
    percentile = None if percentile_raw is None else int(percentile_raw)
    window_days_raw = config.get("label_episode_window_days")
    window_days = None if window_days_raw is None else int(window_days_raw)
    return {
        "threshold": float(config.get("threshold", config.get("cost_floor_pct", 0.0)) or 0.0),
        "cost_floor_pct": float(config.get("cost_floor_pct", config.get("threshold", 0.0)) or 0.0),
        "label_percentile": percentile,
        "label_episode_window_days": window_days,
        "label_threshold_mode": str(config.get("label_threshold_mode") or "fixed_threshold"),
    }


def _dataset_build_kwargs_for_label_config(label_config: dict[str, Any] | None) -> dict[str, Any]:
    payload = _label_config_payload(label_config)
    kwargs: dict[str, Any] = {"min_total_spread_pct": float(payload["cost_floor_pct"])}
    if payload["label_threshold_mode"] == "rolling_pair_percentile":
        kwargs.update(
            {
                "label_cost_floor_pct": float(payload["cost_floor_pct"]),
                "label_percentile": int(payload["label_percentile"]) if payload["label_percentile"] is not None else 70,
                "label_episode_window_days": int(payload["label_episode_window_days"]) if payload["label_episode_window_days"] is not None else 5,
            }
        )
    return kwargs


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return path


def _iter_ndjson(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_hourly_health_samples(
    state_path: Path,
    *,
    scope_start_ts: float | None = None,
    scope_end_ts: float | None = None,
) -> list[dict[str, Any]]:
    if not Path(state_path).is_file():
        return []
    where_clauses: list[str] = []
    params: list[float] = []
    scoped_start = float(scope_start_ts or 0.0)
    scoped_end = float(scope_end_ts or 0.0)
    if scoped_end > 0.0 and scoped_start > 0.0 and scoped_end >= scoped_start:
        # Keep only hourly digests that intersect the certification scope window.
        where_clauses.append("hour_end_ts >= ? AND hour_start_ts <= ?")
        params.extend([scoped_start, scoped_end])
    elif scoped_end > 0.0:
        where_clauses.append("hour_start_ts <= ?")
        params.append(scoped_end)
    elif scoped_start > 0.0:
        where_clauses.append("hour_end_ts >= ?")
        params.append(scoped_start)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    try:
        with sqlite3.connect(state_path, timeout=30.0) as conn:
            rows = list(
                conn.execute(
                    f"""
                    SELECT hour_start_ts, hour_end_ts, records_total, records_rejected, rejection_rate_pct,
                           exchanges_active, exchanges_circuit_open_json, pairs_with_records, pairs_with_gaps,
                           cross_exchange_flags, avg_book_age_json, episode_count, quality_verdict, created_at
                    FROM tracker_hourly_health
                    {where_sql}
                    ORDER BY hour_start_ts ASC
                    """,
                    params,
                )
            )
    except sqlite3.Error:
        return []
    payload: list[dict[str, Any]] = []
    for row in rows:
        payload.append(
            {
                "hour_start_ts": float(row[0] or 0.0),
                "hour_end_ts": float(row[1] or 0.0),
                "records_total": int(row[2] or 0),
                "records_rejected": int(row[3] or 0),
                "rejection_rate_pct": float(row[4] or 0.0),
                "exchanges_active": int(row[5] or 0),
                "exchanges_circuit_open": list(json.loads(str(row[6] or "[]"))),
                "pairs_with_records": int(row[7] or 0),
                "pairs_with_gaps": int(row[8] or 0),
                "cross_exchange_flags": int(row[9] or 0),
                "avg_book_age_sec": dict(json.loads(str(row[10] or "{}"))),
                "episode_count": int(row[11] or 0),
                "quality_verdict": str(row[12] or "healthy"),
                "created_at": float(row[13] or 0.0),
            }
        )
    return payload


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (float(percentile) / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[int(rank)])
    return float(ordered[lower] + ((ordered[upper] - ordered[lower]) * (rank - lower)))


def _median(values: Iterable[float]) -> float:
    return _percentile([float(value) for value in values], 50.0)


def _mad(values: Iterable[float], center: float | None = None) -> float:
    ordered = [float(value) for value in values]
    if not ordered:
        return 0.0
    base = float(center) if center is not None else _median(ordered)
    return _median([abs(value - base) for value in ordered])


def _score_shift(left: list[float], right: list[float]) -> dict[str, float]:
    if len(left) < 2 or len(right) < 2:
        return {"mean_delta": 0.0, "std_ratio": 0.0, "score": 0.0}
    left_np = np.asarray(left, dtype=float)
    right_np = np.asarray(right, dtype=float)
    left_std = float(np.std(left_np))
    right_std = float(np.std(right_np))
    mean_delta = abs(float(np.mean(left_np)) - float(np.mean(right_np))) / max(left_std, right_std, 1e-6)
    std_ratio = max(left_std, right_std, 1e-6) / max(min(left_std, right_std), 1e-6)
    return {"mean_delta": float(mean_delta), "std_ratio": float(std_ratio), "score": float(max(mean_delta, std_ratio))}


def _scaled_density_threshold(pair_count: int) -> float:
    return float(min(100.0, max(1.0, 200.0 / math.sqrt(max(int(pair_count), 1)))))


def _episode_outlier_rate(values: list[float]) -> tuple[float, float]:
    normalized = [float(value) for value in values if math.isfinite(float(value))]
    if not normalized:
        return 0.0, 0.0
    nonzero_values = [value for value in normalized if abs(value) > 1e-12]
    zero_rate = float((len(normalized) - len(nonzero_values)) / len(normalized))
    if len(nonzero_values) <= 1:
        return 0.0, zero_rate
    center = _median(nonzero_values)
    spread = max(_mad(nonzero_values, center), 0.01)
    outlier_rate = float(
        sum(1 for value in nonzero_values if abs(value - center) > 10.0 * spread)
        / max(len(nonzero_values), 1)
    )
    return outlier_rate, zero_rate


def _table_count(conn: sqlite3.Connection, table_name: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    if row is None or int(row[0]) <= 0:
        return 0
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def _placeholders(values: list[int]) -> str:
    return ",".join("?" for _ in values)


def _safe_in(
    conn: sqlite3.Connection,
    column_expr: str,
    values: Iterable[int],
    *,
    temp_table_prefix: str,
) -> tuple[str, list[int]]:
    normalized = sorted({int(value) for value in values if int(value) > 0})
    if not normalized:
        return "", []
    if len(normalized) <= 999:
        return f"{column_expr} IN ({_placeholders(normalized)})", normalized
    temp_table_name = f"temp_{temp_table_prefix}_{time.time_ns()}"
    conn.execute(f"CREATE TEMP TABLE {temp_table_name} (id INTEGER PRIMARY KEY)")
    conn.executemany(
        f"INSERT OR IGNORE INTO {temp_table_name}(id) VALUES (?)",
        ((value,) for value in normalized),
    )
    return f"{column_expr} IN (SELECT id FROM {temp_table_name})", []


def _fetch_scope_ids(
    conn: sqlite3.Connection,
    *,
    selected_session_ids: list[int] | None,
    selected_block_ids: list[int] | None,
    allow_legacy_sessions: bool,
) -> dict[str, Any]:
    if _table_count(conn, "tracker_capture_sessions") <= 0 or _table_count(conn, "tracker_pair_blocks") <= 0:
        return {
            "quality_fix_activated_at": 0.0,
            "effective_session_ids": [],
            "effective_block_ids": [],
            "legacy_session_ids": [],
            "legacy_sessions_excluded": 0,
            "effective_session_scope": {"mode": "legacy_or_unscoped", "session_ids": [], "block_ids": []},
            "runtime_session_rows": [],
            "hot_path_rejection_stats": {},
        }

    meta_rows = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM tracker_meta")}
    quality_fix_activated_at = float(meta_rows.get("quality_fix_activated_at") or 0.0)
    rejection_stats: dict[str, Any] = {}
    raw_rejection_payload = str(meta_rows.get("hot_path_rejection_stats") or "").strip()
    if raw_rejection_payload:
        try:
            parsed = json.loads(raw_rejection_payload)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            rejection_stats = parsed

    session_rows = list(
        conn.execute(
            """
            SELECT id, started_at, ended_at, status
            FROM tracker_capture_sessions
            ORDER BY started_at ASC, id ASC
            """
        )
    )
    all_session_ids = [int(row["id"]) for row in session_rows]
    legacy_session_ids = [
        int(row["id"])
        for row in session_rows
        if quality_fix_activated_at > 0.0 and float(row["started_at"] or 0.0) < quality_fix_activated_at
    ]

    if selected_block_ids:
        normalized_block_ids = sorted({int(block_id) for block_id in selected_block_ids if int(block_id) > 0})
        if normalized_block_ids:
            where_clause, where_params = _safe_in(
                conn,
                "id",
                normalized_block_ids,
                temp_table_prefix="selected_blocks",
            )
            block_rows = list(
                conn.execute(
                    f"""
                    SELECT id, session_id
                    FROM tracker_pair_blocks
                    WHERE {where_clause}
                    ORDER BY id ASC
                    """,
                    where_params,
                )
            )
            effective_block_ids = [int(row["id"]) for row in block_rows]
            effective_session_ids = sorted({int(row["session_id"]) for row in block_rows if int(row["session_id"]) > 0})
        else:
            effective_block_ids = []
            effective_session_ids = []
        scope_mode = "selected_blocks"
    elif selected_session_ids:
        effective_block_ids = []
        effective_session_ids = sorted({int(session_id) for session_id in selected_session_ids if int(session_id) > 0})
        scope_mode = "selected_sessions"
    elif allow_legacy_sessions:
        effective_block_ids = []
        effective_session_ids = list(all_session_ids)
        scope_mode = "all_sessions_override"
    else:
        effective_block_ids = []
        effective_session_ids = [
            int(row["id"])
            for row in session_rows
            if quality_fix_activated_at <= 0.0 or float(row["started_at"] or 0.0) >= quality_fix_activated_at
        ]
        scope_mode = "post_quality_fix"

    runtime_session_rows = [
        {
            "id": int(row["id"]),
            "started_at": float(row["started_at"] or 0.0),
            "ended_at": float(row["ended_at"] or 0.0),
            "status": str(row["status"] or ""),
        }
        for row in session_rows
        if int(row["id"]) in set(effective_session_ids)
    ]
    return {
        "quality_fix_activated_at": float(quality_fix_activated_at),
        "effective_session_ids": list(effective_session_ids),
        "effective_block_ids": list(effective_block_ids),
        "legacy_session_ids": list(legacy_session_ids),
        "legacy_sessions_excluded": int(len(set(legacy_session_ids) - set(effective_session_ids))),
        "effective_session_scope": {
            "mode": scope_mode,
            "session_ids": list(effective_session_ids),
            "block_ids": list(effective_block_ids),
        },
        "runtime_session_rows": runtime_session_rows,
        "hot_path_rejection_stats": rejection_stats,
    }


def collect_sqlite_integrity(
    db_path: Path,
    *,
    selected_session_ids: list[int] | None = None,
    selected_block_ids: list[int] | None = None,
) -> dict[str, object]:
    if not db_path.exists():
        return {
            "db_exists": False,
            "db_path": str(db_path),
            "db_size_bytes": 0,
            "table_counts": {},
            "scope": {"session_ids": [], "block_ids": []},
            "anomalies": {
                "zero_record_blocks": 0,
                "record_count_mismatches": 0,
                "range_mismatches": 0,
                "missing_event_blocks": 0,
                "open_blocks_after_close": 0,
            },
        }

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        table_counts = {
            table_name: _table_count(conn, table_name)
            for table_name in (
                "tracker_pairs",
                "tracker_records",
                "tracker_events",
                "tracker_pair_blocks",
                "tracker_capture_sessions",
                "tracker_pair_episodes",
                "ml_training_runs",
                "ml_training_run_blocks",
                "ml_training_run_sessions",
            )
        }
        block_filters: list[str] = []
        block_params: list[Any] = []
        normalized_session_ids = sorted({int(value) for value in selected_session_ids or [] if int(value) > 0})
        normalized_block_ids = sorted({int(value) for value in selected_block_ids or [] if int(value) > 0})
        if normalized_session_ids:
            clause, params = _safe_in(
                conn,
                "b.session_id",
                normalized_session_ids,
                temp_table_prefix="scope_sessions",
            )
            block_filters.append(clause)
            block_params.extend(params)
        if normalized_block_ids:
            clause, params = _safe_in(
                conn,
                "b.id",
                normalized_block_ids,
                temp_table_prefix="scope_blocks",
            )
            block_filters.append(clause)
            block_params.extend(params)
        where_clause = f"WHERE {' AND '.join(block_filters)}" if block_filters else ""
        block_rows = list(conn.execute(f"SELECT b.id, b.session_id FROM tracker_pair_blocks b {where_clause}", block_params)) if table_counts.get("tracker_pair_blocks", 0) else []
        scope_block_ids = [int(row["id"]) for row in block_rows]
        scope_session_ids = sorted({int(row["session_id"]) for row in block_rows if int(row["session_id"]) > 0})
        block_where_clause, block_where_params = _safe_in(
            conn,
            "b.id",
            scope_block_ids,
            temp_table_prefix="integrity_blocks",
        )
        block_where = f"WHERE {block_where_clause}" if block_where_clause else ""

        zero_record_blocks = int(
            conn.execute(
                f"SELECT COUNT(*) FROM tracker_pair_blocks b {block_where} {'AND' if block_where else 'WHERE'} b.record_count <= 0",
                block_where_params,
            ).fetchone()[0]
        ) if table_counts.get("tracker_pair_blocks", 0) else 0
        record_count_mismatches = int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT b.id
                    FROM tracker_pair_blocks b
                    LEFT JOIN tracker_records r ON r.block_id = b.id
                    {block_where}
                    GROUP BY b.id
                    HAVING COUNT(r.ts) != b.record_count
                )
                """,
                block_where_params,
            ).fetchone()[0]
        ) if table_counts.get("tracker_pair_blocks", 0) else 0
        range_mismatches = int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT b.id
                    FROM tracker_pair_blocks b
                    LEFT JOIN tracker_records r ON r.block_id = b.id
                    {block_where}
                    GROUP BY b.id
                    HAVING COUNT(r.ts) > 0
                       AND (ABS(MIN(r.ts) - b.start_ts) > 1e-9 OR ABS(MAX(r.ts) - b.end_ts) > 1e-9)
                )
                """,
                block_where_params,
            ).fetchone()[0]
        ) if table_counts.get("tracker_pair_blocks", 0) else 0
        missing_event_blocks = 0
        if table_counts.get("tracker_events", 0):
            event_filters: list[str] = []
            event_params: list[Any] = []
            if scope_session_ids:
                clause, params = _safe_in(
                    conn,
                    "e.session_id",
                    scope_session_ids,
                    temp_table_prefix="event_sessions",
                )
                event_filters.append(clause)
                event_params.extend(params)
            if scope_block_ids:
                clause, params = _safe_in(
                    conn,
                    "e.block_id",
                    scope_block_ids,
                    temp_table_prefix="event_blocks",
                )
                event_filters.append(f"(e.block_id IS NULL OR {clause})")
                event_params.extend(params)
            event_where = f"WHERE {' AND '.join(event_filters)}" if event_filters else ""
            missing_event_blocks = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) FROM (
                        SELECT e.rowid
                        FROM tracker_events e
                        LEFT JOIN tracker_pair_blocks b ON b.id = e.block_id
                        {event_where}
                          AND (
                            e.block_id IS NULL
                            OR b.id IS NULL
                            OR b.session_id != e.session_id
                            OR e.ts < b.start_ts
                            OR e.ts > b.end_ts
                          )
                    )
                    """,
                    event_params,
                ).fetchone()[0]
            )
        open_blocks_after_close = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM tracker_pair_blocks b
                JOIN tracker_capture_sessions s ON s.id = b.session_id
                {block_where}
                {'AND' if block_where else 'WHERE'} b.is_open = 1 AND s.status != 'open'
                """,
                block_where_params,
            ).fetchone()[0]
        ) if table_counts.get("tracker_pair_blocks", 0) else 0
    return {
        "db_exists": True,
        "db_path": str(db_path),
        "db_size_bytes": int(db_path.stat().st_size),
        "table_counts": table_counts,
        "scope": {"session_ids": scope_session_ids, "block_ids": scope_block_ids},
        "anomalies": {
            "zero_record_blocks": zero_record_blocks,
            "record_count_mismatches": record_count_mismatches,
            "range_mismatches": range_mismatches,
            "missing_event_blocks": missing_event_blocks,
            "open_blocks_after_close": open_blocks_after_close,
        },
    }


def _scope_records(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for block in blocks:
        pair_id = str(block.get("pair_id") or block.get("pair_key") or "")
        session_id = int(block.get("session_id") or 0)
        block_id = int(block.get("block_id") or 0)
        for index, record in enumerate(block.get("records", [])):
            if not isinstance(record, dict):
                continue
            records.append(
                {
                    "pair_id": pair_id,
                    "session_id": session_id,
                    "block_id": block_id,
                    "timestamp": float(record.get("timestamp", record.get("ts", index)) or 0.0),
                    "entry_spread": float(record.get("entry_spread", record.get("entry", record.get("entry_spread_pct", 0.0))) or 0.0),
                    "exit_spread": float(record.get("exit_spread", record.get("exit", record.get("exit_spread_pct", 0.0))) or 0.0),
                }
            )
    records.sort(key=lambda item: (float(item["timestamp"]), str(item["pair_id"]), int(item["block_id"])))
    return records


def _scope_episodes(blocks: list[dict[str, Any]]) -> list[Any]:
    episodes: list[Any] = []
    for block in blocks:
        block_id = int(block.get("block_id") or 0)
        session_id = int(block.get("session_id") or 0)
        block_records = [
            SpreadRecord(
                timestamp=float(record.get("timestamp", record.get("ts", index)) or 0.0),
                entry_spread_pct=float(record.get("entry_spread", record.get("entry", record.get("entry_spread_pct", 0.0))) or 0.0),
                exit_spread_pct=float(record.get("exit_spread", record.get("exit", record.get("exit_spread_pct", 0.0))) or 0.0),
                session_id=session_id,
                block_id=block_id,
            )
            for index, record in enumerate(block.get("records", []))
            if isinstance(record, dict)
        ]
        episodes.extend(compute_closed_episodes(block_records))
    return sorted(episodes, key=lambda item: (float(item.end_ts), float(item.start_ts)))


def _build_block_scope_clause(
    conn: sqlite3.Connection,
    *,
    selected_session_ids: list[int] | None,
    selected_block_ids: list[int] | None,
    selected_only: bool,
    closed_only: bool,
    block_alias: str = "b",
) -> tuple[str, list[Any]]:
    block_filter: list[str] = []
    params: list[Any] = []
    if selected_only:
        block_filter.append(f"{block_alias}.selected_for_training = 1")
    if closed_only:
        block_filter.append(f"{block_alias}.is_open = 0")
    if selected_session_ids is not None:
        normalized_session_ids = sorted({int(session_id) for session_id in selected_session_ids if int(session_id) > 0})
        if not normalized_session_ids:
            return "WHERE 1 = 0", []
        clause, clause_params = _safe_in(
            conn,
            f"{block_alias}.session_id",
            normalized_session_ids,
            temp_table_prefix=f"{block_alias}_scope_sessions",
        )
        block_filter.append(clause)
        params.extend(clause_params)
    if selected_block_ids is not None:
        normalized_block_ids = sorted({int(block_id) for block_id in selected_block_ids if int(block_id) > 0})
        if not normalized_block_ids:
            return "WHERE 1 = 0", []
        clause, clause_params = _safe_in(
            conn,
            f"{block_alias}.id",
            normalized_block_ids,
            temp_table_prefix=f"{block_alias}_scope_blocks",
        )
        block_filter.append(clause)
        params.extend(clause_params)
    return (f"WHERE {' AND '.join(block_filter)}" if block_filter else ""), params


def _stream_quick_sqlite_metrics(
    state_path: Path,
    *,
    selected_session_ids: list[int] | None,
    selected_block_ids: list[int] | None,
    selected_only: bool,
    closed_only: bool,
    sequence_length: int,
) -> dict[str, Any] | None:
    if Path(state_path).suffix.lower() == ".json" or not Path(state_path).exists():
        return None

    with sqlite3.connect(state_path, timeout=30.0) as conn:
        conn.row_factory = sqlite3.Row
        if not _sqlite_has_blocks(conn):
            return None

        where_clause, params = _build_block_scope_clause(
            conn,
            selected_session_ids=selected_session_ids,
            selected_block_ids=selected_block_ids,
            selected_only=selected_only,
            closed_only=closed_only,
        )

        summary_row = conn.execute(
            f"""
            SELECT MIN(r.ts) AS min_ts, MAX(r.ts) AS max_ts, COUNT(r.ts) AS record_count
            FROM tracker_pair_blocks b
            JOIN tracker_records r ON r.block_id = b.id
            {where_clause}
            """,
            params,
        ).fetchone()
        min_ts = float(summary_row["min_ts"] or 0.0) if summary_row is not None else 0.0
        max_ts = float(summary_row["max_ts"] or 0.0) if summary_row is not None else 0.0
        total_records = int(summary_row["record_count"] or 0) if summary_row is not None else 0

        record_counts: list[int] = []
        durations: list[float] = []
        median_intervals: list[float] = []
        max_to_median_ratios: list[float] = []
        irregular_block_count = 0
        session_metrics: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: {
                "record_counts": [],
                "durations": [],
                "median_intervals": [],
                "max_to_median_ratios": [],
                "irregular_flags": [],
            }
        )
        pair_ids: set[str] = set()
        pair_record_counts: Counter[str] = Counter()
        pair_checkpoint_presence: dict[str, set[int]] = defaultdict(set)
        episode_total_spreads: list[float] = []
        episode_durations: list[float] = []
        episode_peaks: list[float] = []
        episode_exits: list[float] = []
        pair_hour_records: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
        bilateral_zero_count = 0

        def _finalize_block(current_meta: dict[str, Any] | None, current_records: list[tuple[float, float, float]]) -> None:
            nonlocal irregular_block_count
            if current_meta is None:
                return
            record_count = int(current_meta.get("record_count") or len(current_records))
            start_ts = float(current_meta.get("start_ts") or 0.0)
            end_ts = float(current_meta.get("end_ts") or 0.0)
            if current_records and (start_ts <= 0.0 or end_ts <= 0.0):
                start_ts = float(current_records[0][0])
                end_ts = float(current_records[-1][0])
            block_duration = max(0.0, end_ts - start_ts)
            intervals = [
                float(curr[0]) - float(prev[0])
                for prev, curr in zip(current_records, current_records[1:])
            ]
            median_interval = float(np.median(np.asarray(intervals, dtype=float))) if intervals else 0.0
            max_interval = max(intervals) if intervals else 0.0
            max_to_median_ratio = (
                float(max_interval) / max(float(median_interval), 1e-6)
                if intervals and median_interval > 0.0
                else 0.0
            )
            is_irregular = bool(intervals and max_to_median_ratio > 3.0)
            if is_irregular:
                irregular_block_count += 1
            record_counts.append(record_count)
            durations.append(block_duration)
            if median_interval > 0.0:
                median_intervals.append(float(median_interval))
            if max_to_median_ratio > 0.0:
                max_to_median_ratios.append(float(max_to_median_ratio))
            session_id = int(current_meta.get("session_id") or 0)
            bucket = session_metrics[session_id]
            bucket["record_counts"].append(record_count)
            bucket["durations"].append(block_duration)
            if median_interval > 0.0:
                bucket["median_intervals"].append(float(median_interval))
            if max_to_median_ratio > 0.0:
                bucket["max_to_median_ratios"].append(float(max_to_median_ratio))
            bucket["irregular_flags"].append(1.0 if is_irregular else 0.0)
            if not current_records:
                return
            spread_records = [
                SpreadRecord(
                    timestamp=float(timestamp),
                    entry_spread_pct=float(entry_spread),
                    exit_spread_pct=float(exit_spread),
                    session_id=session_id,
                    block_id=int(current_meta.get("block_id") or 0),
                )
                for timestamp, entry_spread, exit_spread in current_records
            ]
            for episode in compute_closed_episodes(spread_records):
                episode_total_spreads.append(float(getattr(episode, "total_spread", 0.0) or 0.0))
                episode_durations.append(float(getattr(episode, "duration_sec", 0.0) or 0.0))
                episode_peaks.append(float(getattr(episode, "peak_entry_spread", 0.0) or 0.0))
                episode_exits.append(float(getattr(episode, "exit_spread_at_close", 0.0) or 0.0))

        block_cursor = conn.execute(
            f"""
            SELECT
                b.id AS block_id,
                b.session_id,
                b.start_ts,
                b.end_ts,
                b.record_count,
                p.symbol,
                p.buy_ex,
                p.buy_mt,
                p.sell_ex,
                p.sell_mt,
                r.ts,
                r.entry_spread_pct,
                r.exit_spread_pct
            FROM tracker_pair_blocks b
            JOIN tracker_pairs p ON p.id = b.pair_id
            LEFT JOIN tracker_records r ON r.block_id = b.id
            {where_clause}
            ORDER BY b.id ASC, r.ts ASC
            """,
            params,
        )

        current_block_id = 0
        current_meta: dict[str, Any] | None = None
        current_records: list[tuple[float, float, float]] = []
        for row in block_cursor:
            block_id = int(row["block_id"] or 0)
            if current_block_id and block_id != current_block_id:
                _finalize_block(current_meta, current_records)
                current_records = []
                current_meta = None
            if current_meta is None:
                pair_key = f"{row['symbol']}|{row['buy_ex']}|{row['buy_mt']}|{row['sell_ex']}|{row['sell_mt']}"
                current_meta = {
                    "block_id": block_id,
                    "session_id": int(row["session_id"] or 0),
                    "pair_id": pair_key,
                    "start_ts": float(row["start_ts"] or 0.0),
                    "end_ts": float(row["end_ts"] or 0.0),
                    "record_count": int(row["record_count"] or 0),
                }
                current_block_id = block_id
            if row["ts"] is None:
                continue
            ts_value = float(row["ts"] or 0.0)
            entry_value = float(row["entry_spread_pct"] or 0.0)
            exit_value = float(row["exit_spread_pct"] or 0.0)
            current_records.append((ts_value, entry_value, exit_value))
            pair_id = str(current_meta["pair_id"])
            pair_ids.add(pair_id)
            pair_record_counts[pair_id] += 1
            if min_ts > 0.0:
                pair_checkpoint_presence[pair_id].add(int((ts_value - min_ts) // CHECKPOINT_WINDOW_SEC))
            pair_hour_records[(pair_id, int(ts_value // 3600))].append((ts_value, entry_value, exit_value))
            if abs(entry_value) <= 1e-12 and abs(exit_value) <= 1e-12:
                bilateral_zero_count += 1
        _finalize_block(current_meta, current_records)

        pair_hour_counts: Counter[str] = Counter()
        frozen_entry_hours: Counter[str] = Counter()
        frozen_exit_hours: Counter[str] = Counter()
        unresponsive_exit_hours: Counter[str] = Counter()
        entry_outlier_hours: Counter[str] = Counter()
        exit_outlier_hours: Counter[str] = Counter()
        pearson_by_pair: dict[str, float] = {}
        def _finalize_pair_hour(
            pair_id: str | None,
            entry_values: list[float],
            exit_values: list[float],
        ) -> None:
            if not pair_id:
                return
            pair_hour_counts[pair_id] += 1
            entry_std = float(np.std(np.asarray(entry_values, dtype=float))) if entry_values else 0.0
            exit_std = float(np.std(np.asarray(exit_values, dtype=float))) if exit_values else 0.0
            if entry_std <= 1e-4 or _zero_return_ratio(entry_values) >= 0.80:
                frozen_entry_hours[pair_id] += 1
            if exit_std <= 1e-4 or _zero_return_ratio(exit_values) >= 0.80:
                frozen_exit_hours[pair_id] += 1
            if entry_std > 1e-4 and exit_std <= 1e-4:
                unresponsive_exit_hours[pair_id] += 1
            entry_median = _median(entry_values)
            exit_median = _median(exit_values)
            entry_mad = max(_mad(entry_values, entry_median), 0.01)
            exit_mad = max(_mad(exit_values, exit_median), 0.01)
            if entry_values and sum(1 for value in entry_values if abs(value - entry_median) > 10.0 * entry_mad) / len(entry_values) > 0.05:
                entry_outlier_hours[pair_id] += 1
            if exit_values and sum(1 for value in exit_values if abs(value - exit_median) > 10.0 * exit_mad) / len(exit_values) > 0.05:
                exit_outlier_hours[pair_id] += 1
            if len(entry_values) >= 2 and len(exit_values) >= 2:
                if float(np.std(np.asarray(entry_values, dtype=float))) <= 1e-12 or float(np.std(np.asarray(exit_values, dtype=float))) <= 1e-12:
                    pearson_by_pair[pair_id] = 0.0
                else:
                    pearson_by_pair[pair_id] = float(np.corrcoef(np.asarray(entry_values), np.asarray(exit_values))[0, 1])

        for (pair_id, _hour_bucket), bucket_records in pair_hour_records.items():
            bucket_records.sort(key=lambda item: item[0])
            _finalize_pair_hour(
                pair_id,
                [float(item[1]) for item in bucket_records],
                [float(item[2]) for item in bucket_records],
            )

    session_summaries: dict[str, Any] = {}
    eligible_sessions = 0
    for session_id, metrics in sorted(session_metrics.items()):
        session_record_counts = [int(value) for value in metrics["record_counts"]]
        session_threshold_counts = _block_threshold_counts(session_record_counts)
        eligible_block_count = int(sum(1 for record_count in session_record_counts if record_count >= int(sequence_length)))
        if eligible_block_count > 0:
            eligible_sessions += 1
        session_summaries[str(session_id)] = {
            "num_blocks": len(session_record_counts),
            "record_count_quantiles": _quantile_summary(session_record_counts),
            "duration_sec_quantiles": _quantile_summary([float(value) for value in metrics["durations"]]),
            "inter_record_interval_sec_quantiles": _quantile_summary([float(value) for value in metrics["median_intervals"]]),
            "max_to_median_interval_ratio_quantiles": _quantile_summary([float(value) for value in metrics["max_to_median_ratios"]]),
            "irregular_block_count": int(sum(1 for value in metrics["irregular_flags"] if value > 0.0)),
            "max_record_count": max(session_record_counts) if session_record_counts else 0,
            "record_count_threshold_counts": session_threshold_counts,
            "eligible_blocks_for_sequence_length": eligible_block_count,
        }

    checkpoint_count = int(((max_ts - min_ts) // CHECKPOINT_WINDOW_SEC) + 1) if total_records > 0 and max_ts >= min_ts else 0
    return {
        "min_ts": float(min_ts),
        "max_ts": float(max_ts),
        "pair_ids": sorted(pair_ids),
        "pair_record_counts": pair_record_counts,
        "pair_checkpoint_presence": pair_checkpoint_presence,
        "checkpoint_count": int(checkpoint_count),
        "records_total": int(total_records),
        "episode_total_spreads": episode_total_spreads,
        "episode_durations": episode_durations,
        "episode_peaks": episode_peaks,
        "episode_exits": episode_exits,
        "pair_hour_counts": pair_hour_counts,
        "frozen_entry_hours": frozen_entry_hours,
        "frozen_exit_hours": frozen_exit_hours,
        "unresponsive_exit_hours": unresponsive_exit_hours,
        "entry_outlier_hours": entry_outlier_hours,
        "exit_outlier_hours": exit_outlier_hours,
        "pearson_by_pair": pearson_by_pair,
        "bilateral_zero_rate": float(bilateral_zero_count / max(total_records, 1)),
        "num_blocks": int(len(record_counts)),
        "block_diagnostics": {
            "record_count_quantiles": _quantile_summary(record_counts),
            "duration_sec_quantiles": _quantile_summary(durations),
            "inter_record_interval_sec_quantiles": _quantile_summary(median_intervals),
            "max_to_median_interval_ratio_quantiles": _quantile_summary(max_to_median_ratios),
            "irregular_block_count": int(irregular_block_count),
            "record_count_threshold_counts": _block_threshold_counts(record_counts),
            "feature_window_feasibility": {
                "sequence_length": int(sequence_length),
                "eligible_blocks_for_sequence_length": int(sum(1 for value in record_counts if int(value) >= int(sequence_length))),
                "eligible_sessions_with_any_eligible_block": int(eligible_sessions),
            },
            "session_summaries": session_summaries,
        },
    }


def _session_duration_hours(session_rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> float:
    total_sec = 0.0
    for row in session_rows:
        started_at = float(row.get("started_at") or 0.0)
        ended_at = float(row.get("ended_at") or 0.0)
        if ended_at > started_at > 0.0:
            total_sec += ended_at - started_at
    if total_sec > 0.0:
        return total_sec / 3600.0
    if len(records) >= 2:
        return max((float(records[-1]["timestamp"]) - float(records[0]["timestamp"])) / 3600.0, 1.0 / 3600.0)
    return 1.0 / 3600.0


def _checkpoint_buckets(records: list[dict[str, Any]], window_sec: int = CHECKPOINT_WINDOW_SEC) -> list[dict[str, Any]]:
    if not records:
        return []
    first_ts = float(records[0]["timestamp"])
    start_anchor = math.floor(first_ts / float(window_sec)) * float(window_sec)
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        bucket_index = int((float(record["timestamp"]) - start_anchor) // float(window_sec))
        buckets[bucket_index].append(record)
    result: list[dict[str, Any]] = []
    for bucket_index in sorted(buckets):
        bucket_records = sorted(buckets[bucket_index], key=lambda item: float(item["timestamp"]))
        result.append(
            {
                "index": int(bucket_index),
                "start_ts": float(start_anchor + bucket_index * float(window_sec)),
                "end_ts": float(start_anchor + (bucket_index + 1) * float(window_sec)),
                "records": bucket_records,
                "pairs": sorted({str(item["pair_id"]) for item in bucket_records}),
            }
        )
    return result


def _pair_hour_buckets(records: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        hour_bucket = int(float(record["timestamp"]) // 3600)
        buckets[(str(record["pair_id"]), hour_bucket)].append(record)
    return buckets


def _zero_return_ratio(values: list[float]) -> float:
    if len(values) <= 1:
        return 1.0
    zero_returns = sum(1 for left, right in zip(values, values[1:]) if abs(float(right) - float(left)) <= 1e-9)
    return float(zero_returns / max(len(values) - 1, 1))


def _build_gate(
    gate_id: str,
    title: str,
    *,
    status: str,
    details: dict[str, Any],
    failure_reasons: list[str] | None = None,
    warnings: list[str] | None = None,
    recommendations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "title": title,
        "status": status,
        "failure_reasons": list(dict.fromkeys(failure_reasons or [])),
        "warnings": list(dict.fromkeys(warnings or [])),
        "recommendations": list(dict.fromkeys(recommendations or [])),
        "details": details,
    }


def _find_runtime_audit_package(runtime_audit_dir: Path | None, *, state_path: Path | None = None) -> dict[str, Any]:
    candidates: list[Path] = []
    if runtime_audit_dir is not None:
        base = Path(runtime_audit_dir)
        if base.is_file() and base.name == "summary.json":
            candidates.append(base.parent)
        elif (base / "summary.json").is_file():
            candidates.append(base)
    if not candidates:
        if state_path is not None:
            resolved_state_path = Path(state_path).resolve()
            project_root = Path(__file__).resolve().parent.parent.parent
            if resolved_state_path.parent.name == "config":
                default_root = resolved_state_path.parent.parent / "runtime_audit"
            elif resolved_state_path.is_relative_to(project_root):
                default_root = project_root / "out" / "runtime_audit"
            else:
                default_root = resolved_state_path.parent / "runtime_audit"
        else:
            default_root = Path(__file__).resolve().parent.parent.parent / "out" / "runtime_audit"
        if default_root.is_dir():
            candidates.extend(path.parent for path in default_root.glob("**/summary.json"))
    candidates = [path for path in candidates if (path / "summary.json").is_file()]
    if not candidates:
        return {"path": "", "summary": {}, "events": [], "alerts": [], "samples": []}
    package_dir = max(candidates, key=lambda item: (item.stat().st_mtime, str(item)))
    return {
        "path": str(package_dir),
        "summary": json.loads((package_dir / "summary.json").read_text(encoding="utf-8")),
        "events": _iter_ndjson(package_dir / "events.ndjson"),
        "alerts": _iter_ndjson(package_dir / "alerts.ndjson"),
        "samples": _iter_ndjson(package_dir / "samples.ndjson"),
    }


def run_training_certification(
    *,
    state_file: Path,
    artifact_dir: Path,
    sequence_length: int,
    prediction_horizon_sec: int,
    thresholds: list[float] | None = None,
    label_percentiles: list[int] | None = None,
    selected_session_ids: list[int] | None = None,
    selected_block_ids: list[int] | None = None,
    allow_cross_session_merge: bool = False,
    max_session_gap_sec: float | None = None,
    regime_shift_score_threshold: float | None = 3.0,
    certification_mode: str = "full",
    max_certification_duration_sec: int = 300,
    label_episode_window_days: int = 5,
    allow_legacy_sessions: bool = False,
    runtime_audit_dir: Path | None = None,
    run_reconnection_stress: bool = False,
    preflight_fn: Callable[..., dict[str, Any]],
    dataset_fingerprint_fn: Callable[..., str],
) -> dict[str, Any]:
    state_path = Path(state_file)
    artifact_root = Path(artifact_dir)
    gate_dir = artifact_root / "gate_results"
    started_at = time.perf_counter()
    threshold_values = _normalize_threshold_values(thresholds)
    adaptive_label_mode = _adaptive_label_requested(thresholds=thresholds, label_percentiles=label_percentiles)
    effective_label_percentiles = _normalize_label_percentiles(label_percentiles) if adaptive_label_mode else []
    operational_min_total_spread_pct = min(threshold_values) if threshold_values else 0.0

    with sqlite3.connect(state_path, timeout=30.0) as conn:
        conn.row_factory = sqlite3.Row
        scope = _fetch_scope_ids(
            conn,
            selected_session_ids=selected_session_ids,
            selected_block_ids=selected_block_ids,
            allow_legacy_sessions=allow_legacy_sessions,
        )

    effective_session_ids = list(scope["effective_session_ids"])
    effective_block_ids = list(scope["effective_block_ids"])
    runtime_session_rows = list(scope["runtime_session_rows"])
    rejection_stats = dict(scope["hot_path_rejection_stats"])
    runtime_package = (
        {"path": "", "summary": {}, "events": [], "alerts": [], "samples": []}
        if str(certification_mode).lower() == "quick"
        else _find_runtime_audit_package(runtime_audit_dir, state_path=state_path)
    )
    quick_sqlite_metrics = (
        _stream_quick_sqlite_metrics(
            state_path,
            selected_block_ids=effective_block_ids or None,
            selected_session_ids=effective_session_ids or None,
            selected_only=False,
            closed_only=True,
            sequence_length=sequence_length,
        )
        if str(certification_mode).lower() == "quick"
        else None
    )
    if quick_sqlite_metrics is None:
        blocks, _, _ = _load_blocks_from_sqlite(
            state_path,
            selected_block_ids=effective_block_ids or None,
            selected_session_ids=effective_session_ids or None,
            selected_only=False,
            closed_only=True,
        )
        records = _scope_records(blocks)
        episodes = _scope_episodes(blocks)
        checkpoint_buckets = _checkpoint_buckets(records)
        pair_ids = sorted({str(record["pair_id"]) for record in records if str(record["pair_id"])})
        block_diagnostics = _build_block_diagnostics(blocks, sequence_length=sequence_length)
    else:
        blocks = []
        records = []
        episodes = []
        checkpoint_buckets = []
        pair_ids = list(quick_sqlite_metrics["pair_ids"])
        block_diagnostics = dict(quick_sqlite_metrics["block_diagnostics"])
    session_hours = _session_duration_hours(runtime_session_rows, records)
    gate_results: dict[str, Any] = {}
    dataset_fingerprints: dict[str, str] = {}

    def _check_timeout() -> None:
        if (time.perf_counter() - started_at) > float(max_certification_duration_sec):
            raise TimeoutError("certification_timeout")

    def _persist_gate(gate: dict[str, Any]) -> dict[str, Any]:
        gate_results[str(gate["gate_id"])] = gate
        write_json(gate_dir / f"{gate['gate_id']}.json", gate)
        return gate

    def _skip_gate(gate_id: str, title: str, reason: str) -> dict[str, Any]:
        return _persist_gate(_build_gate(gate_id, title, status="SKIPPED", details={"reason": reason}, warnings=[reason] if reason == "runtime_audit_unavailable" else []))

    try:
        integrity = collect_sqlite_integrity(state_path, selected_session_ids=effective_session_ids or None, selected_block_ids=effective_block_ids or None)
        anomalies = dict(integrity.get("anomalies", {}))
        structural_integrity_keys = ("record_count_mismatches", "range_mismatches", "open_blocks_after_close")
        integrity_failures = [key for key in structural_integrity_keys if int(anomalies.get(key, 0) or 0) > 0]
        integrity_warnings = [
            key
            for key in ("zero_record_blocks", "missing_event_blocks")
            if int(anomalies.get(key, 0) or 0) > 0
        ]
        integrity_status = "FAIL" if integrity_failures else ("WARNING" if integrity_warnings else "PASS")
        _persist_gate(
            _build_gate(
                "gate_01_sqlite_integrity",
                "SQLite integrity",
                status=integrity_status,
                failure_reasons=["integrity_anomaly"] if integrity_failures else [],
                warnings=integrity_warnings,
                details={
                    **integrity,
                    "structural_integrity_failures": integrity_failures,
                    "integrity_warnings": integrity_warnings,
                },
            )
        )

        _check_timeout()
        bundle = (
            build_dataset_bundle(
                state_path=state_path,
                sequence_length=sequence_length,
                prediction_horizon_sec=prediction_horizon_sec,
                selected_block_ids=effective_block_ids or None,
                selected_session_ids=effective_session_ids or None,
                selected_only=False,
                closed_only=True,
                allow_cross_session_merge=allow_cross_session_merge,
                max_session_gap_sec=max_session_gap_sec,
                regime_shift_score_threshold=regime_shift_score_threshold,
                **_dataset_build_kwargs_for_label_config(
                    {
                        "threshold": operational_min_total_spread_pct,
                        "cost_floor_pct": operational_min_total_spread_pct,
                        "label_percentile": (
                            max(effective_label_percentiles)
                            if adaptive_label_mode and effective_label_percentiles
                            else None
                        ),
                        "label_episode_window_days": int(max(label_episode_window_days, 1)) if adaptive_label_mode else None,
                        "label_threshold_mode": "rolling_pair_percentile" if adaptive_label_mode else "fixed_threshold",
                    }
                ),
            )
            if records and str(certification_mode).lower() != "quick"
            else None
        )
        if bundle is not None:
            block_diagnostics = dict(bundle.summary.get("block_diagnostics", {}))
        total_blocks = max(
            int(quick_sqlite_metrics["num_blocks"]) if quick_sqlite_metrics is not None else len(blocks),
            1,
        )
        interval_p50 = float(block_diagnostics.get("inter_record_interval_sec_quantiles", {}).get("p50", 0.0))
        ratio_p90 = float(block_diagnostics.get("max_to_median_interval_ratio_quantiles", {}).get("p90", 0.0))
        irregular_rate = float(block_diagnostics.get("irregular_block_count", 0) / total_blocks)
        gate02_failed = (
            interval_p50 < GATE_02_MIN_INTERVAL_SEC
            or interval_p50 > GATE_02_MAX_INTERVAL_SEC
            or ratio_p90 > 3.0
            or irregular_rate >= 0.10
        )
        _persist_gate(
            _build_gate(
                "gate_02_intra_block_temporal_regularity",
                "Intra-block temporal regularity",
                status="FAIL" if gate02_failed else "PASS",
                failure_reasons=["excessive_intra_block_irregularity"] if gate02_failed else [],
                details={
                    "total_blocks": total_blocks,
                    "median_interval_p50_sec": interval_p50,
                    "max_to_median_ratio_p90": ratio_p90,
                    "irregular_block_rate": irregular_rate,
                    "expected_interval_floor_sec": GATE_02_MIN_INTERVAL_SEC,
                    "expected_interval_ceiling_sec": GATE_02_MAX_INTERVAL_SEC,
                    "block_diagnostics": block_diagnostics,
                },
            )
        )

        _check_timeout()
        pair_record_counts = (
            Counter(quick_sqlite_metrics["pair_record_counts"])
            if quick_sqlite_metrics is not None
            else Counter(str(record["pair_id"]) for record in records if str(record["pair_id"]))
        )
        completeness_pair_ids = sorted(
            pair_id
            for pair_id, record_count in pair_record_counts.items()
            if int(record_count) >= GATE_03_MIN_RECORDS_FOR_COMPLETENESS
        )
        completeness_pair_set = set(completeness_pair_ids)
        disappeared_pairs = []
        intermittent_pairs = []
        if quick_sqlite_metrics is not None:
            checkpoint_count = int(quick_sqlite_metrics["checkpoint_count"])
            pair_presence_sets = dict(quick_sqlite_metrics["pair_checkpoint_presence"])
            for pair_id in completeness_pair_ids:
                presence_set = {int(value) for value in pair_presence_sets.get(pair_id, set())}
                if not presence_set:
                    continue
                first = min(presence_set)
                last = max(presence_set)
                if checkpoint_count > 0 and last < checkpoint_count - 1:
                    disappeared_pairs.append(pair_id)
                if len(presence_set) < (last - first + 1):
                    intermittent_pairs.append(pair_id)
            records_per_pair_per_hour = float(
                int(quick_sqlite_metrics["records_total"])
                / max(len(pair_ids) * max(session_hours, 1.0 / 3600.0), 1.0)
            )
        else:
            pair_presence = {pair_id: [pair_id in set(bucket["pairs"]) for bucket in checkpoint_buckets] for pair_id in pair_ids}
            for pair_id, presence in pair_presence.items():
                if pair_id not in completeness_pair_set:
                    continue
                if not any(presence):
                    continue
                first = presence.index(True)
                last = len(presence) - 1 - presence[::-1].index(True)
                if last < len(presence) - 1:
                    disappeared_pairs.append(pair_id)
                if any(not value for value in presence[first:last + 1]):
                    intermittent_pairs.append(pair_id)
            records_per_pair_per_hour = float(len(records) / max(len(pair_ids) * max(session_hours, 1.0 / 3600.0), 1.0))
        active_pair_count = max(len(completeness_pair_ids), 1)
        disappeared_rate = float(len(disappeared_pairs) / active_pair_count)
        intermittent_rate = float(len(intermittent_pairs) / active_pair_count)
        density_threshold = _scaled_density_threshold(len(pair_ids))
        recommendations: list[str] = []
        if rejection_stats and (
            disappeared_rate >= GATE_03_DISAPPEARED_WARN_RATE
            or intermittent_rate >= GATE_03_INTERMITTENT_WARN_RATE
            or records_per_pair_per_hour < density_threshold
        ):
            top_pair = next(iter(dict(rejection_stats.get("rejection_rate_by_pair") or {}).items()), None)
            top_exchange = next(iter(dict(rejection_stats.get("rejection_rate_by_exchange") or {}).items()), None)
            if top_pair or top_exchange:
                recommendations.append(f"hot_path_rejections dominate completeness pressure: top_pair={top_pair} top_exchange={top_exchange}")
        completeness_failures = []
        completeness_warnings = []
        if disappeared_rate >= GATE_03_DISAPPEARED_FAIL_RATE or intermittent_rate >= GATE_03_INTERMITTENT_FAIL_RATE:
            completeness_failures.append("pair_completeness_degraded")
        elif disappeared_rate >= GATE_03_DISAPPEARED_WARN_RATE or intermittent_rate >= GATE_03_INTERMITTENT_WARN_RATE:
            completeness_warnings.append("pair_completeness_degraded")
        if records_per_pair_per_hour < density_threshold:
            completeness_failures.append("insufficient_record_density")
        _persist_gate(
            _build_gate(
                "gate_03_completeness",
                "Completeness",
                status="FAIL" if completeness_failures else ("WARNING" if completeness_warnings else "PASS"),
                failure_reasons=completeness_failures,
                warnings=completeness_warnings,
                recommendations=recommendations,
                details={
                    "active_pair_count": len(pair_ids),
                    "checkpoint_count": int(quick_sqlite_metrics["checkpoint_count"]) if quick_sqlite_metrics is not None else len(checkpoint_buckets),
                    "records_per_pair_per_hour": records_per_pair_per_hour,
                    "records_per_pair_per_hour_threshold": density_threshold,
                    "completeness_pair_count": len(completeness_pair_ids),
                    "completeness_pair_min_records": GATE_03_MIN_RECORDS_FOR_COMPLETENESS,
                    "disappeared_pair_warn_rate": GATE_03_DISAPPEARED_WARN_RATE,
                    "disappeared_pair_fail_rate": GATE_03_DISAPPEARED_FAIL_RATE,
                    "intermittent_pair_warn_rate": GATE_03_INTERMITTENT_WARN_RATE,
                    "intermittent_pair_fail_rate": GATE_03_INTERMITTENT_FAIL_RATE,
                    "disappeared_pairs": disappeared_pairs[:50],
                    "intermittent_pairs": intermittent_pairs[:50],
                    "disappeared_pair_rate": disappeared_rate,
                    "intermittent_pair_rate": intermittent_rate,
                    "hot_path_rejection_stats": rejection_stats,
                },
            )
        )

        _check_timeout()
        if certification_mode == "quick":
            _skip_gate("gate_04_checkpoint_stationarity", "Checkpoint stationarity", "quick_mode_skipped")
            _skip_gate("gate_05_intra_soak_feature_drift", "Intra-soak feature drift", "quick_mode_skipped")
        else:
            shifts: list[dict[str, Any]] = []
            for left, right in zip(checkpoint_buckets, checkpoint_buckets[1:]):
                for feature_name in ("entry_spread", "exit_spread"):
                    metrics = _score_shift([float(item[feature_name]) for item in left["records"]], [float(item[feature_name]) for item in right["records"]])
                    if float(metrics["score"]) > 3.0:
                        shifts.append({"feature": feature_name, "left_checkpoint_index": int(left["index"]), "right_checkpoint_index": int(right["index"]), "left_start_ts": float(left["start_ts"]), "right_start_ts": float(right["start_ts"]), **metrics})
            _persist_gate(_build_gate("gate_04_checkpoint_stationarity", "Checkpoint stationarity", status="WARNING" if shifts else "PASS", warnings=["checkpoint_regime_shift_detected"] if shifts else [], details={"shifted_checkpoint_pairs": shifts}))
            drifted_features: list[dict[str, Any]] = []
            if bundle is not None and int(bundle.summary.get("num_samples", 0)) >= 2:
                order = np.argsort(np.asarray(bundle.timestamps, dtype=float))
                feature_view = bundle.X.detach().cpu().numpy()[order, -1, :]
                midpoint = feature_view.shape[0] // 2
                first_half = feature_view[:midpoint]
                second_half = feature_view[midpoint:]
                feature_names = list(getattr(bundle, "feature_names", FEATURE_NAMES[: feature_view.shape[1]]))
                if first_half.size and second_half.size:
                    for index, feature_name in enumerate(feature_names):
                        metrics = _score_shift(first_half[:, index].tolist(), second_half[:, index].tolist())
                        if float(metrics["score"]) > 3.0:
                            drifted_features.append({"feature": feature_name, **metrics})
            gate05_status = "FAIL" if len(drifted_features) > 3 else ("WARNING" if drifted_features else "PASS")
            _persist_gate(_build_gate("gate_05_intra_soak_feature_drift", "Intra-soak feature drift", status=gate05_status, failure_reasons=["intra_soak_feature_drift"] if len(drifted_features) > 3 else [], warnings=["intra_soak_feature_drift"] if 0 < len(drifted_features) <= 3 else [], details={"drifted_features": drifted_features, "feature_count": len(getattr(bundle, "feature_names", FEATURE_NAMES)) if bundle is not None else len(FEATURE_NAMES)}))

        _check_timeout()
        total_spreads = (
            [float(value) for value in quick_sqlite_metrics["episode_total_spreads"]]
            if quick_sqlite_metrics is not None
            else [float(getattr(item, "total_spread", 0.0) or 0.0) for item in episodes]
        )
        durations = (
            [float(value) for value in quick_sqlite_metrics["episode_durations"]]
            if quick_sqlite_metrics is not None
            else [float(getattr(item, "duration_sec", 0.0) or 0.0) for item in episodes]
        )
        episode_count = len(total_spreads)
        episodes_per_hour = float(episode_count / max(session_hours, 1.0 / 3600.0))
        gate06_failures = []
        gate06_warnings = []
        if episode_count <= 0:
            gate06_failures.append("no_qualified_episodes")
        elif episodes_per_hour < 0.5:
            gate06_failures.append("insufficient_episode_yield")
        elif episodes_per_hour < 1.0:
            gate06_warnings.append("insufficient_episode_yield")
        duration_p50 = _percentile(durations, 50.0)
        total_spread_p50 = _percentile(total_spreads, 50.0)
        if total_spreads and (
            duration_p50 <= GATE_06_MIN_EPISODE_DURATION_SEC
            or duration_p50 >= float(prediction_horizon_sec)
        ):
            gate06_failures.append("episode_quality_degraded")
        _persist_gate(
            _build_gate(
                "gate_06_episode_yield",
                "Episode yield",
                status="FAIL" if gate06_failures else ("WARNING" if gate06_warnings else "PASS"),
                failure_reasons=gate06_failures,
                warnings=gate06_warnings,
                details={
                    "episode_count": episode_count,
                    "episodes_per_hour": episodes_per_hour,
                    "minimum_duration_sec": GATE_06_MIN_EPISODE_DURATION_SEC,
                    "prediction_horizon_sec": float(prediction_horizon_sec),
                    "total_spread_quantiles": {"p50": total_spread_p50, "p90": _percentile(total_spreads, 90.0)},
                    "duration_sec_quantiles": {"p50": duration_p50, "p90": _percentile(durations, 90.0)},
                },
            )
        )

        _check_timeout()
        if certification_mode == "quick":
            quick_scope_start = float(quick_sqlite_metrics.get("min_ts", 0.0) or 0.0) if quick_sqlite_metrics is not None else 0.0
            quick_scope_end = float(quick_sqlite_metrics.get("max_ts", 0.0) or 0.0) if quick_sqlite_metrics is not None else 0.0
            hourly_health = _load_hourly_health_samples(
                state_path,
                scope_start_ts=quick_scope_start,
                scope_end_ts=quick_scope_end,
            )
            if not hourly_health:
                _skip_gate("gate_07_book_health", "Book health", "quick_mode_skipped")
            else:
                gate07_failures = []
                unhealthy_count = sum(1 for item in hourly_health if str(item.get("quality_verdict") or "") == "unhealthy")
                degraded_count = sum(1 for item in hourly_health if str(item.get("quality_verdict") or "") == "degraded")
                rejection_rates = [float(item.get("rejection_rate_pct", 0.0) or 0.0) for item in hourly_health]
                if unhealthy_count > 0:
                    gate07_failures.append("hourly_health_unhealthy")
                if _percentile(rejection_rates, 95.0) > 50.0:
                    gate07_failures.append("hourly_rejection_rate_failed")
                _persist_gate(
                    _build_gate(
                        "gate_07_book_health",
                        "Book health",
                        status="FAIL" if gate07_failures else ("WARNING" if degraded_count > 0 else "PASS"),
                        failure_reasons=gate07_failures,
                        warnings=["hourly_health_degraded"] if degraded_count > 0 and not gate07_failures else [],
                        details={
                            "hourly_health_sample_count": len(hourly_health),
                            "unhealthy_hours": unhealthy_count,
                            "degraded_hours": degraded_count,
                            "rejection_rate_pct_p95": _percentile(rejection_rates, 95.0),
                        },
                    )
                )
            _skip_gate("gate_09_runtime_audit_sqlite_consistency", "Runtime audit to SQLite consistency", "quick_mode_skipped")
            _skip_gate("gate_11_runtime_audit_health", "Runtime audit health", "quick_mode_skipped")
        else:
            book_samples = [dict(sample.get("extra", {}).get("book_health") or {}) for sample in runtime_package.get("samples", []) if str(sample.get("kind") or "") == "system_sample" and isinstance(sample.get("extra"), dict) and isinstance(sample.get("extra", {}).get("book_health"), dict)]
            if not book_samples:
                _skip_gate("gate_07_book_health", "Book health", "runtime_audit_unavailable")
            else:
                median_ages = [float(item.get("median_book_age_sec") or 0.0) for item in book_samples]
                p95_ages = [float(item.get("p95_book_age_sec") or 0.0) for item in book_samples]
                max_ages = [float(item.get("max_book_age_sec") or 0.0) for item in book_samples]
                asymmetry = [float(item.get("p95_book_asymmetry_sec") or 0.0) for item in book_samples]
                invalid_rates = [float(item.get("invalid_quote_rate") or 0.0) for item in book_samples]
                gate07_failures = []
                if _percentile(median_ages, 50.0) > 6.0 or _percentile(p95_ages, 95.0) > 12.0 or max(max_ages, default=0.0) > 30.0:
                    gate07_failures.append("book_freshness_failed")
                if _percentile(asymmetry, 95.0) > 10.0:
                    gate07_failures.append("book_asymmetry_failed")
                if _percentile(invalid_rates, 95.0) >= 0.01:
                    gate07_failures.append("empty_book_rate_failed")
                _persist_gate(_build_gate("gate_07_book_health", "Book health", status="FAIL" if gate07_failures else "PASS", failure_reasons=gate07_failures, details={"sample_count": len(book_samples), "median_book_age_sec_p50": _percentile(median_ages, 50.0), "p95_book_age_sec_p95": _percentile(p95_ages, 95.0), "max_book_age_sec": max(max_ages, default=0.0), "p95_book_asymmetry_sec": _percentile(asymmetry, 95.0), "invalid_quote_rate_p95": _percentile(invalid_rates, 95.0)}))

            system_samples = [sample for sample in runtime_package.get("samples", []) if str(sample.get("kind") or "") == "system_sample"]
            if not system_samples:
                _skip_gate("gate_09_runtime_audit_sqlite_consistency", "Runtime audit to SQLite consistency", "runtime_audit_unavailable")
            else:
                latest_sample = system_samples[-1]
                tracker_stats = dict(latest_sample.get("tracker_stats") or {})
                sqlite_pairs = int(integrity.get("table_counts", {}).get("tracker_pairs", 0) or 0)
                sqlite_records = int(integrity.get("table_counts", {}).get("tracker_records", 0) or 0)
                runtime_pairs = int(tracker_stats.get("pairs_persisted", tracker_stats.get("pairs_in_memory", 0)) or 0)
                runtime_records = int(tracker_stats.get("records_total", 0) or 0)
                pair_divergence = abs(runtime_pairs - sqlite_pairs) / max(sqlite_pairs, 1)
                record_divergence = abs(runtime_records - sqlite_records) / max(sqlite_records, 1)
                latest_session_id = max((int(row.get("id") or 0) for row in runtime_session_rows), default=0)
                active_session_id = int(tracker_stats.get("active_session_id", 0) or 0)
                gate09_failures = []
                if pair_divergence >= 0.05 or (active_session_id > 0 and latest_session_id > 0 and active_session_id != latest_session_id):
                    gate09_failures.append("audit_sqlite_inconsistency")
                if record_divergence >= 0.01:
                    gate09_failures.append("tracker_sqlite_count_divergence")
                _persist_gate(_build_gate("gate_09_runtime_audit_sqlite_consistency", "Runtime audit to SQLite consistency", status="FAIL" if gate09_failures else "PASS", failure_reasons=gate09_failures, details={"runtime_pairs": runtime_pairs, "sqlite_pairs": sqlite_pairs, "pair_divergence": pair_divergence, "runtime_records": runtime_records, "sqlite_records": sqlite_records, "record_divergence": record_divergence, "active_session_id": active_session_id, "latest_sqlite_session_id": latest_session_id}))

            if not runtime_package.get("path"):
                _persist_gate(_build_gate("gate_11_runtime_audit_health", "Runtime audit health", status="WARNING", warnings=["runtime_audit_unavailable"], details={"runtime_audit_package_path": ""}))
            else:
                runtime_summary = dict(runtime_package.get("summary") or {})
                finished_at_raw = str(runtime_summary.get("finished_at_utc") or "")
                finished_at = 0.0
                if finished_at_raw:
                    try:
                        finished_at = datetime.fromisoformat(finished_at_raw.replace("Z", "+00:00")).timestamp()
                    except ValueError:
                        finished_at = 0.0
                stale = finished_at > 0.0 and (time.time() - finished_at) > DEFAULT_RUNTIME_AUDIT_STALENESS_SEC
                disconnect_count = int(sum(1 for alert in runtime_package.get("alerts", []) if str(alert.get("code") or "") == "exchange_disconnected"))
                reconnect_count = int(sum(1 for alert in runtime_package.get("alerts", []) if str(alert.get("code") or "") == "exchange_reconnected"))
                runtime_errors = [alert for alert in runtime_package.get("alerts", []) if str(alert.get("code") or "") == "runtime_error" and str(alert.get("component") or "") in {"calc_broadcast", "tracker_persist", "runtime_audit_loop"}]
                inference_events = [event for event in runtime_package.get("events", []) if str(event.get("kind") or "") == "inference"]
                history_points_bug = any(int(event.get("history_points", 0) or 0) <= 0 for event in inference_events)
                rss_values = [float(sample.get("rss_bytes") or 0.0) for sample in runtime_package.get("samples", []) if str(sample.get("kind") or "") == "system_sample"]
                severe_memory_growth = len(rss_values) >= 2 and rss_values[-1] > rss_values[0] * 1.5 and (rss_values[-1] - rss_values[0]) > 250 * 1024 * 1024
                duration_hours = max(float(runtime_summary.get("duration_sec") or 0.0) / 3600.0, 1.0 / 3600.0)
                excessive_reconnect_noise = float((disconnect_count + reconnect_count) / duration_hours) > 6.0
                gate11_failures = []
                if stale:
                    gate11_failures.append("runtime_audit_stale")
                if disconnect_count > reconnect_count or excessive_reconnect_noise or runtime_errors or history_points_bug or severe_memory_growth:
                    gate11_failures.append("runtime_health_failed")
                _persist_gate(_build_gate("gate_11_runtime_audit_health", "Runtime audit health", status="FAIL" if gate11_failures else "PASS", failure_reasons=gate11_failures, details={"runtime_audit_package_path": str(runtime_package.get("path") or ""), "disconnect_count": disconnect_count, "reconnect_count": reconnect_count, "runtime_error_count": len(runtime_errors), "history_points_bug": bool(history_points_bug), "severe_memory_growth": bool(severe_memory_growth), "stale": bool(stale)}))

        _check_timeout()
        if not run_reconnection_stress:
            _persist_gate(_build_gate("gate_08_reconnection_stress", "Reconnection stress", status="SKIPPED", details={"executed": False, "reason": "manual_gate_skipped"}))
        else:
            disconnects = int(sum(1 for alert in runtime_package.get("alerts", []) if str(alert.get("code") or "") == "exchange_disconnected"))
            reconnects = int(sum(1 for alert in runtime_package.get("alerts", []) if str(alert.get("code") or "") == "exchange_reconnected"))
            _persist_gate(_build_gate("gate_08_reconnection_stress", "Reconnection stress", status="FAIL" if disconnects > reconnects else "PASS", failure_reasons=["reconnection_stress_failed"] if disconnects > reconnects else [], details={"executed": True, "disconnect_count": disconnects, "reconnect_count": reconnects}))

        _check_timeout()
        if str(certification_mode).lower() == "quick":
            _persist_gate(
                _build_gate(
                    "gate_10_dual_mode_preflight",
                    "Dual-mode preflight",
                    status="SKIPPED",
                    warnings=["dual_preflight_skipped_in_quick_mode"],
                    details={"executed": False, "reason": "quick_mode"},
                )
            )
        else:
            dual_summary: list[dict[str, Any]] = []
            qualifying_configs: list[str] = []
            failure_counter: Counter[str] = Counter()
            for config in DEFAULT_DUAL_PREFLIGHT_CONFIGS:
                config_entry: dict[str, Any] = {"sequence_length": int(config["sequence_length"]), "prediction_horizon_sec": int(config["prediction_horizon_sec"])}
                for merge_enabled, key in ((False, "merge_off"), (True, "merge_on")):
                    preflight = preflight_fn(
                        state_file=state_path,
                        sequence_length=int(config["sequence_length"]),
                        prediction_horizon_sec=int(config["prediction_horizon_sec"]),
                        thresholds=list(threshold_values),
                        label_percentiles=list(effective_label_percentiles) if adaptive_label_mode else None,
                        label_episode_window_days=int(max(label_episode_window_days, 1)),
                        selected_session_ids=effective_session_ids or None,
                        selected_block_ids=effective_block_ids or None,
                        allow_cross_session_merge=merge_enabled,
                        max_session_gap_sec=max_session_gap_sec,
                        regime_shift_score_threshold=regime_shift_score_threshold,
                        min_train_positive_samples=1,
                        min_val_positive_samples=1,
                        min_test_positive_samples=1,
                    )
                    fingerprint_threshold = float(preflight.get("selected_threshold") or operational_min_total_spread_pct)
                    selected_label_config = _label_config_payload(preflight.get("selected_label_config"))
                    try:
                        config_bundle = build_dataset_bundle(
                            state_path=state_path,
                            sequence_length=int(config["sequence_length"]),
                            prediction_horizon_sec=int(config["prediction_horizon_sec"]),
                            selected_block_ids=effective_block_ids or None,
                            selected_session_ids=effective_session_ids or None,
                            selected_only=False,
                            closed_only=True,
                            allow_cross_session_merge=merge_enabled,
                            max_session_gap_sec=max_session_gap_sec,
                            regime_shift_score_threshold=regime_shift_score_threshold,
                            **_dataset_build_kwargs_for_label_config(
                                selected_label_config or {
                                    "threshold": fingerprint_threshold,
                                    "cost_floor_pct": fingerprint_threshold,
                                    "label_percentile": None,
                                    "label_episode_window_days": None,
                                    "label_threshold_mode": "fixed_threshold",
                                }
                            ),
                        )
                        fingerprint = dataset_fingerprint_fn(
                            state_path=state_path,
                            bundle=config_bundle,
                            min_total_spread_pct=fingerprint_threshold,
                            label_cost_floor_pct=(
                                float(selected_label_config["cost_floor_pct"])
                                if selected_label_config
                                and selected_label_config.get("label_threshold_mode") == "rolling_pair_percentile"
                                else None
                            ),
                            label_percentile=(
                                int(selected_label_config["label_percentile"])
                                if selected_label_config and selected_label_config.get("label_percentile") is not None
                                else None
                            ),
                            label_episode_window_days=(
                                int(selected_label_config["label_episode_window_days"])
                                if selected_label_config and selected_label_config.get("label_episode_window_days") is not None
                                else None
                            ),
                            label_threshold_mode=str(selected_label_config.get("label_threshold_mode") or "fixed_threshold"),
                            selected_session_ids=effective_session_ids or None,
                            selected_block_ids=effective_block_ids or None,
                            allow_cross_session_merge=merge_enabled,
                            max_session_gap_sec=max_session_gap_sec,
                            regime_shift_score_threshold=regime_shift_score_threshold,
                        )
                    except Exception:
                        fingerprint = ""
                    fingerprint_key = f"seq{config['sequence_length']}_h{config['prediction_horizon_sec']}_{key}"
                    if fingerprint:
                        dataset_fingerprints[fingerprint_key] = fingerprint
                    reason_list = [
                        reason
                        for threshold_entry in dict(preflight.get("thresholds", {})).values()
                        if isinstance(threshold_entry, dict)
                        for reason in list(threshold_entry.get("failure_reasons") or [])
                    ]
                    config_entry[key] = {
                        "qualifies_for_training": bool(preflight.get("qualifies_for_training")),
                        "selected_threshold": preflight.get("selected_threshold"),
                        "selected_label_config": dict(selected_label_config),
                        "selection_mode": preflight.get("selection_mode"),
                        "failure_reasons": reason_list,
                        "dataset_fingerprint": fingerprint,
                    }
                    if bool(preflight.get("qualifies_for_training")):
                        qualifying_configs.append(f"{fingerprint_key}:{preflight.get('selected_threshold')}")
                    else:
                        failure_counter.update(reason_list)
                dual_summary.append(config_entry)
            _persist_gate(_build_gate("gate_10_dual_mode_preflight", "Dual-mode preflight", status="PASS" if qualifying_configs else "FAIL", failure_reasons=list(failure_counter.keys()) if not qualifying_configs else [], details={"qualifying_configs": qualifying_configs, "configs": dual_summary, "dataset_fingerprints": dataset_fingerprints}))

        _check_timeout()
        if quick_sqlite_metrics is not None:
            pair_record_counts = Counter(quick_sqlite_metrics["pair_record_counts"])
            pair_hour_counts = Counter(quick_sqlite_metrics["pair_hour_counts"])
            frozen_entry_hours = Counter(quick_sqlite_metrics["frozen_entry_hours"])
            frozen_exit_hours = Counter(quick_sqlite_metrics["frozen_exit_hours"])
            unresponsive_exit_hours = Counter(quick_sqlite_metrics["unresponsive_exit_hours"])
            entry_outlier_hours = Counter(quick_sqlite_metrics["entry_outlier_hours"])
            exit_outlier_hours = Counter(quick_sqlite_metrics["exit_outlier_hours"])
            pearson_by_pair = dict(quick_sqlite_metrics["pearson_by_pair"])
            episode_peaks = [float(value) for value in quick_sqlite_metrics["episode_peaks"]]
            episode_exits = [float(value) for value in quick_sqlite_metrics["episode_exits"]]
            bilateral_zero_rate = float(quick_sqlite_metrics["bilateral_zero_rate"])
        else:
            pair_hour_buckets = _pair_hour_buckets(records)
            pair_record_counts = Counter(str(record["pair_id"]) for record in records if str(record["pair_id"]))
            pair_hour_counts = Counter()
            frozen_entry_hours = Counter()
            frozen_exit_hours = Counter()
            unresponsive_exit_hours = Counter()
            entry_outlier_hours = Counter()
            exit_outlier_hours = Counter()
            pearson_by_pair: dict[str, float] = {}
            for (pair_id, _hour_bucket), bucket_records in pair_hour_buckets.items():
                pair_hour_counts[pair_id] += 1
                entry_values = [float(item["entry_spread"]) for item in bucket_records]
                exit_values = [float(item["exit_spread"]) for item in bucket_records]
                entry_std = float(np.std(np.asarray(entry_values, dtype=float))) if entry_values else 0.0
                exit_std = float(np.std(np.asarray(exit_values, dtype=float))) if exit_values else 0.0
                if entry_std <= 1e-4 or _zero_return_ratio(entry_values) >= 0.80:
                    frozen_entry_hours[pair_id] += 1
                if exit_std <= 1e-4 or _zero_return_ratio(exit_values) >= 0.80:
                    frozen_exit_hours[pair_id] += 1
                if entry_std > 1e-4 and exit_std <= 1e-4:
                    unresponsive_exit_hours[pair_id] += 1
                entry_median = _median(entry_values)
                exit_median = _median(exit_values)
                entry_mad = max(_mad(entry_values, entry_median), 0.01)
                exit_mad = max(_mad(exit_values, exit_median), 0.01)
                if entry_values and sum(1 for value in entry_values if abs(value - entry_median) > 10.0 * entry_mad) / len(entry_values) > 0.05:
                    entry_outlier_hours[pair_id] += 1
                if exit_values and sum(1 for value in exit_values if abs(value - exit_median) > 10.0 * exit_mad) / len(exit_values) > 0.05:
                    exit_outlier_hours[pair_id] += 1
                if len(entry_values) >= 2 and len(exit_values) >= 2:
                    if float(np.std(np.asarray(entry_values, dtype=float))) <= 1e-12 or float(np.std(np.asarray(exit_values, dtype=float))) <= 1e-12:
                        pearson_by_pair[pair_id] = 0.0
                    else:
                        pearson_by_pair[pair_id] = float(np.corrcoef(np.asarray(entry_values), np.asarray(exit_values))[0, 1])
            episode_peaks = [float(getattr(item, "peak_entry_spread", 0.0) or 0.0) for item in episodes]
            episode_exits = [float(getattr(item, "exit_spread_at_close", 0.0) or 0.0) for item in episodes]
            bilateral_zero_rate = float(sum(1 for item in records if abs(float(item["entry_spread"])) <= 1e-12 and abs(float(item["exit_spread"])) <= 1e-12) / max(len(records), 1))
        active_quality_pairs = [
            pair_id
            for pair_id, total_hours in pair_hour_counts.items()
            if float(pair_record_counts.get(pair_id, 0) / max(total_hours, 1)) >= GATE_12_MIN_RECORDS_PER_ACTIVE_HOUR
        ]
        active_quality_pair_set = set(active_quality_pairs)
        pair_count = max(len(active_quality_pairs), 1)
        frozen_entry_pairs = [
            pair_id
            for pair_id, total_hours in pair_hour_counts.items()
            if pair_id in active_quality_pair_set and float(frozen_entry_hours.get(pair_id, 0) / max(total_hours, 1)) > 0.50
        ]
        frozen_exit_pairs = [
            pair_id
            for pair_id, total_hours in pair_hour_counts.items()
            if pair_id in active_quality_pair_set and float(frozen_exit_hours.get(pair_id, 0) / max(total_hours, 1)) > 0.50
        ]
        frozen_pairs = sorted(set(frozen_entry_pairs) | set(frozen_exit_pairs))
        eligible_pairs = [pair_id for pair_id in active_quality_pairs if pair_id not in set(frozen_pairs)]
        unresponsive_pairs = [pair_id for pair_id in eligible_pairs if float(unresponsive_exit_hours.get(pair_id, 0) / max(pair_hour_counts.get(pair_id, 0), 1)) > 0.05]
        entry_outlier_pairs = [
            pair_id
            for pair_id, total_hours in pair_hour_counts.items()
            if pair_id in active_quality_pair_set
            and float(entry_outlier_hours.get(pair_id, 0) / max(total_hours, 1)) > GATE_12_OUTLIER_HOUR_RATE_THRESHOLD
        ]
        exit_outlier_pairs = [
            pair_id
            for pair_id, total_hours in pair_hour_counts.items()
            if pair_id in active_quality_pair_set
            and float(exit_outlier_hours.get(pair_id, 0) / max(total_hours, 1)) > GATE_12_OUTLIER_HOUR_RATE_THRESHOLD
        ]
        gate12_failures = []
        frozen_entry_rate = float(len(frozen_entry_pairs) / pair_count)
        frozen_exit_rate = float(len(frozen_exit_pairs) / pair_count)
        frozen_pair_rate = float(len(frozen_pairs) / pair_count)
        unresponsive_exit_rate = float(len(unresponsive_pairs) / max(len(eligible_pairs), 1)) if eligible_pairs else 0.0
        entry_outlier_pair_rate = float(len(entry_outlier_pairs) / pair_count)
        exit_outlier_pair_rate = float(len(exit_outlier_pairs) / pair_count)
        if len(frozen_pairs) / pair_count > GATE_12_PAIR_QUALITY_FAIL_RATE:
            if frozen_entry_rate > GATE_12_PAIR_QUALITY_FAIL_RATE:
                gate12_failures.append("entry_stale_quotes")
            if frozen_exit_rate > GATE_12_PAIR_QUALITY_FAIL_RATE:
                gate12_failures.append("exit_stale_quotes")
            gate12_failures.append("entry_or_exit_stale_quotes")
        if eligible_pairs and unresponsive_exit_rate > 0.05:
            gate12_failures.append("exit_unresponsive_while_entry_active")
        if entry_outlier_pair_rate > GATE_12_PAIR_QUALITY_FAIL_RATE:
            gate12_failures.append("entry_outliers_unfiltered")
        if exit_outlier_pair_rate > GATE_12_PAIR_QUALITY_FAIL_RATE:
            gate12_failures.append("exit_outliers_unfiltered")
        if bilateral_zero_rate >= 0.001:
            gate12_failures.append("bilateral_zero_quote_records")
        if episode_peaks and float(np.std(np.asarray(episode_peaks, dtype=float))) <= 0.0:
            gate12_failures.append("episode_entry_frozen")
        if episode_exits and float(np.std(np.asarray(episode_exits, dtype=float))) <= 0.0:
            gate12_failures.append("episode_exit_frozen")
        entry_episode_outlier_rate, entry_episode_zero_rate = _episode_outlier_rate(episode_peaks)
        exit_episode_outlier_rate, exit_episode_zero_rate = _episode_outlier_rate(episode_exits)
        if entry_episode_outlier_rate > GATE_12_EPISODE_OUTLIER_FAIL_RATE:
            gate12_failures.append("episode_entry_corrupted")
        if exit_episode_outlier_rate > GATE_12_EPISODE_OUTLIER_FAIL_RATE:
            gate12_failures.append("episode_exit_corrupted")
        _persist_gate(
            _build_gate(
                "gate_12_entry_exit_quality",
                "Entry and exit quality",
                status="FAIL" if gate12_failures else "PASS",
                failure_reasons=gate12_failures,
                details={
                    "pair_count": len(pair_hour_counts),
                    "active_quality_pair_count": len(active_quality_pairs),
                    "active_quality_pair_min_records_per_hour": GATE_12_MIN_RECORDS_PER_ACTIVE_HOUR,
                    "outlier_hour_rate_threshold": GATE_12_OUTLIER_HOUR_RATE_THRESHOLD,
                    "pair_quality_fail_rate_threshold": GATE_12_PAIR_QUALITY_FAIL_RATE,
                    "episode_outlier_fail_rate_threshold": GATE_12_EPISODE_OUTLIER_FAIL_RATE,
                    "frozen_entry_pair_count": len(frozen_entry_pairs),
                    "frozen_exit_pair_count": len(frozen_exit_pairs),
                    "frozen_pair_rate": frozen_pair_rate,
                    "entry_frozen_rate": frozen_entry_rate,
                    "exit_frozen_rate": frozen_exit_rate,
                    "frozen_entry_pairs": frozen_entry_pairs[:50],
                    "frozen_exit_pairs": frozen_exit_pairs[:50],
                    "eligible_pairs_for_responsiveness": len(eligible_pairs),
                    "unresponsive_exit_pair_count": len(unresponsive_pairs),
                    "unresponsive_exit_rate": unresponsive_exit_rate,
                    "unresponsive_exit_pairs": unresponsive_pairs[:50],
                    "entry_outlier_pair_count": len(entry_outlier_pairs),
                    "entry_outlier_rate": entry_outlier_pair_rate,
                    "exit_outlier_pair_count": len(exit_outlier_pairs),
                    "exit_outlier_rate": exit_outlier_pair_rate,
                    "entry_outlier_pairs": entry_outlier_pairs[:50],
                    "exit_outlier_pairs": exit_outlier_pairs[:50],
                    "bilateral_zero_rate": bilateral_zero_rate,
                    "pearson_correlation_by_pair": pearson_by_pair,
                    "episode_count": episode_count,
                    "episode_entry_outlier_rate": entry_episode_outlier_rate,
                    "episode_exit_outlier_rate": exit_episode_outlier_rate,
                    "episode_entry_zero_rate": entry_episode_zero_rate,
                    "episode_exit_zero_rate": exit_episode_zero_rate,
                },
            )
        )
    except TimeoutError:
        pass

    elapsed = time.perf_counter() - started_at
    failure_reasons = [reason for gate in gate_results.values() for reason in list(gate.get("failure_reasons") or [])]
    warnings = [warning for gate in gate_results.values() if str(gate.get("status") or "") in {"WARNING", "SKIPPED"} for warning in list(gate.get("warnings") or [])]
    recommendations = [recommendation for gate in gate_results.values() for recommendation in list(gate.get("recommendations") or [])]
    if elapsed > float(max_certification_duration_sec):
        failure_reasons.append("certification_timeout")
    verdict = "FAILED" if failure_reasons else ("CERTIFIED_WITH_WARNINGS" if warnings else "CERTIFIED")
    certification_digest_payload = {
        "effective_session_scope": dict(scope.get("effective_session_scope") or {}),
        "gate_results": gate_results,
        "config": {
            "sequence_length": int(sequence_length),
            "prediction_horizon_sec": int(prediction_horizon_sec),
            "thresholds": list(threshold_values),
            "allow_cross_session_merge": bool(allow_cross_session_merge),
            "max_session_gap_sec": None if max_session_gap_sec is None else float(max_session_gap_sec),
            "regime_shift_score_threshold": None if regime_shift_score_threshold is None else float(regime_shift_score_threshold),
            "certification_mode": str(certification_mode),
            "allow_legacy_sessions": bool(allow_legacy_sessions),
        },
        "dataset_fingerprints": dataset_fingerprints,
    }
    certification_id = hashlib.sha1(
        json.dumps(certification_digest_payload, sort_keys=True, default=_json_default, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "certified": verdict != "FAILED",
        "verdict": verdict,
        "failure_reasons": list(dict.fromkeys(failure_reasons)),
        "warnings": list(dict.fromkeys(warnings)),
        "gate_results": gate_results,
        "recommendations": list(dict.fromkeys(recommendations)),
        "certification_mode": str(certification_mode),
        "certification_duration_sec": float(round(elapsed, 6)),
        "quality_fix_activated_at": float(scope.get("quality_fix_activated_at") or 0.0),
        "legacy_sessions_excluded": int(scope.get("legacy_sessions_excluded") or 0),
        "legacy_session_ids": list(scope.get("legacy_session_ids") or []),
        "effective_session_scope": dict(scope.get("effective_session_scope") or {}),
        "dataset_fingerprints": dict(dataset_fingerprints),
        "runtime_audit_package_path": str(runtime_package.get("path") or ""),
        "certification_id": certification_id,
        "generated_at_utc": _utc_now_iso(),
    }
    write_json(artifact_root / "data_certification.json", payload)
    return payload
