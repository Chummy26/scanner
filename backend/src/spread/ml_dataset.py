from __future__ import annotations

import bisect
import json
import math
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .feature_contracts import FEATURE_NAMES, build_feature_rows
from .spread_tracker import SpreadRecord, TrackerEpisode, compute_closed_episodes

_SQLITE_BLOCK_FETCH_CHUNK = 2000
_DEFAULT_REGIME_SHIFT_SCORE_THRESHOLD = 3.0
_MIN_REGIME_WINDOW_RECORDS = 5
_MAX_REGIME_WINDOW_RECORDS = 20


@dataclass(slots=True)
class DatasetBundle:
    X: torch.Tensor
    y_class: torch.Tensor
    y_eta: torch.Tensor
    label_thresholds: list[float]
    pair_ids: list[str]
    block_ids: list[int]
    session_ids: list[int]
    timestamps: list[float]
    label_end_timestamps: list[float]
    last_entries: list[float]
    feature_names: list[str]
    summary: dict[str, Any]

    def subset(self, indices: list[int], split_name: str) -> "DatasetBundle":
        return DatasetBundle(
            X=self.X[indices],
            y_class=self.y_class[indices],
            y_eta=self.y_eta[indices],
            label_thresholds=[self.label_thresholds[index] for index in indices],
            pair_ids=[self.pair_ids[index] for index in indices],
            block_ids=[self.block_ids[index] for index in indices],
            session_ids=[self.session_ids[index] for index in indices],
            timestamps=[self.timestamps[index] for index in indices],
            label_end_timestamps=[self.label_end_timestamps[index] for index in indices],
            last_entries=[self.last_entries[index] for index in indices],
            feature_names=list(self.feature_names),
            summary={
                "split_name": split_name,
                "num_samples": len(indices),
                "num_positive_samples": int(self.y_class[indices].sum().item()) if indices else 0,
                "num_negative_samples": len(indices) - int(self.y_class[indices].sum().item()) if indices else 0,
                "positive_rate": (
                    float(self.y_class[indices].mean().item())
                    if indices
                    else 0.0
                ),
                "label_threshold_p50": (
                    float(np.percentile(np.asarray([self.label_thresholds[index] for index in indices], dtype=float), 50))
                    if indices
                    else 0.0
                ),
                "num_pairs": len({self.pair_ids[index] for index in indices}),
            },
        )


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def canonicalize_record(record: dict[str, Any], fallback_ts: float) -> dict[str, float]:
    if "timestamp" in record and "entry_spread" in record and "exit_spread" in record:
        return record  # Already canonical — zero allocation
    return {
        "timestamp": _coerce_float(record.get("ts"), fallback_ts),
        "entry_spread": _coerce_float(record.get("entry", record.get("entry_spread_pct", 0.0)), 0.0),
        "exit_spread": _coerce_float(record.get("exit", record.get("exit_spread_pct", 0.0)), 0.0),
    }

def _linear_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    pct = max(0.0, min(100.0, float(percentile)))
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + ((ordered[upper] - ordered[lower]) * (rank - lower)))


def _bucket_count(
    value: float,
    buckets: list[tuple[str, float, float]],
    counts: dict[str, int],
) -> None:
    numeric_value = float(value)
    for label, lower, upper in buckets:
        if lower <= numeric_value < upper:
            counts[label] = int(counts.get(label, 0)) + 1
            return
    if buckets:
        counts[buckets[-1][0]] = int(counts.get(buckets[-1][0], 0)) + 1


def _quantile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "p10": _linear_percentile(values, 10.0),
        "p25": _linear_percentile(values, 25.0),
        "p50": _linear_percentile(values, 50.0),
        "p75": _linear_percentile(values, 75.0),
        "p90": _linear_percentile(values, 90.0),
        "max": max(float(value) for value in values),
    }


def _load_pairs_from_json(state_path: Path) -> tuple[dict[str, Any], float]:
    payload = json.loads(Path(state_path).read_text(encoding="utf-8"))
    return payload.get("pairs", {}), _coerce_float(payload.get("saved_at"), 0.0)


def _load_pairs_from_sqlite(state_path: Path) -> tuple[dict[str, Any], float]:
    pairs: dict[str, Any] = {}
    saved_at = 0.0
    conn = sqlite3.connect(state_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        meta_rows = {
            row["key"]: row["value"]
            for row in conn.execute("SELECT key, value FROM tracker_meta")
        }
        saved_at = _coerce_float(meta_rows.get("last_flush_at"), 0.0)

        pair_rows = list(
            conn.execute(
                """
                SELECT id, symbol, buy_ex, buy_mt, sell_ex, sell_mt,
                       last_state, last_crossover_ts, last_seen_ts, history_enabled
                FROM tracker_pairs
                ORDER BY symbol, buy_ex, buy_mt, sell_ex, sell_mt
                """
            )
        )
        pair_payloads: dict[int, dict[str, Any]] = {}
        for row in pair_rows:
            pair_id = f"{row['symbol']}|{row['buy_ex']}|{row['buy_mt']}|{row['sell_ex']}|{row['sell_mt']}"
            payload = {
                "last_state": int(row["last_state"]),
                "last_crossover_ts": _coerce_float(row["last_crossover_ts"], 0.0),
                "last_seen_ts": _coerce_float(row["last_seen_ts"], 0.0),
                "history_enabled": bool(row["history_enabled"]),
                "inverted_events": [],
                "entry_events": [],
                "exit_events": [],
                "records": [],
            }
            pairs[pair_id] = payload
            pair_payloads[int(row["id"])] = payload

        for row in conn.execute(
            "SELECT pair_id, event_type, ts FROM tracker_events ORDER BY pair_id, ts ASC"
        ):
            payload = pair_payloads.get(int(row["pair_id"]))
            if payload is None:
                continue
            event_type = str(row["event_type"])
            target = f"{event_type}_events"
            if target in payload:
                payload[target].append(_coerce_float(row["ts"], 0.0))

        for row in conn.execute(
            """
            SELECT pair_id, ts, entry_spread_pct, exit_spread_pct
            FROM tracker_records
            ORDER BY pair_id, ts ASC
            """
        ):
            payload = pair_payloads.get(int(row["pair_id"]))
            if payload is None:
                continue
            payload["records"].append(
                {
                    "timestamp": _coerce_float(row["ts"], 0.0),
                    "entry_spread": _coerce_float(row["entry_spread_pct"], 0.0),
                    "exit_spread": _coerce_float(row["exit_spread_pct"], 0.0),
                }
            )
    finally:
        conn.close()
    return pairs, saved_at


def _sqlite_has_blocks(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'tracker_pair_blocks'"
    ).fetchone()
    return row is not None


def _placeholders(values: list[int]) -> str:
    return ",".join("?" for _ in values)


def _safe_in(
    conn: sqlite3.Connection,
    column_expr: str,
    values: list[int],
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


def _load_blocks_from_sqlite(
    state_path: Path,
    *,
    selected_block_ids: list[int] | None = None,
    selected_session_ids: list[int] | None = None,
    selected_only: bool = True,
    closed_only: bool = True,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    saved_at = 0.0
    conn = sqlite3.connect(state_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-262144")  # 256MB cache (4x previous)
    # mmap the entire file for read-heavy workloads
    try:
        db_size = Path(state_path).stat().st_size
        conn.execute(f"PRAGMA mmap_size={max(268435456, db_size + 10 * 1024 * 1024)}")
    except OSError:
        conn.execute("PRAGMA mmap_size=268435456")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA query_only=1")
    try:
        if not _sqlite_has_blocks(conn):
            pairs, saved_at = _load_pairs_from_sqlite(state_path)
            legacy_blocks = []
            for pair_id, payload in pairs.items():
                legacy_blocks.append(
                    {
                        "block_id": 0,
                        "session_id": 0,
                        "pair_id": pair_id,
                        "pair_key": pair_id,
                        "records": list(payload.get("records", [])),
                        "inverted_events": list(payload.get("inverted_events", [])),
                        "boundary_reason": "legacy_pair",
                        "selected_for_training": True,
                        "is_open": False,
                    }
                )
            return legacy_blocks, saved_at, {
                "num_blocks": len(legacy_blocks),
                "num_sessions": 0,
                "state_storage_kind": "sqlite_legacy_pairs",
                "selected_only": False,
                "closed_only": False,
            }

        meta_rows = {
            row["key"]: row["value"]
            for row in conn.execute("SELECT key, value FROM tracker_meta")
        }
        saved_at = _coerce_float(meta_rows.get("last_flush_at"), 0.0)

        block_filter: list[str] = []
        params: list[Any] = []
        if selected_only:
            block_filter.append("b.selected_for_training = 1")
        if closed_only:
            block_filter.append("b.is_open = 0")
        if selected_session_ids is not None:
            normalized_session_ids = [int(session_id) for session_id in selected_session_ids if int(session_id) > 0]
            if not normalized_session_ids:
                return [], saved_at, {
                    "num_blocks": 0,
                    "num_sessions": 0,
                    "state_storage_kind": "sqlite_blocks",
                    "selected_only": selected_only,
                    "closed_only": closed_only,
                }
            clause, clause_params = _safe_in(
                conn,
                "b.session_id",
                normalized_session_ids,
                temp_table_prefix="session_ids",
            )
            if clause:
                block_filter.append(clause)
                params.extend(clause_params)
        if selected_block_ids is not None:
            normalized_ids = [int(block_id) for block_id in selected_block_ids if int(block_id) > 0]
            if not normalized_ids:
                return [], saved_at, {
                    "num_blocks": 0,
                    "num_sessions": 0,
                    "state_storage_kind": "sqlite_blocks",
                    "selected_only": selected_only,
                    "closed_only": closed_only,
                }
            clause, clause_params = _safe_in(
                conn,
                "b.id",
                normalized_ids,
                temp_table_prefix="block_ids",
            )
            if clause:
                block_filter.append(clause)
                params.extend(clause_params)
        where_clause = f"WHERE {' AND '.join(block_filter)}" if block_filter else ""
        block_rows = list(
            conn.execute(
                f"""
                SELECT
                    b.id,
                    b.session_id,
                    b.start_ts,
                    b.end_ts,
                    b.record_count,
                    b.boundary_reason,
                    b.selected_for_training,
                    b.is_open,
                    p.symbol,
                    p.buy_ex,
                    p.buy_mt,
                    p.sell_ex,
                    p.sell_mt
                FROM tracker_pair_blocks b
                JOIN tracker_pairs p ON p.id = b.pair_id
                {where_clause}
                ORDER BY b.session_id ASC, p.symbol ASC, b.start_ts ASC, b.id ASC
                """,
                params,
            )
        )
        if not block_rows:
            return [], saved_at, {
                "num_blocks": 0,
                "num_sessions": 0,
                "state_storage_kind": "sqlite_blocks",
                "selected_only": selected_only,
                "closed_only": closed_only,
            }
        blocks_by_id: dict[int, dict[str, Any]] = {}
        block_ids = []
        session_ids = set()
        for row in block_rows:
            block_id = int(row["id"])
            pair_key = f"{row['symbol']}|{row['buy_ex']}|{row['buy_mt']}|{row['sell_ex']}|{row['sell_mt']}"
            blocks_by_id[block_id] = {
                "block_id": block_id,
                "session_id": int(row["session_id"]),
                "pair_id": pair_key,
                "pair_key": pair_key,
                "records": [],
                "inverted_events": [],
                "boundary_reason": str(row["boundary_reason"]),
                "selected_for_training": bool(row["selected_for_training"]),
                "is_open": bool(row["is_open"]),
                "record_count": int(row["record_count"]),
                "start_ts": _coerce_float(row["start_ts"], 0.0),
                "end_ts": _coerce_float(row["end_ts"], 0.0),
            }
            block_ids.append(block_id)
            session_ids.add(int(row["session_id"]))

        for chunk_start in range(0, len(block_ids), _SQLITE_BLOCK_FETCH_CHUNK):
            chunk_ids = block_ids[chunk_start : chunk_start + _SQLITE_BLOCK_FETCH_CHUNK]
            if not chunk_ids:
                continue
            placeholders = ",".join("?" for _ in chunk_ids)
            for row in conn.execute(
                f"""
                SELECT block_id, ts, entry_spread_pct, exit_spread_pct
                FROM tracker_records
                WHERE block_id IN ({placeholders})
                ORDER BY block_id ASC, ts ASC
                """,
                chunk_ids,
            ):
                payload = blocks_by_id.get(int(row["block_id"]))
                if payload is None:
                    continue
                payload["records"].append(
                    {
                        "timestamp": _coerce_float(row["ts"], 0.0),
                        "entry_spread": _coerce_float(row["entry_spread_pct"], 0.0),
                        "exit_spread": _coerce_float(row["exit_spread_pct"], 0.0),
                    }
                )

            for row in conn.execute(
                f"""
                SELECT block_id, ts
                FROM tracker_events
                WHERE event_type = 'inverted' AND block_id IN ({placeholders})
                ORDER BY block_id ASC, ts ASC
                """,
                chunk_ids,
            ):
                payload = blocks_by_id.get(int(row["block_id"]))
                if payload is None:
                    continue
                payload["inverted_events"].append(_coerce_float(row["ts"], 0.0))
    finally:
        conn.close()

    blocks = list(blocks_by_id.values())
    return blocks, saved_at, {
        "num_blocks": len(blocks),
        "num_sessions": len(session_ids),
        "state_storage_kind": "sqlite_blocks",
        "selected_only": selected_only,
        "closed_only": closed_only,
    }


def _load_pairs_payload(state_path: Path) -> tuple[dict[str, Any], float, str]:
    suffix = state_path.suffix.lower()
    if suffix == ".json":
        pairs, saved_at = _load_pairs_from_json(state_path)
        return pairs, saved_at, "json"
    pairs, saved_at = _load_pairs_from_sqlite(state_path)
    return pairs, saved_at, "sqlite"


def _load_tracker_gap_threshold_sec(state_path: Path) -> float | None:
    if Path(state_path).suffix.lower() == ".json" or not Path(state_path).exists():
        return None
    conn = sqlite3.connect(state_path, timeout=30.0)
    try:
        row = conn.execute(
            "SELECT value FROM tracker_meta WHERE key = 'gap_threshold_sec'"
        ).fetchone()
        if row is None:
            return None
        return float(row[0])
    except Exception:
        return None
    finally:
        conn.close()


def _build_closed_episodes(
    records: list[dict[str, float]],
    *,
    block_id: int,
    session_id: int,
) -> list[Any]:
    if not records:
        return []
    return sorted(
        compute_closed_episodes(
            [
                SpreadRecord(
                    timestamp=float(record["timestamp"]),
                    entry_spread_pct=float(record["entry_spread"]),
                    exit_spread_pct=float(record["exit_spread"]),
                    session_id=int(session_id),
                    block_id=int(block_id),
                )
                for record in records
            ]
        ),
        key=lambda episode: (float(episode.end_ts), float(episode.start_ts)),
    )


def _build_closed_episodes_for_blocks(blocks: list[dict[str, Any]]) -> list[Any]:
    if not blocks:
        return []
    ordered_blocks = sorted(
        blocks,
        key=lambda block: (
            float(block.get("start_ts") or 0.0),
            float(block.get("end_ts") or 0.0),
            int(block.get("block_id") or 0),
        ),
    )
    records: list[SpreadRecord] = []
    synthetic_to_storage_block_id: dict[int, int] = {}
    for synthetic_block_id, block in enumerate(ordered_blocks, start=1):
        storage_block_id = int(block.get("block_id") or 0)
        synthetic_to_storage_block_id[synthetic_block_id] = storage_block_id
        session_id = int(block.get("session_id") or 0)
        normalized_records = [
            canonicalize_record(record, fallback_ts=float(index))
            for index, record in enumerate(block.get("records", []))
            if isinstance(record, dict)
        ]
        normalized_records.sort(key=lambda record: record["timestamp"])
        records.extend(
            SpreadRecord(
                timestamp=float(record["timestamp"]),
                entry_spread_pct=float(record["entry_spread"]),
                exit_spread_pct=float(record["exit_spread"]),
                session_id=session_id,
                block_id=synthetic_block_id,
            )
            for record in normalized_records
        )
    episodes = compute_closed_episodes(records)
    for episode in episodes:
        episode.block_id = int(synthetic_to_storage_block_id.get(int(episode.block_id or 0), int(episode.block_id or 0)))
    return sorted(
        episodes,
        key=lambda episode: (float(episode.end_ts), float(episode.start_ts)),
    )


def _build_closed_episodes_from_records(
    records: list[dict[str, float | int]],
) -> list[Any]:
    """Build episodes from already-normalized records (with block_id, session_id).

    Computes episodes directly from dict records — avoids 6.3M SpreadRecord
    allocations and the redundant canonicalize + sort of _build_closed_episodes_for_blocks.
    Uses the same algorithm as compute_closed_episodes but on dicts.
    """
    if not records:
        return []
    # Assign synthetic block_ids starting from 1 (episode detector skips block_id=0
    # and resets state on block_id transitions).
    storage_to_synthetic: dict[int, int] = {}
    counter = 0
    tagged: list[tuple[int, dict[str, float | int]]] = []
    for r in records:
        storage_bid = int(r.get("block_id") or 0)
        syn = storage_to_synthetic.get(storage_bid)
        if syn is None:
            counter += 1
            syn = counter
            storage_to_synthetic[storage_bid] = syn
        tagged.append((syn, r))
    synthetic_to_storage = {v: k for k, v in storage_to_synthetic.items()}

    # Sort by (synthetic_block_id, timestamp) — same order compute_closed_episodes uses
    tagged.sort(key=lambda item: (item[0], float(item[1]["timestamp"])))

    # Inline episode detection — same algorithm as compute_closed_episodes
    _BASELINE_WINDOW = 32
    _SOURCE_VERSION = "recurring_v1"
    episodes: list[Any] = []
    baseline: deque[float] = deque(maxlen=_BASELINE_WINDOW)
    last_entry: float | None = None
    current_bid = 0
    active: TrackerEpisode | None = None

    for syn_bid, r in tagged:
        if syn_bid != current_bid:
            baseline.clear()
            last_entry = None
            current_bid = syn_bid
            active = None

        current_entry = float(r["entry_spread"])
        n = len(baseline)
        if n > 0:
            s = sorted(baseline)
            mid = n >> 1
            bmed = s[mid] if (n & 1) else (s[mid - 1] + s[mid]) * 0.5
            ds = sorted(abs(v - bmed) for v in baseline)
            bmad = max(ds[mid] if (n & 1) else (ds[mid - 1] + ds[mid]) * 0.5, 0.0)
        else:
            bmed = current_entry
            bmad = 0.0

        act_thresh = bmed + max(1.0 * bmad, 0.05)
        rel_thresh = bmed + max(0.25 * bmad, 0.02)

        if active is None:
            if last_entry is not None and last_entry <= act_thresh and current_entry > act_thresh:
                storage_bid = int(synthetic_to_storage.get(syn_bid, syn_bid))
                active = TrackerEpisode(
                    start_ts=float(r["timestamp"]),
                    peak_ts=float(r["timestamp"]),
                    end_ts=0.0,
                    duration_sec=0.0,
                    peak_entry_spread=current_entry,
                    exit_spread_at_close=float(r.get("exit_spread") or 0.0),
                    baseline_median=float(bmed),
                    baseline_mad=float(bmad),
                    activation_threshold=float(act_thresh),
                    release_threshold=float(rel_thresh),
                    session_id=int(r.get("session_id") or 0),
                    block_id=storage_bid,
                    source_version=_SOURCE_VERSION,
                    is_closed=False,
                )
        else:
            if current_entry >= active.peak_entry_spread:
                active.peak_entry_spread = current_entry
                active.peak_ts = float(r["timestamp"])
            if current_entry <= active.release_threshold:
                active.end_ts = float(r["timestamp"])
                active.duration_sec = max(0.0, active.end_ts - active.start_ts)
                active.exit_spread_at_close = float(r.get("exit_spread") or 0.0)
                active.is_closed = True
                episodes.append(active)
                active = None

        baseline.append(current_entry)
        last_entry = current_entry

    return sorted(
        episodes,
        key=lambda ep: (float(ep.end_ts), float(ep.start_ts)),
    )


def _windowed_entry_spreads(records: list[dict[str, Any]], *, tail: bool, limit: int) -> list[float]:
    # Fast path: records already canonical and sorted (from _normalized_block_records)
    if records and isinstance(records[0], dict) and "timestamp" in records[0]:
        if tail:
            return [float(r["entry_spread"]) for r in records[-max(int(limit), 0):]]
        return [float(r["entry_spread"]) for r in records[: max(int(limit), 0)]]
    # Slow path: legacy format
    normalized = [
        canonicalize_record(record, fallback_ts=float(index))
        for index, record in enumerate(records)
        if isinstance(record, dict)
    ]
    normalized.sort(key=lambda record: record["timestamp"])
    if tail:
        normalized = normalized[-max(int(limit), 0):]
    else:
        normalized = normalized[: max(int(limit), 0)]
    return [float(record["entry_spread"]) for record in normalized]


def _normalized_block_records(block: dict[str, Any]) -> list[dict[str, float | int]]:
    block_id = int(block.get("block_id") or 0)
    session_id = int(block.get("session_id") or 0)
    raw_records = block.get("records", [])
    if not raw_records:
        return []
    # Fast path: canonical keys from SQLite loader — already sorted by ts ASC.
    # Enrich in-place (add block_id/session_id) to avoid 6.3M dict allocations.
    first = raw_records[0]
    if isinstance(first, dict) and "timestamp" in first:
        for r in raw_records:
            r["block_id"] = block_id
            r["session_id"] = session_id
        return raw_records
    # Slow path: legacy format — needs coercion and sort
    result: list[dict[str, float | int]] = []
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            continue
        ts = _coerce_float(record.get("ts"), float(index))
        entry = _coerce_float(record.get("entry", record.get("entry_spread_pct", 0.0)), 0.0)
        exit_ = _coerce_float(record.get("exit", record.get("exit_spread_pct", 0.0)), 0.0)
        result.append({
            "timestamp": ts,
            "entry_spread": entry,
            "exit_spread": exit_,
            "block_id": block_id,
            "session_id": session_id,
        })
    result.sort(key=lambda record: record["timestamp"])
    return result


def _regime_window_size(left_record_count: int, right_record_count: int) -> int:
    shortest = max(0, min(int(left_record_count), int(right_record_count)))
    if shortest <= 0:
        return 0
    suggested = max(shortest // 2, _MIN_REGIME_WINDOW_RECORDS)
    return max(1, min(_MAX_REGIME_WINDOW_RECORDS, suggested, shortest))


def _regime_shift_metrics(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
) -> dict[str, Any]:
    left_count = sum(1 for record in left_records if isinstance(record, dict))
    right_count = sum(1 for record in right_records if isinstance(record, dict))
    window_size = _regime_window_size(left_count, right_count)
    if window_size <= 0:
        return {
            "window_size": 0,
            "left_count": int(left_count),
            "right_count": int(right_count),
            "mean_delta": 0.0,
            "std_ratio": 0.0,
            "score": 0.0,
        }
    left_values = _windowed_entry_spreads(left_records, tail=True, limit=window_size)
    right_values = _windowed_entry_spreads(right_records, tail=False, limit=window_size)
    if not left_values or not right_values:
        return {
            "window_size": 0,
            "left_count": int(left_count),
            "right_count": int(right_count),
            "mean_delta": 0.0,
            "std_ratio": 0.0,
            "score": 0.0,
        }
    left_np = np.asarray(left_values, dtype=float)
    right_np = np.asarray(right_values, dtype=float)
    left_mean = float(np.mean(left_np))
    right_mean = float(np.mean(right_np))
    left_std = float(np.std(left_np))
    right_std = float(np.std(right_np))
    mean_delta = abs(left_mean - right_mean) / max(left_std, right_std, 1e-6)
    std_ratio = max(left_std, right_std, 1e-6) / max(min(left_std, right_std), 1e-6)
    return {
        "window_size": int(window_size),
        "left_count": int(left_count),
        "right_count": int(right_count),
        "mean_delta": float(mean_delta),
        "std_ratio": float(std_ratio),
        "score": float(max(mean_delta, std_ratio)),
    }


def _merge_summary(values: list[float]) -> dict[str, float]:
    return {
        "p10": _linear_percentile(values, 10.0),
        "p25": _linear_percentile(values, 25.0),
        "p50": _linear_percentile(values, 50.0),
        "p75": _linear_percentile(values, 75.0),
        "p90": _linear_percentile(values, 90.0),
        "max": max((float(value) for value in values), default=0.0),
    } if values else {
        "p10": 0.0,
        "p25": 0.0,
        "p50": 0.0,
        "p75": 0.0,
        "p90": 0.0,
        "max": 0.0,
    }


def _build_pair_segments(
    blocks: list[dict[str, Any]],
    *,
    allow_cross_session_merge: bool,
    max_session_gap_sec: float | None,
    regime_shift_score_threshold: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blocks_by_pair_session: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        pair_id = str(block.get("pair_id") or "")
        session_id = int(block.get("session_id") or 0)
        blocks_by_pair_session[(pair_id, session_id)].append(block)

    session_segments_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostics: dict[str, Any] = {
        "candidate_count": 0,
        "applied_count": 0,
        "rejected_gap_count": 0,
        "rejected_regime_shift_count": 0,
        "gap_sec_quantiles": _merge_summary([]),
        "score_quantiles": _merge_summary([]),
        "window_size_quantiles": _merge_summary([]),
        "threshold": None if regime_shift_score_threshold is None else float(regime_shift_score_threshold),
        "max_session_gap_sec": None if max_session_gap_sec is None else float(max_session_gap_sec),
        "calibration_log": [],
    }
    gap_values: list[float] = []
    score_values: list[float] = []
    window_sizes: list[float] = []

    for (pair_id, session_id), pair_session_blocks in blocks_by_pair_session.items():
        ordered_blocks = sorted(
            pair_session_blocks,
            key=lambda block: (
                float(block.get("start_ts") or 0.0),
                float(block.get("end_ts") or 0.0),
                int(block.get("block_id") or 0),
            ),
        )
        records: list[dict[str, float | int]] = []
        observable_end_ts = 0.0
        start_ts = 0.0
        for block in ordered_blocks:
            normalized_records = _normalized_block_records(block)
            if normalized_records and start_ts <= 0.0:
                start_ts = float(normalized_records[0]["timestamp"])
            if normalized_records:
                observable_end_ts = max(observable_end_ts, float(normalized_records[-1]["timestamp"]))
            records.extend(normalized_records)
        session_segments_by_pair[pair_id].append(
            {
                "pair_id": pair_id,
                "session_ids": [int(session_id)],
                "blocks": ordered_blocks,
                "records": records,
                "episodes": _build_closed_episodes_from_records(records),
                "observable_end_ts": float(observable_end_ts),
                "start_ts": float(start_ts),
                "end_ts": float(observable_end_ts),
                "cross_session_boundaries": [],
            }
        )

    merged_segments: list[dict[str, Any]] = []
    for pair_id, segments in session_segments_by_pair.items():
        ordered_segments = sorted(
            segments,
            key=lambda segment: (
                float(segment.get("start_ts") or 0.0),
                float(segment.get("end_ts") or 0.0),
                int(segment.get("session_ids", [0])[0]),
            ),
        )
        if not allow_cross_session_merge or len(ordered_segments) <= 1:
            merged_segments.extend(ordered_segments)
            continue
        active_segment = dict(ordered_segments[0])
        for next_segment in ordered_segments[1:]:
            left_end_ts = float(active_segment.get("end_ts") or 0.0)
            right_start_ts = float(next_segment.get("start_ts") or 0.0)
            gap_sec = max(0.0, right_start_ts - left_end_ts)
            metrics = _regime_shift_metrics(
                list(active_segment.get("records") or []),
                list(next_segment.get("records") or []),
            )
            diagnostics["candidate_count"] += 1
            gap_values.append(float(gap_sec))
            score_values.append(float(metrics["score"]))
            window_sizes.append(float(metrics["window_size"]))
            diagnostics["calibration_log"].append(
                {
                    "pair_id": pair_id,
                    "left_session_id": int(active_segment.get("session_ids", [0])[-1]),
                    "right_session_id": int(next_segment.get("session_ids", [0])[0]),
                    "gap_sec": float(gap_sec),
                    "regime_shift_score": float(metrics["score"]),
                    "regime_window_size": int(metrics["window_size"]),
                    "mean_delta": float(metrics["mean_delta"]),
                    "std_ratio": float(metrics["std_ratio"]),
                }
            )
            merge_allowed = True
            if max_session_gap_sec is not None and float(gap_sec) > float(max_session_gap_sec):
                diagnostics["rejected_gap_count"] += 1
                merge_allowed = False
            if (
                merge_allowed
                and regime_shift_score_threshold is not None
                and float(metrics["score"]) > float(regime_shift_score_threshold)
            ):
                diagnostics["rejected_regime_shift_count"] += 1
                merge_allowed = False
            if merge_allowed:
                diagnostics["applied_count"] += 1
                active_segment["session_ids"] = list(active_segment.get("session_ids") or []) + list(next_segment.get("session_ids") or [])
                active_segment["blocks"] = list(active_segment.get("blocks") or []) + list(next_segment.get("blocks") or [])
                active_segment["records"] = list(active_segment.get("records") or []) + list(next_segment.get("records") or [])
                active_segment["episodes"] = list(active_segment.get("episodes") or []) + list(next_segment.get("episodes") or [])
                active_segment["observable_end_ts"] = float(next_segment.get("observable_end_ts") or active_segment.get("observable_end_ts") or 0.0)
                active_segment["end_ts"] = float(next_segment.get("end_ts") or active_segment.get("end_ts") or 0.0)
                active_segment["cross_session_boundaries"] = list(active_segment.get("cross_session_boundaries") or []) + [
                    {
                        "left_session_id": int(active_segment.get("session_ids", [0])[-2]),
                        "right_session_id": int(next_segment.get("session_ids", [0])[0]),
                        "gap_sec": float(gap_sec),
                        "regime_shift_score": float(metrics["score"]),
                        "regime_window_size": int(metrics["window_size"]),
                    }
                ]
            else:
                merged_segments.append(active_segment)
                active_segment = dict(next_segment)
        merged_segments.append(active_segment)

    diagnostics["gap_sec_quantiles"] = _merge_summary(gap_values)
    diagnostics["score_quantiles"] = _merge_summary(score_values)
    diagnostics["window_size_quantiles"] = _merge_summary(window_sizes)
    return merged_segments, diagnostics


def compute_label_threshold(
    pair_key: str,
    episodes: list[Any],
    *,
    cost_floor_pct: float,
    percentile: float,
) -> float:
    del pair_key
    total_spreads = [
        float(getattr(episode, "total_spread", 0.0) or 0.0)
        for episode in episodes
    ]
    if not total_spreads:
        return float(max(0.0, cost_floor_pct))
    return float(max(float(cost_floor_pct), np.percentile(np.asarray(total_spreads, dtype=float), float(percentile))))


class _EpisodeIndex:
    """Pre-indexed episode data for fast lookups during window sliding."""

    __slots__ = ("episodes", "_end_ts_sorted", "_sort_order", "_start_ts", "_end_ts", "_total_spread")

    def __init__(self, episodes: list[Any]) -> None:
        self.episodes = episodes
        if not episodes:
            self._end_ts_sorted = np.empty(0, dtype=np.float64)
            self._sort_order = np.empty(0, dtype=np.intp)
            self._start_ts = np.empty(0, dtype=np.float64)
            self._end_ts = np.empty(0, dtype=np.float64)
            self._total_spread = np.empty(0, dtype=np.float64)
            return
        self._end_ts = np.array([float(getattr(ep, "end_ts", 0.0) or 0.0) for ep in episodes], dtype=np.float64)
        self._start_ts = np.array([float(ep.start_ts) for ep in episodes], dtype=np.float64)
        self._total_spread = np.array([float(ep.total_spread) for ep in episodes], dtype=np.float64)
        self._sort_order = np.argsort(self._end_ts)
        self._end_ts_sorted = self._end_ts[self._sort_order]

    def prior_episodes_for_threshold(self, current_ts: float, history_window_sec: float) -> list[Any]:
        lo = bisect.bisect_left(self._end_ts_sorted, current_ts - history_window_sec)
        hi = bisect.bisect_right(self._end_ts_sorted, current_ts)
        return [self.episodes[self._sort_order[i]] for i in range(lo, hi)]

    def future_episodes_with_spreads(
        self, current_ts: float, horizon_end_ts: float, pair_threshold: float,
    ) -> tuple[list[float], Any | None, float]:
        mask = (self._start_ts > current_ts) & (self._end_ts <= horizon_end_ts)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return [], None, 0.0
        spreads = self._total_spread[idxs]
        future_total_spreads = spreads.tolist()
        peak = float(spreads.max())
        qualified_mask = spreads >= pair_threshold
        if not np.any(qualified_mask):
            return future_total_spreads, None, peak
        first_qualified_pos = int(np.argmax(qualified_mask))
        return future_total_spreads, self.episodes[idxs[first_qualified_pos]], peak


def _label_window_from_episodes(
    current_ts: float,
    episodes: list[Any],
    *,
    prediction_horizon_sec: int,
    min_total_spread_pct: float,
    adaptive_threshold_enabled: bool = False,
    pair_key: str = "",
    label_cost_floor_pct: float | None = None,
    label_percentile: float = 70.0,
    label_episode_window_days: int = 5,
    _episode_index: _EpisodeIndex | None = None,
) -> dict[str, Any]:
    horizon_end_ts = float(current_ts) + float(prediction_horizon_sec)
    effective_cost_floor = float(max(0.0, label_cost_floor_pct if label_cost_floor_pct is not None else min_total_spread_pct or 0.0))
    pair_threshold = float(max(0.0, min_total_spread_pct or 0.0))
    threshold_support = 0
    ep_idx = _episode_index
    if adaptive_threshold_enabled:
        history_window_sec = max(int(label_episode_window_days), 1) * 24 * 60 * 60
        if ep_idx is not None:
            prior_episodes = ep_idx.prior_episodes_for_threshold(current_ts, float(history_window_sec))
        else:
            prior_episodes = [
                episode
                for episode in episodes
                if float(getattr(episode, "end_ts", 0.0) or 0.0) <= float(current_ts)
                and float(getattr(episode, "end_ts", 0.0) or 0.0) >= (float(current_ts) - float(history_window_sec))
            ]
        threshold_support = int(len(prior_episodes))
        pair_threshold = compute_label_threshold(
            pair_key,
            prior_episodes,
            cost_floor_pct=effective_cost_floor,
            percentile=label_percentile,
        )
    if ep_idx is not None:
        future_total_spreads, first_qualified, peak = ep_idx.future_episodes_with_spreads(
            current_ts, horizon_end_ts, pair_threshold,
        )
    else:
        future_episodes = [
            episode
            for episode in episodes
            if float(episode.start_ts) > float(current_ts)
            and float(episode.end_ts) <= horizon_end_ts
        ]
        future_total_spreads = [float(episode.total_spread) for episode in future_episodes]
        peak = max(future_total_spreads) if future_total_spreads else 0.0
        qualified = [ep for ep in future_episodes if float(ep.total_spread) >= float(pair_threshold)]
        first_qualified = qualified[0] if qualified else None
    if first_qualified is not None:
        return {
            "y_class": 1.0,
            "y_eta": float(first_qualified.end_ts) - float(current_ts),
            "timeout_reason": "qualified_episode",
            "label_threshold": float(pair_threshold),
            "label_threshold_support": int(threshold_support),
            "peak_future_total_spread": peak,
            "qualified_episode_total_spread": float(first_qualified.total_spread),
            "future_episode_total_spreads": future_total_spreads,
        }
    return {
        "y_class": 0.0,
        "y_eta": 0.0,
        "timeout_reason": "sub_threshold_only" if future_total_spreads else "no_future_episode",
        "label_threshold": float(pair_threshold),
        "label_threshold_support": int(threshold_support),
        "peak_future_total_spread": peak,
        "qualified_episode_total_spread": 0.0,
        "future_episode_total_spreads": future_total_spreads,
    }


def _empty_label_audit() -> dict[str, Any]:
    return {
        "positive_entry_spread_buckets": {
            "lt_0_30": 0,
            "0_30_to_0_50": 0,
            "0_50_to_1_00": 0,
            "1_00_to_2_00": 0,
            "ge_2_00": 0,
        },
        "future_episode_total_spread_quantiles": {
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        },
        "timeout_peak_future_total_spread_buckets": {
            "lt_0_30": 0,
            "0_30_to_0_50": 0,
            "0_50_to_0_80": 0,
            "0_80_to_1_00": 0,
            "ge_1_00": 0,
        },
        "timeouts_without_future_episode": 0,
        "timeouts_with_only_sub_threshold_episode": 0,
        "right_censored_windows": 0,
    }


def _finalize_label_audit(
    positive_entry_spread_buckets: dict[str, int],
    future_episode_total_spreads: list[float],
    timeout_peak_future_total_spread_buckets: dict[str, int],
    *,
    timeouts_without_future_episode: int,
    timeouts_with_only_sub_threshold_episode: int,
    right_censored_windows: int,
) -> dict[str, Any]:
    return {
        "positive_entry_spread_buckets": {
            "lt_0_30": int(positive_entry_spread_buckets.get("lt_0_30", 0)),
            "0_30_to_0_50": int(positive_entry_spread_buckets.get("0_30_to_0_50", 0)),
            "0_50_to_1_00": int(positive_entry_spread_buckets.get("0_50_to_1_00", 0)),
            "1_00_to_2_00": int(positive_entry_spread_buckets.get("1_00_to_2_00", 0)),
            "ge_2_00": int(positive_entry_spread_buckets.get("ge_2_00", 0)),
        },
        "future_episode_total_spread_quantiles": _quantile_summary(future_episode_total_spreads),
        "timeout_peak_future_total_spread_buckets": {
            "lt_0_30": int(timeout_peak_future_total_spread_buckets.get("lt_0_30", 0)),
            "0_30_to_0_50": int(timeout_peak_future_total_spread_buckets.get("0_30_to_0_50", 0)),
            "0_50_to_0_80": int(timeout_peak_future_total_spread_buckets.get("0_50_to_0_80", 0)),
            "0_80_to_1_00": int(timeout_peak_future_total_spread_buckets.get("0_80_to_1_00", 0)),
            "ge_1_00": int(timeout_peak_future_total_spread_buckets.get("ge_1_00", 0)),
        },
        "timeouts_without_future_episode": int(timeouts_without_future_episode),
        "timeouts_with_only_sub_threshold_episode": int(timeouts_with_only_sub_threshold_episode),
        "right_censored_windows": int(right_censored_windows),
    }


def _empty_block_diagnostics(sequence_length: int) -> dict[str, Any]:
    return {
        "record_count_quantiles": _quantile_summary([]),
        "duration_sec_quantiles": _quantile_summary([]),
        "inter_record_interval_sec_quantiles": _quantile_summary([]),
        "max_to_median_interval_ratio_quantiles": _quantile_summary([]),
        "irregular_block_count": 0,
        "record_count_threshold_counts": {
            "ge_8": 0,
            "ge_10": 0,
            "ge_12": 0,
            "ge_15": 0,
            "ge_20": 0,
        },
        "feature_window_feasibility": {
            "sequence_length": int(sequence_length),
            "eligible_blocks_for_sequence_length": 0,
            "eligible_sessions_with_any_eligible_block": 0,
        },
        "session_summaries": {},
    }


def _block_threshold_counts(record_counts: list[int]) -> dict[str, int]:
    thresholds = {
        "ge_8": 8,
        "ge_10": 10,
        "ge_12": 12,
        "ge_15": 15,
        "ge_20": 20,
    }
    return {
        label: int(sum(1 for record_count in record_counts if int(record_count) >= minimum))
        for label, minimum in thresholds.items()
    }


def _pure_python_median(values: list[float]) -> float:
    """Fast median for small lists — avoids numpy overhead."""
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return values[0]
    if n == 2:
        return (values[0] + values[1]) * 0.5
    values.sort()
    mid = n >> 1
    if n & 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) * 0.5


def _build_block_diagnostics(blocks: list[dict[str, Any]], *, sequence_length: int) -> dict[str, Any]:
    if not blocks:
        return _empty_block_diagnostics(sequence_length)

    record_counts: list[int] = []
    durations: list[float] = []
    median_intervals: list[float] = []
    max_to_median_ratios: list[float] = []
    irregular_block_count = 0
    # Use tuples instead of dicts for per-block session data (avoids 1.675M dict allocations)
    # (record_count, duration_sec, median_interval, max_to_median_ratio, is_irregular)
    session_block_tuples: dict[int, list[tuple[int, float, float, float, bool]]] = defaultdict(list)

    for block in blocks:
        record_count = int(block.get("record_count") or len(block.get("records", [])))
        start_ts = float(block.get("start_ts") or 0.0)
        end_ts = float(block.get("end_ts") or 0.0)
        median_interval = 0.0
        max_to_median_ratio = 0.0
        is_irregular = False
        if record_count >= 2:
            raw_records = block.get("records", [])
            if raw_records:
                # Compute intervals directly using pure Python (numpy has too much
                # overhead for typical 3-4 element arrays at 1.675M blocks)
                prev_ts = None
                intervals: list[float] = []
                first_ts = 0.0
                last_ts = 0.0
                for r in raw_records:
                    if not isinstance(r, dict):
                        continue
                    ts = float(r.get("timestamp") or r.get("ts") or 0.0)
                    if prev_ts is not None:
                        intervals.append(ts - prev_ts)
                    else:
                        first_ts = ts
                    last_ts = ts
                    prev_ts = ts
                if start_ts <= 0.0 or end_ts <= 0.0:
                    start_ts = first_ts
                    end_ts = last_ts
                if intervals:
                    max_interval = max(intervals)
                    median_interval = _pure_python_median(intervals)
                    if median_interval > 0.0:
                        max_to_median_ratio = max_interval / median_interval
                        is_irregular = max_to_median_ratio > 3.0
        elif record_count == 1 and (start_ts <= 0.0 or end_ts <= 0.0):
            raw_records = block.get("records", [])
            if raw_records:
                r = raw_records[0]
                if isinstance(r, dict):
                    ts = float(r.get("timestamp") or r.get("ts") or 0.0)
                    start_ts = ts
                    end_ts = ts
        duration = max(0.0, end_ts - start_ts)
        record_counts.append(record_count)
        durations.append(duration)
        if median_interval > 0.0:
            median_intervals.append(median_interval)
        if max_to_median_ratio > 0.0:
            max_to_median_ratios.append(max_to_median_ratio)
        if is_irregular:
            irregular_block_count += 1
        session_block_tuples[int(block.get("session_id") or 0)].append(
            (record_count, duration, median_interval, max_to_median_ratio, is_irregular)
        )

    eligible_sessions = 0
    session_summaries: dict[str, Any] = {}
    # Tuple indices: 0=record_count, 1=duration, 2=median_interval, 3=max_to_median_ratio, 4=is_irregular
    for session_id, items in sorted(session_block_tuples.items()):
        session_record_counts = [t[0] for t in items]
        session_threshold_counts = _block_threshold_counts(session_record_counts)
        eligible_block_count = int(sum(1 for rc in session_record_counts if rc >= int(sequence_length)))
        if eligible_block_count > 0:
            eligible_sessions += 1
        session_summaries[str(session_id)] = {
            "num_blocks": len(items),
            "record_count_quantiles": _quantile_summary(session_record_counts),
            "duration_sec_quantiles": _quantile_summary([t[1] for t in items]),
            "inter_record_interval_sec_quantiles": _quantile_summary(
                [t[2] for t in items if t[2] > 0.0]
            ),
            "max_to_median_interval_ratio_quantiles": _quantile_summary(
                [t[3] for t in items if t[3] > 0.0]
            ),
            "irregular_block_count": int(sum(1 for t in items if t[4])),
            "max_record_count": max(session_record_counts) if session_record_counts else 0,
            "record_count_threshold_counts": session_threshold_counts,
            "eligible_blocks_for_sequence_length": eligible_block_count,
        }

    return {
        "record_count_quantiles": _quantile_summary(record_counts),
        "duration_sec_quantiles": _quantile_summary(durations),
        "inter_record_interval_sec_quantiles": _quantile_summary(median_intervals),
        "max_to_median_interval_ratio_quantiles": _quantile_summary(max_to_median_ratios),
        "irregular_block_count": int(irregular_block_count),
        "record_count_threshold_counts": _block_threshold_counts(record_counts),
        "feature_window_feasibility": {
            "sequence_length": int(sequence_length),
            "eligible_blocks_for_sequence_length": int(sum(1 for record_count in record_counts if record_count >= int(sequence_length))),
            "eligible_sessions_with_any_eligible_block": int(eligible_sessions),
        },
        "session_summaries": session_summaries,
    }


def build_dataset_bundle(
    state_path: Path,
    sequence_length: int = 15,
    prediction_horizon_sec: int = 14_400,
    *,
    min_total_spread_pct: float = 0.0,
    label_cost_floor_pct: float | None = None,
    label_percentile: float | None = None,
    label_episode_window_days: int | None = None,
    selected_block_ids: list[int] | None = None,
    selected_session_ids: list[int] | None = None,
    selected_only: bool | None = None,
    closed_only: bool = True,
    allow_cross_session_merge: bool = False,
    max_session_gap_sec: float | None = None,
    regime_shift_score_threshold: float | None = _DEFAULT_REGIME_SHIFT_SCORE_THRESHOLD,
    _preloaded_blocks: tuple[list[dict[str, Any]], float, dict[str, Any]] | None = None,
    _precomputed_pair_segments: tuple[list[dict[str, Any]], dict[str, Any]] | None = None,
    _precomputed_segment_features: dict[int, list[list[float]]] | None = None,
) -> DatasetBundle:
    X_samples: list[list[list[float]]] = []
    y_class: list[float] = []
    y_eta: list[float] = []
    sample_label_thresholds: list[float] = []
    pair_ids: list[str] = []
    block_ids: list[int] = []
    session_ids: list[int] = []
    timestamps: list[float] = []
    label_end_timestamps: list[float] = []
    last_entries: list[float] = []
    state_path = Path(state_path)
    storage_kind = "json"
    saved_at = 0.0
    block_summary: dict[str, Any] = {"num_blocks": 0, "num_sessions": 0}
    positive_entry_spread_buckets: dict[str, int] = {}
    timeout_peak_future_total_spread_buckets: dict[str, int] = {}
    future_episode_total_spreads: list[float] = []
    timeouts_without_future_episode = 0
    timeouts_with_only_sub_threshold_episode = 0
    skipped_windows_right_censored = 0
    pair_label_thresholds: dict[str, list[float]] = defaultdict(list)

    if _preloaded_blocks is not None:
        blocks, saved_at, block_summary = _preloaded_blocks
        storage_kind = str(block_summary.get("state_storage_kind", "preloaded"))
    elif state_path.suffix.lower() == ".json":
        pairs, saved_at = _load_pairs_from_json(state_path)
        blocks = []
        for pair_id, payload in pairs.items():
            blocks.append(
                {
                    "block_id": 0,
                    "session_id": 0,
                    "pair_id": pair_id,
                    "records": list(payload.get("records", [])),
                    "inverted_events": list(payload.get("inverted_events", [])),
                }
            )
        storage_kind = "json"
        block_summary = {"num_blocks": len(blocks), "num_sessions": 0}
    else:
        blocks, saved_at, block_summary = _load_blocks_from_sqlite(
            state_path,
            selected_block_ids=selected_block_ids,
            selected_session_ids=selected_session_ids,
            selected_only=True if selected_only is None else bool(selected_only),
            closed_only=closed_only,
        )
        storage_kind = str(block_summary.get("state_storage_kind", "sqlite"))

    legacy_min_total_spread_pct = max(0.0, float(min_total_spread_pct or 0.0))
    adaptive_threshold_enabled = any(value is not None for value in (label_cost_floor_pct, label_percentile, label_episode_window_days))
    effective_label_cost_floor_pct = (
        max(0.0, float(label_cost_floor_pct))
        if label_cost_floor_pct is not None
        else (legacy_min_total_spread_pct if legacy_min_total_spread_pct > 0.0 else 1.0)
    )
    effective_label_percentile = float(label_percentile if label_percentile is not None else 70.0)
    effective_label_episode_window_days = max(1, int(label_episode_window_days if label_episode_window_days is not None else 5))
    effective_min_total_spread_pct = effective_label_cost_floor_pct if adaptive_threshold_enabled else legacy_min_total_spread_pct
    effective_max_session_gap_sec = max_session_gap_sec
    if bool(allow_cross_session_merge) and effective_max_session_gap_sec is None:
        effective_max_session_gap_sec = _load_tracker_gap_threshold_sec(state_path)
    block_diagnostics = _build_block_diagnostics(blocks, sequence_length=sequence_length)
    if _precomputed_pair_segments is not None:
        pair_segments, cross_session_merge_diagnostics = _precomputed_pair_segments
    else:
        pair_segments, cross_session_merge_diagnostics = _build_pair_segments(
            blocks,
            allow_cross_session_merge=bool(allow_cross_session_merge),
            max_session_gap_sec=effective_max_session_gap_sec,
            regime_shift_score_threshold=regime_shift_score_threshold,
        )
    segment_episodes = [
        episode
        for segment in pair_segments
        for episode in list(segment.get("episodes") or [])
    ]
    episode_diagnostics = {
        "episode_count": int(len(segment_episodes)),
        "total_spread_quantiles": _quantile_summary([float(getattr(episode, "total_spread", 0.0) or 0.0) for episode in segment_episodes]),
        "duration_sec_quantiles": _quantile_summary([float(getattr(episode, "duration_sec", 0.0) or 0.0) for episode in segment_episodes]),
    }

    skipped_blocks_too_short = sum(
        1
        for block in blocks
        if int(block.get("record_count") or len(block.get("records", []))) < int(sequence_length)
    )
    skipped_windows_cross_session_boundary = 0
    cross_block_window_count = 0
    cross_session_window_count = 0
    for _seg_idx, segment in enumerate(pair_segments):
        episodes = list(segment.get("episodes") or [])
        observable_end_ts = float(segment.get("observable_end_ts") or 0.0)
        segment_records = list(segment.get("records") or [])
        if len(segment_records) < sequence_length:
            continue
        normalized_pair_id = str(segment.get("pair_id") or "")
        if _precomputed_segment_features is not None and _seg_idx in _precomputed_segment_features:
            segment_feature_rows = _precomputed_segment_features[_seg_idx]
        else:
            segment_feature_rows = build_feature_rows(
                segment_records,
                feature_names=list(FEATURE_NAMES),
                episodes=episodes,
            )
            if _precomputed_segment_features is not None:
                _precomputed_segment_features[_seg_idx] = segment_feature_rows
        _seg_session_ids = np.array([int(r.get("session_id") or 0) for r in segment_records], dtype=np.int64)
        _seg_block_ids = np.array([int(r.get("block_id") or 0) for r in segment_records], dtype=np.int64)
        _ep_index = _EpisodeIndex(episodes)
        for start in range(len(segment_records) - sequence_length + 1):
            window = segment_records[start : start + sequence_length]
            win_sessions = _seg_session_ids[start : start + sequence_length]
            if win_sessions[0] != win_sessions[-1] or np.unique(win_sessions).size > 1:
                cross_session_window_count += 1
                skipped_windows_cross_session_boundary += 1
                continue
            win_blocks = _seg_block_ids[start : start + sequence_length]
            if win_blocks[0] != win_blocks[-1] or np.unique(win_blocks).size > 1:
                cross_block_window_count += 1
            current_ts = float(window[-1]["timestamp"])
            if observable_end_ts > 0.0 and (float(current_ts) + float(prediction_horizon_sec)) > observable_end_ts:
                skipped_windows_right_censored += 1
                continue
            label = _label_window_from_episodes(
                current_ts,
                episodes,
                prediction_horizon_sec=prediction_horizon_sec,
                min_total_spread_pct=effective_min_total_spread_pct,
                adaptive_threshold_enabled=adaptive_threshold_enabled,
                pair_key=normalized_pair_id,
                label_cost_floor_pct=effective_label_cost_floor_pct,
                label_percentile=effective_label_percentile,
                label_episode_window_days=effective_label_episode_window_days,
                _episode_index=_ep_index,
            )

            X_samples.append(segment_feature_rows[start : start + sequence_length])
            y_class.append(float(label["y_class"]))
            y_eta.append(float(label["y_eta"]))
            sample_label_thresholds.append(float(label["label_threshold"]))
            pair_ids.append(normalized_pair_id)
            block_ids.append(int(window[-1].get("block_id") or 0))
            session_ids.append(int(window[-1].get("session_id") or 0))
            timestamps.append(current_ts)
            label_end_timestamps.append(float(current_ts) + float(prediction_horizon_sec))
            last_entries.append(float(window[-1]["entry_spread"]))
            pair_label_thresholds[normalized_pair_id].append(float(label["label_threshold"]))

            future_episode_total_spreads.extend(float(value) for value in label["future_episode_total_spreads"])
            if float(label["y_class"]) >= 0.5:
                _bucket_count(
                    float(window[-1]["entry_spread"]),
                    [
                        ("lt_0_30", -float("inf"), 0.30),
                        ("0_30_to_0_50", 0.30, 0.50),
                        ("0_50_to_1_00", 0.50, 1.00),
                        ("1_00_to_2_00", 1.00, 2.00),
                        ("ge_2_00", 2.00, float("inf")),
                    ],
                    positive_entry_spread_buckets,
                )
            else:
                _bucket_count(
                    float(label["peak_future_total_spread"]),
                    [
                        ("lt_0_30", -float("inf"), 0.30),
                        ("0_30_to_0_50", 0.30, 0.50),
                        ("0_50_to_0_80", 0.50, 0.80),
                        ("0_80_to_1_00", 0.80, 1.00),
                        ("ge_1_00", 1.00, float("inf")),
                    ],
                    timeout_peak_future_total_spread_buckets,
                )
                if label["timeout_reason"] == "no_future_episode":
                    timeouts_without_future_episode += 1
                elif label["timeout_reason"] == "sub_threshold_only":
                    timeouts_with_only_sub_threshold_episode += 1

    if X_samples:
        X_tensor = torch.from_numpy(np.array(X_samples, dtype=np.float32))
        y_class_tensor = torch.from_numpy(np.array(y_class, dtype=np.float32))
        y_eta_tensor = torch.from_numpy(np.array(y_eta, dtype=np.float32))
    else:
        X_tensor = torch.empty((0, sequence_length, len(FEATURE_NAMES)), dtype=torch.float32)
        y_class_tensor = torch.empty((0,), dtype=torch.float32)
        y_eta_tensor = torch.empty((0,), dtype=torch.float32)
    label_threshold_summary = {
        pair_id: {
            "latest": float(values[-1]),
            "min": float(min(values)),
            "p50": float(np.percentile(np.asarray(values, dtype=float), 50)),
            "p90": float(np.percentile(np.asarray(values, dtype=float), 90)),
            "max": float(max(values)),
            "samples": int(len(values)),
        }
        for pair_id, values in sorted(pair_label_thresholds.items())
        if values
    }

    summary = {
        "num_samples": len(X_samples),
        "num_positive_samples": int(sum(y_class)),
        "num_negative_samples": len(X_samples) - int(sum(y_class)),
        "num_pairs": len(set(pair_ids)),
        "num_blocks": int(block_summary.get("num_blocks", 0)),
        "num_sessions": int(block_summary.get("num_sessions", 0)),
        "blocks_used": len(set(block_ids)),
        "sessions_used": len(set(session_ids)),
        "block_ids_used": sorted({int(block_id) for block_id in block_ids if int(block_id) > 0}),
        "session_ids_used": sorted({int(session_id) for session_id in session_ids if int(session_id) > 0}),
        "selected_block_ids": sorted({int(block_id) for block_id in (selected_block_ids or []) if int(block_id) > 0}),
        "selected_session_ids": sorted({int(session_id) for session_id in (selected_session_ids or []) if int(session_id) > 0}),
        "skipped_blocks_too_short": int(skipped_blocks_too_short),
        "skipped_windows_right_censored": int(skipped_windows_right_censored),
        "skipped_windows_cross_session_boundary": int(skipped_windows_cross_session_boundary),
        "num_cross_block_windows": int(cross_block_window_count),
        "num_cross_session_windows": int(cross_session_window_count),
        "prediction_horizon_sec": int(prediction_horizon_sec),
        "sequence_length": int(sequence_length),
        "min_total_spread_pct": float(effective_min_total_spread_pct),
        "label_threshold_mode": "rolling_pair_percentile" if adaptive_threshold_enabled else "fixed_threshold",
        "label_cost_floor_pct": float(effective_label_cost_floor_pct),
        "label_percentile": float(effective_label_percentile),
        "label_episode_window_days": int(effective_label_episode_window_days),
        "label_thresholds": label_threshold_summary,
        "cross_session_merge_enabled": bool(allow_cross_session_merge),
        "cross_session_merges_applied": int(cross_session_merge_diagnostics.get("applied_count", 0)),
        "cross_session_gap_threshold_sec": None if effective_max_session_gap_sec is None else float(effective_max_session_gap_sec),
        "cross_session_boundaries": int(cross_session_merge_diagnostics.get("applied_count", 0)),
        "cross_session_merge_diagnostics": dict(cross_session_merge_diagnostics),
        "episode_diagnostics": episode_diagnostics,
        "labeling_method": "rolling_pair_percentile_take_profit_time_barrier" if adaptive_threshold_enabled else "episode_take_profit_time_barrier",
        "labeling_timeout_only": True,
        "label_audit": _finalize_label_audit(
            positive_entry_spread_buckets,
            future_episode_total_spreads,
            timeout_peak_future_total_spread_buckets,
            timeouts_without_future_episode=timeouts_without_future_episode,
            timeouts_with_only_sub_threshold_episode=timeouts_with_only_sub_threshold_episode,
            right_censored_windows=skipped_windows_right_censored,
        ),
        "block_diagnostics": block_diagnostics,
        "state_saved_at": float(saved_at),
        "state_storage_kind": storage_kind,
    }
    return DatasetBundle(
        X=X_tensor,
        y_class=y_class_tensor,
        y_eta=y_eta_tensor,
        label_thresholds=sample_label_thresholds,
        pair_ids=pair_ids,
        block_ids=block_ids,
        session_ids=session_ids,
        timestamps=timestamps,
        label_end_timestamps=label_end_timestamps,
        last_entries=last_entries,
        feature_names=list(FEATURE_NAMES),
        summary=summary,
    )


def _largest_remainder_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    raw_counts = [ratio * total for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    remainder = total - sum(counts)
    ranked = sorted(
        range(len(raw_counts)),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in ranked[:remainder]:
        counts[index] += 1
    if total >= 3:
        for index in range(3):
            if counts[index] == 0:
                donor = max(range(3), key=lambda candidate: counts[candidate])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1
    return counts


def build_group_splits(
    bundle: DatasetBundle,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    *,
    prediction_horizon_sec: int | None = None,
) -> dict[str, DatasetBundle]:
    if bundle.summary["num_samples"] == 0:
        empty = bundle.subset([], "empty")
        return {"train": empty, "val": empty, "test": empty}
    horizon_sec = int(prediction_horizon_sec or bundle.summary.get("prediction_horizon_sec", 14_400))

    def _pair_positive_lookup() -> dict[str, bool]:
        pair_has_positive: dict[str, bool] = {}
        for pair_id, label in zip(bundle.pair_ids, bundle.y_class.tolist()):
            pair_has_positive[pair_id] = pair_has_positive.get(pair_id, False) or bool(label >= 0.5)
        return pair_has_positive

    def _next_start_after_horizon(ordered_indices: list[int]) -> list[int | None]:
        result: list[int | None] = [None] * len(ordered_indices)
        pointer = 0
        ordered_timestamps = [float(bundle.timestamps[index]) for index in ordered_indices]
        for idx, current_ts in enumerate(ordered_timestamps):
            pointer = max(pointer, idx + 1)
            while pointer < len(ordered_timestamps) and ordered_timestamps[pointer] < current_ts + horizon_sec:
                pointer += 1
            result[idx] = pointer if pointer < len(ordered_timestamps) else None
        return result

    def _attach_summary(
        train_indices: list[int],
        val_indices: list[int],
        test_indices: list[int],
        *,
        split_mode: str,
        embargo_samples: int,
        fallback_reason: str,
        train_session_ids: list[int] | None = None,
        val_session_ids: list[int] | None = None,
        test_session_ids: list[int] | None = None,
    ) -> dict[str, DatasetBundle]:
        train_bundle = bundle.subset(train_indices, "train")
        val_bundle = bundle.subset(val_indices, "val")
        test_bundle = bundle.subset(test_indices, "test")
        split_pairs = {
            "train": set(train_bundle.pair_ids),
            "val": set(val_bundle.pair_ids),
            "test": set(test_bundle.pair_ids),
        }
        pair_has_positive = _pair_positive_lookup()
        overlap = len(split_pairs["train"] & split_pairs["val"])
        overlap += len(split_pairs["train"] & split_pairs["test"])
        overlap += len(split_pairs["val"] & split_pairs["test"])
        train_end_label_ts = max(train_bundle.label_end_timestamps) if train_bundle.label_end_timestamps else 0.0
        val_end_label_ts = max(val_bundle.label_end_timestamps) if val_bundle.label_end_timestamps else 0.0
        val_start_ts = min(val_bundle.timestamps) if val_bundle.timestamps else 0.0
        test_start_ts = min(test_bundle.timestamps) if test_bundle.timestamps else 0.0
        min_cross_split_gap_sec = min(
            max(0.0, val_start_ts - train_end_label_ts) if train_bundle.timestamps and val_bundle.timestamps else float("inf"),
            max(0.0, test_start_ts - val_end_label_ts) if val_bundle.timestamps and test_bundle.timestamps else float("inf"),
        )
        if min_cross_split_gap_sec == float("inf"):
            min_cross_split_gap_sec = 0.0
        split_summary = {
            "pair_overlap": overlap,
            "train_pairs": len(split_pairs["train"]),
            "val_pairs": len(split_pairs["val"]),
            "test_pairs": len(split_pairs["test"]),
            "train_positive_pairs": sum(1 for pid in split_pairs["train"] if pair_has_positive.get(pid, False)),
            "val_positive_pairs": sum(1 for pid in split_pairs["val"] if pair_has_positive.get(pid, False)),
            "test_positive_pairs": sum(1 for pid in split_pairs["test"] if pair_has_positive.get(pid, False)),
            "train_negative_pairs": sum(1 for pid in split_pairs["train"] if not pair_has_positive.get(pid, False)),
            "val_negative_pairs": sum(1 for pid in split_pairs["val"] if not pair_has_positive.get(pid, False)),
            "test_negative_pairs": sum(1 for pid in split_pairs["test"] if not pair_has_positive.get(pid, False)),
            "train_start_ts": min(train_bundle.timestamps) if train_bundle.timestamps else 0.0,
            "train_end_ts": max(train_bundle.timestamps) if train_bundle.timestamps else 0.0,
            "train_end_label_ts": train_end_label_ts,
            "val_start_ts": val_start_ts,
            "val_end_ts": max(val_bundle.timestamps) if val_bundle.timestamps else 0.0,
            "val_end_label_ts": val_end_label_ts,
            "test_start_ts": test_start_ts,
            "test_end_ts": max(test_bundle.timestamps) if test_bundle.timestamps else 0.0,
            "embargo_samples": int(embargo_samples),
            "embargo_time_sec": int(horizon_sec),
            "split_mode": split_mode,
            "split_mode_fallback_reason": str(fallback_reason),
            "train_session_ids": list(train_session_ids or []),
            "val_session_ids": list(val_session_ids or []),
            "test_session_ids": list(test_session_ids or []),
            "purged_temporal_separation_ok": bool(
                (not train_bundle.timestamps or not val_bundle.timestamps or train_end_label_ts <= val_start_ts)
                and (not val_bundle.timestamps or not test_bundle.timestamps or val_end_label_ts <= test_start_ts)
            ),
            "min_cross_split_gap_sec": float(min_cross_split_gap_sec),
            "global_temporal_order": bool(
                (max(train_bundle.timestamps) if train_bundle.timestamps else 0.0) <= (min(val_bundle.timestamps) if val_bundle.timestamps else float("inf"))
                and (max(val_bundle.timestamps) if val_bundle.timestamps else 0.0) <= (min(test_bundle.timestamps) if test_bundle.timestamps else float("inf"))
            ),
        }
        for split_bundle in (train_bundle, val_bundle, test_bundle):
            split_bundle.summary["split_summary"] = split_summary
        return {"train": train_bundle, "val": val_bundle, "test": test_bundle}

    valid_session_ids = [int(session_id) for session_id in bundle.session_ids if int(session_id) > 0]
    unique_sessions = sorted(
        set(valid_session_ids),
        key=lambda session_id: min(
            bundle.timestamps[index]
            for index, candidate in enumerate(bundle.session_ids)
            if int(candidate) == session_id
        ),
    ) if valid_session_ids else []
    if len(unique_sessions) >= 3:
        session_to_indices: dict[int, list[int]] = {}
        for index, session_id in enumerate(bundle.session_ids):
            normalized = int(session_id)
            if normalized <= 0:
                continue
            session_to_indices.setdefault(normalized, []).append(index)
        session_counts = _largest_remainder_counts(len(unique_sessions), (train_ratio, val_ratio, test_ratio))
        train_sessions = unique_sessions[: session_counts[0]]
        val_sessions = unique_sessions[session_counts[0] : session_counts[0] + session_counts[1]]
        test_sessions = unique_sessions[session_counts[0] + session_counts[1] :]
        if train_sessions and val_sessions and test_sessions:
            session_ranges = {
                session_id: {
                    "start_ts": min(bundle.timestamps[index] for index in session_to_indices.get(session_id, [])),
                    "end_label_ts": max(bundle.label_end_timestamps[index] for index in session_to_indices.get(session_id, [])),
                }
                for session_id in unique_sessions
            }
            session_split_ok = bool(
                session_ranges[train_sessions[-1]]["end_label_ts"] <= session_ranges[val_sessions[0]]["start_ts"]
                and session_ranges[val_sessions[-1]]["end_label_ts"] <= session_ranges[test_sessions[0]]["start_ts"]
            )
            if session_split_ok:
                return _attach_summary(
                    [index for session_id in train_sessions for index in session_to_indices.get(session_id, [])],
                    [index for session_id in val_sessions for index in session_to_indices.get(session_id, [])],
                    [index for session_id in test_sessions for index in session_to_indices.get(session_id, [])],
                    split_mode="session_chronological",
                    embargo_samples=0,
                    fallback_reason="",
                    train_session_ids=list(train_sessions),
                    val_session_ids=list(val_sessions),
                    test_session_ids=list(test_sessions),
                )
            session_fallback_reason = "session_gap_below_horizon"
        else:
            session_fallback_reason = "insufficient_session_groups"
    else:
        session_fallback_reason = "insufficient_sessions"

    ordered_indices = sorted(range(len(bundle.timestamps)), key=lambda index: (bundle.timestamps[index], index))
    target_counts = _largest_remainder_counts(len(ordered_indices), (train_ratio, val_ratio, test_ratio))
    next_start = _next_start_after_horizon(ordered_indices)
    valid_val_end_positions = [index for index, candidate in enumerate(next_start) if candidate is not None]

    chosen: tuple[int, int, int] | None = None
    best_score: tuple[float, float, float, float] | None = None
    for train_last_index, val_start in enumerate(next_start):
        if val_start is None:
            continue
        train_count = train_last_index + 1
        if train_count <= 0 or val_start >= len(ordered_indices) - 1:
            continue
        target_val_last = min(len(ordered_indices) - 2, max(val_start, val_start + target_counts[1] - 1))
        candidate_positions = sorted(
            {
                candidate
                for candidate in (
                    max(val_start, min(target_val_last, len(ordered_indices) - 2)),
                    max(val_start, min(target_val_last - 1, len(ordered_indices) - 2)),
                    max(val_start, min(target_val_last + 1, len(ordered_indices) - 2)),
                )
                if candidate < len(ordered_indices) - 1
            }
        )
        candidate_positions.extend(
            [
                candidate
                for candidate in valid_val_end_positions
                if candidate >= val_start and candidate < len(ordered_indices) - 1 and candidate not in candidate_positions
            ][:3]
        )
        for val_last_index in candidate_positions:
            test_start = next_start[val_last_index]
            if test_start is None:
                continue
            val_count = (val_last_index - val_start) + 1
            test_count = len(ordered_indices) - test_start
            if min(train_count, val_count, test_count) <= 0:
                continue
            embargo_samples = max(0, val_start - train_count) + max(0, test_start - (val_last_index + 1))
            score = (
                abs(train_count - target_counts[0]) + abs(val_count - target_counts[1]) + abs(test_count - target_counts[2]),
                float(embargo_samples),
                abs(float(bundle.timestamps[ordered_indices[val_start]]) - float(bundle.label_end_timestamps[ordered_indices[train_last_index]])),
                abs(float(bundle.timestamps[ordered_indices[test_start]]) - float(bundle.label_end_timestamps[ordered_indices[val_last_index]])),
            )
            if best_score is None or score < best_score:
                chosen = (train_last_index, val_start, val_last_index)
                best_score = score

    if chosen is None:
        counts = _largest_remainder_counts(len(ordered_indices), (train_ratio, val_ratio, test_ratio))
        train_end = counts[0]
        val_end = train_end + counts[1]
        return _attach_summary(
            ordered_indices[:train_end],
            ordered_indices[train_end:val_end],
            ordered_indices[val_end:],
            split_mode="sample_chronological_purged",
            embargo_samples=0,
            fallback_reason=session_fallback_reason or "purging_failed",
        )

    train_last_index, val_start, val_last_index = chosen
    test_start = next_start[val_last_index]
    assert test_start is not None
    embargo_samples = max(0, val_start - (train_last_index + 1)) + max(0, test_start - (val_last_index + 1))
    return _attach_summary(
        ordered_indices[: train_last_index + 1],
        ordered_indices[val_start : val_last_index + 1],
        ordered_indices[test_start:],
        split_mode="sample_chronological_purged",
        embargo_samples=embargo_samples,
        fallback_reason=session_fallback_reason,
    )


def compute_feature_stats(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = X.reshape(-1, X.shape[-1])
    mean = flattened.mean(dim=0)
    std = flattened.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def normalize_features(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (X - mean.view(1, 1, -1)) / std.view(1, 1, -1)
