from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .spread_tracker import SpreadRecord, compute_closed_episodes

FEATURE_NAMES = [
    "entry_spread", "exit_spread",
    "delta_entry", "delta_exit",
    "delta2_entry", "delta2_exit",
    "rolling_std_entry", "rolling_std_exit",
    "zscore_entry", "zscore_exit",
]

_ROLLING_WINDOW = 5
_SQLITE_BLOCK_FETCH_CHUNK = 500


@dataclass(slots=True)
class DatasetBundle:
    X: torch.Tensor
    y_class: torch.Tensor
    y_eta: torch.Tensor
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
        return {
            "timestamp": _coerce_float(record.get("timestamp"), fallback_ts),
            "entry_spread": _coerce_float(record.get("entry_spread"), 0.0),
            "exit_spread": _coerce_float(record.get("exit_spread"), 0.0),
        }
    return {
        "timestamp": _coerce_float(record.get("ts"), fallback_ts),
        "entry_spread": _coerce_float(record.get("entry", record.get("entry_spread_pct", 0.0)), 0.0),
        "exit_spread": _coerce_float(record.get("exit", record.get("exit_spread_pct", 0.0)), 0.0),
    }


def _rolling_std_zscore(buffer: deque[float], current: float) -> tuple[float, float]:
    if len(buffer) < 2:
        return 0.0, 0.0
    mean = sum(buffer) / len(buffer)
    variance = sum((v - mean) ** 2 for v in buffer) / len(buffer)
    std = math.sqrt(variance)
    if std < 1e-6:
        return 0.0, 0.0
    zscore = (current - mean) / std
    return std, zscore


def _feature_window(window: list[dict[str, float]]) -> list[list[float]]:
    features: list[list[float]] = []
    previous_entry: float | None = None
    previous_exit: float | None = None
    previous_delta_entry: float | None = None
    previous_delta_exit: float | None = None
    entry_buffer: deque[float] = deque(maxlen=_ROLLING_WINDOW)
    exit_buffer: deque[float] = deque(maxlen=_ROLLING_WINDOW)
    for record in window:
        entry_spread = record["entry_spread"]
        exit_spread = record["exit_spread"]
        delta_entry = 0.0 if previous_entry is None else entry_spread - previous_entry
        delta_exit = 0.0 if previous_exit is None else exit_spread - previous_exit
        delta2_entry = 0.0 if previous_delta_entry is None else delta_entry - previous_delta_entry
        delta2_exit = 0.0 if previous_delta_exit is None else delta_exit - previous_delta_exit
        entry_buffer.append(entry_spread)
        exit_buffer.append(exit_spread)
        rolling_std_entry, zscore_entry = _rolling_std_zscore(entry_buffer, entry_spread)
        rolling_std_exit, zscore_exit = _rolling_std_zscore(exit_buffer, exit_spread)
        features.append([
            entry_spread, exit_spread,
            delta_entry, delta_exit,
            delta2_entry, delta2_exit,
            rolling_std_entry, rolling_std_exit,
            zscore_entry, zscore_exit,
        ])
        previous_entry = entry_spread
        previous_exit = exit_spread
        previous_delta_entry = delta_entry
        previous_delta_exit = delta_exit
    return features


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
                    "ts": _coerce_float(row["ts"], 0.0),
                    "entry": _coerce_float(row["entry_spread_pct"], 0.0),
                    "exit": _coerce_float(row["exit_spread_pct"], 0.0),
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
            placeholders = ",".join("?" for _ in normalized_session_ids)
            block_filter.append(f"b.session_id IN ({placeholders})")
            params.extend(normalized_session_ids)
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
            placeholders = ",".join("?" for _ in normalized_ids)
            block_filter.append(f"b.id IN ({placeholders})")
            params.extend(normalized_ids)
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
                        "ts": _coerce_float(row["ts"], 0.0),
                        "entry": _coerce_float(row["entry_spread_pct"], 0.0),
                        "exit": _coerce_float(row["exit_spread_pct"], 0.0),
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


def _label_window_from_episodes(
    current_ts: float,
    episodes: list[Any],
    *,
    prediction_horizon_sec: int,
    min_total_spread_pct: float,
) -> dict[str, Any]:
    horizon_end_ts = float(current_ts) + float(prediction_horizon_sec)
    future_episodes = [
        episode
        for episode in episodes
        if float(episode.start_ts) > float(current_ts)
        and float(episode.end_ts) <= horizon_end_ts
    ]
    future_total_spreads = [float(episode.total_spread) for episode in future_episodes]
    qualified_episodes = [
        episode
        for episode in future_episodes
        if float(episode.total_spread) >= float(min_total_spread_pct or 0.0)
    ]
    if qualified_episodes:
        first_qualified = qualified_episodes[0]
        return {
            "y_class": 1.0,
            "y_eta": float(first_qualified.end_ts) - float(current_ts),
            "timeout_reason": "qualified_episode",
            "peak_future_total_spread": max(future_total_spreads) if future_total_spreads else 0.0,
            "qualified_episode_total_spread": float(first_qualified.total_spread),
            "future_episode_total_spreads": future_total_spreads,
        }
    return {
        "y_class": 0.0,
        "y_eta": 0.0,
        "timeout_reason": "sub_threshold_only" if future_episodes else "no_future_episode",
        "peak_future_total_spread": max(future_total_spreads) if future_total_spreads else 0.0,
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


def _build_block_diagnostics(blocks: list[dict[str, Any]], *, sequence_length: int) -> dict[str, Any]:
    if not blocks:
        return _empty_block_diagnostics(sequence_length)

    record_counts: list[int] = []
    durations: list[float] = []
    session_blocks: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for block in blocks:
        records = [record for record in block.get("records", []) if isinstance(record, dict)]
        record_count = int(block.get("record_count") or len(records))
        start_ts = float(block.get("start_ts") or 0.0)
        end_ts = float(block.get("end_ts") or 0.0)
        if (start_ts <= 0.0 or end_ts <= 0.0) and records:
            normalized_records = [
                canonicalize_record(record, fallback_ts=float(index))
                for index, record in enumerate(records)
            ]
            normalized_records.sort(key=lambda record: record["timestamp"])
            start_ts = float(normalized_records[0]["timestamp"])
            end_ts = float(normalized_records[-1]["timestamp"])
        record_counts.append(record_count)
        durations.append(max(0.0, end_ts - start_ts))
        session_blocks[int(block.get("session_id") or 0)].append(
            {
                "record_count": record_count,
                "duration_sec": max(0.0, end_ts - start_ts),
            }
        )

    eligible_sessions = 0
    session_summaries: dict[str, Any] = {}
    for session_id, items in sorted(session_blocks.items()):
        session_record_counts = [int(item["record_count"]) for item in items]
        session_threshold_counts = _block_threshold_counts(session_record_counts)
        eligible_block_count = int(sum(1 for record_count in session_record_counts if record_count >= int(sequence_length)))
        if eligible_block_count > 0:
            eligible_sessions += 1
        session_summaries[str(session_id)] = {
            "num_blocks": len(items),
            "record_count_quantiles": _quantile_summary(session_record_counts),
            "duration_sec_quantiles": _quantile_summary([float(item["duration_sec"]) for item in items]),
            "max_record_count": max(session_record_counts) if session_record_counts else 0,
            "record_count_threshold_counts": session_threshold_counts,
            "eligible_blocks_for_sequence_length": eligible_block_count,
        }

    return {
        "record_count_quantiles": _quantile_summary(record_counts),
        "duration_sec_quantiles": _quantile_summary(durations),
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
    selected_block_ids: list[int] | None = None,
    selected_session_ids: list[int] | None = None,
    selected_only: bool | None = None,
    closed_only: bool = True,
) -> DatasetBundle:
    X_samples: list[list[list[float]]] = []
    y_class: list[float] = []
    y_eta: list[float] = []
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

    if state_path.suffix.lower() == ".json":
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

    min_total_spread_pct = max(0.0, float(min_total_spread_pct or 0.0))
    block_diagnostics = _build_block_diagnostics(blocks, sequence_length=sequence_length)
    episodes_by_pair_session: dict[tuple[str, int], list[Any]] = {}
    blocks_by_pair_session: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    observable_end_by_pair_session: dict[tuple[str, int], float] = {}
    for block in blocks:
        pair_id = str(block.get("pair_id") or "")
        session_id = int(block.get("session_id") or 0)
        blocks_by_pair_session[(pair_id, session_id)].append(block)
    for key, pair_session_blocks in blocks_by_pair_session.items():
        episodes_by_pair_session[key] = _build_closed_episodes_for_blocks(pair_session_blocks)
        observable_end_ts = 0.0
        for block in pair_session_blocks:
            records = [
                canonicalize_record(record, fallback_ts=float(index))
                for index, record in enumerate(block.get("records", []))
                if isinstance(record, dict)
            ]
            records.sort(key=lambda record: record["timestamp"])
            if records:
                observable_end_ts = max(observable_end_ts, float(records[-1]["timestamp"]))
        observable_end_by_pair_session[key] = float(observable_end_ts)

    skipped_blocks_too_short = 0
    for block in blocks:
        normalized_session_id = int(block.get("session_id") or 0)
        normalized_pair_id = str(block.get("pair_id") or "")
        records = [
            canonicalize_record(record, fallback_ts=float(index))
            for index, record in enumerate(block.get("records", []))
            if isinstance(record, dict)
        ]
        if len(records) < sequence_length:
            skipped_blocks_too_short += 1
            continue
        records.sort(key=lambda record: record["timestamp"])
        episodes = episodes_by_pair_session.get((normalized_pair_id, normalized_session_id), [])
        observable_end_ts = float(observable_end_by_pair_session.get((normalized_pair_id, normalized_session_id), 0.0))

        for start in range(len(records) - sequence_length + 1):
            window = records[start : start + sequence_length]
            current_ts = window[-1]["timestamp"]
            if observable_end_ts > 0.0 and (float(current_ts) + float(prediction_horizon_sec)) > observable_end_ts:
                skipped_windows_right_censored += 1
                continue
            label = _label_window_from_episodes(
                current_ts,
                episodes,
                prediction_horizon_sec=prediction_horizon_sec,
                min_total_spread_pct=min_total_spread_pct,
            )

            X_samples.append(_feature_window(window))
            y_class.append(float(label["y_class"]))
            y_eta.append(float(label["y_eta"]))
            pair_ids.append(str(block.get("pair_id") or ""))
            block_ids.append(int(block.get("block_id") or 0))
            session_ids.append(normalized_session_id)
            timestamps.append(current_ts)
            label_end_timestamps.append(float(current_ts) + float(prediction_horizon_sec))
            last_entries.append(window[-1]["entry_spread"])

            future_episode_total_spreads.extend(float(value) for value in label["future_episode_total_spreads"])
            if float(label["y_class"]) >= 0.5:
                _bucket_count(
                    window[-1]["entry_spread"],
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
        X_tensor = torch.tensor(X_samples, dtype=torch.float32)
        y_class_tensor = torch.tensor(y_class, dtype=torch.float32)
        y_eta_tensor = torch.tensor(y_eta, dtype=torch.float32)
    else:
        X_tensor = torch.empty((0, sequence_length, len(FEATURE_NAMES)), dtype=torch.float32)
        y_class_tensor = torch.empty((0,), dtype=torch.float32)
        y_eta_tensor = torch.empty((0,), dtype=torch.float32)

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
        "num_cross_block_windows": 0,
        "prediction_horizon_sec": int(prediction_horizon_sec),
        "sequence_length": int(sequence_length),
        "min_total_spread_pct": float(min_total_spread_pct),
        "labeling_method": "episode_take_profit_time_barrier",
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
