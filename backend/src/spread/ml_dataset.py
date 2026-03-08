from __future__ import annotations

import json
import math
import sqlite3
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

FEATURE_NAMES = [
    "entry_spread", "exit_spread",
    "delta_entry", "delta_exit",
    "delta2_entry", "delta2_exit",
    "rolling_std_entry", "rolling_std_exit",
    "zscore_entry", "zscore_exit",
]

_ROLLING_WINDOW = 5


@dataclass(slots=True)
class DatasetBundle:
    X: torch.Tensor
    y_class: torch.Tensor
    y_eta: torch.Tensor
    pair_ids: list[str]
    block_ids: list[int]
    session_ids: list[int]
    timestamps: list[float]
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
            last_entries=[self.last_entries[index] for index in indices],
            feature_names=list(self.feature_names),
            summary={
                "split_name": split_name,
                "num_samples": len(indices),
                "num_positive_samples": int(self.y_class[indices].sum().item()) if indices else 0,
                "num_negative_samples": len(indices) - int(self.y_class[indices].sum().item()) if indices else 0,
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

        placeholders = ",".join("?" for _ in block_ids)
        for row in conn.execute(
            f"""
            SELECT block_id, ts, entry_spread_pct, exit_spread_pct
            FROM tracker_records
            WHERE block_id IN ({placeholders})
            ORDER BY block_id ASC, ts ASC
            """,
            block_ids,
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
            block_ids,
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


def build_dataset_bundle(
    state_path: Path,
    sequence_length: int = 15,
    prediction_horizon_sec: int = 14_400,
    *,
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
    last_entries: list[float] = []
    state_path = Path(state_path)
    storage_kind = "json"
    saved_at = 0.0
    block_summary: dict[str, Any] = {"num_blocks": 0, "num_sessions": 0}

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

    skipped_blocks_too_short = 0
    for block in blocks:
        records = [
            canonicalize_record(record, fallback_ts=float(index))
            for index, record in enumerate(block.get("records", []))
            if isinstance(record, dict)
        ]
        if len(records) < sequence_length + 1:
            skipped_blocks_too_short += 1
            continue
        records.sort(key=lambda record: record["timestamp"])
        inv_events = sorted(_coerce_float(ts, 0.0) for ts in block.get("inverted_events", []))

        for start in range(len(records) - sequence_length):
            window = records[start : start + sequence_length]
            current_ts = window[-1]["timestamp"]
            future_events = [
                event_ts
                for event_ts in inv_events
                if current_ts < event_ts <= current_ts + prediction_horizon_sec
            ]
            did_invert = 1.0 if future_events else 0.0
            eta_seconds = (future_events[0] - current_ts) if future_events else 0.0

            X_samples.append(_feature_window(window))
            y_class.append(did_invert)
            y_eta.append(eta_seconds)
            pair_ids.append(str(block.get("pair_id") or ""))
            block_ids.append(int(block.get("block_id") or 0))
            session_ids.append(int(block.get("session_id") or 0))
            timestamps.append(current_ts)
            last_entries.append(window[-1]["entry_spread"])

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
        "num_cross_block_windows": 0,
        "prediction_horizon_sec": int(prediction_horizon_sec),
        "sequence_length": int(sequence_length),
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
) -> dict[str, DatasetBundle]:
    if bundle.summary["num_samples"] == 0:
        empty = bundle.subset([], "empty")
        return {"train": empty, "val": empty, "test": empty}

    valid_session_ids = [int(session_id) for session_id in bundle.session_ids if int(session_id) > 0]
    unique_sessions = sorted(set(valid_session_ids), key=lambda session_id: min(
        bundle.timestamps[index] for index, candidate in enumerate(bundle.session_ids) if int(candidate) == session_id
    )) if valid_session_ids else []
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
            split_indices = {
                "train": [index for session_id in train_sessions for index in session_to_indices.get(session_id, [])],
                "val": [index for session_id in val_sessions for index in session_to_indices.get(session_id, [])],
                "test": [index for session_id in test_sessions for index in session_to_indices.get(session_id, [])],
            }
            train_bundle = bundle.subset(split_indices["train"], "train")
            val_bundle = bundle.subset(split_indices["val"], "val")
            test_bundle = bundle.subset(split_indices["test"], "test")
            split_pairs = {
                "train": set(train_bundle.pair_ids),
                "val": set(val_bundle.pair_ids),
                "test": set(test_bundle.pair_ids),
            }
            split_summary = {
                "pair_overlap": len((split_pairs["train"] & split_pairs["val"]) | (split_pairs["train"] & split_pairs["test"]) | (split_pairs["val"] & split_pairs["test"])),
                "train_pairs": len(split_pairs["train"]),
                "val_pairs": len(split_pairs["val"]),
                "test_pairs": len(split_pairs["test"]),
                "train_positive_pairs": 0,
                "val_positive_pairs": 0,
                "test_positive_pairs": 0,
                "train_negative_pairs": 0,
                "val_negative_pairs": 0,
                "test_negative_pairs": 0,
                "train_start_ts": min(train_bundle.timestamps) if train_bundle.timestamps else 0.0,
                "train_end_ts": max(train_bundle.timestamps) if train_bundle.timestamps else 0.0,
                "val_start_ts": min(val_bundle.timestamps) if val_bundle.timestamps else 0.0,
                "val_end_ts": max(val_bundle.timestamps) if val_bundle.timestamps else 0.0,
                "test_start_ts": min(test_bundle.timestamps) if test_bundle.timestamps else 0.0,
                "test_end_ts": max(test_bundle.timestamps) if test_bundle.timestamps else 0.0,
                "embargo_samples": 0,
                "split_mode": "session_chronological",
                "train_session_ids": list(train_sessions),
                "val_session_ids": list(val_sessions),
                "test_session_ids": list(test_sessions),
            }
            pair_has_positive: dict[str, bool] = {}
            for pair_id, label in zip(bundle.pair_ids, bundle.y_class.tolist()):
                pair_has_positive[pair_id] = pair_has_positive.get(pair_id, False) or bool(label >= 0.5)
            for split_name, pair_set in split_pairs.items():
                split_summary[f"{split_name}_positive_pairs"] = sum(1 for pid in pair_set if pair_has_positive.get(pid, False))
                split_summary[f"{split_name}_negative_pairs"] = sum(1 for pid in pair_set if not pair_has_positive.get(pid, False))
            split_summary["global_temporal_order"] = bool(
                split_summary["train_end_ts"] <= split_summary["val_start_ts"]
                and split_summary["val_end_ts"] <= split_summary["test_start_ts"]
            )
            train_bundle.summary["split_summary"] = split_summary
            val_bundle.summary["split_summary"] = split_summary
            test_bundle.summary["split_summary"] = split_summary
            return {"train": train_bundle, "val": val_bundle, "test": test_bundle}

    pair_has_positive: dict[str, bool] = {}
    for pair_id, label in zip(bundle.pair_ids, bundle.y_class.tolist()):
        pair_has_positive[pair_id] = pair_has_positive.get(pair_id, False) or bool(label >= 0.5)

    ordered_indices = sorted(range(len(bundle.timestamps)), key=lambda index: (bundle.timestamps[index], index))
    ratios = (train_ratio, val_ratio, test_ratio)
    max_embargo = max(0, min(int(bundle.summary.get("sequence_length", 1)) - 1, max(0, (len(ordered_indices) - 3) // 2)))
    chosen: tuple[int, int, int, int, int] | None = None
    for embargo_samples in range(max_embargo, -1, -1):
        effective_total = len(ordered_indices) - (2 * embargo_samples)
        if effective_total < 3:
            continue
        train_count, val_count, test_count = _largest_remainder_counts(effective_total, ratios)
        train_end = train_count
        val_start = train_end + embargo_samples
        val_end = val_start + val_count
        test_start = val_end + embargo_samples
        if train_end <= 0 or val_end <= val_start or test_start >= len(ordered_indices):
            continue
        chosen = (embargo_samples, train_end, val_start, val_end, test_start)
        if test_count > 0:
            break

    if chosen is None:
        counts = _largest_remainder_counts(len(ordered_indices), ratios)
        train_end = counts[0]
        val_end = train_end + counts[1]
        chosen = (0, train_end, train_end, val_end, val_end)

    embargo_samples, train_end, val_start, val_end, test_start = chosen
    split_indices = {
        "train": ordered_indices[:train_end],
        "val": ordered_indices[val_start:val_end],
        "test": ordered_indices[test_start:],
    }
    train_bundle = bundle.subset(split_indices["train"], "train")
    val_bundle = bundle.subset(split_indices["val"], "val")
    test_bundle = bundle.subset(split_indices["test"], "test")

    split_pairs = {
        "train": set(train_bundle.pair_ids),
        "val": set(val_bundle.pair_ids),
        "test": set(test_bundle.pair_ids),
    }

    overlap = len(split_pairs["train"] & split_pairs["val"])
    overlap += len(split_pairs["train"] & split_pairs["test"])
    overlap += len(split_pairs["val"] & split_pairs["test"])

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
        "val_start_ts": min(val_bundle.timestamps) if val_bundle.timestamps else 0.0,
        "val_end_ts": max(val_bundle.timestamps) if val_bundle.timestamps else 0.0,
        "test_start_ts": min(test_bundle.timestamps) if test_bundle.timestamps else 0.0,
        "test_end_ts": max(test_bundle.timestamps) if test_bundle.timestamps else 0.0,
        "embargo_samples": int(embargo_samples),
        "split_mode": "sample_chronological",
    }
    split_summary["global_temporal_order"] = bool(
        split_summary["train_end_ts"] <= split_summary["val_start_ts"]
        and split_summary["val_end_ts"] <= split_summary["test_start_ts"]
    )
    train_bundle.summary["split_summary"] = split_summary
    val_bundle.summary["split_summary"] = split_summary
    test_bundle.summary["split_summary"] = split_summary

    return {"train": train_bundle, "val": val_bundle, "test": test_bundle}


def compute_feature_stats(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = X.reshape(-1, X.shape[-1])
    mean = flattened.mean(dim=0)
    std = flattened.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def normalize_features(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (X - mean.view(1, 1, -1)) / std.view(1, 1, -1)
