import json
import sqlite3
from pathlib import Path

import pytest
import torch

from src.spread.train_model import (
    FocalLoss,
    _build_split_summary,
    _derive_focal_alpha,
    _loader_worker_count,
    build_dataset_bundle,
    build_group_splits,
    run_clean_training_cycle,
    run_training_loop,
    run_threshold_preflight,
)
from src.spread.ml_dataset import FEATURE_NAMES, _label_window_from_episodes, _load_blocks_from_sqlite
from src.spread.spread_tracker import SpreadTracker, TrackerEpisode


def test_loader_worker_count_disables_multiprocessing_on_windows(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ARBML_DATALOADER_WORKERS", raising=False)
    monkeypatch.setattr("src.spread.train_model.os.name", "nt")

    assert _loader_worker_count(use_pin=True, sample_count=50_000, batch_size=256) == 0


def test_loader_worker_count_honors_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ARBML_DATALOADER_WORKERS", "2")

    assert _loader_worker_count(use_pin=True, sample_count=50_000, batch_size=256) == 2


def _make_pair(base_ts: int, positive: bool, offset: float) -> dict:
    records = []
    inverted_events = []
    entries_positive = [
        0.10,
        0.11,
        0.09,
        0.10,
        0.12,
        1.05,
        0.98,
        0.36,
        -0.12,
        -0.18,
        -0.08,
        0.06,
        0.10,
        0.14,
        0.12,
        0.10,
    ]
    exits_positive = [
        -0.22,
        -0.21,
        -0.20,
        -0.20,
        -0.18,
        -0.16,
        -0.14,
        -0.08,
        0.18,
        0.14,
        0.18,
        0.10,
        0.02,
        -0.03,
        -0.06,
        -0.05,
    ]
    entries_negative = [
        0.02,
        0.03,
        0.05,
        0.06,
        0.08,
        0.10,
        0.11,
        0.10,
        0.09,
        0.08,
        0.07,
        0.06,
        0.05,
        0.04,
        0.03,
        0.02,
    ]
    exits_negative = [
        -0.08,
        -0.09,
        -0.10,
        -0.10,
        -0.09,
        -0.08,
        -0.08,
        -0.09,
        -0.10,
        -0.11,
        -0.11,
        -0.10,
        -0.09,
        -0.09,
        -0.08,
        -0.08,
    ]

    entries = entries_positive if positive else entries_negative
    exits = exits_positive if positive else exits_negative
    for index, (entry, exit_spread) in enumerate(zip(entries, exits)):
        ts = float(base_ts + index * 60)
        records.append(
            {
                "ts": ts,
                "entry": round(entry + offset, 6),
                "exit": round(exit_spread - offset * 0.5, 6),
            }
        )
        if positive and index == 8:
            inverted_events.append(ts)

    return {
        "last_state": -1 if positive else 1,
        "last_crossover_ts": float(inverted_events[0]) if inverted_events else 0.0,
        "last_seen_ts": float(records[-1]["ts"]),
        "history_enabled": True,
        "inverted_events": inverted_events,
        "entry_events": [],
        "exit_events": [],
        "records": records,
    }


def _write_tracker_state(path: Path) -> Path:
    pairs = {}
    for index in range(9):
        pair_id = f"TOKEN{index:02d}|buy{index}|spot|sell{index}|futures"
        pairs[pair_id] = _make_pair(
            base_ts=1_700_100_000 + index * 10_000,
            positive=index % 2 == 0,
            offset=index * 0.01,
        )
    path.write_text(
        json.dumps({"saved_at": 1_700_200_000.0, "window_sec": 604800, "pairs": pairs}),
        encoding="utf-8",
    )
    return path


def _write_low_total_spread_state(path: Path) -> Path:
    base_ts = 1_701_000_000
    records = [
        {"ts": float(base_ts + 0), "entry": 0.10, "exit": -0.20},
        {"ts": float(base_ts + 60), "entry": 0.11, "exit": -0.19},
        {"ts": float(base_ts + 120), "entry": 0.09, "exit": -0.18},
        {"ts": float(base_ts + 180), "entry": 0.23, "exit": -0.18},
        {"ts": float(base_ts + 240), "entry": 0.11, "exit": -0.17},
        {"ts": float(base_ts + 300), "entry": -0.10, "exit": 0.15},
        {"ts": float(base_ts + 360), "entry": -0.08, "exit": 0.14},
        {"ts": float(base_ts + 420), "entry": -0.05, "exit": 0.12},
    ]
    payload = {
        "saved_at": float(base_ts + 420),
        "window_sec": 604800,
        "pairs": {
            "LOW00|buy|spot|sell|futures": {
                "last_state": -1,
                "last_crossover_ts": float(base_ts + 300),
                "last_seen_ts": float(base_ts + 420),
                "history_enabled": True,
                "inverted_events": [float(base_ts + 300)],
                "entry_events": [],
                "exit_events": [],
                "records": records,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_multi_pair_low_total_spread_state(path: Path, *, count: int = 10) -> Path:
    template_path = _write_low_total_spread_state(path.with_name(f"{path.stem}_template.json"))
    template_payload = json.loads(template_path.read_text(encoding="utf-8"))
    template_pair = next(iter(template_payload["pairs"].values()))
    base_records = template_pair["records"]
    pairs: dict[str, dict] = {}
    last_seen_ts = 0.0
    for index in range(count):
        offset = index * 10_000
        shifted_records = [
            {
                "ts": float(record["ts"] + offset),
                "entry": float(record["entry"]),
                "exit": float(record["exit"]),
            }
            for record in base_records
        ]
        last_seen_ts = max(last_seen_ts, shifted_records[-1]["ts"])
        pairs[f"LOW{index:02d}|buy|spot|sell|futures"] = {
            "last_state": int(template_pair["last_state"]),
            "last_crossover_ts": float(template_pair["last_crossover_ts"] + offset),
            "last_seen_ts": float(shifted_records[-1]["ts"]),
            "history_enabled": True,
            "inverted_events": [float(ts + offset) for ts in template_pair["inverted_events"]],
            "entry_events": [],
            "exit_events": [],
            "records": shifted_records,
        }
    payload = {
        "saved_at": last_seen_ts,
        "window_sec": template_payload["window_sec"],
        "pairs": pairs,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_tracker_sqlite(path: Path, pairs: dict[str, dict]) -> Path:
    tracker = SpreadTracker(
        window_sec=604800,
        record_interval_sec=60.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=path,
    )
    max_ts = 0.0
    for pair_id, payload in pairs.items():
        symbol, buy_ex, buy_mt, sell_ex, sell_mt = pair_id.split("|")
        for record in payload.get("records", []):
            ts_value = float(record.get("ts", 0.0))
            max_ts = max(max_ts, ts_value)
            tracker.record_spread(
                symbol,
                buy_ex,
                buy_mt,
                sell_ex,
                sell_mt,
                record.get("entry", 0.0),
                record.get("exit", 0.0),
                now_ts=ts_value,
            )
    assert tracker.flush_to_storage(now_ts=max_ts, force=True)
    tracker.close_active_session(ended_at=max_ts)
    return path


def _write_multi_session_sqlite(path: Path, *, num_sessions: int = 4) -> Path:
    for session_index in range(num_sessions):
        tracker = SpreadTracker(
            window_sec=604800,
            record_interval_sec=15.0,
            max_records_per_pair=0,
            epsilon_pct=0.0,
            history_enable_entry_spread_pct=0.0,
            track_enable_entry_spread_pct=0.0,
            db_path=path,
            gap_threshold_sec=60.0,
        )
        base_ts = 1_700_900_000 + session_index * 100_000
        for pair_offset, symbol in enumerate((f"BTC{session_index}", f"ETH{session_index}")):
            start = base_ts + pair_offset * 1_000
            for index in range(40):
                tracker.record_spread(
                    symbol,
                    "mexc",
                    "spot",
                    "gate",
                    "futures",
                    0.30 + (index * 0.01),
                    -0.10 + (index * 0.005),
                    now_ts=float(start + index * 15),
                )
        assert tracker.flush_to_storage(now_ts=float(base_ts + 3_000), force=True)
        tracker.close_active_session(ended_at=float(base_ts + 3_000))
    return path


def _write_near_threshold_state(path: Path, *, count: int = 12) -> Path:
    base_ts = 1_702_000_000
    template_records = [
        {"ts": float(base_ts + 0), "entry": 0.12, "exit": -0.18},
        {"ts": float(base_ts + 60), "entry": 0.14, "exit": -0.17},
        {"ts": float(base_ts + 120), "entry": 0.18, "exit": -0.14},
        {"ts": float(base_ts + 180), "entry": 0.82, "exit": -0.09},
        {"ts": float(base_ts + 240), "entry": 0.68, "exit": -0.04},
        {"ts": float(base_ts + 300), "entry": 0.54, "exit": 0.02},
        {"ts": float(base_ts + 360), "entry": -0.04, "exit": 0.08},
        {"ts": float(base_ts + 420), "entry": -0.07, "exit": 0.10},
        {"ts": float(base_ts + 480), "entry": -0.05, "exit": 0.11},
    ]
    pairs: dict[str, dict] = {}
    last_seen_ts = 0.0
    for index in range(count):
        offset = index * 10_000
        shifted_records = [
            {
                "ts": float(record["ts"] + offset),
                "entry": float(record["entry"]),
                "exit": float(record["exit"]),
            }
            for record in template_records
        ]
        last_seen_ts = max(last_seen_ts, shifted_records[-1]["ts"])
        pairs[f"NEAR{index:02d}|buy|spot|sell|futures"] = {
            "last_state": -1,
            "last_crossover_ts": float(base_ts + 360 + offset),
            "last_seen_ts": float(shifted_records[-1]["ts"]),
            "history_enabled": True,
            "inverted_events": [float(base_ts + 360 + offset)],
            "entry_events": [],
            "exit_events": [],
            "records": shifted_records,
        }
    path.write_text(
        json.dumps({"saved_at": last_seen_ts, "window_sec": 604800, "pairs": pairs}),
        encoding="utf-8",
    )
    return path


def _write_many_gap_blocks_sqlite(path: Path, *, num_blocks: int = 1_100) -> Path:
    tracker = SpreadTracker(
        window_sec=30 * 24 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=path,
        gap_threshold_sec=30.0,
    )
    pair = ("BTC", "mexc", "spot", "gate", "futures")
    for index in range(num_blocks):
        tracker.record_spread(
            *pair,
            0.20 + ((index % 5) * 0.01),
            -0.10 + ((index % 3) * 0.01),
            now_ts=float(index * 120),
        )
    assert tracker.flush_to_storage(now_ts=float(num_blocks * 120), force=True)
    tracker.close_active_session(ended_at=float(num_blocks * 120))
    return path


def _write_same_session_cross_block_label_sqlite(path: Path) -> Path:
    tracker = SpreadTracker(
        window_sec=30 * 24 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=path,
        gap_threshold_sec=60.0,
    )
    pair = ("BTC", "mexc", "spot", "gate", "futures")
    base_ts = 1_704_000_000
    first_block_entries = [0.10, 0.12, 0.11, 0.10, 0.09]
    for index, entry in enumerate(first_block_entries):
        tracker.record_spread(*pair, entry, -0.20, now_ts=float(base_ts + (index * 15)))

    second_block_records = [
        (0.10, -0.20),
        (0.12, -0.18),
        (0.11, -0.17),
        (0.92, -0.10),
        (0.84, -0.04),
        (0.08, 0.22),
        (0.06, 0.20),
    ]
    for index, (entry, exit_spread) in enumerate(second_block_records):
        tracker.record_spread(
            *pair,
            entry,
            exit_spread,
            now_ts=float(base_ts + 180 + (index * 15)),
        )

    assert tracker.flush_to_storage(now_ts=float(base_ts + 360), force=True)
    tracker.close_active_session(ended_at=float(base_ts + 360))
    return path


def _write_exact_sequence_length_cross_block_label_sqlite(path: Path) -> Path:
    tracker = SpreadTracker(
        window_sec=30 * 24 * 60 * 60,
        record_interval_sec=60.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=path,
        gap_threshold_sec=120.0,
    )
    pair = ("BTC", "mexc", "spot", "gate", "futures")
    for index in range(15):
        tracker.record_spread(
            *pair,
            0.10 + (index * 0.01),
            -0.08,
            now_ts=float(index * 60),
        )

    second_block_records = [
        (0.10, -0.08),
        (0.12, -0.08),
        (0.15, -0.07),
        (0.55, -0.04),
        (0.72, -0.02),
        (0.30, 0.05),
        (0.12, 0.08),
    ]
    base_ts = float((15 * 60) + 180)
    for index, (entry, exit_spread) in enumerate(second_block_records):
        tracker.record_spread(
            *pair,
            entry,
            exit_spread,
            now_ts=base_ts + (index * 60),
        )

    assert tracker.flush_to_storage(now_ts=base_ts + ((len(second_block_records) - 1) * 60), force=True)
    tracker.close_active_session(ended_at=base_ts + (len(second_block_records) * 60))
    return path


def _write_right_censored_session_sqlite(path: Path) -> Path:
    tracker = SpreadTracker(
        window_sec=30 * 24 * 60 * 60,
        record_interval_sec=60.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=path,
        gap_threshold_sec=120.0,
    )
    pair = ("BTC", "mexc", "spot", "gate", "futures")
    for index in range(20):
        tracker.record_spread(
            *pair,
            0.10 + (index * 0.01),
            -0.05,
            now_ts=float(index * 60),
        )
    assert tracker.flush_to_storage(now_ts=float(19 * 60), force=True)
    tracker.close_active_session(ended_at=float(20 * 60))
    return path


def _write_block_length_sqlite(path: Path, *, block_lengths: list[int]) -> Path:
    tracker = SpreadTracker(
        window_sec=30 * 24 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=path,
        gap_threshold_sec=60.0,
    )
    pair = ("ETH", "mexc", "spot", "gate", "futures")
    base_ts = 1_705_000_000
    cursor_ts = base_ts
    for block_index, block_length in enumerate(block_lengths):
        for record_index in range(block_length):
            tracker.record_spread(
                *pair,
                0.20 + (record_index * 0.01),
                -0.10 + (record_index * 0.002),
                now_ts=float(cursor_ts + (record_index * 15)),
            )
        cursor_ts += (block_length * 15) + 120

    assert tracker.flush_to_storage(now_ts=float(cursor_ts), force=True)
    tracker.close_active_session(ended_at=float(cursor_ts))
    return path


def _write_multi_session_sqlite_with_spacing(
    path: Path,
    *,
    num_sessions: int = 4,
    session_spacing_sec: int = 90,
) -> Path:
    for session_index in range(num_sessions):
        tracker = SpreadTracker(
            window_sec=604800,
            record_interval_sec=15.0,
            max_records_per_pair=0,
            epsilon_pct=0.0,
            history_enable_entry_spread_pct=0.0,
            track_enable_entry_spread_pct=0.0,
            db_path=path,
            gap_threshold_sec=60.0,
        )
        base_ts = 1_703_000_000 + (session_index * session_spacing_sec)
        for pair_offset, symbol in enumerate((f"BTCG{session_index}", f"ETHG{session_index}")):
            start = base_ts + pair_offset * 10
            for index in range(40):
                tracker.record_spread(
                    symbol,
                    "mexc",
                    "spot",
                    "gate",
                    "futures",
                    0.30 + (index * 0.01),
                    -0.08 + (index * 0.005),
                    now_ts=float(start + index * 15),
                )
        assert tracker.flush_to_storage(now_ts=float(base_ts + 400), force=True)
        tracker.close_active_session(ended_at=float(base_ts + 400))
    return path


def _write_same_pair_multi_session_merge_sqlite(
    path: Path,
    *,
    session_gap_sec: int = 90,
) -> Path:
    pair = ("BTC", "mexc", "spot", "gate", "futures")
    base_ts = 1_706_000_000
    tracker = SpreadTracker(
        window_sec=604800,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=path,
        gap_threshold_sec=60.0,
    )
    session_payloads = [
        [
            (0.10, -0.20),
            (0.11, -0.19),
            (0.12, -0.18),
            (0.11, -0.18),
            (0.10, -0.17),
            (0.09, -0.17),
        ],
        [
            (0.12, -0.16),
            (0.14, -0.15),
            (0.19, -0.12),
            (0.82, -0.09),
            (0.78, -0.03),
            (0.08, 0.22),
            (0.07, 0.20),
        ],
        [
            (0.10, -0.15),
            (0.11, -0.14),
            (0.10, -0.13),
            (0.12, -0.12),
            (0.11, -0.11),
            (0.09, -0.10),
        ],
    ]
    for session_index, records in enumerate(session_payloads):
        session_start = base_ts + (session_index * session_gap_sec)
        for record_index, (entry_spread, exit_spread) in enumerate(records):
            tracker.record_spread(
                *pair,
                entry_spread,
                exit_spread,
                now_ts=float(session_start + (record_index * 15)),
            )
        ended_at = float(session_start + ((len(records) - 1) * 15))
        assert tracker.flush_to_storage(now_ts=ended_at, force=True)
        tracker.close_active_session(ended_at=ended_at)
        if session_index < len(session_payloads) - 1:
            tracker._open_runtime_session()
    return path


def _write_same_pair_mergeable_groups_sqlite(path: Path) -> Path:
    pair = ("BTC", "mexc", "spot", "gate", "futures")
    tracker = SpreadTracker(
        window_sec=604800,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=path,
        gap_threshold_sec=60.0,
    )
    session_offsets = [0, 90, 1_000, 1_090]
    session_payloads = [
        [(0.10, -0.20), (0.11, -0.19), (0.12, -0.18), (0.11, -0.18), (0.10, -0.17), (0.09, -0.17)],
        [(0.12, -0.16), (0.14, -0.15), (0.19, -0.12), (0.82, -0.09), (0.78, -0.03), (0.08, 0.22), (0.07, 0.20)],
        [(0.10, -0.20), (0.11, -0.19), (0.10, -0.18), (0.09, -0.18), (0.08, -0.17), (0.07, -0.17)],
        [(0.11, -0.16), (0.12, -0.15), (0.18, -0.12), (0.85, -0.09), (0.80, -0.03), (0.09, 0.21), (0.08, 0.19)],
    ]
    base_ts = 1_707_000_000
    for session_index, records in enumerate(session_payloads):
        session_start = base_ts + session_offsets[session_index]
        for record_index, (entry_spread, exit_spread) in enumerate(records):
            tracker.record_spread(
                *pair,
                entry_spread,
                exit_spread,
                now_ts=float(session_start + (record_index * 15)),
            )
        ended_at = float(session_start + ((len(records) - 1) * 15))
        assert tracker.flush_to_storage(now_ts=ended_at, force=True)
        tracker.close_active_session(ended_at=ended_at)
        if session_index < len(session_payloads) - 1:
            tracker._open_runtime_session()
    return path


def _write_same_pair_mergeable_session_chain_sqlite(path: Path) -> Path:
    pair = ("BTC", "mexc", "spot", "gate", "futures")
    tracker = SpreadTracker(
        window_sec=604800,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=path,
        gap_threshold_sec=60.0,
    )
    base_ts = 1_708_000_000
    group_offsets = [0, 1_000, 2_000]
    session_payload = [
        [(0.10, -0.20), (0.11, -0.19), (0.12, -0.18), (0.11, -0.18), (0.10, -0.17), (0.09, -0.17)],
        [(0.12, -0.16), (0.14, -0.15), (0.19, -0.12), (0.82, -0.09), (0.78, -0.03), (0.08, 0.22), (0.07, 0.20)],
    ]
    for group_offset in group_offsets:
        for local_session_index, records in enumerate(session_payload):
            session_start = base_ts + group_offset + (local_session_index * 90)
            for record_index, (entry_spread, exit_spread) in enumerate(records):
                tracker.record_spread(
                    *pair,
                    entry_spread,
                    exit_spread,
                    now_ts=float(session_start + (record_index * 15)),
                )
            ended_at = float(session_start + ((len(records) - 1) * 15))
            assert tracker.flush_to_storage(now_ts=ended_at, force=True)
            tracker.close_active_session(ended_at=ended_at)
            if not (group_offset == group_offsets[-1] and local_session_index == len(session_payload) - 1):
                tracker._open_runtime_session()
    return path


def _write_overlapping_tracker_state(path: Path) -> Path:
    pairs = {}
    for index in range(6):
        pair_id = f"OVER{index:02d}|buy{index}|spot|sell{index}|futures"
        pairs[pair_id] = _make_pair(
            base_ts=1_700_500_000,
            positive=index % 2 == 0,
            offset=index * 0.01,
        )
    path.write_text(
        json.dumps({"saved_at": 1_700_600_000.0, "window_sec": 604800, "pairs": pairs}),
        encoding="utf-8",
    )
    return path


def _episode(*, start_ts: float, end_ts: float, peak_entry_spread: float, exit_spread_at_close: float) -> TrackerEpisode:
    return TrackerEpisode(
        start_ts=start_ts,
        peak_ts=start_ts,
        end_ts=end_ts,
        duration_sec=max(0.0, end_ts - start_ts),
        peak_entry_spread=peak_entry_spread,
        exit_spread_at_close=exit_spread_at_close,
        baseline_median=0.1,
        baseline_mad=0.01,
        activation_threshold=0.15,
        release_threshold=0.12,
        session_id=1,
        block_id=1,
        is_closed=True,
    )


def test_build_dataset_bundle_reads_tracker_sqlite_schema(tmp_path: Path):
    json_state = _write_tracker_state(tmp_path / "tracker_state.json")
    payload = json.loads(json_state.read_text(encoding="utf-8"))
    state_path = _write_tracker_sqlite(tmp_path / "tracker_history.sqlite", payload["pairs"])

    bundle = build_dataset_bundle(
        state_path=state_path,
        sequence_length=4,
        prediction_horizon_sec=240,
    )

    assert bundle.summary["num_samples"] > 0
    assert bundle.summary["num_positive_samples"] > 0
    assert bundle.summary["num_negative_samples"] > 0
    assert float(bundle.X.abs().sum().item()) > 0.0
    assert bundle.X.shape[2] == len(FEATURE_NAMES)
    assert bundle.feature_names == FEATURE_NAMES
    assert bundle.summary["state_storage_kind"] == "sqlite_blocks"
    assert bundle.summary["num_blocks"] > 0
    assert bundle.summary["num_sessions"] == 1
    assert bundle.summary["num_cross_block_windows"] == 0
    assert bundle.block_ids
    assert bundle.session_ids


def test_load_blocks_from_sqlite_chunks_large_block_queries(tmp_path: Path):
    sqlite_path = _write_many_gap_blocks_sqlite(tmp_path / "tracker_history_many_blocks.sqlite", num_blocks=1_100)

    blocks, _, summary = _load_blocks_from_sqlite(sqlite_path, selected_only=False, closed_only=True)

    assert len(blocks) >= 1_000
    assert summary["num_blocks"] == len(blocks)


def test_build_dataset_bundle_still_reads_legacy_tracker_json_schema(tmp_path: Path):
    state_path = _write_tracker_state(tmp_path / "tracker_state.json")

    bundle = build_dataset_bundle(
        state_path=state_path,
        sequence_length=4,
        prediction_horizon_sec=240,
    )

    assert bundle.summary["num_samples"] > 0
    assert bundle.summary["state_storage_kind"] == "json"


def test_build_dataset_bundle_filters_low_total_spread_inversions_from_positive_labels(tmp_path: Path):
    state_path = _write_low_total_spread_state(tmp_path / "tracker_state_low_total.json")

    unfiltered_bundle = build_dataset_bundle(
        state_path=state_path,
        sequence_length=3,
        prediction_horizon_sec=180,
        min_total_spread_pct=0.0,
    )
    filtered_bundle = build_dataset_bundle(
        state_path=state_path,
        sequence_length=3,
        prediction_horizon_sec=180,
        min_total_spread_pct=1.0,
    )

    assert unfiltered_bundle.summary["num_positive_samples"] > 0
    assert filtered_bundle.summary["num_samples"] == unfiltered_bundle.summary["num_samples"]
    assert filtered_bundle.summary["num_positive_samples"] == 0
    assert filtered_bundle.summary["num_negative_samples"] == filtered_bundle.summary["num_samples"]
    assert filtered_bundle.summary["min_total_spread_pct"] == 1.0
    assert filtered_bundle.summary["labeling_method"] == "episode_take_profit_time_barrier"
    assert filtered_bundle.summary["label_audit"]["timeouts_with_only_sub_threshold_episode"] > 0


def test_build_dataset_bundle_labels_future_episode_from_later_block_same_session(tmp_path: Path):
    sqlite_path = _write_same_session_cross_block_label_sqlite(tmp_path / "tracker_history_cross_block.sqlite")
    blocks, _, _ = _load_blocks_from_sqlite(sqlite_path, selected_only=False, closed_only=True)

    assert len(blocks) == 2
    assert len({int(block["session_id"]) for block in blocks}) == 1
    earlier_block_id = int(min(blocks, key=lambda block: float(block["start_ts"]))["block_id"])

    bundle = build_dataset_bundle(
        state_path=sqlite_path,
        sequence_length=3,
        prediction_horizon_sec=240,
        min_total_spread_pct=1.0,
    )

    positive_block_ids = {
        int(bundle.block_ids[index])
        for index, value in enumerate(bundle.y_class.tolist())
        if float(value) >= 0.5
    }
    assert earlier_block_id in positive_block_ids


def test_build_dataset_bundle_allows_exact_sequence_length_block_with_future_label_in_same_session(tmp_path: Path):
    sqlite_path = _write_exact_sequence_length_cross_block_label_sqlite(
        tmp_path / "tracker_history_exact_sequence_cross_block.sqlite"
    )

    bundle = build_dataset_bundle(
        state_path=sqlite_path,
        sequence_length=15,
        prediction_horizon_sec=600,
        min_total_spread_pct=0.5,
    )

    assert bundle.summary["num_samples"] == 1
    assert bundle.summary["num_positive_samples"] == 1
    assert bundle.summary["block_diagnostics"]["feature_window_feasibility"]["eligible_blocks_for_sequence_length"] == 1
    assert bundle.summary["block_diagnostics"]["feature_window_feasibility"]["eligible_sessions_with_any_eligible_block"] == 1


def test_build_dataset_bundle_discards_right_censored_tail_windows(tmp_path: Path):
    sqlite_path = _write_right_censored_session_sqlite(tmp_path / "tracker_history_right_censored.sqlite")

    bundle = build_dataset_bundle(
        state_path=sqlite_path,
        sequence_length=15,
        prediction_horizon_sec=14_400,
        min_total_spread_pct=0.5,
    )

    assert bundle.summary["num_samples"] == 0
    assert bundle.summary["skipped_windows_right_censored"] == 6
    assert bundle.summary["label_audit"]["right_censored_windows"] == 6


def test_label_window_requires_qualified_episode_to_close_within_horizon():
    result = _label_window_from_episodes(
        180.0,
        episodes=[
            _episode(
                start_ts=200.0,
                end_ts=360.0,
                peak_entry_spread=0.92,
                exit_spread_at_close=0.18,
            )
        ],
        prediction_horizon_sec=120,
        min_total_spread_pct=1.0,
    )

    assert result["y_class"] == 0.0
    assert result["y_eta"] == 0.0
    assert result["timeout_reason"] == "no_future_episode"


def test_label_window_ignores_episode_that_started_before_current_ts():
    result = _label_window_from_episodes(
        150.0,
        episodes=[
            _episode(
                start_ts=120.0,
                end_ts=220.0,
                peak_entry_spread=0.88,
                exit_spread_at_close=0.20,
            )
        ],
        prediction_horizon_sec=120,
        min_total_spread_pct=1.0,
    )

    assert result["y_class"] == 0.0
    assert result["y_eta"] == 0.0
    assert result["timeout_reason"] == "no_future_episode"


def test_label_window_uses_time_until_episode_close_as_eta():
    result = _label_window_from_episodes(
        150.0,
        episodes=[
            _episode(
                start_ts=200.0,
                end_ts=260.0,
                peak_entry_spread=0.90,
                exit_spread_at_close=0.22,
            )
        ],
        prediction_horizon_sec=120,
        min_total_spread_pct=1.0,
    )

    assert result["y_class"] == 1.0
    assert result["y_eta"] == 110.0
    assert result["qualified_episode_total_spread"] >= 1.0


def test_label_window_adaptive_threshold_falls_back_to_cost_floor_without_prior_support():
    result = _label_window_from_episodes(
        150.0,
        episodes=[
            _episode(
                start_ts=200.0,
                end_ts=240.0,
                peak_entry_spread=0.90,
                exit_spread_at_close=0.20,
            )
        ],
        prediction_horizon_sec=120,
        min_total_spread_pct=1.0,
        adaptive_threshold_enabled=True,
        pair_key="BTC|mexc|spot|gate|futures",
        label_cost_floor_pct=1.0,
        label_percentile=70.0,
        label_episode_window_days=5,
    )

    assert result["y_class"] == 1.0
    assert result["label_threshold"] == pytest.approx(1.0)
    assert result["label_threshold_support"] == 0


def test_label_window_adaptive_threshold_uses_only_prior_episodes_without_future_leakage():
    result = _label_window_from_episodes(
        150.0,
        episodes=[
            _episode(
                start_ts=60.0,
                end_ts=120.0,
                peak_entry_spread=0.90,
                exit_spread_at_close=0.20,
            ),
            _episode(
                start_ts=200.0,
                end_ts=260.0,
                peak_entry_spread=3.80,
                exit_spread_at_close=0.40,
            ),
        ],
        prediction_horizon_sec=120,
        min_total_spread_pct=0.8,
        adaptive_threshold_enabled=True,
        pair_key="DOGE|mexc|spot|gate|futures",
        label_cost_floor_pct=0.8,
        label_percentile=70.0,
        label_episode_window_days=5,
    )

    assert result["y_class"] == 1.0
    assert result["label_threshold"] == pytest.approx(1.1)
    assert result["label_threshold_support"] == 1


def test_label_window_adaptive_threshold_raises_bar_for_high_spread_pairs():
    result = _label_window_from_episodes(
        500.0,
        episodes=[
            _episode(start_ts=100.0, end_ts=150.0, peak_entry_spread=1.0, exit_spread_at_close=0.2),
            _episode(start_ts=180.0, end_ts=220.0, peak_entry_spread=2.0, exit_spread_at_close=0.2),
            _episode(start_ts=260.0, end_ts=300.0, peak_entry_spread=3.0, exit_spread_at_close=0.2),
            _episode(start_ts=540.0, end_ts=580.0, peak_entry_spread=2.1, exit_spread_at_close=0.1),
        ],
        prediction_horizon_sec=120,
        min_total_spread_pct=1.0,
        adaptive_threshold_enabled=True,
        pair_key="SHIB|mexc|spot|gate|futures",
        label_cost_floor_pct=1.0,
        label_percentile=70.0,
        label_episode_window_days=5,
    )

    assert result["y_class"] == 0.0
    assert result["timeout_reason"] == "sub_threshold_only"
    assert result["label_threshold_support"] == 3
    assert result["label_threshold"] > 2.2


def test_build_dataset_bundle_reports_adaptive_label_threshold_metadata(tmp_path: Path):
    state_path = _write_tracker_state(tmp_path / "tracker_state_adaptive.json")

    bundle = build_dataset_bundle(
        state_path=state_path,
        sequence_length=3,
        prediction_horizon_sec=240,
        label_cost_floor_pct=0.8,
        label_percentile=70,
        label_episode_window_days=5,
    )

    assert bundle.summary["label_threshold_mode"] == "rolling_pair_percentile"
    assert bundle.summary["label_cost_floor_pct"] == pytest.approx(0.8)
    assert bundle.summary["label_percentile"] == pytest.approx(70.0)
    assert bundle.summary["label_episode_window_days"] == 5
    assert len(bundle.label_thresholds) == bundle.summary["num_samples"]
    assert bundle.summary["label_thresholds"]


def test_focal_loss_alpha_high_weights_positive_class_more_than_negative():
    positive_loss = FocalLoss(alpha=0.9, gamma=0.0)(
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
    )
    negative_loss = FocalLoss(alpha=0.9, gamma=0.0)(
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
    )

    assert positive_loss.item() > negative_loss.item()
    assert _derive_focal_alpha(0.07) == pytest.approx(0.93)
    assert _derive_focal_alpha(0.90) == pytest.approx(0.25)


def test_group_splits_handle_overlapping_pair_histories_with_temporal_embargo(tmp_path: Path):
    json_state = _write_overlapping_tracker_state(tmp_path / "tracker_state_overlap.json")
    payload = json.loads(json_state.read_text(encoding="utf-8"))
    state_path = _write_tracker_sqlite(tmp_path / "tracker_history_overlap.sqlite", payload["pairs"])
    bundle = build_dataset_bundle(
        state_path=state_path,
        sequence_length=4,
        prediction_horizon_sec=240,
    )

    splits = build_group_splits(bundle)

    assert splits["train"].timestamps
    assert splits["val"].timestamps
    assert splits["test"].timestamps
    assert max(splits["train"].timestamps) < min(splits["val"].timestamps)
    assert max(splits["val"].timestamps) < min(splits["test"].timestamps)
    assert splits["train"].summary["split_summary"]["embargo_samples"] >= 1


def test_split_summary_reports_unique_pair_overlap_rate(tmp_path: Path):
    json_state = _write_overlapping_tracker_state(tmp_path / "tracker_state_overlap.json")
    payload = json.loads(json_state.read_text(encoding="utf-8"))
    state_path = _write_tracker_sqlite(tmp_path / "tracker_history_overlap.sqlite", payload["pairs"])
    bundle = build_dataset_bundle(
        state_path=state_path,
        sequence_length=4,
        prediction_horizon_sec=240,
    )

    summary = _build_split_summary(build_group_splits(bundle))

    assert 0.0 <= summary["pair_overlap_rate"] <= 1.0
    assert summary["pair_overlap"] <= summary["train_pairs"] + summary["val_pairs"] + summary["test_pairs"]
    assert summary["pair_overlap_pairwise_sum"] >= summary["pair_overlap"]
    assert "pair_overlap_breakdown" in summary


def test_group_splits_preserve_global_temporal_order(tmp_path: Path):
    pairs = {}
    for index in range(9):
        pair_id = f"POS{index:02d}|buy|spot|sell|futures"
        pairs[pair_id] = _make_pair(
            base_ts=1_700_000_000 + index * 3_000,
            positive=True,
            offset=index * 0.01,
        )
    for index in range(3):
        pair_id = f"NEG{index:02d}|buy|spot|sell|futures"
        pairs[pair_id] = _make_pair(
            base_ts=1_800_000_000 + index * 3_000,
            positive=False,
            offset=index * 0.01,
        )
    state_path = tmp_path / "tracker_state_temporal.json"
    state_path.write_text(
        json.dumps({"saved_at": 1_800_200_000.0, "window_sec": 604800, "pairs": pairs}),
        encoding="utf-8",
    )
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    sqlite_path = _write_tracker_sqlite(tmp_path / "tracker_history_temporal.sqlite", payload["pairs"])
    bundle = build_dataset_bundle(
        state_path=sqlite_path,
        sequence_length=4,
        prediction_horizon_sec=240,
    )

    splits = build_group_splits(bundle)

    train_max = max(splits["train"].timestamps)
    val_min = min(splits["val"].timestamps)
    val_max = max(splits["val"].timestamps)
    test_min = min(splits["test"].timestamps)
    assert train_max <= val_min
    assert val_max <= test_min


def test_group_splits_use_session_boundaries_when_multiple_sessions_exist(tmp_path: Path):
    sqlite_path = _write_multi_session_sqlite(tmp_path / "tracker_history_sessions.sqlite", num_sessions=4)
    bundle = build_dataset_bundle(
        state_path=sqlite_path,
        sequence_length=4,
        prediction_horizon_sec=240,
    )

    splits = build_group_splits(bundle)
    summary = splits["train"].summary["split_summary"]

    assert summary["split_mode"] == "session_chronological"
    assert set(splits["train"].session_ids).isdisjoint(set(splits["val"].session_ids))
    assert set(splits["train"].session_ids).isdisjoint(set(splits["test"].session_ids))
    assert set(splits["val"].session_ids).isdisjoint(set(splits["test"].session_ids))
    assert set(summary["train_session_ids"]) == set(splits["train"].session_ids)
    assert set(summary["val_session_ids"]) == set(splits["val"].session_ids)
    assert set(summary["test_session_ids"]) == set(splits["test"].session_ids)


def test_group_splits_fall_back_to_purged_sample_mode_when_session_gap_is_below_horizon(tmp_path: Path):
    sqlite_path = _write_multi_session_sqlite_with_spacing(
        tmp_path / "tracker_history_sessions_tight.sqlite",
        num_sessions=4,
        session_spacing_sec=230,
    )
    bundle = build_dataset_bundle(
        state_path=sqlite_path,
        sequence_length=4,
        prediction_horizon_sec=240,
    )

    splits = build_group_splits(bundle, prediction_horizon_sec=240)
    summary = splits["train"].summary["split_summary"]

    assert summary["split_mode"] == "sample_chronological_purged"
    assert summary["split_mode_fallback_reason"] == "session_gap_below_horizon"
    assert summary["purged_temporal_separation_ok"] is True
    assert summary["embargo_time_sec"] == 240


def test_build_dataset_bundle_cross_session_merge_recovers_future_labels_without_crossing_feature_blocks(tmp_path: Path):
    sqlite_path = _write_same_pair_multi_session_merge_sqlite(
        tmp_path / "tracker_history_cross_session_merge.sqlite",
        session_gap_sec=90,
    )

    plain_bundle = build_dataset_bundle(
        state_path=sqlite_path,
        sequence_length=4,
        prediction_horizon_sec=105,
        allow_cross_session_merge=False,
    )
    merged_bundle = build_dataset_bundle(
        state_path=sqlite_path,
        sequence_length=4,
        prediction_horizon_sec=105,
        allow_cross_session_merge=True,
        regime_shift_score_threshold=None,
    )

    assert merged_bundle.summary["cross_session_merge_enabled"] is True
    assert merged_bundle.summary["cross_session_merges_applied"] >= 1
    assert merged_bundle.summary["num_cross_block_windows"] == 0
    assert merged_bundle.summary["skipped_windows_cross_session_boundary"] > 0
    assert merged_bundle.summary["num_positive_samples"] > plain_bundle.summary["num_positive_samples"]


def test_group_splits_preserve_purging_when_cross_session_merge_is_enabled(tmp_path: Path):
    sqlite_path = _write_same_pair_mergeable_session_chain_sqlite(
        tmp_path / "tracker_history_cross_session_split.sqlite",
    )

    bundle = build_dataset_bundle(
        state_path=sqlite_path,
        sequence_length=4,
        prediction_horizon_sec=105,
        allow_cross_session_merge=True,
        regime_shift_score_threshold=None,
    )
    splits = build_group_splits(bundle, prediction_horizon_sec=105)
    summary = splits["train"].summary["split_summary"]

    assert summary["purged_temporal_separation_ok"] is True
    assert summary["train_end_label_ts"] <= summary["val_start_ts"]
    assert summary["val_end_label_ts"] <= summary["test_start_ts"]


def test_threshold_preflight_reports_cross_session_merge_mode_and_failure_reasons(tmp_path: Path):
    sqlite_path = _write_same_pair_multi_session_merge_sqlite(
        tmp_path / "tracker_history_cross_session_preflight.sqlite",
        session_gap_sec=90,
    )

    preflight_off = run_threshold_preflight(
        state_file=sqlite_path,
        sequence_length=4,
        prediction_horizon_sec=105,
        thresholds=[0.0],
        min_train_positive_samples=1,
        min_val_positive_samples=1,
        min_test_positive_samples=1,
        allow_cross_session_merge=False,
    )
    preflight_on = run_threshold_preflight(
        state_file=sqlite_path,
        sequence_length=4,
        prediction_horizon_sec=105,
        thresholds=[0.0],
        min_train_positive_samples=1,
        min_val_positive_samples=1,
        min_test_positive_samples=1,
        allow_cross_session_merge=True,
    )

    assert preflight_off["cross_session_merge_enabled"] is False
    assert preflight_on["cross_session_merge_enabled"] is True
    assert "failure_reasons" in preflight_off["thresholds"]["0.0"]
    assert "cross_session_merge_diagnostics" in preflight_on["thresholds"]["0.0"]


def test_dataset_fingerprint_is_stable_when_only_sqlite_mtime_changes(tmp_path: Path):
    sqlite_path = _write_tracker_sqlite(
        tmp_path / "tracker_history_fingerprint.sqlite",
        json.loads(_write_tracker_state(tmp_path / "tracker_state_fingerprint.json").read_text(encoding="utf-8"))["pairs"],
    )
    artifact_dir_a = tmp_path / "artifacts_a"
    artifact_dir_b = tmp_path / "artifacts_b"

    report_a = run_training_loop(
        state_file=sqlite_path,
        artifact_dir=artifact_dir_a,
        sequence_length=4,
        prediction_horizon_sec=240,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        batch_size=8,
        max_epochs=2,
        patience=1,
        learning_rate=0.01,
        min_train_positive_samples=0,
        min_val_positive_samples=0,
        min_test_positive_samples=0,
    )
    sqlite_path.touch()
    report_b = run_training_loop(
        state_file=sqlite_path,
        artifact_dir=artifact_dir_b,
        sequence_length=4,
        prediction_horizon_sec=240,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        batch_size=8,
        max_epochs=2,
        patience=1,
        learning_rate=0.01,
        min_train_positive_samples=0,
        min_val_positive_samples=0,
        min_test_positive_samples=0,
    )

    assert report_a["artifact_metadata"]["dataset_fingerprint"] == report_b["artifact_metadata"]["dataset_fingerprint"]


def test_run_training_loop_saves_artifacts_and_beats_negative_baseline(tmp_path: Path):
    json_state = _write_tracker_state(tmp_path / "tracker_state.json")
    payload = json.loads(json_state.read_text(encoding="utf-8"))
    state_path = _write_tracker_sqlite(tmp_path / "tracker_history.sqlite", payload["pairs"])
    artifact_dir = tmp_path / "artifacts"

    report = run_training_loop(
        state_file=state_path,
        artifact_dir=artifact_dir,
        sequence_length=4,
        prediction_horizon_sec=240,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        batch_size=8,
        max_epochs=6,
        patience=3,
        learning_rate=0.01,
        min_train_positive_samples=1,
        min_val_positive_samples=0,
        min_test_positive_samples=0,
    )

    assert report["model_status"] == "trained"
    assert report["split_summary"]["global_temporal_order"] is True
    assert artifact_dir.joinpath("best_lstm_model.pth").is_file()
    assert artifact_dir.joinpath("best_lstm_model.meta.json").is_file()
    assert report["metrics"]["test"]["recall"] > report["baselines"]["always_negative"]["recall"]
    assert report["metrics"]["test"]["average_precision"] >= report["baselines"]["always_negative"]["average_precision"]
    assert "val_threshold" in report["metrics"]
    assert "val" not in report["metrics"]
    assert "confusion_matrix" in report["metrics"]["test"]
    assert "false_positive_rate" in report["metrics"]["test"]
    assert "false_negative_rate" in report["metrics"]["test"]
    assert "specificity" in report["metrics"]["test"]
    assert "balanced_accuracy" in report["metrics"]["test"]
    assert "brier_score" in report["metrics"]["test"]
    assert "log_loss" in report["metrics"]["test"]
    assert "threshold_selection" in report
    assert "calibration" in report
    assert "subgroup_metrics" in report
    assert "temporal_walk_forward" in report
    assert "validation_partition" in report
    assert "label_audit" in report
    assert "median_absolute_error" in report["eta_metrics"]["test"]
    assert "p90_absolute_error" in report["eta_metrics"]["test"]
    assert report["validation_partition"]["mode"] == "chronological"
    assert report["validation_partition"]["calibration_end_ts"] <= report["validation_partition"]["selection_start_ts"]
    assert report["training"]["positive_rate_train"] > 0.0
    assert 0.25 <= report["training"]["focal_alpha_effective"] <= 0.95
    assert "val_threshold" in report["calibration"]
    assert "val" not in report["calibration"]
    assert "high_confidence" in report["calibration"]["test"]
    assert report["split_summary"]["purged_temporal_separation_ok"] is True
    assert report["dataset_summary"]["labeling_method"] == "episode_take_profit_time_barrier"
    assert "future_episode_total_spread_quantiles" in report["label_audit"]

    metadata_payload = json.loads(artifact_dir.joinpath("best_lstm_model.meta.json").read_text(encoding="utf-8"))
    assert metadata_payload["execute_threshold"] == report["thresholds"]["execute_threshold"]
    assert metadata_payload["strong_threshold"] == report["thresholds"]["strong_threshold"]
    assert metadata_payload["input_size"] == len(FEATURE_NAMES)
    assert len(metadata_payload["feature_names"]) == len(FEATURE_NAMES)
    assert metadata_payload["training_config"]["max_epochs"] == 6
    assert metadata_payload["training_config"]["labeling_method"] == "episode_take_profit_time_barrier"
    assert metadata_payload["training_config"]["labeling_timeout_only"] is True
    assert metadata_payload["training_config"]["positive_rate_train"] > 0.0
    assert 0.25 <= metadata_payload["training_config"]["focal_alpha_effective"] <= 0.95
    assert metadata_payload["validation_metrics"] == report["metrics"]["val_threshold"]
    assert metadata_payload["trained_at_utc"]
    assert metadata_payload["dataset_fingerprint"]
    assert metadata_payload["feature_schema_hash"]
    assert metadata_payload["dataset_summary"]["state_storage_kind"] == "sqlite_blocks"
    assert 0.0 <= report["split_summary"]["pair_overlap_rate"] <= 1.0
    assert report["split_summary"]["pair_overlap_pairwise_sum"] >= report["split_summary"]["pair_overlap"]

    audit_text = Path(report["artifacts"]["audit_path"]).read_text(encoding="utf-8")
    assert "## Calibration" in audit_text
    assert "## Temporal Audit" in audit_text
    assert "## Label Audit" in audit_text
    assert "Crítico" in audit_text or "Alto" in audit_text or "Médio" in audit_text
    assert "Correções Aplicadas" not in audit_text
    assert "Unique pair overlap rate" in audit_text


def test_threshold_preflight_selects_highest_threshold_that_passes_guardrails(tmp_path: Path):
    state_path = _write_near_threshold_state(tmp_path / "tracker_state_near_threshold.json", count=18)

    preflight = run_threshold_preflight(
        state_file=state_path,
        sequence_length=3,
        prediction_horizon_sec=240,
        thresholds=[1.0, 0.8, 0.7],
        min_train_positive_samples=1,
        min_val_positive_samples=1,
        min_test_positive_samples=1,
        output_path=tmp_path / "threshold_preflight.json",
    )

    assert preflight["selected_threshold"] == pytest.approx(0.8)
    assert Path(preflight["output_path"]).is_file()
    assert preflight["thresholds"]["1.0"]["guardrail_ok"] is False
    assert preflight["thresholds"]["0.8"]["qualifies_for_training"] is True
    assert "split_positive_rates" in preflight["thresholds"]["0.8"]


def test_threshold_preflight_reports_block_window_feasibility(tmp_path: Path):
    sqlite_path = _write_block_length_sqlite(
        tmp_path / "tracker_history_block_lengths.sqlite",
        block_lengths=[8, 12, 20],
    )

    preflight = run_threshold_preflight(
        state_file=sqlite_path,
        sequence_length=15,
        prediction_horizon_sec=240,
        thresholds=[1.0],
        min_train_positive_samples=0,
        min_val_positive_samples=0,
        min_test_positive_samples=0,
    )

    block_diagnostics = preflight["block_diagnostics"]
    assert block_diagnostics["record_count_threshold_counts"]["ge_8"] == 3
    assert block_diagnostics["record_count_threshold_counts"]["ge_10"] == 2
    assert block_diagnostics["record_count_threshold_counts"]["ge_12"] == 2
    assert block_diagnostics["record_count_threshold_counts"]["ge_15"] == 1
    assert block_diagnostics["record_count_threshold_counts"]["ge_20"] == 1
    assert block_diagnostics["feature_window_feasibility"]["sequence_length"] == 15
    assert block_diagnostics["feature_window_feasibility"]["eligible_blocks_for_sequence_length"] == 1
    assert block_diagnostics["feature_window_feasibility"]["eligible_sessions_with_any_eligible_block"] == 1


def test_clean_training_cycle_archives_existing_artifacts_and_writes_preflight(monkeypatch, tmp_path: Path):
    state_path = _write_multi_session_sqlite(tmp_path / "tracker_history_clean_cycle.sqlite", num_sessions=4)
    artifact_dir = tmp_path / "artifacts_clean_cycle"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.joinpath("best_lstm_model.pth").write_bytes(b"legacy-model")
    artifact_dir.joinpath("best_lstm_model.meta.json").write_text(json.dumps({"version": "legacy"}), encoding="utf-8")

    def _fake_certify(**kwargs):
        target_dir = Path(kwargs["artifact_dir"])
        gate_dir = target_dir / "gate_results"
        gate_dir.mkdir(parents=True, exist_ok=True)
        certification = {
            "certified": True,
            "verdict": "CERTIFIED",
            "failure_reasons": [],
            "warnings": [],
            "gate_results": {
                "gate_10_dual_mode_preflight": {"metrics": {"qualifying_configs": ["seq4_h240_merge_off"]}},
            },
            "recommendations": [],
            "certification_mode": kwargs.get("certification_mode", "full"),
            "certification_duration_sec": 0.01,
            "certification_id": "test-certification-id",
            "runtime_audit_package_path": "",
            "preflight_off_vs_on_summary": {"qualifying_configs": ["seq4_h240_merge_off"]},
            "effective_session_scope": {
                "effective_session_ids": [1, 2, 3, 4],
                "selected_session_ids": [],
                "selected_block_ids": [],
                "legacy_session_ids": [],
                "legacy_sessions_excluded": 0,
            },
            "legacy_sessions_excluded": 0,
            "legacy_session_ids": [],
        }
        (target_dir / "data_certification.json").write_text(json.dumps(certification), encoding="utf-8")
        (gate_dir / "gate_12_entry_exit_quality.json").write_text(json.dumps({"status": "PASSED"}), encoding="utf-8")
        return certification

    def _fake_preflight(**kwargs):
        output_path = Path(kwargs["output_path"])
        payload = {
            "qualifies_for_training": True,
            "selected_threshold": 0.8,
            "selection_mode": "primary",
            "thresholds": {"0.8": {"qualifies_for_training": True, "failure_reasons": []}},
            "block_diagnostics": {},
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _fake_training_loop(**kwargs):
        target_dir = Path(kwargs["artifact_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)
        meta_path = target_dir / "best_lstm_model.meta.json"
        meta_path.write_text(json.dumps({"training_config": {}, "dataset_fingerprint": "fp"}), encoding="utf-8")
        report_path = target_dir / "best_lstm_model.report.json"
        report_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        audit_path = target_dir / "best_lstm_model.audit.md"
        audit_path.write_text("# audit\n", encoding="utf-8")
        return {
            "training": {},
            "artifact_metadata": {"training_config": {}},
            "artifacts": {
                "metadata_path": str(meta_path),
                "report_path": str(report_path),
                "audit_path": str(audit_path),
            },
        }

    monkeypatch.setattr("src.spread.train_model.certify_data_for_training", _fake_certify)
    monkeypatch.setattr("src.spread.train_model.run_threshold_preflight", _fake_preflight)
    monkeypatch.setattr("src.spread.train_model.run_training_loop", _fake_training_loop)

    report = run_clean_training_cycle(
        state_file=state_path,
        artifact_dir=artifact_dir,
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[1.0, 0.8, 0.7],
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        batch_size=8,
        max_epochs=3,
        patience=2,
        learning_rate=0.01,
        min_train_positive_samples=1,
        min_val_positive_samples=0,
        min_test_positive_samples=0,
    )

    legacy_dir = artifact_dir / "legacy_artifacts"
    assert report["preflight"]["selected_threshold"] in {1.0, 0.8, 0.7}
    assert report["data_certification"]["certification_id"]
    assert artifact_dir.joinpath("threshold_preflight.json").is_file()
    assert artifact_dir.joinpath("data_certification.json").is_file()
    assert artifact_dir.joinpath("gate_results", "gate_12_entry_exit_quality.json").is_file()
    assert legacy_dir.is_dir()
    assert any(path.is_dir() for path in legacy_dir.iterdir())


def test_run_training_loop_accepts_session_chronological_split(tmp_path: Path):
    sqlite_path = _write_multi_session_sqlite(tmp_path / "tracker_history_sessions.sqlite", num_sessions=4)
    artifact_dir = tmp_path / "artifacts_sessions"

    report = run_training_loop(
        state_file=sqlite_path,
        artifact_dir=artifact_dir,
        sequence_length=4,
        prediction_horizon_sec=240,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        batch_size=8,
        max_epochs=3,
        patience=2,
        learning_rate=0.01,
        min_train_positive_samples=0,
        min_val_positive_samples=0,
        min_test_positive_samples=0,
    )

    assert report["model_status"] == "trained"
    assert report["split_summary"]["split_mode"] == "session_chronological"
    assert report["split_summary"]["split_mode_fallback_reason"] == ""
    assert report["dataset_summary"]["sessions_used"] == 4
    assert set(report["split_summary"]["train_session_ids"]).isdisjoint(report["split_summary"]["val_session_ids"])
    assert set(report["split_summary"]["train_session_ids"]).isdisjoint(report["split_summary"]["test_session_ids"])


def test_run_training_loop_aborts_when_filtered_dataset_has_too_few_positive_samples(tmp_path: Path):
    state_path = _write_multi_pair_low_total_spread_state(tmp_path / "tracker_state_low_total_many.json", count=12)

    with pytest.raises(ValueError, match="positive samples"):
        run_training_loop(
            state_file=state_path,
            artifact_dir=tmp_path / "artifacts_guardrail",
            sequence_length=3,
            prediction_horizon_sec=180,
            min_total_spread_pct=1.0,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            batch_size=4,
            max_epochs=2,
            patience=1,
            learning_rate=0.01,
            min_train_positive_samples=1,
            min_val_positive_samples=1,
            min_test_positive_samples=1,
        )


def test_build_dataset_bundle_restores_cross_block_windows_within_same_session(tmp_path: Path):
    tracker = SpreadTracker(
        window_sec=10 * 24 * 60 * 60,
        record_interval_sec=60.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=tmp_path / "tracker_history.sqlite",
    )
    pair = ("BTC", "mexc", "spot", "gate", "futures")
    for index in range(26):
        tracker.record_spread(*pair, 0.20 + (index * 0.01), -0.10, now_ts=float(index * 60))
    gap_start = 7 * 24 * 60 * 60
    for index in range(5):
        tracker.record_spread(*pair, 0.30 + (index * 0.01), -0.05, now_ts=float(gap_start + (index * 60)))
    assert tracker.flush_to_storage(now_ts=float(gap_start + (4 * 60)), force=True)
    tracker.close_active_session(ended_at=float(gap_start + (4 * 60)))

    bundle = build_dataset_bundle(
        state_path=tmp_path / "tracker_history.sqlite",
        sequence_length=4,
        prediction_horizon_sec=240,
    )

    assert bundle.summary["num_blocks"] == 2
    assert bundle.summary["blocks_used"] == 2
    assert bundle.summary["num_cross_block_windows"] > 0
    assert bundle.summary["skipped_windows_right_censored"] > 0
    assert sorted(set(bundle.block_ids)) == bundle.summary["block_ids_used"]
    assert all(block_id > 0 for block_id in bundle.block_ids)


def test_load_blocks_from_sqlite_supports_large_selected_scopes(tmp_path: Path):
    sqlite_path = _write_multi_session_sqlite(tmp_path / "tracker_history_large_blocks.sqlite", num_sessions=2)

    conn = sqlite3.connect(sqlite_path)
    try:
        block_ids = [int(row[0]) for row in conn.execute("SELECT id FROM tracker_pair_blocks ORDER BY id ASC LIMIT 3")]
        session_ids = [
            int(row[0])
            for row in conn.execute("SELECT DISTINCT session_id FROM tracker_pair_blocks ORDER BY session_id ASC")
        ]
    finally:
        conn.close()

    assert block_ids
    assert session_ids

    large_block_ids = block_ids + list(range(1000, 2300))
    blocks, saved_at, summary = _load_blocks_from_sqlite(
        sqlite_path,
        selected_block_ids=large_block_ids,
        selected_only=False,
        closed_only=False,
    )

    assert saved_at > 0.0
    assert summary["num_blocks"] >= len(block_ids)
    loaded_block_ids = {block["block_id"] for block in blocks}
    for block_id in block_ids:
        assert block_id in loaded_block_ids

    large_session_ids = session_ids + list(range(2000, 3200))
    session_blocks, _, session_summary = _load_blocks_from_sqlite(
        sqlite_path,
        selected_session_ids=large_session_ids,
        selected_only=False,
        closed_only=False,
    )
    assert session_blocks
    assert session_summary["num_blocks"] == len(session_blocks)
    loaded_session_ids = {block["session_id"] for block in session_blocks}
    assert loaded_session_ids.issubset(set(large_session_ids))
