import sqlite3
from pathlib import Path

import pytest

from src.spread.spread_tracker import SpreadTracker


def test_tracker_sqlite_flush_and_reload_round_trip(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=120,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )

    pair = ("BTC", "mexc", "spot", "gate", "futures")
    tracker.record_spread(*pair, 0.42, -0.30, now_ts=0.0)
    tracker.record_spread(*pair, -0.15, 0.12, now_ts=15.0)
    tracker.record_spread(*pair, 0.51, -0.22, now_ts=30.0)
    assert tracker.flush_to_storage(now_ts=30.0, force=True)

    restored = SpreadTracker(
        window_sec=120,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )
    assert restored.load_from_storage(now_ts=30.0) == 1

    history = restored.get_history(*pair, limit=10)
    stats = restored.get_pair_stats(*pair)
    storage = restored.get_storage_stats()

    assert len(history) == 3
    assert stats is not None
    assert stats["inverted_count"] == 2
    assert stats["record_count"] == 3
    assert storage["pairs_persisted"] == 1
    assert storage["records_total"] == 3
    assert storage["events_total"] == 5


def test_tracker_uses_15s_interval_and_derives_record_cap(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )

    pair = ("ETH", "mexc", "spot", "gate", "futures")
    for ts in (0.0, 10.0, 15.0, 30.0, 45.0, 60.0, 75.0):
        tracker.record_spread(*pair, 0.5, -0.2, now_ts=ts)

    history = tracker.get_history(*pair, limit=20)

    assert tracker.max_records_per_pair == 4
    assert [round(item["timestamp"], 1) for item in history] == [30.0, 45.0, 60.0, 75.0]


def test_tracker_keeps_8d_storage_while_pruning_12h_memory_window(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=8 * 24 * 60 * 60,
        memory_window_sec=12 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )

    pair = ("XRP", "mexc", "spot", "gate", "futures")
    tracker.record_spread(*pair, 0.30, -0.15, now_ts=0.0)
    tracker.record_spread(*pair, 0.42, -0.10, now_ts=float((12 * 60 * 60) + 15))
    assert tracker.flush_to_storage(now_ts=float((12 * 60 * 60) + 15), force=True)

    history = tracker.get_history(*pair, limit=10)
    training_config = tracker.get_training_config()
    with sqlite3.connect(db_path, timeout=30.0) as conn:
        records_total = int(conn.execute("SELECT COUNT(*) FROM tracker_records").fetchone()[0])

    assert len(history) == 1
    assert history[0]["timestamp"] == pytest.approx(float((12 * 60 * 60) + 15))
    assert records_total == 2
    assert training_config["storage_window_sec"] == 8 * 24 * 60 * 60
    assert training_config["tracker_memory_window_sec"] == 12 * 60 * 60

    restored = SpreadTracker(
        window_sec=8 * 24 * 60 * 60,
        memory_window_sec=12 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )
    assert restored.load_from_storage(now_ts=float((12 * 60 * 60) + 15)) == 1
    assert len(restored.get_history(*pair, limit=10)) == 1
    assert restored.get_storage_stats()["records_total"] == 2


def test_tracker_prune_removes_stale_pairs_from_sqlite(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=30,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )

    stale_pair = ("SOL", "mexc", "spot", "gate", "futures")
    fresh_pair = ("ADA", "mexc", "spot", "gate", "futures")
    tracker.record_spread(*stale_pair, 0.6, -0.3, now_ts=0.0)
    tracker.record_spread(*fresh_pair, 0.7, -0.2, now_ts=50.0)
    assert tracker.flush_to_storage(now_ts=50.0, force=True)

    tracker.prune(now_ts=50.0)
    assert tracker.flush_to_storage(now_ts=50.0)

    restored = SpreadTracker(
        window_sec=30,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )
    assert restored.load_from_storage(now_ts=50.0) == 1
    assert restored.get_pair_stats(*fresh_pair) is not None
    assert restored.get_pair_stats(*stale_pair) is None


def test_tracker_restart_preserves_existing_last_flush_at(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=120,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )

    pair = ("BTC", "mexc", "spot", "gate", "futures")
    tracker.record_spread(*pair, 0.42, -0.30, now_ts=0.0)
    tracker.record_spread(*pair, -0.15, 0.12, now_ts=15.0)
    assert tracker.flush_to_storage(now_ts=15.0, force=True)

    with sqlite3.connect(db_path, timeout=30.0) as conn:
        expected = float(conn.execute("SELECT value FROM tracker_meta WHERE key = 'last_flush_at'").fetchone()[0])

    restarted = SpreadTracker(
        window_sec=120,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )

    assert restarted._last_flush_at == pytest.approx(expected)
    with sqlite3.connect(db_path, timeout=30.0) as conn:
        persisted = float(conn.execute("SELECT value FROM tracker_meta WHERE key = 'last_flush_at'").fetchone()[0])
    assert persisted == pytest.approx(expected)


def test_tracker_zero_last_flush_only_rewrites_current_memory_window(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=1_000,
        memory_window_sec=30,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )

    pair = ("ETH", "mexc", "spot", "gate", "futures")
    tracker.record_spread(*pair, 0.40, -0.20, now_ts=0.0)
    tracker.record_spread(*pair, 0.45, -0.18, now_ts=15.0)
    assert tracker.flush_to_storage(now_ts=15.0, force=True)

    restarted = SpreadTracker(
        window_sec=1_000,
        memory_window_sec=30,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )
    assert restarted.load_from_storage(now_ts=100.0) == 1

    restarted._last_flush_at = 0.0
    restarted.record_spread(*pair, 0.55, -0.10, now_ts=100.0)
    assert restarted.flush_to_storage(now_ts=100.0)

    with sqlite3.connect(db_path, timeout=30.0) as conn:
        rows = conn.execute(
            """
            SELECT ts, entry_spread_pct, exit_spread_pct
            FROM tracker_records
            ORDER BY ts ASC
            """
        ).fetchall()

    assert rows == [
        (0.0, 0.4, -0.2),
        (15.0, 0.45, -0.18),
        (100.0, 0.55, -0.1),
    ]


def test_tracker_creates_new_block_when_gap_exceeds_threshold(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=7200,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=45.0,
    )

    pair = ("BTC", "mexc", "spot", "gate", "futures")
    tracker.record_spread(*pair, 0.40, -0.20, now_ts=0.0)
    tracker.record_spread(*pair, 0.45, -0.18, now_ts=15.0)
    tracker.record_spread(*pair, 0.50, -0.15, now_ts=120.0)
    assert tracker.flush_to_storage(now_ts=120.0, force=True)
    tracker.close_active_session(ended_at=120.0)

    listing = tracker.list_training_blocks()
    assert listing["summary"]["total_sessions"] == 1
    assert listing["summary"]["total_blocks"] == 2
    blocks = listing["sessions"][0]["blocks"]
    assert [block["boundary_reason"] for block in blocks] == ["initial", "auto_gap"]
    assert all(block["is_open"] is False for block in blocks)


def test_tracker_split_merge_and_training_run_snapshot(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=7200,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=120.0,
    )

    pair = ("ETH", "mexc", "spot", "gate", "futures")
    for ts in (0.0, 15.0, 30.0, 45.0, 60.0):
        tracker.record_spread(*pair, 0.5, -0.2, now_ts=ts)
    assert tracker.flush_to_storage(now_ts=60.0, force=True)
    tracker.close_active_session(ended_at=60.0)

    listing = tracker.list_training_blocks()
    original_block = listing["sessions"][0]["blocks"][0]
    split_result = tracker.split_block(int(original_block["id"]), 30.0)
    blocks_after_split = tracker.list_training_blocks()["sessions"][0]["blocks"]
    assert len(blocks_after_split) == 2
    assert split_result["new_block_id"] in {block["id"] for block in blocks_after_split}

    first_block_id = int(blocks_after_split[0]["id"])
    with pytest.raises(ValueError, match="clean-cycle block gates"):
        tracker.create_training_run(
            block_ids=[block["id"] for block in blocks_after_split],
            sequence_length=2,
            prediction_horizon_sec=30,
        )

    merge_result = tracker.merge_next_block(first_block_id)
    merged_blocks = tracker.list_training_blocks()["sessions"][0]["blocks"]
    assert merge_result["merged_block_id"] not in {block["id"] for block in merged_blocks}
    assert len(merged_blocks) == 1


def test_close_active_session_clamps_ended_at_after_started_at(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=7200,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )

    tracker.record_spread("BTC", "mexc", "spot", "gate", "futures", 0.4, -0.2, now_ts=1.0)
    assert tracker.flush_to_storage(now_ts=1.0, force=True)
    tracker.close_active_session(ended_at=1.0)

    session = tracker.list_training_blocks()["sessions"][0]
    assert session["ended_at"] >= session["started_at"]


def test_tracker_extracts_adaptive_episodes_and_recurring_context_for_positive_regime_pair(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=24 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=60.0,
    )

    pair = ("BTC", "mexc", "spot", "gate", "futures")
    entries = [
        0.20, 0.21, 0.19, 0.20, 0.22,
        0.45, 0.62, 0.31, 0.22,
        0.21, 0.20, 0.23, 0.24,
        0.58, 0.66, 0.35, 0.24, 0.21,
    ]
    exits = [
        -0.08, -0.07, -0.09, -0.08, -0.07,
        -0.04, -0.02, 0.03, 0.08,
        -0.08, -0.08, -0.07, -0.06,
        -0.03, -0.01, 0.04, 0.09, 0.10,
    ]
    for index, (entry_spread, exit_spread) in enumerate(zip(entries, exits)):
        tracker.record_spread(*pair, entry_spread, exit_spread, now_ts=float(index * 15))
    assert tracker.flush_to_storage(now_ts=float((len(entries) - 1) * 15), force=True)
    tracker.close_active_session(ended_at=float(len(entries) * 15))

    episodes = tracker.get_pair_episodes(*pair, limit=10)
    context = tracker.get_pair_recurring_context(*pair, current_entry=0.64, now_ts=float(len(entries) * 15))

    assert len(episodes) >= 2
    assert all(episode["is_closed"] is True for episode in episodes)
    assert context["range_status"] == "ready_short"
    assert context["empirical_support_short"] >= 2
    assert context["recommended_entry_range"] != "--"
    assert context["entry_position_label"] in {"inside_outer", "inside_core"}


def test_tracker_exposes_total_spread_and_filters_low_value_episodes_from_qualified_context(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=24 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=120.0,
        min_total_spread_pct=1.0,
    )

    pair = ("LTC", "mexc", "spot", "gate", "futures")
    cycle = [
        (0.10, -0.20),
        (0.11, -0.19),
        (0.09, -0.18),
        (0.23, -0.18),
        (0.11, -0.17),
        (-0.10, 0.15),
    ]
    ts = 0.0
    for _ in range(3):
        for entry_spread, exit_spread in cycle:
            tracker.record_spread(*pair, entry_spread, exit_spread, now_ts=ts)
            ts += 15.0
    assert tracker.flush_to_storage(now_ts=ts, force=True)
    tracker.close_active_session(ended_at=ts)

    episodes = tracker.get_pair_episodes(*pair, limit=10)
    context = tracker.get_pair_recurring_context(*pair, current_entry=0.23, now_ts=ts)

    assert len(episodes) >= 3
    assert all("total_spread" in episode for episode in episodes)
    assert all(float(episode["total_spread"]) < 1.0 for episode in episodes)
    assert context["raw_empirical_support_short"] >= 2
    assert context["empirical_support_short"] == 0
    assert context["median_total_spread"] < 1.0
    assert context["range_status"] == "insufficient_empirical_context"


def test_tracker_rebuilds_episodes_when_block_is_split_and_merged(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=24 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=120.0,
    )

    pair = ("ETH", "mexc", "spot", "gate", "futures")
    entries = [0.20, 0.21, 0.20, 0.48, 0.61, 0.34, 0.22, 0.21, 0.24, 0.57, 0.69, 0.36, 0.23, 0.21]
    exits = [-0.08, -0.08, -0.07, -0.03, -0.01, 0.05, 0.09, 0.09, -0.07, -0.02, -0.01, 0.04, 0.08, 0.08]
    for index, (entry_spread, exit_spread) in enumerate(zip(entries, exits)):
        tracker.record_spread(*pair, entry_spread, exit_spread, now_ts=float(index * 15))
    assert tracker.flush_to_storage(now_ts=float((len(entries) - 1) * 15), force=True)
    tracker.close_active_session(ended_at=float(len(entries) * 15))

    before = tracker.get_pair_episodes(*pair, limit=10)
    assert len(before) >= 2

    block_id = tracker.list_training_blocks()["sessions"][0]["blocks"][0]["id"]
    tracker.split_block(int(block_id), 142.5)
    after_split = tracker.get_pair_episodes(*pair, limit=10)
    assert len(after_split) < len(before)

    merged_block_id = tracker.list_training_blocks()["sessions"][0]["blocks"][0]["id"]
    tracker.merge_next_block(int(merged_block_id))
    after_merge = tracker.get_pair_episodes(*pair, limit=10)
    assert len(after_merge) == len(before)


def test_tracker_excludes_open_episode_from_recurring_context(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=24 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=120.0,
    )

    pair = ("SOL", "mexc", "spot", "gate", "futures")
    entries = [
        0.20, 0.21, 0.19, 0.20, 0.22,
        0.46, 0.63, 0.34, 0.23,
        0.21, 0.20, 0.22, 0.24,
        0.59, 0.68, 0.36, 0.24,
        0.23, 0.22, 0.24, 0.27,
        0.61, 0.72, 0.74, 0.75, 0.73,
    ]
    exits = [
        -0.08, -0.07, -0.08, -0.07, -0.06,
        -0.03, -0.01, 0.04, 0.08,
        -0.08, -0.08, -0.07, -0.06,
        -0.02, -0.01, 0.05, 0.09,
        -0.08, -0.07, -0.06, -0.05,
        -0.02, 0.00, 0.01, 0.01, 0.02,
    ]
    for index, (entry_spread, exit_spread) in enumerate(zip(entries, exits)):
        tracker.record_spread(*pair, entry_spread, exit_spread, now_ts=float(index * 15))
    assert tracker.flush_to_storage(now_ts=float((len(entries) - 1) * 15), force=True)
    tracker.close_active_session(ended_at=float(len(entries) * 15))

    episodes = tracker.get_pair_episodes(*pair, limit=10)
    context = tracker.get_pair_recurring_context(*pair, current_entry=0.74, now_ts=float(len(entries) * 15))

    assert len(episodes) == 2
    assert all(episode["is_closed"] is True for episode in episodes)
    assert context["empirical_support_short"] == 2
    assert context["range_status"] == "ready_short"


def test_tracker_does_not_mark_pair_dirty_when_no_persistable_state_changed(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=24 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=120.0,
    )

    pair = ("BTC", "mexc", "spot", "gate", "futures")
    tracker.record_spread(*pair, 0.50, -0.20, now_ts=0.0)
    assert tracker._dirty_pairs
    assert tracker.flush_to_storage(now_ts=0.0, force=True)
    assert not tracker._dirty_pairs

    tracker.record_spread(*pair, 0.50, -0.20, now_ts=5.0)

    assert not tracker._dirty_pairs
    assert tracker.get_storage_stats()["records_total"] == 1


def test_tracker_record_spread_hot_path_does_not_touch_sqlite_until_flush(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=24 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=45.0,
    )

    pair = ("BTC", "mexc", "spot", "gate", "futures")
    real_connect = tracker._connect

    def _fail_connect():
        raise AssertionError("hot path touched sqlite")

    monkeypatch.setattr(tracker, "_connect", _fail_connect)
    tracker.record_spread(*pair, 0.42, -0.20, now_ts=0.0)
    tracker.record_spread(*pair, 0.47, -0.16, now_ts=15.0)
    tracker.record_spread(*pair, 0.55, -0.08, now_ts=90.0)

    stats = tracker.get_pair_stats(*pair)
    assert stats is not None
    assert stats["record_count"] == 3

    monkeypatch.setattr(tracker, "_connect", real_connect)
    assert tracker.flush_to_storage(now_ts=90.0, force=True)
    tracker.close_active_session(ended_at=90.0)

    listing = tracker.list_training_blocks()
    assert listing["summary"]["total_blocks"] == 2
    assert listing["sessions"][0]["blocks"][0]["boundary_reason"] == "initial"
    assert listing["sessions"][0]["blocks"][1]["boundary_reason"] == "auto_gap"


def test_tracker_rejects_invalid_numeric_records_and_tracks_aggregates(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=24 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=45.0,
    )

    pair = ("BTC", "mexc", "spot", "gate", "futures")
    tracker.record_spread(*pair, float("nan"), -0.20, now_ts=0.0)
    tracker.record_spread(*pair, 0.42, float("inf"), now_ts=15.0)
    assert tracker.flush_to_storage(now_ts=15.0, force=True)

    storage = tracker.get_storage_stats()
    pair_id = "BTC|mexc|spot|gate|futures"

    assert storage["records_total"] == 0
    assert storage["invalid_record_rejections_total"] == 2
    assert storage["invalid_entry_rejections_total"] == 1
    assert storage["invalid_exit_rejections_total"] == 1
    assert storage["rejection_rate_by_pair"][pair_id] == pytest.approx(1.0)
    assert storage["rejection_rate_by_exchange"]["mexc"] == pytest.approx(1.0)
    assert storage["rejection_rate_by_exchange"]["gate"] == pytest.approx(1.0)


def test_tracker_batch_record_flush_materializes_runtime_block_ids(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=24 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=45.0,
    )

    pair = ("ETH", "mexc", "spot", "gate", "futures")
    tracker.batch_record([(*pair, 0.44, -0.20)], now_ts=0.0)
    tracker.batch_record([(*pair, 0.49, -0.15)], now_ts=15.0)
    tracker.batch_record([(*pair, 0.58, -0.08)], now_ts=90.0)

    stats = tracker.get_pair_stats_obj(*pair)
    assert stats is not None
    pre_flush_block_ids = {record.block_id for record in stats.records}
    assert all(block_id < 0 for block_id in pre_flush_block_ids)

    assert tracker.flush_to_storage(now_ts=90.0, force=True)

    stats = tracker.get_pair_stats_obj(*pair)
    assert stats is not None
    post_flush_block_ids = {record.block_id for record in stats.records}
    assert all(block_id > 0 for block_id in post_flush_block_ids)
    assert stats.current_block_id > 0

    tracker.close_active_session(ended_at=90.0)
    storage = tracker.get_storage_stats()
    assert storage["records_total"] == 3
    assert storage["blocks_total"] == 2


def test_tracker_quality_report_flags_structural_anomalies(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=7200,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )
    pair = ("BTC", "mexc", "spot", "gate", "futures")
    for ts in (0.0, 15.0, 30.0, 45.0):
        tracker.record_spread(*pair, 0.4, -0.2, now_ts=ts)
    assert tracker.flush_to_storage(now_ts=45.0, force=True)
    tracker.close_active_session(ended_at=45.0)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        block_id = int(conn.execute("SELECT id FROM tracker_pair_blocks ORDER BY id ASC LIMIT 1").fetchone()["id"])
        session_id = int(conn.execute("SELECT id FROM tracker_capture_sessions ORDER BY id ASC LIMIT 1").fetchone()["id"])
        pair_id = int(conn.execute("SELECT id FROM tracker_pairs ORDER BY id ASC LIMIT 1").fetchone()["id"])
        conn.execute(
            """
            INSERT INTO tracker_pair_blocks(
                pair_id, session_id, start_ts, end_ts, record_count, max_gap_sec, boundary_reason,
                selected_for_training, disabled_reason, manual_override, notes, is_open, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, 0, 0.0, 'auto_gap', 1, '', 0, '', 0, ?, ?)
            """,
            (pair_id, session_id, 120.0, 120.0, 120.0, 120.0),
        )
        conn.execute("UPDATE tracker_events SET block_id = NULL WHERE block_id = ?", (block_id,))
        conn.execute(
            """
            INSERT INTO tracker_events(pair_id, event_type, ts, session_id, block_id)
            VALUES(?, 'entry', ?, ?, NULL)
            """,
            (pair_id, 300.0, session_id),
        )
        conn.commit()

    report = tracker.get_training_quality_report(sequence_length=4, prediction_horizon_sec=30)

    assert report["summary"]["zero_record_blocks"] >= 1
    assert report["summary"]["missing_event_blocks"] >= 1
    assert report["sessions"][0]["critical_anomaly_count"] >= 2


def test_tracker_reset_training_cycle_archives_existing_db_and_reopens_runtime_session(tmp_path: Path):
    db_path = tmp_path / "tracker_history.sqlite"
    tracker = SpreadTracker(
        window_sec=7200,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )
    pair = ("BTC", "mexc", "spot", "gate", "futures")
    tracker.record_spread(*pair, 0.4, -0.2, now_ts=0.0)
    tracker.record_spread(*pair, 0.5, -0.1, now_ts=15.0)
    assert tracker.flush_to_storage(now_ts=15.0, force=True)

    result = tracker.reset_training_cycle(archive_root=tmp_path / "archive")

    assert Path(result["archive_dir"]).is_dir()
    assert Path(result["manifest_path"]).is_file()
    assert Path(result["new_db_path"]).is_file()
    assert result["new_active_session_id"] > 0
    fresh_stats = tracker.get_storage_stats()
    assert fresh_stats["records_total"] == 0


def test_tracker_rejects_invalid_numeric_records_before_persistence_and_tracks_aggregates(tmp_path: Path):
    tracker = SpreadTracker(
        window_sec=3600,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=tmp_path / "tracker.sqlite",
    )
    pair = ("BTC", "mexc", "spot", "gate", "futures")
    captured_events: list[dict] = []
    tracker.add_event_listener(captured_events.append)

    tracker.record_spread(*pair, 0.25, float("nan"), now_ts=15.0)
    tracker.record_spread(*pair, 0.30, -0.10, now_ts=30.0)
    assert tracker.flush_to_storage(now_ts=30.0, force=True)

    history = tracker.get_history(*pair, limit=10)
    storage = tracker.get_storage_stats()

    assert len(history) == 1
    assert history[0]["entry_spread"] == pytest.approx(0.3)
    assert history[0]["exit_spread"] == pytest.approx(-0.1)
    assert storage["attempted_records_total"] == 2
    assert storage["invalid_record_rejections_total"] == 1
    assert storage["invalid_exit_rejections_total"] == 1
    assert storage["invalid_entry_rejections_total"] == 0
    assert storage["rejection_rate_by_pair"]["BTC|mexc|spot|gate|futures"] == pytest.approx(0.5)
    assert storage["rejection_rate_by_exchange"]["mexc"] == pytest.approx(0.5)
    assert storage["rejection_rate_by_exchange"]["gate"] == pytest.approx(0.5)
    assert any(event.get("kind") == "tracker_rejection_stats" for event in captured_events)


def test_tracker_hot_path_stats_update_incrementally_before_flush(tmp_path: Path):
    tracker = SpreadTracker(
        window_sec=3600,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=tmp_path / "tracker.sqlite",
    )
    pair = ("BTC", "mexc", "spot", "gate", "futures")

    tracker.record_spread(*pair, 0.25, float("nan"), now_ts=15.0)
    tracker.record_spread(*pair, 0.30, -0.10, now_ts=30.0)

    stats = tracker._hot_path_rejection_stats
    pair_id = "BTC|mexc|spot|gate|futures"

    assert stats["attempted_records_total"] == 2
    assert stats["invalid_record_rejections_total"] == 1
    assert stats["pair_attempts"][pair_id] == 2
    assert stats["pair_rejections"][pair_id] == 1
    assert stats["rejection_rate_total"] == pytest.approx(0.5)
    assert stats["rejection_rate_by_pair"][pair_id] == pytest.approx(0.5)
    assert stats["rejection_rate_by_exchange"]["mexc"] == pytest.approx(0.5)
    assert stats["rejection_rate_by_exchange"]["gate"] == pytest.approx(0.5)


def test_tracker_rejects_invalid_numeric_records_before_persistence(tmp_path: Path):
    db_path = tmp_path / "tracker_invalid.sqlite"
    tracker = SpreadTracker(
        window_sec=3600,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
    )

    tracker.record_spread("BTC", "mexc", "spot", "gate", "futures", 0.25, float("nan"), now_ts=15.0)
    assert tracker.flush_to_storage(now_ts=15.0, force=True)

    storage = tracker.get_storage_stats()

    assert storage["records_total"] == 0
    assert storage["invalid_record_rejections_total"] == 1
    assert storage["invalid_entry_rejections_total"] == 0
    assert storage["invalid_exit_rejections_total"] == 1
