from pathlib import Path

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
    run_payload = tracker.create_training_run(
        block_ids=[block["id"] for block in blocks_after_split],
        sequence_length=2,
        prediction_horizon_sec=240,
    )
    assert run_payload["selected_block_count"] == 2
    assert [block["block_id"] for block in run_payload["blocks"]] == [block["id"] for block in blocks_after_split]

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
