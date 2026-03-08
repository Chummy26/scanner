import json
from pathlib import Path

from src.spread.train_model import (
    _build_split_summary,
    build_dataset_bundle,
    build_group_splits,
    run_training_loop,
)
from src.spread.ml_dataset import FEATURE_NAMES
from src.spread.spread_tracker import SpreadTracker


def _make_pair(base_ts: int, positive: bool, offset: float) -> dict:
    records = []
    inverted_events = []
    entries_positive = [
        0.08,
        0.12,
        0.18,
        0.28,
        0.42,
        0.58,
        0.74,
        0.88,
        0.92,
        -0.12,
        -0.18,
        -0.08,
        0.06,
        0.10,
        0.14,
        0.12,
    ]
    exits_positive = [
        -0.40,
        -0.36,
        -0.31,
        -0.26,
        -0.21,
        -0.17,
        -0.12,
        -0.07,
        -0.02,
        0.14,
        0.18,
        0.10,
        0.02,
        -0.03,
        -0.06,
        -0.04,
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
        if positive and index == 9:
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
            for index in range(12):
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


def test_build_dataset_bundle_still_reads_legacy_tracker_json_schema(tmp_path: Path):
    state_path = _write_tracker_state(tmp_path / "tracker_state.json")

    bundle = build_dataset_bundle(
        state_path=state_path,
        sequence_length=4,
        prediction_horizon_sec=240,
    )

    assert bundle.summary["num_samples"] > 0
    assert bundle.summary["state_storage_kind"] == "json"


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
    )

    assert report["model_status"] == "trained"
    assert report["split_summary"]["global_temporal_order"] is True
    assert artifact_dir.joinpath("best_lstm_model.pth").is_file()
    assert artifact_dir.joinpath("best_lstm_model.meta.json").is_file()
    assert report["metrics"]["test"]["recall"] > report["baselines"]["always_negative"]["recall"]
    assert report["metrics"]["test"]["average_precision"] >= report["baselines"]["always_negative"]["average_precision"]
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
    assert "median_absolute_error" in report["eta_metrics"]["test"]
    assert "p90_absolute_error" in report["eta_metrics"]["test"]
    assert report["validation_partition"]["mode"] == "chronological"
    assert report["validation_partition"]["calibration_end_ts"] <= report["validation_partition"]["selection_start_ts"]

    metadata_payload = json.loads(artifact_dir.joinpath("best_lstm_model.meta.json").read_text(encoding="utf-8"))
    assert metadata_payload["execute_threshold"] == report["thresholds"]["execute_threshold"]
    assert metadata_payload["strong_threshold"] == report["thresholds"]["strong_threshold"]
    assert metadata_payload["input_size"] == len(FEATURE_NAMES)
    assert len(metadata_payload["feature_names"]) == len(FEATURE_NAMES)
    assert metadata_payload["training_config"]["max_epochs"] == 6
    assert metadata_payload["trained_at_utc"]
    assert metadata_payload["dataset_fingerprint"]
    assert metadata_payload["feature_schema_hash"]
    assert metadata_payload["dataset_summary"]["state_storage_kind"] == "sqlite_blocks"
    assert 0.0 <= report["split_summary"]["pair_overlap_rate"] <= 1.0
    assert report["split_summary"]["pair_overlap_pairwise_sum"] >= report["split_summary"]["pair_overlap"]

    audit_text = Path(report["artifacts"]["audit_path"]).read_text(encoding="utf-8")
    assert "## Calibration" in audit_text
    assert "## Temporal Audit" in audit_text
    assert "Crítico" in audit_text or "Alto" in audit_text or "Médio" in audit_text
    assert "Correções Aplicadas" not in audit_text
    assert "Unique pair overlap rate" in audit_text


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
    )

    assert report["model_status"] == "trained"
    assert report["split_summary"]["split_mode"] == "session_chronological"
    assert report["dataset_summary"]["sessions_used"] == 4
    assert set(report["split_summary"]["train_session_ids"]).isdisjoint(report["split_summary"]["val_session_ids"])
    assert set(report["split_summary"]["train_session_ids"]).isdisjoint(report["split_summary"]["test_session_ids"])


def test_build_dataset_bundle_never_crosses_temporal_gap_blocks(tmp_path: Path):
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
    assert bundle.summary["num_cross_block_windows"] == 0
    assert sorted(set(bundle.block_ids)) == bundle.summary["block_ids_used"]
    assert all(block_id > 0 for block_id in bundle.block_ids)
