from pathlib import Path

import torch

from src.spread.ml_dataset import FEATURE_NAMES
from src.spread.ml_analyzer import SpreadMLAnalyzer, SpreadSequenceLSTM
from src.spread.ml_model import (
    ModelArtifactMetadata,
    current_feature_schema_hash,
    save_artifact_bundle,
)


def _structured_history() -> list[dict]:
    history = []
    timestamp = 1_700_000_000
    cycles = [
        (0.42, -0.30),
        (0.58, -0.24),
        (0.73, -0.14),
        (-0.08, 0.12),
        (0.48, -0.28),
        (0.64, -0.19),
        (0.79, -0.09),
        (-0.05, 0.16),
        (0.44, -0.27),
        (0.62, -0.20),
        (0.76, -0.08),
        (-0.03, 0.14),
        (0.40, -0.25),
        (0.59, -0.18),
        (0.72, -0.07),
        (-0.02, 0.11),
    ]
    for entry_spread, exit_spread in cycles:
        history.append(
            {
                "timestamp": float(timestamp),
                "entry_spread": float(entry_spread),
                "exit_spread": float(exit_spread),
            }
        )
        timestamp += 60
    return history


_LEGACY_FEATURE_NAMES = ["entry_spread", "exit_spread", "delta_entry", "delta_exit"]


def _save_ready_artifact(
    artifact_dir: Path,
    sequence_length: int = 12,
    *,
    execute_threshold: float = 0.45,
    strong_threshold: float = 0.70,
    platt_scale: float = 1.0,
    platt_bias: float = 0.0,
    selected_threshold: float | None = None,
) -> None:
    model = SpreadSequenceLSTM(input_sz=len(FEATURE_NAMES), hidden_sz=8, num_layers=1, dropout=0.0)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    metadata = ModelArtifactMetadata(
        version="test-bundle",
        sequence_length=sequence_length,
        input_size=len(FEATURE_NAMES),
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        feature_names=list(FEATURE_NAMES),
        feature_mean=[0.0] * len(FEATURE_NAMES),
        feature_std=[1.0] * len(FEATURE_NAMES),
        platt_scale=platt_scale,
        platt_bias=platt_bias,
        execute_threshold=execute_threshold,
        strong_threshold=strong_threshold,
        min_history_points=sequence_length,
        min_empirical_events=2,
        validation_metrics={"average_precision": 0.5},
        test_metrics={"average_precision": 0.5},
        baselines={"always_negative": {"recall": 0.0}},
        dataset_summary={"num_samples": 10},
        split_summary={"pair_overlap": 0},
        training_config={
            "prediction_horizon_sec": 14_400,
            "selected_threshold": float(selected_threshold) if selected_threshold is not None else execute_threshold,
            "min_total_spread_pct": float(selected_threshold) if selected_threshold is not None else execute_threshold,
        },
        use_attention=True,
        trained_at_utc="2026-03-07T00:00:00+00:00",
        dataset_fingerprint="test-dataset-fingerprint",
        feature_schema_hash=current_feature_schema_hash(),
    )
    save_artifact_bundle(model, metadata, artifact_dir)


def _save_legacy_artifact(artifact_dir: Path, sequence_length: int = 12) -> None:
    """Save a v2-style artifact with 4 features and no attention (backward compat)."""
    num_features = len(_LEGACY_FEATURE_NAMES)
    model = SpreadSequenceLSTM(input_sz=num_features, hidden_sz=8, num_layers=1, dropout=0.0, use_attention=False)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    metadata = ModelArtifactMetadata(
        version="legacy-v2-test",
        sequence_length=sequence_length,
        input_size=num_features,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        feature_names=list(_LEGACY_FEATURE_NAMES),
        feature_mean=[0.0] * num_features,
        feature_std=[1.0] * num_features,
        platt_scale=1.0,
        platt_bias=0.0,
        execute_threshold=0.45,
        strong_threshold=0.70,
        min_history_points=sequence_length,
        min_empirical_events=2,
        validation_metrics={"average_precision": 0.5},
        test_metrics={"average_precision": 0.5},
        baselines={"always_negative": {"recall": 0.0}},
        dataset_summary={"num_samples": 10},
        split_summary={"pair_overlap": 0},
        training_config={"prediction_horizon_sec": 14_400},
        use_attention=False,
    )
    save_artifact_bundle(model, metadata, artifact_dir)


def test_analyzer_returns_safe_wait_without_artifact(tmp_path: Path):
    analyzer = SpreadMLAnalyzer(sequence_length=12, artifact_dir=tmp_path)

    result = analyzer.analyze_pair(0.62, _structured_history())

    assert result is not None
    assert result["model_status"] == "missing_artifact"
    assert result["signal_action"] == "WAIT"
    assert result["ml_score"] == 0
    assert result["success_probability"] == 0.0
    assert "recommended_entry_range" in result
    assert "recommended_exit_range" in result
    assert "signal_reason" in result


def test_analyzer_loads_artifact_and_preserves_ml_context_contract(tmp_path: Path):
    _save_ready_artifact(tmp_path, sequence_length=12)
    analyzer = SpreadMLAnalyzer(sequence_length=12, artifact_dir=tmp_path)

    result = analyzer.analyze_pair(0.62, _structured_history())

    assert result is not None
    assert result["model_status"] == "ready"
    assert result["signal_action"] in {"WAIT", "EXECUTE", "STRONG_EXECUTE"}
    assert 0.0 <= result["inversion_probability"] <= 1.0
    assert 0.0 <= result["success_probability"] <= 100.0
    assert result["history_points"] >= 12
    assert "recommended_entry_range" in result
    assert "recommended_exit_range" in result
    assert "estimated_time_to_close" in result
    assert "ml_score" in result
    assert "model_version" in result
    assert "context_percentile" in result
    assert "signal_reason" in result
    assert "empirical_support" in result
    assert "context_strength" in result
    assert "is_entry_inside_range" in result
    assert "execute_threshold_used" in result
    assert "strong_threshold_used" in result
    assert "drift_status" in result
    assert "drifted_features" in result
    assert "inference_latency_ms" in result
    assert "artifact_feature_count" in result


def test_analyzer_fails_closed_on_threshold_mismatch(tmp_path: Path):
    _save_ready_artifact(tmp_path, sequence_length=12, execute_threshold=0.45, selected_threshold=0.80)

    analyzer = SpreadMLAnalyzer(sequence_length=12, artifact_dir=tmp_path, min_total_spread_pct=1.0)
    result = analyzer.analyze_pair(0.62, _structured_history())

    assert analyzer.model_status == "threshold_mismatch"
    assert result is not None
    assert result["model_status"] == "threshold_mismatch"
    assert result["signal_action"] == "WAIT"
    assert result["signal_reason_code"] == "threshold_mismatch"


def test_analyzer_backward_compat_loads_legacy_4_feature_artifact(tmp_path: Path):
    _save_legacy_artifact(tmp_path, sequence_length=12)
    analyzer = SpreadMLAnalyzer(sequence_length=12, artifact_dir=tmp_path)

    assert analyzer.model_status == "stale_artifact"
    assert analyzer.metadata is not None
    assert analyzer.metadata.use_attention is False
    assert analyzer.metadata.feature_names == _LEGACY_FEATURE_NAMES
    assert analyzer.metadata.input_size == 4

    result = analyzer.analyze_pair(0.62, _structured_history())

    assert result is not None
    assert result["model_status"] == "stale_artifact"
    assert result["signal_action"] == "WAIT"
    assert result["artifact_feature_count"] == 4
    assert "feature schema mismatch" in result["signal_reason"]


def test_analyzer_allows_legacy_artifact_only_with_explicit_opt_in(tmp_path: Path):
    _save_legacy_artifact(tmp_path, sequence_length=12)
    analyzer = SpreadMLAnalyzer(sequence_length=12, artifact_dir=tmp_path, allow_stale_artifacts=True)

    assert analyzer.model_status == "ready"
    assert analyzer.metadata is not None
    assert analyzer.metadata.use_attention is False
    assert analyzer.metadata.feature_names == _LEGACY_FEATURE_NAMES

    result = analyzer.analyze_pair(0.62, _structured_history())

    assert result is not None
    assert result["model_status"] == "ready"
    assert result["artifact_feature_count"] == 4


def test_analyzer_blocks_execute_when_probability_is_high_but_recurring_context_is_missing(tmp_path: Path):
    _save_ready_artifact(tmp_path, sequence_length=12, execute_threshold=0.45, strong_threshold=0.70)
    analyzer = SpreadMLAnalyzer(sequence_length=12, artifact_dir=tmp_path)

    flat_history = [
        {
            "timestamp": float(1_700_000_000 + (index * 60)),
            "entry_spread": 0.18 + ((index % 3) * 0.01),
            "exit_spread": -0.08 + ((index % 2) * 0.01),
        }
        for index in range(16)
    ]

    result = analyzer.analyze_pair(0.22, flat_history)

    assert result is not None
    assert result["model_status"] == "ready"
    assert result["inversion_probability"] > 0.45
    assert result["range_status"] == "insufficient_empirical_context"
    assert result["recommended_entry_range"] == "--"
    assert result["signal_action"] == "WAIT"


def test_analyzer_executes_only_when_recurring_context_is_ready(tmp_path: Path):
    from src.spread.spread_tracker import SpreadTracker

    db_path = tmp_path / "tracker.sqlite"
    tracker = SpreadTracker(
        window_sec=24 * 60 * 60,
        record_interval_sec=15.0,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=600.0,
    )
    pair = ("ETH", "mexc", "spot", "gate", "futures")
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
    tracker.flush_to_storage(now_ts=float((len(entries) - 1) * 15), force=True)
    tracker.close_active_session(ended_at=float(len(entries) * 15))

    _save_ready_artifact(
        tmp_path,
        sequence_length=12,
        execute_threshold=0.45,
        strong_threshold=0.85,
        selected_threshold=1.0,
    )
    analyzer = SpreadMLAnalyzer(sequence_length=12, artifact_dir=tmp_path)
    analyzer.attach_tracker(tracker)

    result = analyzer.analyze_pair(
        0.64,
        tracker.get_history(*pair, limit=100),
        pair_key=pair,
    )

    assert result is not None
    assert result["range_status"] == "ready_short"
    assert result["signal_action"] == "EXECUTE"
    assert result["recommended_entry_range"] != "--"


def test_analyzer_allows_strong_execute_when_eta_gate_is_disabled(tmp_path: Path):
    from src.spread.spread_tracker import SpreadTracker

    db_path = tmp_path / "tracker.sqlite"
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
    pair = ("SOL", "mexc", "spot", "gate", "futures")
    ts = 0.0
    cycle_entries = [
        (0.20, -0.08),
        (0.21, -0.07),
        (0.19, -0.09),
        (0.20, -0.08),
        (0.22, -0.07),
        (0.45, -0.04),
        (0.62, -0.02),
        (0.31, 0.03),
        (0.22, 0.08),
        (0.21, -0.08),
        (0.20, -0.08),
        (0.23, -0.07),
        (0.24, -0.06),
        (0.58, -0.03),
        (0.66, -0.01),
        (0.35, 0.04),
        (0.24, 0.09),
        (0.21, 0.10),
    ]
    for _ in range(3):
        for entry_spread, exit_spread in cycle_entries:
            tracker.record_spread(*pair, entry_spread, exit_spread, now_ts=ts)
            ts += 45.0
    tracker.flush_to_storage(now_ts=ts, force=True)
    tracker.close_active_session(ended_at=ts)

    _save_ready_artifact(tmp_path, sequence_length=12, execute_threshold=0.45, strong_threshold=0.45)
    analyzer = SpreadMLAnalyzer(sequence_length=12, artifact_dir=tmp_path)
    analyzer.attach_tracker(tracker)

    result = analyzer.analyze_pair(
        0.64,
        tracker.get_history(*pair, limit=100),
        pair_key=pair,
    )

    assert result is not None
    assert result["range_status"] == "ready_short"
    assert result["eta_alignment_status"] == "disabled_contract_mismatch"
    assert result["signal_action"] == "STRONG_EXECUTE"


def test_analyzer_waits_when_median_total_spread_is_below_operational_floor(tmp_path: Path):
    from src.spread.spread_tracker import SpreadTracker

    db_path = tmp_path / "tracker.sqlite"
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
    tracker.flush_to_storage(now_ts=ts, force=True)
    tracker.close_active_session(ended_at=ts)

    _save_ready_artifact(
        tmp_path,
        sequence_length=12,
        execute_threshold=0.45,
        strong_threshold=0.85,
        selected_threshold=1.0,
    )
    analyzer = SpreadMLAnalyzer(sequence_length=12, artifact_dir=tmp_path, min_total_spread_pct=1.0)
    analyzer.attach_tracker(tracker)

    result = analyzer.analyze_pair(
        0.23,
        tracker.get_history(*pair, limit=100),
        pair_key=pair,
    )

    assert result is not None
    assert result["signal_action"] == "WAIT"
    assert result["signal_reason_code"] == "median_total_spread_below_threshold"
    assert result["median_total_spread"] < result["min_total_spread_threshold"]
    assert result["raw_empirical_support_short"] >= 2
