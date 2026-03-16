import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.spread import training_certification as tc
from src.spread.ml_model import ARTIFACT_BASENAME, get_artifact_paths
from src.spread.train_model import run_clean_training_cycle


@dataclass
class DummyEpisode:
    peak_entry_spread: float
    exit_spread_at_close: float
    duration_sec: float
    end_ts: float = 0.0
    start_ts: float = 0.0

    @property
    def total_spread(self) -> float:
        return float(self.peak_entry_spread + self.exit_spread_at_close)


def _make_records(pair_id: str, *, points: int = 120, start_ts: float = 0.0, step_sec: float = 30.0) -> list[dict]:
    return [
        {
            "pair_id": pair_id,
            "session_id": 101,
            "block_id": 1001,
            "timestamp": float(start_ts + index * step_sec),
            "entry_spread": 0.20 + (index % 7) * 0.01,
            "exit_spread": -0.12 + ((index + 3) % 5) * 0.01,
        }
        for index in range(points)
    ]


def _make_bundle(
    num_samples: int = 48,
    *,
    irregular_blocks: int = 0,
    interval_p50: float = 15.0,
    ratio_p90: float = 1.2,
) -> SimpleNamespace:
    feature_count = len(tc.FEATURE_NAMES)
    half_samples = max(num_samples // 2, 1)
    half_rows = torch.arange(half_samples * 2 * feature_count, dtype=torch.float32).reshape(half_samples, 2, feature_count)
    half_rows = (half_rows / 1000.0) + 0.25
    rows = torch.cat([half_rows, half_rows.clone()], dim=0)
    if rows.shape[0] < num_samples:
        rows = torch.cat([rows, half_rows[:1].clone()], dim=0)
    rows = rows[:num_samples]
    return SimpleNamespace(
        X=rows,
        y_class=torch.ones(num_samples, dtype=torch.float32),
        y_eta=torch.full((num_samples,), 120.0, dtype=torch.float32),
        feature_names=list(tc.FEATURE_NAMES),
        timestamps=[float(index * 60) for index in range(num_samples)],
        summary={
            "num_samples": num_samples,
            "block_diagnostics": {
                "inter_record_interval_sec_quantiles": {"p50": interval_p50},
                "max_to_median_interval_ratio_quantiles": {"p90": ratio_p90},
                "irregular_block_count": irregular_blocks,
            },
        },
    )


def _install_base_mocks(
    monkeypatch,
    tmp_path: Path,
    *,
    records: list[dict],
    episodes: list[DummyEpisode],
    runtime_hours: float = 2.0,
    rejection_stats: dict | None = None,
    runtime_package: dict | None = None,
    bundle: SimpleNamespace | None = None,
    integrity_anomalies: dict | None = None,
):
    state_path = tmp_path / "tracker.sqlite"
    sqlite3.connect(state_path).close()
    monkeypatch.setattr(
        tc,
        "_fetch_scope_ids",
        lambda conn, **kwargs: {
            "quality_fix_activated_at": 100.0,
            "effective_session_ids": [101],
            "effective_block_ids": [],
            "legacy_session_ids": [1],
            "legacy_sessions_excluded": 1,
            "effective_session_scope": {"mode": "post_quality_fix", "session_ids": [101], "block_ids": []},
            "runtime_session_rows": [{"id": 101, "started_at": 1_000.0, "ended_at": 1_000.0 + runtime_hours * 3600.0, "status": "closed"}],
            "hot_path_rejection_stats": rejection_stats or {},
        },
    )
    monkeypatch.setattr(
        tc,
        "collect_sqlite_integrity",
        lambda *args, **kwargs: {
            "db_exists": True,
            "db_path": str(state_path),
            "db_size_bytes": 0,
            "table_counts": {"tracker_pairs": len({item["pair_id"] for item in records}), "tracker_records": len(records)},
            "scope": {"session_ids": [101], "block_ids": []},
            "anomalies": {
                "zero_record_blocks": 0,
                "record_count_mismatches": 0,
                "range_mismatches": 0,
                "missing_event_blocks": 0,
                "open_blocks_after_close": 0,
                **(integrity_anomalies or {}),
            },
        },
    )
    monkeypatch.setattr(tc, "_load_blocks_from_sqlite", lambda *args, **kwargs: ([{"block_id": 1001, "session_id": 101, "pair_id": "stub", "records": []}], 0.0, {}))
    monkeypatch.setattr(tc, "_scope_records", lambda blocks: list(records))
    monkeypatch.setattr(tc, "_scope_episodes", lambda blocks: list(episodes))
    monkeypatch.setattr(
        tc,
        "_find_runtime_audit_package",
        lambda runtime_audit_dir=None, **kwargs: runtime_package or {"path": "", "summary": {}, "events": [], "alerts": [], "samples": []},
    )
    monkeypatch.setattr(tc, "build_dataset_bundle", lambda **kwargs: bundle or _make_bundle())
    return state_path


def _preflight_ok(**kwargs):
    threshold = float((kwargs.get("thresholds") or [0.8])[0])
    return {
        "qualifies_for_training": True,
        "selected_threshold": threshold,
        "selection_mode": "primary",
        "thresholds": {f"{threshold:.1f}": {"failure_reasons": [], "qualifies_for_training": True}},
    }


def _fingerprint(**kwargs):
    return f"fp-{int(bool(kwargs.get('allow_cross_session_merge')))}-{kwargs.get('min_total_spread_pct')}"


def test_certification_missing_runtime_audit_is_non_blocking_and_writes_artifacts(tmp_path: Path, monkeypatch):
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=_make_records("PAIR_A") + _make_records("PAIR_B", start_ts=15.0),
        episodes=[DummyEpisode(1.20, 0.20, 120.0), DummyEpisode(1.15, 0.18, 140.0), DummyEpisode(1.25, 0.22, 160.0)],
        runtime_hours=1.0,
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="full",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    assert payload["verdict"] == "CERTIFIED_WITH_WARNINGS"
    assert payload["certification_id"]
    assert (tmp_path / "artifacts" / "data_certification.json").is_file()
    assert (tmp_path / "artifacts" / "gate_results" / "gate_12_entry_exit_quality.json").is_file()
    assert payload["gate_results"]["gate_11_runtime_audit_health"]["warnings"] == ["runtime_audit_unavailable"]


def test_certification_gate1_treats_missing_event_blocks_as_warning(tmp_path: Path, monkeypatch):
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=_make_records("PAIR_A") + _make_records("PAIR_B", start_ts=15.0),
        episodes=[DummyEpisode(1.20, 0.20, 120.0), DummyEpisode(1.15, 0.18, 140.0)],
        integrity_anomalies={"missing_event_blocks": 7},
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="full",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate01 = payload["gate_results"]["gate_01_sqlite_integrity"]
    assert gate01["status"] == "WARNING"
    assert gate01["failure_reasons"] == []
    assert gate01["warnings"] == ["missing_event_blocks"]
    assert gate01["details"]["structural_integrity_failures"] == []


def test_certification_gate2_accepts_runtime_interval_near_30_seconds(tmp_path: Path, monkeypatch):
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=_make_records("PAIR_A") + _make_records("PAIR_B", start_ts=15.0),
        episodes=[DummyEpisode(1.20, 0.20, 120.0), DummyEpisode(1.15, 0.18, 140.0)],
        bundle=_make_bundle(interval_p50=32.0, ratio_p90=1.9),
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="full",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate02 = payload["gate_results"]["gate_02_intra_block_temporal_regularity"]
    assert gate02["status"] == "PASS"
    assert gate02["details"]["median_interval_p50_sec"] == pytest.approx(32.0)
    assert gate02["details"]["expected_interval_ceiling_sec"] == pytest.approx(40.0)


def test_certification_gate3_reports_hot_path_rejection_context(tmp_path: Path, monkeypatch):
    sparse_records = _make_records("PAIR_A", points=8, step_sec=600.0) + _make_records("PAIR_B", points=8, start_ts=1800.0, step_sec=600.0)
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=sparse_records,
        episodes=[DummyEpisode(1.10, 0.20, 120.0), DummyEpisode(1.12, 0.21, 120.0)],
        rejection_stats={
            "rejection_rate_by_pair": {"PAIR_A": 0.42},
            "rejection_rate_by_exchange": {"gate": 0.35},
        },
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="full",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate03 = payload["gate_results"]["gate_03_completeness"]
    assert gate03["status"] == "FAIL"
    assert gate03["recommendations"]
    assert "top_pair" in gate03["recommendations"][0]


def test_certification_gate3_scales_density_and_filters_sparse_pairs_in_quick_mode(tmp_path: Path, monkeypatch):
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=[],
        episodes=[],
        runtime_hours=2.0,
        rejection_stats={
            "rejection_rate_by_pair": {"PAIR_0000": 0.42},
            "rejection_rate_by_exchange": {"gate": 0.35},
        },
    )
    pair_ids = [f"PAIR_{index:04d}" for index in range(400)]
    pair_record_counts = Counter({pair_id: 24 for pair_id in pair_ids})
    pair_checkpoint_presence = {pair_id: {0, 1, 2, 3} for pair_id in pair_ids}
    for pair_id in pair_ids[:80]:
        pair_checkpoint_presence[pair_id] = {0, 1, 2}
    for pair_id in pair_ids[80:140]:
        pair_checkpoint_presence[pair_id] = {0, 1, 3}

    monkeypatch.setattr(
        tc,
        "_stream_quick_sqlite_metrics",
        lambda *args, **kwargs: {
            "pair_ids": pair_ids,
            "pair_record_counts": pair_record_counts,
            "pair_checkpoint_presence": pair_checkpoint_presence,
            "checkpoint_count": 4,
            "records_total": 9_600,
            "episode_total_spreads": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
            "episode_durations": [45.0, 50.0, 55.0, 60.0, 65.0, 70.0],
            "episode_peaks": [1.0 + (index % 7) * 0.01 for index in range(60)],
            "episode_exits": [0.2 + (index % 5) * 0.01 for index in range(60)],
            "pair_hour_counts": Counter({pair_id: 2 for pair_id in pair_ids}),
            "frozen_entry_hours": Counter(),
            "frozen_exit_hours": Counter(),
            "unresponsive_exit_hours": Counter(),
            "entry_outlier_hours": Counter(),
            "exit_outlier_hours": Counter(),
            "pearson_by_pair": {},
            "bilateral_zero_rate": 0.0,
            "num_blocks": 800,
            "block_diagnostics": {
                "inter_record_interval_sec_quantiles": {"p50": 31.0},
                "max_to_median_interval_ratio_quantiles": {"p90": 1.8},
                "irregular_block_count": 0,
            },
        },
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="quick",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate03 = payload["gate_results"]["gate_03_completeness"]
    assert gate03["status"] == "WARNING"
    assert gate03["failure_reasons"] == []
    assert gate03["warnings"] == ["pair_completeness_degraded"]
    assert gate03["details"]["completeness_pair_count"] == 400
    assert gate03["details"]["records_per_pair_per_hour_threshold"] == pytest.approx(10.0)
    assert gate03["details"]["records_per_pair_per_hour"] == pytest.approx(12.0)


def test_certification_gate3_treats_route_ephemerality_as_warning_not_fail(tmp_path: Path, monkeypatch):
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=[],
        episodes=[],
        runtime_hours=2.0,
    )
    pair_ids = [f"PAIR_{index:02d}" for index in range(10)]
    pair_record_counts = Counter({pair_id: 20 for pair_id in pair_ids})
    pair_checkpoint_presence = {pair_id: {0} for pair_id in pair_ids}
    pair_checkpoint_presence[pair_ids[-1]] = {0, 1, 2, 3}

    monkeypatch.setattr(
        tc,
        "_stream_quick_sqlite_metrics",
        lambda *args, **kwargs: {
            "pair_ids": pair_ids,
            "pair_record_counts": pair_record_counts,
            "pair_checkpoint_presence": pair_checkpoint_presence,
            "checkpoint_count": 4,
            "records_total": 2_000,
            "episode_total_spreads": [0.20, 0.25, 0.30, 0.35],
            "episode_durations": [45.0, 50.0, 55.0, 60.0],
            "episode_peaks": [1.0, 1.1, 1.2, 1.3],
            "episode_exits": [0.2, 0.3, 0.4, 0.5],
            "pair_hour_counts": Counter({pair_id: 2 for pair_id in pair_ids}),
            "frozen_entry_hours": Counter(),
            "frozen_exit_hours": Counter(),
            "unresponsive_exit_hours": Counter(),
            "entry_outlier_hours": Counter(),
            "exit_outlier_hours": Counter(),
            "pearson_by_pair": {},
            "bilateral_zero_rate": 0.0,
            "num_blocks": 20,
            "block_diagnostics": {
                "inter_record_interval_sec_quantiles": {"p50": 31.0},
                "max_to_median_interval_ratio_quantiles": {"p90": 1.8},
                "irregular_block_count": 0,
            },
        },
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="quick",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate03 = payload["gate_results"]["gate_03_completeness"]
    assert gate03["status"] == "PASS"
    assert gate03["failure_reasons"] == []
    assert gate03["warnings"] == []
    assert gate03["details"]["completeness_pair_count"] == 1
    assert gate03["details"]["excluded_ephemeral_pairs_count"] == 9


def test_certification_gate3_excludes_pairs_from_degraded_exchange_health(tmp_path: Path, monkeypatch):
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=[],
        episodes=[],
        runtime_hours=2.0,
        rejection_stats={
            "rejection_rate_by_pair": {"AAA|gate|futures|xt|futures": 1.0},
            "rejection_rate_by_exchange": {"xt": 0.75},
        },
    )
    pair_ids = [
        "AAA|gate|futures|xt|futures",
        "BBB|gate|futures|mexc|futures",
    ]
    pair_record_counts = Counter({pair_id: 20 for pair_id in pair_ids})
    pair_checkpoint_presence = {
        pair_ids[0]: {0, 2},
        pair_ids[1]: {0, 1, 2, 3},
    }
    monkeypatch.setattr(
        tc,
        "_stream_quick_sqlite_metrics",
        lambda *args, **kwargs: {
            "pair_ids": pair_ids,
            "pair_record_counts": pair_record_counts,
            "pair_checkpoint_presence": pair_checkpoint_presence,
            "checkpoint_count": 4,
            "records_total": 500,
            "episode_total_spreads": [0.20, 0.25, 0.30, 0.35],
            "episode_durations": [45.0, 50.0, 55.0, 60.0],
            "episode_peaks": [1.0, 1.1, 1.2, 1.3],
            "episode_exits": [0.2, 0.3, 0.4, 0.5],
            "pair_hour_counts": Counter({pair_id: 2 for pair_id in pair_ids}),
            "frozen_entry_hours": Counter(),
            "frozen_exit_hours": Counter(),
            "unresponsive_exit_hours": Counter(),
            "entry_outlier_hours": Counter(),
            "exit_outlier_hours": Counter(),
            "pearson_by_pair": {},
            "bilateral_zero_rate": 0.0,
            "num_blocks": 20,
            "min_ts": 0.0,
            "max_ts": 7_200.0,
            "block_diagnostics": {
                "inter_record_interval_sec_quantiles": {"p50": 31.0},
                "max_to_median_interval_ratio_quantiles": {"p90": 1.8},
                "irregular_block_count": 0,
            },
        },
    )
    monkeypatch.setattr(
        tc,
        "_load_hourly_health_samples",
        lambda *args, **kwargs: [
            {
                "quality_verdict": "degraded",
                "exchanges_circuit_open": ["xt"],
            }
        ],
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="quick",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate03 = payload["gate_results"]["gate_03_completeness"]
    assert gate03["status"] == "PASS"
    assert gate03["failure_reasons"] == []
    assert gate03["details"]["completeness_pair_count_raw"] == 2
    assert gate03["details"]["completeness_pair_count"] == 1
    assert gate03["details"]["excluded_pairs_by_exchange_health_count"] == 1
    assert gate03["details"]["excluded_exchanges_by_health"] == ["xt"]


@pytest.mark.parametrize(("runtime_hours", "expected_status"), [(1.5, "WARNING"), (4.0, "FAIL")])
def test_certification_gate6_distinguishes_warning_from_fail(tmp_path: Path, monkeypatch, runtime_hours: float, expected_status: str):
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=_make_records("PAIR_A"),
        episodes=[DummyEpisode(1.20, 0.20, 120.0)],
        runtime_hours=runtime_hours,
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="full",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate06 = payload["gate_results"]["gate_06_episode_yield"]
    assert gate06["status"] == expected_status


def test_certification_gate6_no_longer_requires_spread_median_above_threshold(tmp_path: Path, monkeypatch):
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=_make_records("PAIR_A"),
        episodes=[
            DummyEpisode(0.05, -0.02, 45.0),
            DummyEpisode(0.04, -0.01, 50.0),
            DummyEpisode(0.03, -0.01, 55.0),
        ],
        runtime_hours=1.0,
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="full",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate06 = payload["gate_results"]["gate_06_episode_yield"]
    assert gate06["status"] == "PASS"
    assert gate06["failure_reasons"] == []
    assert gate06["details"]["total_spread_quantiles"]["p50"] < 0.8


def test_certification_gate05_fails_on_degenerate_features(tmp_path: Path, monkeypatch):
    degenerate_bundle = SimpleNamespace(
        X=torch.zeros((32, 2, len(tc.FEATURE_NAMES)), dtype=torch.float32),
        y_class=torch.ones(32, dtype=torch.float32),
        y_eta=torch.full((32,), 120.0, dtype=torch.float32),
        feature_names=list(tc.FEATURE_NAMES),
        timestamps=[float(index * 60) for index in range(32)],
        summary={
            "num_samples": 32,
            "block_diagnostics": {
                "inter_record_interval_sec_quantiles": {"p50": 15.0},
                "max_to_median_interval_ratio_quantiles": {"p90": 1.2},
                "irregular_block_count": 0,
            },
        },
    )
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=_make_records("PAIR_A"),
        episodes=[DummyEpisode(1.20, 0.20, 120.0), DummyEpisode(1.15, 0.18, 140.0)],
        bundle=degenerate_bundle,
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="full",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate05 = payload["gate_results"]["gate_05_intra_soak_feature_drift"]
    assert gate05["status"] == "FAIL"
    assert "degenerate_features_detected" in gate05["failure_reasons"]


def test_certification_gate06_fails_on_incomplete_closed_episode_fields(tmp_path: Path, monkeypatch):
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=_make_records("PAIR_A"),
        episodes=[
            DummyEpisode(1.20, 0.20, 120.0),
            DummyEpisode(1.15, 0.18, -1.0),
            DummyEpisode(1.25, 0.22, -2.0),
        ],
        runtime_hours=1.0,
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="full",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate06 = payload["gate_results"]["gate_06_episode_yield"]
    assert gate06["status"] == "FAIL"
    assert "episode_field_completeness_failed" in gate06["failure_reasons"]


def test_certification_gate12b_excludes_pairs_failed_by_12a(tmp_path: Path, monkeypatch):
    frozen_pair = [
        {"pair_id": "PAIR_A", "session_id": 101, "block_id": 1001, "timestamp": float(index * 60), "entry_spread": 0.25, "exit_spread": -0.10}
        for index in range(20)
    ]
    unresponsive_pair = [
        {"pair_id": "PAIR_B", "session_id": 101, "block_id": 1001, "timestamp": float(index * 60), "entry_spread": 0.20 + index * 0.01, "exit_spread": -0.12}
        for index in range(20)
    ]
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=frozen_pair + unresponsive_pair,
        episodes=[DummyEpisode(1.20, 0.20, 120.0), DummyEpisode(1.25, 0.22, 140.0)],
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="full",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate12 = payload["gate_results"]["gate_12_entry_exit_quality"]
    assert gate12["details"]["eligible_pairs_for_responsiveness"] == 0
    assert gate12["details"]["unresponsive_exit_pairs"] == []
    assert "entry_or_exit_stale_quotes" in gate12["failure_reasons"]


def test_certification_gate12_uses_active_pairs_and_episode_outlier_rates(tmp_path: Path, monkeypatch):
    records: list[dict] = []
    for pair_index in range(20):
        pair_id = f"ACTIVE_{pair_index:02d}"
        for minute in range(12):
            records.append(
                {
                    "pair_id": pair_id,
                    "session_id": 101,
                    "block_id": 1001 + pair_index,
                    "timestamp": float(minute * 60),
                    "entry_spread": 0.20 + pair_index * 0.001 + minute * 0.002,
                    "exit_spread": -0.12 + (minute % 4) * 0.01,
                }
            )
    for pair_index in range(3):
        pair_id = f"ACTIVE_FROZEN_{pair_index:02d}"
        for minute in range(12):
            records.append(
                {
                    "pair_id": pair_id,
                    "session_id": 101,
                    "block_id": 2001 + pair_index,
                    "timestamp": float(minute * 60),
                    "entry_spread": 0.25,
                    "exit_spread": -0.10,
                }
            )
    for pair_index in range(30):
        pair_id = f"SPARSE_{pair_index:02d}"
        for minute in range(3):
            records.append(
                {
                    "pair_id": pair_id,
                    "session_id": 101,
                    "block_id": 3001 + pair_index,
                    "timestamp": float(minute * 60),
                    "entry_spread": 0.22,
                    "exit_spread": -0.11,
                }
            )

    episodes = [
        DummyEpisode(1.0 + (index % 7) * 0.01, 0.20 + (index % 5) * 0.01, 120.0 + (index % 3) * 10.0)
        for index in range(98)
    ]
    episodes.extend([DummyEpisode(0.0, 0.0, 120.0), DummyEpisode(0.0, 0.0, 130.0)])

    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=records,
        episodes=episodes,
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="full",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate12 = payload["gate_results"]["gate_12_entry_exit_quality"]
    assert gate12["status"] == "PASS"
    assert gate12["failure_reasons"] == []
    assert gate12["details"]["active_quality_pair_count"] == 23
    assert gate12["details"]["eligible_pairs_for_responsiveness"] == 20
    assert gate12["details"]["episode_entry_outlier_rate"] == pytest.approx(0.0)
    assert gate12["details"]["episode_entry_zero_rate"] == pytest.approx(0.02)


def test_certification_gate12_ignores_sparse_outlier_hours_below_pair_threshold(tmp_path: Path, monkeypatch):
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=[],
        episodes=[DummyEpisode(1.0, 0.2, 120.0) for _ in range(40)],
        runtime_hours=4.0,
    )
    pair_ids = [f"PAIR_{index:02d}" for index in range(20)]
    pair_record_counts = Counter({pair_id: 100 for pair_id in pair_ids})
    pair_hour_counts = Counter({pair_id: 10 for pair_id in pair_ids})
    entry_outlier_hours = Counter({pair_id: 1 for pair_id in pair_ids[:4]})
    exit_outlier_hours = Counter({pair_id: 1 for pair_id in pair_ids[4:8]})

    monkeypatch.setattr(
        tc,
        "_stream_quick_sqlite_metrics",
        lambda *args, **kwargs: {
            "pair_ids": pair_ids,
            "pair_record_counts": pair_record_counts,
            "pair_checkpoint_presence": {pair_id: {0, 1, 2, 3} for pair_id in pair_ids},
            "checkpoint_count": 4,
            "records_total": sum(pair_record_counts.values()),
            "episode_total_spreads": [0.2, 0.3, 0.4, 0.5],
            "episode_durations": [60.0, 75.0, 90.0, 105.0],
            "episode_peaks": [1.0 + (index % 5) * 0.02 for index in range(40)],
            "episode_exits": [0.2 + (index % 5) * 0.01 for index in range(40)],
            "pair_hour_counts": pair_hour_counts,
            "frozen_entry_hours": Counter(),
            "frozen_exit_hours": Counter(),
            "unresponsive_exit_hours": Counter(),
            "entry_outlier_hours": entry_outlier_hours,
            "exit_outlier_hours": exit_outlier_hours,
            "pearson_by_pair": {},
            "bilateral_zero_rate": 0.0,
            "num_blocks": 200,
            "block_diagnostics": {
                "inter_record_interval_sec_quantiles": {"p50": 31.0},
                "max_to_median_interval_ratio_quantiles": {"p90": 1.8},
                "irregular_block_count": 0,
            },
        },
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="quick",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate12 = payload["gate_results"]["gate_12_entry_exit_quality"]
    assert gate12["status"] == "PASS"
    assert gate12["failure_reasons"] == []
    assert gate12["details"]["entry_outlier_pair_count"] == 0
    assert gate12["details"]["exit_outlier_pair_count"] == 0
    assert gate12["details"]["outlier_hour_rate_threshold"] == pytest.approx(0.15)


def test_certification_gate12_fails_when_outlier_pair_rate_exceeds_threshold(tmp_path: Path, monkeypatch):
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=[],
        episodes=[DummyEpisode(1.0, 0.2, 120.0) for _ in range(40)],
        runtime_hours=4.0,
    )
    pair_ids = [f"PAIR_{index:02d}" for index in range(20)]
    pair_record_counts = Counter({pair_id: 100 for pair_id in pair_ids})
    pair_hour_counts = Counter({pair_id: 10 for pair_id in pair_ids})
    entry_outlier_hours = Counter({pair_id: 2 for pair_id in pair_ids[:4]})
    exit_outlier_hours = Counter({pair_id: 2 for pair_id in pair_ids[4:8]})

    monkeypatch.setattr(
        tc,
        "_stream_quick_sqlite_metrics",
        lambda *args, **kwargs: {
            "pair_ids": pair_ids,
            "pair_record_counts": pair_record_counts,
            "pair_checkpoint_presence": {pair_id: {0, 1, 2, 3} for pair_id in pair_ids},
            "checkpoint_count": 4,
            "records_total": sum(pair_record_counts.values()),
            "episode_total_spreads": [0.2, 0.3, 0.4, 0.5],
            "episode_durations": [60.0, 75.0, 90.0, 105.0],
            "episode_peaks": [1.0 + (index % 5) * 0.02 for index in range(40)],
            "episode_exits": [0.2 + (index % 5) * 0.01 for index in range(40)],
            "pair_hour_counts": pair_hour_counts,
            "frozen_entry_hours": Counter(),
            "frozen_exit_hours": Counter(),
            "unresponsive_exit_hours": Counter(),
            "entry_outlier_hours": entry_outlier_hours,
            "exit_outlier_hours": exit_outlier_hours,
            "pearson_by_pair": {},
            "bilateral_zero_rate": 0.0,
            "num_blocks": 200,
            "block_diagnostics": {
                "inter_record_interval_sec_quantiles": {"p50": 31.0},
                "max_to_median_interval_ratio_quantiles": {"p90": 1.8},
                "irregular_block_count": 0,
            },
        },
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="quick",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate12 = payload["gate_results"]["gate_12_entry_exit_quality"]
    assert gate12["status"] == "FAIL"
    assert "entry_outliers_unfiltered" in gate12["failure_reasons"]
    assert "exit_outliers_unfiltered" in gate12["failure_reasons"]
    assert gate12["details"]["entry_outlier_pair_count"] == 4
    assert gate12["details"]["exit_outlier_pair_count"] == 4
    assert gate12["details"]["entry_outlier_rate"] == pytest.approx(0.2)
    assert gate12["details"]["exit_outlier_rate"] == pytest.approx(0.2)


def test_run_clean_training_cycle_blocks_on_failed_certification(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "src.spread.train_model._load_blocks_from_sqlite",
        lambda *args, **kwargs: ([], 0.0, {"num_blocks": 0, "num_sessions": 0}),
    )
    monkeypatch.setattr(
        "src.spread.train_model.certify_data_for_training",
        lambda **kwargs: {"certified": False, "failure_reasons": ["runtime_health_failed"]},
    )

    with pytest.raises(ValueError, match="runtime_health_failed"):
        run_clean_training_cycle(state_file=tmp_path / "missing.sqlite", artifact_dir=tmp_path / "artifacts")


def test_run_clean_training_cycle_propagates_certification_id_to_report_and_metadata(tmp_path: Path, monkeypatch):
    certification = {
        "certified": True,
        "verdict": "CERTIFIED",
        "failure_reasons": [],
        "warnings": [],
        "certification_id": "cert-123",
        "runtime_audit_package_path": "",
        "effective_session_scope": {"session_ids": [101], "block_ids": []},
        "gate_results": {"gate_10_dual_mode_preflight": {"details": {"qualifying_configs": ["seq4_h240_merge_off:0.8"]}}},
    }

    monkeypatch.setattr("src.spread.train_model.certify_data_for_training", lambda **kwargs: certification)
    monkeypatch.setattr(
        "src.spread.train_model._load_blocks_from_sqlite",
        lambda *args, **kwargs: ([], 0.0, {"num_blocks": 0, "num_sessions": 0}),
    )
    monkeypatch.setattr(
        "src.spread.train_model.run_threshold_preflight",
        lambda **kwargs: {"qualifies_for_training": True, "selected_threshold": 0.8},
    )

    def _fake_training_loop(**kwargs):
        artifact_dir = Path(kwargs["artifact_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _, meta_path = get_artifact_paths(artifact_dir)
        meta_path.write_text(json.dumps({"training_config": {}}), encoding="utf-8")
        return {
            "artifact_metadata": {"training_config": {}},
            "training": {},
            "artifacts": {"report_path": str(artifact_dir / f"{ARTIFACT_BASENAME}.report.json")},
        }

    monkeypatch.setattr("src.spread.train_model.run_training_loop", _fake_training_loop)

    report = run_clean_training_cycle(
        state_file=tmp_path / "state.sqlite",
        artifact_dir=tmp_path / "artifacts",
    )

    _, meta_path = get_artifact_paths(tmp_path / "artifacts")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    assert report["certification_id"] == "cert-123"
    assert report["artifact_metadata"]["training_config"]["certification_id"] == "cert-123"
    assert metadata["training_config"]["certification_id"] == "cert-123"


def test_certification_quick_mode_skips_dual_preflight(tmp_path: Path, monkeypatch):
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=_make_records("PAIR_A"),
        episodes=[DummyEpisode(1.20, 0.20, 120.0)],
    )

    def _unexpected_preflight(**kwargs):
        raise AssertionError("quick mode should skip dual preflight")

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="quick",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_unexpected_preflight,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate10 = payload["gate_results"]["gate_10_dual_mode_preflight"]
    assert gate10["status"] == "SKIPPED"
    assert gate10["warnings"] == ["dual_preflight_skipped_in_quick_mode"]
    assert gate10["details"]["executed"] is False


def test_certification_quick_mode_avoids_building_dataset_bundle(tmp_path: Path, monkeypatch):
    state_path = _install_base_mocks(
        monkeypatch,
        tmp_path,
        records=_make_records("PAIR_A"),
        episodes=[DummyEpisode(1.20, 0.20, 120.0)],
    )
    monkeypatch.setattr(
        tc,
        "_build_block_diagnostics",
        lambda blocks, sequence_length: {
            "inter_record_interval_sec_quantiles": {"p50": 15.0},
            "max_to_median_interval_ratio_quantiles": {"p90": 1.2},
            "irregular_block_count": 0,
        },
    )
    monkeypatch.setattr(
        tc,
        "build_dataset_bundle",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("quick mode should not build dataset bundle")),
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=4,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="quick",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    assert payload["gate_results"]["gate_02_intra_block_temporal_regularity"]["status"] == "PASS"
    assert payload["gate_results"]["gate_10_dual_mode_preflight"]["status"] == "SKIPPED"


def test_certification_quick_mode_streams_sqlite_blocks_without_legacy_loader(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "quick_stream.sqlite"
    with sqlite3.connect(state_path) as conn:
        conn.execute("CREATE TABLE tracker_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO tracker_meta (key, value) VALUES ('quality_fix_activated_at', '0')")
        conn.execute(
            """
            CREATE TABLE tracker_capture_sessions (
                id INTEGER PRIMARY KEY,
                started_at REAL,
                ended_at REAL,
                status TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE tracker_pairs (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                buy_ex TEXT,
                buy_mt TEXT,
                sell_ex TEXT,
                sell_mt TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE tracker_pair_blocks (
                id INTEGER PRIMARY KEY,
                pair_id INTEGER,
                session_id INTEGER,
                start_ts REAL,
                end_ts REAL,
                record_count INTEGER,
                boundary_reason TEXT,
                selected_for_training INTEGER,
                is_open INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE tracker_records (
                block_id INTEGER,
                ts REAL,
                entry_spread_pct REAL,
                exit_spread_pct REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE tracker_events (
                session_id INTEGER,
                block_id INTEGER,
                event_type TEXT,
                ts REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE tracker_hourly_health (
                hour_start_ts REAL,
                hour_end_ts REAL,
                records_total INTEGER,
                records_rejected INTEGER,
                rejection_rate_pct REAL,
                exchanges_active INTEGER,
                exchanges_circuit_open_json TEXT,
                pairs_with_records INTEGER,
                pairs_with_gaps INTEGER,
                cross_exchange_flags INTEGER,
                avg_book_age_json TEXT,
                episode_count INTEGER,
                quality_verdict TEXT,
                created_at REAL
            )
            """
        )
        conn.execute("INSERT INTO tracker_capture_sessions (id, started_at, ended_at, status) VALUES (101, 1000.0, 4600.0, 'closed')")
        conn.execute("INSERT INTO tracker_pairs (id, symbol, buy_ex, buy_mt, sell_ex, sell_mt) VALUES (1, 'PAIR', 'a', 'spot', 'b', 'spot')")
        conn.execute(
            """
            INSERT INTO tracker_pair_blocks (
                id, pair_id, session_id, start_ts, end_ts, record_count, boundary_reason, selected_for_training, is_open
            ) VALUES (1001, 1, 101, 1000.0, 1060.0, 3, 'closed', 1, 0)
            """
        )
        conn.executemany(
            "INSERT INTO tracker_records (block_id, ts, entry_spread_pct, exit_spread_pct) VALUES (?, ?, ?, ?)",
            [
                (1001, 1000.0, 0.20, -0.10),
                (1001, 1015.0, 0.21, -0.09),
                (1001, 1030.0, 0.22, -0.08),
            ],
        )
        conn.execute(
            """
            INSERT INTO tracker_hourly_health (
                hour_start_ts, hour_end_ts, records_total, records_rejected, rejection_rate_pct, exchanges_active,
                exchanges_circuit_open_json, pairs_with_records, pairs_with_gaps, cross_exchange_flags,
                avg_book_age_json, episode_count, quality_verdict, created_at
            ) VALUES (1000.0, 4600.0, 3, 0, 0.0, 2, '[]', 1, 0, 0, '{}', 0, 'healthy', 4600.0)
            """
        )
        conn.commit()

    monkeypatch.setattr(
        tc,
        "_load_blocks_from_sqlite",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("quick sqlite mode should stream directly from sqlite")),
    )
    monkeypatch.setattr(
        tc,
        "build_dataset_bundle",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("quick sqlite mode should not build dataset bundle")),
    )

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=3,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=None,
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="quick",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    assert payload["gate_results"]["gate_10_dual_mode_preflight"]["status"] == "SKIPPED"
    assert payload["gate_results"]["gate_02_intra_block_temporal_regularity"]["details"]["total_blocks"] == 1


def test_certification_quick_mode_scopes_hourly_health_to_selected_window(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "tracker_history_hourly_scope.sqlite"
    with sqlite3.connect(state_path) as conn:
        conn.execute("CREATE TABLE tracker_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            """
            CREATE TABLE tracker_capture_sessions (
                id INTEGER PRIMARY KEY,
                started_at REAL,
                ended_at REAL,
                status TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE tracker_pairs (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                buy_ex TEXT,
                buy_mt TEXT,
                sell_ex TEXT,
                sell_mt TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE tracker_pair_blocks (
                id INTEGER PRIMARY KEY,
                pair_id INTEGER,
                session_id INTEGER,
                start_ts REAL,
                end_ts REAL,
                record_count INTEGER,
                boundary_reason TEXT,
                selected_for_training INTEGER,
                is_open INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE tracker_records (
                block_id INTEGER,
                ts REAL,
                entry_spread_pct REAL,
                exit_spread_pct REAL
            )
            """
        )
        conn.execute("CREATE TABLE tracker_events (session_id INTEGER, block_id INTEGER, event_type TEXT, ts REAL)")
        conn.execute(
            """
            CREATE TABLE tracker_hourly_health (
                hour_start_ts REAL,
                hour_end_ts REAL,
                records_total INTEGER,
                records_rejected INTEGER,
                rejection_rate_pct REAL,
                exchanges_active INTEGER,
                exchanges_circuit_open_json TEXT,
                pairs_with_records INTEGER,
                pairs_with_gaps INTEGER,
                cross_exchange_flags INTEGER,
                avg_book_age_json TEXT,
                episode_count INTEGER,
                quality_verdict TEXT,
                created_at REAL
            )
            """
        )
        conn.execute("INSERT INTO tracker_capture_sessions (id, started_at, ended_at, status) VALUES (201, 1000.0, 1100.0, 'closed')")
        conn.execute("INSERT INTO tracker_pairs (id, symbol, buy_ex, buy_mt, sell_ex, sell_mt) VALUES (1, 'PAIR', 'a', 'spot', 'b', 'spot')")
        conn.execute(
            """
            INSERT INTO tracker_pair_blocks (
                id, pair_id, session_id, start_ts, end_ts, record_count, boundary_reason, selected_for_training, is_open
            ) VALUES (2001, 1, 201, 1000.0, 1030.0, 3, 'closed', 1, 0)
            """
        )
        conn.executemany(
            "INSERT INTO tracker_records (block_id, ts, entry_spread_pct, exit_spread_pct) VALUES (?, ?, ?, ?)",
            [
                (2001, 1000.0, 0.20, -0.10),
                (2001, 1015.0, 0.21, -0.09),
                (2001, 1030.0, 0.22, -0.08),
            ],
        )
        conn.execute(
            """
            INSERT INTO tracker_hourly_health (
                hour_start_ts, hour_end_ts, records_total, records_rejected, rejection_rate_pct, exchanges_active,
                exchanges_circuit_open_json, pairs_with_records, pairs_with_gaps, cross_exchange_flags,
                avg_book_age_json, episode_count, quality_verdict, created_at
            ) VALUES (900.0, 1200.0, 3, 0, 0.0, 2, '[]', 1, 0, 0, '{}', 0, 'healthy', 1200.0)
            """
        )
        conn.execute(
            """
            INSERT INTO tracker_hourly_health (
                hour_start_ts, hour_end_ts, records_total, records_rejected, rejection_rate_pct, exchanges_active,
                exchanges_circuit_open_json, pairs_with_records, pairs_with_gaps, cross_exchange_flags,
                avg_book_age_json, episode_count, quality_verdict, created_at
            ) VALUES (7200.0, 10800.0, 0, 0, 0.0, 0, '[]', 0, 0, 0, '{}', 0, 'unhealthy', 10800.0)
            """
        )
        conn.commit()

    payload = tc.run_training_certification(
        state_file=state_path,
        artifact_dir=tmp_path / "artifacts",
        sequence_length=3,
        prediction_horizon_sec=240,
        thresholds=[0.8],
        selected_session_ids=[201],
        selected_block_ids=None,
        allow_cross_session_merge=False,
        max_session_gap_sec=None,
        regime_shift_score_threshold=3.0,
        certification_mode="quick",
        max_certification_duration_sec=300,
        allow_legacy_sessions=False,
        runtime_audit_dir=None,
        run_reconnection_stress=False,
        preflight_fn=_preflight_ok,
        dataset_fingerprint_fn=_fingerprint,
    )

    gate07 = payload["gate_results"]["gate_07_book_health"]
    assert gate07["status"] == "PASS"
    assert gate07["failure_reasons"] == []
    assert gate07["details"]["hourly_health_sample_count"] == 1


def test_scope_and_integrity_handle_more_than_999_block_ids(tmp_path: Path):
    db_path = tmp_path / "large_scope.sqlite"
    block_count = 1205
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE tracker_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            """
            CREATE TABLE tracker_capture_sessions (
                id INTEGER PRIMARY KEY,
                started_at REAL,
                ended_at REAL,
                status TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE tracker_pair_blocks (
                id INTEGER PRIMARY KEY,
                session_id INTEGER,
                record_count INTEGER,
                start_ts REAL,
                end_ts REAL,
                is_open INTEGER
            )
            """
        )
        conn.execute("CREATE TABLE tracker_records (block_id INTEGER, ts REAL)")
        conn.execute("CREATE TABLE tracker_events (session_id INTEGER, block_id INTEGER, ts REAL)")
        conn.execute(
            "INSERT INTO tracker_capture_sessions (id, started_at, ended_at, status) VALUES (101, 1000.0, 4600.0, 'closed')"
        )
        conn.execute("INSERT INTO tracker_meta (key, value) VALUES ('quality_fix_activated_at', '0')")
        block_rows = [
            (block_id, 101, 1, float(1000 + block_id), float(1001 + block_id), 0)
            for block_id in range(1, block_count + 1)
        ]
        conn.executemany(
            "INSERT INTO tracker_pair_blocks (id, session_id, record_count, start_ts, end_ts, is_open) VALUES (?, ?, ?, ?, ?, ?)",
            block_rows,
        )
        conn.executemany(
            "INSERT INTO tracker_records (block_id, ts) VALUES (?, ?)",
            [(block_id, float(1000 + block_id)) for block_id in range(1, block_count + 1)],
        )
        conn.executemany(
            "INSERT INTO tracker_events (session_id, block_id, ts) VALUES (?, ?, ?)",
            [(101, block_id, float(1000 + block_id)) for block_id in range(1, block_count + 1)],
        )
        conn.commit()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        scope = tc._fetch_scope_ids(
            conn,
            selected_session_ids=None,
            selected_block_ids=list(range(1, block_count + 1)),
            allow_legacy_sessions=False,
        )

    integrity = tc.collect_sqlite_integrity(
        db_path,
        selected_session_ids=scope["effective_session_ids"],
        selected_block_ids=scope["effective_block_ids"],
    )

    assert len(scope["effective_block_ids"]) == block_count
    assert scope["effective_session_ids"] == [101]
    assert integrity["scope"]["session_ids"] == [101]
    assert len(integrity["scope"]["block_ids"]) == block_count
    assert integrity["anomalies"]["record_count_mismatches"] == 0
    assert integrity["anomalies"]["missing_event_blocks"] == 0
