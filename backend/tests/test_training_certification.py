import json
import sqlite3
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


def _make_bundle(num_samples: int = 48, *, irregular_blocks: int = 0) -> SimpleNamespace:
    feature_count = 10
    rows = torch.full((num_samples, 2, feature_count), 0.25, dtype=torch.float32)
    return SimpleNamespace(
        X=rows,
        y_class=torch.ones(num_samples, dtype=torch.float32),
        y_eta=torch.full((num_samples,), 120.0, dtype=torch.float32),
        timestamps=[float(index * 60) for index in range(num_samples)],
        summary={
            "num_samples": num_samples,
            "block_diagnostics": {
                "inter_record_interval_sec_quantiles": {"p50": 15.0},
                "max_to_median_interval_ratio_quantiles": {"p90": 1.2},
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


def test_run_clean_training_cycle_blocks_on_failed_certification(tmp_path: Path, monkeypatch):
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
