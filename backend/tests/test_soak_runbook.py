import math
from pathlib import Path

from tests.run_arbml_v2_soak import _normalize_stage_arg, _resolve_duration_override
from src.spread.feature_contracts import DEFAULT_FEATURE_CONTRACT_VERSION, V2_MULTISCALE_FEATURE_NAMES
from src.spread.runtime_audit import RuntimeAuditCollector
from src.spread.soak_runbook import (
    _signal_anti_spike_gate,
    audit_snapshot_labeling,
    collect_runtime_audit_checks,
    collect_tracker_sql_checks,
    evaluate_stage1,
    run_feature_history_harness,
    run_memory_window_harness,
    validate_latest_run_payload,
)
from src.spread.spread_tracker import SpreadTracker


def _make_tracker(
    db_path: Path,
    *,
    record_interval_sec: float = 60.0,
    window_sec: int = 8 * 24 * 60 * 60,
    memory_window_sec: int = 12 * 60 * 60,
) -> SpreadTracker:
    return SpreadTracker(
        window_sec=window_sec,
        memory_window_sec=memory_window_sec,
        record_interval_sec=record_interval_sec,
        max_records_per_pair=0,
        epsilon_pct=0.0,
        history_enable_entry_spread_pct=0.0,
        track_enable_entry_spread_pct=0.0,
        db_path=db_path,
        gap_threshold_sec=max(record_interval_sec * 2.0, 60.0),
    )


def _record_series(
    tracker: SpreadTracker,
    *,
    symbol: str,
    market_pair: tuple[str, str, str, str] = ("mexc", "spot", "gate", "futures"),
    start_ts: float,
    step_sec: float,
    values: list[tuple[float, float]],
) -> float:
    buy_ex, buy_mt, sell_ex, sell_mt = market_pair
    last_ts = float(start_ts)
    for index, (entry_spread, exit_spread) in enumerate(values):
        last_ts = float(start_ts) + (float(index) * float(step_sec))
        tracker.record_spread(
            symbol,
            buy_ex,
            buy_mt,
            sell_ex,
            sell_mt,
            float(entry_spread),
            float(exit_spread),
            now_ts=last_ts,
        )
    return last_ts


def _persist_tracker(tracker: SpreadTracker, *, ended_at: float) -> None:
    assert tracker.flush_to_storage(now_ts=ended_at, force=True)
    tracker.close_active_session(ended_at=ended_at)


def test_soak_runner_cli_accepts_numeric_stage_aliases_and_duration_hours():
    assert _normalize_stage_arg("1") == "stage1"
    assert _normalize_stage_arg("2") == "stage2"
    assert _normalize_stage_arg("both") == "both"
    assert _resolve_duration_override(0, 3.0, "stage1") == 3 * 60 * 60
    assert _resolve_duration_override(900, 0.0, "stage1") == 900


def test_collect_tracker_sql_checks_and_feature_history_harness_cover_low_spread_and_multiscale_features(tmp_path: Path):
    db_path = tmp_path / "tracker.sqlite"
    tracker = _make_tracker(db_path, record_interval_sec=60.0)
    base_ts = 1_700_100_000.0
    low_spread_values = [
        (
            0.11 + (0.04 * math.sin(index / 4.0)) + (0.01 if index % 7 == 0 else 0.0),
            -0.07 + (0.03 * math.cos(index / 5.0)),
        )
        for index in range(90)
    ]
    high_spread_values = [
        (
            0.45 + (0.15 * math.sin(index / 6.0)) + (0.03 if index % 9 == 0 else 0.0),
            -0.12 + (0.05 * math.cos(index / 7.0)),
        )
        for index in range(90)
    ]
    last_low_ts = _record_series(
        tracker,
        symbol="LOW",
        start_ts=base_ts,
        step_sec=60.0,
        values=low_spread_values,
    )
    last_high_ts = _record_series(
        tracker,
        symbol="DOGE",
        start_ts=base_ts,
        step_sec=60.0,
        values=high_spread_values,
    )
    ended_at = max(last_low_ts, last_high_ts)
    _persist_tracker(tracker, ended_at=ended_at)
    tracker.save_hourly_health_digest(
        {
            "hour_start_ts": ended_at - 3600.0,
            "hour_end_ts": ended_at,
            "records_total": 180,
            "records_rejected": 8,
            "rejection_rate_pct": 4.4,
            "exchanges_active": 2,
            "exchanges_circuit_open": [],
            "pairs_with_records": 2,
            "pairs_with_gaps": 0,
            "cross_exchange_flags": 1,
            "avg_book_age_sec": {"mexc": 1.2, "gate": 1.4},
            "episode_count": 0,
            "quality_verdict": "healthy",
            "created_at": ended_at,
        }
    )

    sql_checks = collect_tracker_sql_checks(db_path)
    feature_harness = run_feature_history_harness(db_path, limit=15)

    assert sql_checks["ok"] is True
    assert sql_checks["low_spread_records"] > 0
    assert sql_checks["low_spread_pairs_history_enabled"] > 0
    assert sql_checks["min_entry_spread_pct_recent"] < 0.2
    assert sql_checks["stale_pairs"] == []
    assert sql_checks["latest_hourly_health"]["quality_verdict"] == "healthy"

    assert feature_harness["ok"] is True
    assert feature_harness["feature_count"] == len(V2_MULTISCALE_FEATURE_NAMES)
    assert feature_harness["feature_contract_version"] == DEFAULT_FEATURE_CONTRACT_VERSION
    assert feature_harness["cache_invalidated"] is True
    assert feature_harness["nonzero_multiscale_features"]


def test_run_memory_window_harness_keeps_old_records_on_disk_but_not_in_ram(tmp_path: Path):
    db_path = tmp_path / "memory_window.sqlite"
    tracker = _make_tracker(
        db_path,
        record_interval_sec=3600.0,
        window_sec=8 * 24 * 60 * 60,
        memory_window_sec=12 * 60 * 60,
    )
    reference_ts = 1_700_300_000.0
    hourly_values = [
        (0.30 + (0.02 * math.sin(index)), -0.10 + (0.01 * math.cos(index)))
        for index in range(16)
    ]
    start_ts = reference_ts - (15 * 3600.0)
    ended_at = _record_series(
        tracker,
        symbol="BTC",
        start_ts=start_ts,
        step_sec=3600.0,
        values=hourly_values,
    )
    _persist_tracker(tracker, ended_at=ended_at)

    harness = run_memory_window_harness(db_path)

    assert harness["ok"] is True
    assert harness["has_disk_records_older_than_8h"] is True
    assert harness["memory_kept_older_than_13h"] is False
    assert int(harness["memory_record_count"]) <= int(harness["disk_record_count"])
    assert harness["tracker_memory_window_sec"] == 12 * 60 * 60


def test_audit_snapshot_labeling_reports_rolling_thresholds_without_leakage(tmp_path: Path):
    db_path = tmp_path / "snapshot.sqlite"
    tracker = _make_tracker(db_path, record_interval_sec=900.0)
    base_ts = 1_700_500_000.0
    high_cycle = [
        (0.05, -0.25),
        (0.20, -0.20),
        (0.60, -0.10),
        (1.10, 0.05),
        (1.40, 0.10),
        (0.90, 0.20),
        (0.20, 0.25),
        (-0.10, 0.20),
        (-0.05, 0.10),
    ]
    low_cycle = [
        (0.02, -0.18),
        (0.08, -0.14),
        (0.20, -0.05),
        (0.35, 0.05),
        (0.45, 0.10),
        (0.22, 0.12),
        (-0.04, 0.08),
        (-0.02, 0.04),
    ]
    warmup_high = [(0.02, -0.18)] * 20
    warmup_low = [(0.01, -0.12)] * 20
    high_values = warmup_high + (high_cycle * 5)
    low_values = warmup_low + (low_cycle * 5)
    high_end = _record_series(
        tracker,
        symbol="SHIB",
        start_ts=base_ts,
        step_sec=900.0,
        values=high_values,
    )
    low_end = _record_series(
        tracker,
        symbol="BTC",
        start_ts=base_ts,
        step_sec=900.0,
        values=low_values,
    )
    ended_at = max(high_end, low_end)
    _persist_tracker(tracker, ended_at=ended_at)

    audit = audit_snapshot_labeling(
        db_path,
        sequence_length=5,
        label_cost_floor_pct=1.0,
        label_percentile=70,
        label_episode_window_days=5,
        selected_only=False,
        closed_only=True,
    )

    assert audit["ok"] is True
    assert audit["label_threshold_mode"] == "rolling_pair_percentile"
    assert audit["zero_leakage_ok"] is True
    assert audit["fallback_cost_floor_samples"] > 0
    assert audit["sample_count"] > 0
    assert audit["elevated_threshold_pairs"]
    assert any(
        float((payload or {}).get("latest", 0.0)) > 1.0
        for payload in audit["label_thresholds"].values()
    )


def test_validate_latest_run_payload_and_stage_gates_use_runtime_shapes():
    latest_payload = {
        "id": 42,
        "status": "completed",
        "artifact_dir": "C:/tmp/artifacts/run_42",
        "started_at": 1_700_900_000.0,
        "finished_at": 1_700_900_600.0,
        "result": {
            "model_status": "ready",
            "metrics": {"test": {"roc_auc": 0.63}},
            "training": {"platt_scale": 1.25, "platt_bias": -0.35},
            "artifact_metadata": {"feature_contract_version": DEFAULT_FEATURE_CONTRACT_VERSION},
            "training_config": {
                "label_threshold_mode": "rolling_pair_percentile",
                "selected_threshold": 1.0,
                "selected_label_config": {"cost_floor_pct": 1.0, "percentile": 70, "episode_window_days": 5},
            },
        },
    }
    scanner_checkpoint = {
        "metrics": {
            "invalid_probability_count": 0,
            "success_probability_summary": {"count": 4, "p50": 0.61},
            "signal_action_counts": {"WAIT": 8, "EXECUTE": 1},
        }
    }
    validated = validate_latest_run_payload(latest_payload, scanner_checkpoint=scanner_checkpoint)

    assert validated["ok"] is True
    assert validated["auc"] == 0.63
    assert validated["platt_finite"] is True
    assert validated["feature_contract_version"] == DEFAULT_FEATURE_CONTRACT_VERSION
    assert validated["label_threshold_mode"] == "rolling_pair_percentile"

    stage1_checkpoints = [
        {
            "ts": 1_700_000_000.0,
            "metrics": {
                "rss_mb": 800.0,
                "calculate_ms_p95": 20.0,
                "event_loop_lag_ms_p95": 100.0,
                "event_loop_lag_ms_p99": 250.0,
                "tracker_records_enqueued_p95": 0.0,
                "tracker_rejections_enqueued_p95": 0.0,
                "dashboard_low_spread_count": 0,
            },
            "http": {
                "debug_perf": {"ok": True, "status": 200},
                "system_health": {"ok": True, "status": 200},
                "pipeline_status": {"ok": True, "status": 200},
                "scanner_lite": {"ok": True, "status": 200},
            },
        },
        {
            "ts": 1_700_001_200.0,
            "metrics": {
                "rss_mb": 802.0,
                "calculate_ms_p95": 24.0,
                "event_loop_lag_ms_p95": 105.0,
                "event_loop_lag_ms_p99": 260.0,
                "tracker_records_enqueued_p95": 0.0,
                "tracker_rejections_enqueued_p95": 0.0,
                "dashboard_low_spread_count": 0,
            },
            "http": {
                "debug_perf": {"ok": True, "status": 200},
                "system_health": {"ok": True, "status": 200},
                "pipeline_status": {"ok": True, "status": 200},
                "scanner_lite": {"ok": True, "status": 200},
            },
        },
        {
            "ts": 1_700_002_400.0,
            "metrics": {
                "rss_mb": 804.0,
                "calculate_ms_p95": 23.0,
                "event_loop_lag_ms_p95": 110.0,
                "event_loop_lag_ms_p99": 240.0,
                "tracker_records_enqueued_p95": 0.0,
                "tracker_rejections_enqueued_p95": 0.0,
                "dashboard_low_spread_count": 0,
            },
            "http": {
                "debug_perf": {"ok": True, "status": 200},
                "system_health": {"ok": True, "status": 200},
                "pipeline_status": {"ok": True, "status": 200},
                "scanner_lite": {"ok": True, "status": 200},
            },
        },
    ]
    stage1_result = evaluate_stage1(
        stage1_checkpoints,
        sql_checks={
            "low_spread_records": 12,
            "low_spread_pairs_history_enabled": 2,
            "min_entry_spread_pct_recent": 0.07,
            "stale_pairs": [],
            "latest_hourly_health": {"rejection_rate_pct": 4.2, "quality_verdict": "healthy"},
        },
        feature_harness={
            "ok": True,
            "feature_count": len(V2_MULTISCALE_FEATURE_NAMES),
            "cache_invalidated": True,
            "nonzero_multiscale_features": ["entry_std_30m"],
        },
        runtime_audit_checks={
            "ok": True,
            "events_size_mb_per_hour": 1.4,
            "ws_latency_summary_exists": True,
            "disconnect_alert_count": 0,
            "reconnect_alert_count": 0,
            "summary": {"dashboard_validation": {"book_age_p95_sec": {"p95": 4.1}}},
        },
    )

    assert stage1_result["ok"] is True
    assert all(bool(gate["ok"]) for gate in stage1_result["gates"])

    anti_spike_gate = _signal_anti_spike_gate(
        [
            {"ts": 1_700_900_000.0, "metrics": {"signal_action_counts": {"WAIT": 9, "EXECUTE": 1}}},
            {"ts": 1_700_900_300.0, "metrics": {"signal_action_counts": {"WAIT": 8, "EXECUTE": 1}}},
            {"ts": 1_700_900_700.0, "metrics": {"signal_action_counts": {"WAIT": 7, "EXECUTE": 1}}},
            {"ts": 1_700_901_000.0, "metrics": {"signal_action_counts": {"WAIT": 6, "STRONG_EXECUTE": 1}}},
        ],
        deploy_ts=1_700_900_600.0,
    )

    assert anti_spike_gate["ok"] is True
    assert anti_spike_gate["name"] == "signal_gate_anti_spike"


def test_collect_runtime_audit_checks_reads_latest_package(tmp_path: Path):
    collector = RuntimeAuditCollector(
        output_dir=tmp_path / "runtime_audit" / "20260310T000000Z",
        record_interval_sec=15.0,
        gap_threshold_sec=60.0,
        duration_sec=7200,
    )
    collector.record_ws_ingest("mexc", "BTC", "spot", 12.0)
    collector.record_inference(
        pair_id="BTC|mexc|spot|gate|futures",
        result={
            "signal_action": "WAIT",
            "inversion_probability": 0.44,
            "eta_seconds": 900,
            "context_strength": "normal",
            "inference_latency_ms": 8.0,
            "drift_status": "stable",
            "drifted_features": [],
            "model_status": "ready",
        },
        end_to_end_ms=16.0,
    )
    collector.finalize_runtime_only()

    checks = collect_runtime_audit_checks(tmp_path / "runtime_audit")

    assert checks["ok"] is True
    assert checks["ws_latency_summary_exists"] is True
    assert checks["events_size_mb_per_hour"] >= 0.0
    assert checks["book_age_p95_sec"] >= 0.0
