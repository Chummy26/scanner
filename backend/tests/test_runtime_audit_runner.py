import json
from pathlib import Path

from src.spread.spread_tracker import SpreadTracker


def _read_ndjson(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_runtime_audit_collector_writes_events_samples_and_alerts(tmp_path: Path):
    from src.spread.runtime_audit import RuntimeAuditCollector, build_runtime_summary

    db_path = tmp_path / "tracker.sqlite"
    db_path.write_bytes(b"sqlite")
    collector = RuntimeAuditCollector(
        output_dir=tmp_path / "audit",
        record_interval_sec=15.0,
        gap_threshold_sec=60.0,
        duration_sec=120,
    )

    collector.record_ws_ingest("mexc", "BTC", "spot", 3.2)
    collector.record_tracker_record(
        pair_id="BTC|mexc|spot|gate|futures",
        ts=1_700_000_000.0,
        entry=0.31,
        exit=-0.12,
        session_id=3,
        block_id=12,
        record_delta_sec=16.0,
        gap_detected=False,
        monotonic=True,
        invalid_fields=[],
    )
    collector.record_tracker_record(
        pair_id="BTC|mexc|spot|gate|futures",
        ts=1_700_000_080.0,
        entry=0.33,
        exit=-0.08,
        session_id=3,
        block_id=13,
        record_delta_sec=80.0,
        gap_detected=True,
        monotonic=False,
        invalid_fields=["ts"],
    )
    collector.record_inference(
        pair_id="BTC|mexc|spot|gate|futures",
        result={
            "signal_action": "EXECUTE",
            "inversion_probability": 0.55,
            "eta_seconds": 900,
            "context_strength": "normal",
            "inference_latency_ms": 12.4,
            "drift_status": "stable",
            "drifted_features": [],
            "model_status": "ready",
        },
        end_to_end_ms=18.2,
    )
    collector.record_inference(
        pair_id="BTC|mexc|spot|gate|futures",
        result={
            "signal_action": "STRONG_EXECUTE",
            "inversion_probability": 0.91,
            "eta_seconds": 600,
            "context_strength": "strong",
            "inference_latency_ms": 72.5,
            "drift_status": "drifted",
            "drifted_features": ["zscore_entry"],
            "model_status": "ready",
        },
        end_to_end_ms=88.0,
    )
    collector.sample_system(
        pid=1234,
        db_path=db_path,
        tracker_stats={"records_total": 20, "blocks_total": 4, "sessions_total": 2},
        exchange_statuses=[{"name": "mexc", "ws_running": True}],
        extra={"pairs_active": 7},
    )
    paths = collector.finalize()

    api_probe_path = collector.output_dir / "api_probe.ndjson"
    api_probe_path.write_text(
        json.dumps({"kind": "api_probe", "latency_ms": 21.5, "status_code": 200, "ok": True, "payload_count": 3})
        + "\n",
        encoding="utf-8",
    )

    events = _read_ndjson(paths["events_path"])
    alerts = _read_ndjson(paths["alerts_path"])
    samples = _read_ndjson(paths["samples_path"])
    summary = json.loads(paths["summary_path"].read_text(encoding="utf-8"))
    rebuilt_summary = build_runtime_summary(collector.output_dir)

    assert any(event["kind"] == "ws_ingest" for event in events)
    assert any(event["kind"] == "tracker_record" for event in events)
    assert any(event["kind"] == "inference" for event in events)
    assert samples and samples[0]["kind"] == "system_sample"
    assert any(alert["code"] == "gap_detected" for alert in alerts)
    assert any(alert["code"] == "invalid_record" for alert in alerts)
    assert any(alert["code"] == "inference_latency_high" for alert in alerts)
    assert any(alert["code"] == "drift_detected" for alert in alerts)
    assert any(alert["code"] == "signal_probability_jump" for alert in alerts)
    assert summary["counts"]["events"] == len(events)
    assert summary["counts"]["alerts"] == len(alerts)
    assert summary["signal_counts"]["EXECUTE"] == 1
    assert summary["signal_counts"]["STRONG_EXECUTE"] == 1
    assert rebuilt_summary["api_probe_latency_ms"]["count"] == 1


def test_finalize_runtime_audit_package_creates_snapshot_and_reports(tmp_path: Path):
    from src.spread.runtime_audit import RuntimeAuditCollector, finalize_runtime_audit_package

    db_path = tmp_path / "tracker.sqlite"
    base_ts = 1_700_000_000.0
    for session_idx in range(4):
        tracker = SpreadTracker(
            window_sec=4 * 60 * 60,
            record_interval_sec=15.0,
            max_records_per_pair=0,
            epsilon_pct=0.0,
            history_enable_entry_spread_pct=0.0,
            track_enable_entry_spread_pct=0.0,
            db_path=db_path,
            gap_threshold_sec=60.0,
        )
        for idx in range(20):
            tracker.record_spread(
                "BTC",
                "mexc",
                "spot",
                "gate",
                "futures",
                0.20 + (idx * 0.01),
                -0.10 + (idx * 0.005),
                now_ts=base_ts + (session_idx * 10_000) + (idx * 15),
            )
        tracker.flush_to_storage(force=True, now_ts=base_ts + (session_idx * 10_000) + 1_000)
        tracker.close_active_session(ended_at=base_ts + (session_idx * 10_000) + 1_000)

    collector = RuntimeAuditCollector(
        output_dir=tmp_path / "audit",
        record_interval_sec=15.0,
        gap_threshold_sec=60.0,
        duration_sec=120,
    )
    collector.record_inference(
        pair_id="BTC|mexc|spot|gate|futures",
        result={
            "signal_action": "EXECUTE",
            "inversion_probability": 0.62,
            "eta_seconds": 1200,
            "context_strength": "normal",
            "inference_latency_ms": 10.0,
            "drift_status": "stable",
            "drifted_features": [],
            "model_status": "ready",
        },
        end_to_end_ms=15.0,
    )
    collector.finalize()

    def _fake_training_loop(**kwargs):
        artifact_root = Path(kwargs["artifact_dir"])
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "best_lstm_model.pth").write_bytes(b"model")
        (artifact_root / "best_lstm_model.meta.json").write_text(json.dumps({"version": "test-v1"}), encoding="utf-8")
        (artifact_root / "best_lstm_model.report.json").write_text(json.dumps({"metrics": {"test": {"precision": 0.5}}}), encoding="utf-8")
        audit_path = artifact_root / "best_lstm_model.audit.md"
        audit_path.write_text("# audit\n", encoding="utf-8")
        return {
            "metrics": {"test": {"precision": 0.5, "recall": 0.4, "f1": 0.44, "average_precision": 0.52}},
            "calibration": {"test": {"ece": 0.08}},
            "thresholds": {"execute_threshold": 0.45, "strong_threshold": 0.85},
            "artifacts": {
                "model_path": str(artifact_root / "best_lstm_model.pth"),
                "metadata_path": str(artifact_root / "best_lstm_model.meta.json"),
                "report_path": str(artifact_root / "best_lstm_model.report.json"),
                "audit_path": str(audit_path),
            },
        }

    result = finalize_runtime_audit_package(
        output_dir=collector.output_dir,
        state_path=db_path,
        run_training_fn=_fake_training_loop,
        duration_sec=120,
    )

    assert Path(result["sqlite_snapshot"]).is_file()
    assert Path(result["dataset_summary_path"]).is_file()
    assert Path(result["training_report_path"]).is_file()
    assert Path(result["manual_verification_path"]).is_file()
    assert Path(result["final_audit_path"]).is_file()

    dataset_summary = json.loads(Path(result["dataset_summary_path"]).read_text(encoding="utf-8"))
    assert dataset_summary["num_samples"] > 0
    assert dataset_summary["num_cross_block_windows"] == 0
    assert len(dataset_summary["manual_feature_checks"]) >= 1

    final_audit = Path(result["final_audit_path"]).read_text(encoding="utf-8")
    assert "Runtime Overview" in final_audit
    assert "Offline Training" in final_audit


def test_finalize_runtime_audit_package_degrades_gracefully_on_invalid_sqlite(tmp_path: Path):
    from src.spread.runtime_audit import finalize_runtime_audit_package

    db_path = tmp_path / "broken.sqlite"
    db_path.write_bytes(b"not-a-valid-sqlite-schema")

    result = finalize_runtime_audit_package(
        output_dir=tmp_path / "audit",
        state_path=db_path,
        run_training_fn=lambda **kwargs: {"model_status": "should_not_run"},
        duration_sec=30,
    )

    training_report = json.loads(Path(result["training_report_path"]).read_text(encoding="utf-8"))
    dataset_summary = json.loads(Path(result["dataset_summary_path"]).read_text(encoding="utf-8"))
    final_audit = Path(result["final_audit_path"]).read_text(encoding="utf-8")

    assert training_report["model_status"] in {"dataset_failed", "training_failed"}
    assert dataset_summary["model_status"] == "dataset_failed"
    assert "Collection & Integrity" in final_audit
