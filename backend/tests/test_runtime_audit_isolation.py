import json
from pathlib import Path

from src.spread.runtime_audit import default_runtime_audit_dir, finalize_runtime_audit_package
from src.spread.spread_tracker import SpreadTracker


def test_runtime_audit_package_isolated_from_main_model_report(tmp_path: Path):
    output_dir = default_runtime_audit_dir(tmp_path)
    assert output_dir.parent == tmp_path

    db_path = tmp_path / "tracker.sqlite"
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
    base_ts = 1_700_000_000.0
    for idx in range(20):
        tracker.record_spread(
            "BTC",
            "mexc",
            "spot",
            "gate",
            "futures",
            0.10 + (idx * 0.01),
            -0.05 + (idx * 0.005),
            now_ts=base_ts + (idx * 15),
        )
    tracker.flush_to_storage(force=True, now_ts=base_ts + 10_000)
    tracker.close_active_session(ended_at=base_ts + 10_000)

    def _fake_training_loop(**kwargs):
        artifact_root = Path(kwargs["artifact_dir"])
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "best_lstm_model.pth").write_bytes(b"model")
        (artifact_root / "best_lstm_model.meta.json").write_text(json.dumps({"version": "audit-only"}), encoding="utf-8")
        (artifact_root / "best_lstm_model.report.json").write_text(json.dumps({"metrics": {}}), encoding="utf-8")
        audit_path = artifact_root / "best_lstm_model.audit.md"
        audit_path.write_text("# dedicated audit\n", encoding="utf-8")
        return {
            "model_status": "trained",
            "metrics": {},
            "thresholds": {},
            "artifacts": {
                "model_path": str(artifact_root / "best_lstm_model.pth"),
                "metadata_path": str(artifact_root / "best_lstm_model.meta.json"),
                "report_path": str(artifact_root / "best_lstm_model.report.json"),
                "audit_path": str(audit_path),
            },
        }

    result = finalize_runtime_audit_package(
        output_dir=output_dir,
        state_path=db_path,
        run_training_fn=_fake_training_loop,
        duration_sec=120,
    )

    assert "ml_diagnostic_report.md" not in str(result["final_audit_path"])
    assert "ml_diagnostic_report.md" not in str(result["training_report_path"])
    assert Path(result["final_audit_path"]).is_file()
    assert Path(result["training_report_path"]).is_file()
