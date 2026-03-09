from pathlib import Path

from tests import run_10m_diagnostic


def test_runtime_diagnostic_report_is_separate_and_cautious(tmp_path: Path):
    report_path = run_10m_diagnostic.default_report_path(tmp_path)
    report = run_10m_diagnostic.build_report(
        {
            "avg_latency_ms": 11.2,
            "max_latency_ms": 28.4,
            "min_latency_ms": 4.8,
            "p99_latency_ms": 27.1,
            "start_db_kb": 100.0,
            "end_db_kb": 104.0,
            "db_growth_kb": 4.0,
            "total_samples": 120,
            "quality_errors": 1,
            "records_tracked": 9,
            "degraded_rows_seen": 3,
            "artifact_ready_ratio": 0.75,
        }
    )

    assert report_path.name == "best_lstm_model.runtime_diagnostic.md"
    assert "ml_diagnostic_report.md" not in str(report_path)
    assert "não prova qualidade do modelo" in report
    assert "estável" not in report.lower()


def test_runtime_diagnostic_resolves_paths_absolutely(tmp_path: Path):
    report_path, db_path = run_10m_diagnostic.resolve_runtime_paths(
        "out/reports/runtime.md",
        "out/config/tracker.sqlite",
        workspace_root=tmp_path,
    )

    assert report_path.is_absolute()
    assert db_path.is_absolute()
    assert str(report_path).endswith(str(Path("out") / "reports" / "runtime.md"))
    assert str(db_path).endswith(str(Path("out") / "config" / "tracker.sqlite"))
