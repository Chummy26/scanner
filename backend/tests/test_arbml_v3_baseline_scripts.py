from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests import run_arbml_v3_baseline_monitor as monitor_script
from tests import run_arbml_v3_baseline_t0 as t0_script
from tests import run_arbml_v3_baseline_t4 as t4_script
from tests import run_arbml_v3_baseline_t8 as t8_script


def test_t0_missing_db_returns_blocking_finding(tmp_path):
    run_dir = tmp_path / "baseline_run"
    run_dir.mkdir()
    state_path = tmp_path / "missing.sqlite"

    payload = t0_script.run_t0_check(
        run_dir=run_dir,
        state_path=state_path,
        base_url="http://127.0.0.1:8000",
    )

    assert "missing_state_db" in payload["blocking_findings"]
    assert (run_dir / "t0_check.json").is_file()
    assert (run_dir / "t0_check.md").is_file()


def test_monitor_aborts_when_health_probe_fails(tmp_path, monkeypatch):
    run_dir = tmp_path / "baseline_run"
    run_dir.mkdir()
    state_path = tmp_path / "tracker.sqlite"
    state_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        monitor_script.common,
        "collect_runtime_probe",
        lambda base_url: {
            "health_ok": False,
            "perf_ok": False,
            "rejection_rate_pct": 0.0,
            "rss_mb": 0.0,
        },
    )
    monkeypatch.setattr(monitor_script.common, "read_server_lock_status", lambda: {})
    monkeypatch.setattr(
        monitor_script.common,
        "collect_db_overview",
        lambda path: {"max_ts": 0.0, "max_ts_utc": "", "table_counts": {}},
    )
    monkeypatch.setattr(monitor_script.common, "count_records_since", lambda path, since_ts: 0)
    monkeypatch.setattr(monitor_script.common, "count_closed_episodes_since", lambda path, since_ts: 0)
    monkeypatch.setattr(monitor_script.common, "get_process_rss_mb", lambda pid: 0.0)
    monkeypatch.setattr(monitor_script.common, "current_server_pid", lambda lock_status: 0)

    payload = monitor_script.run_monitor(
        run_dir=run_dir,
        state_path=state_path,
        base_url="http://127.0.0.1:8000",
        restart_ts=monitor_script.common.utc_now_ts(),
        duration_hours=8.0,
        interval_minutes=30,
    )

    assert "runtime_health_failed" in payload["failures"]
    assert (run_dir / "runtime_monitor_summary.json").is_file()
    assert (run_dir / "runtime_monitor_summary.md").is_file()


def test_t4_aborts_when_preflight_does_not_qualify(tmp_path, monkeypatch):
    run_dir = tmp_path / "baseline_run"
    run_dir.mkdir()
    state_path = tmp_path / "tracker.sqlite"
    state_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        t4_script.common,
        "collect_runtime_probe",
        lambda base_url: {
            "health_ok": True,
            "perf_ok": True,
            "rejection_rate_pct": 0.0,
        },
    )
    monkeypatch.setattr(
        t4_script.common,
        "run_quick_certification",
        lambda **kwargs: {
            "verdict": "CERTIFIED_WITH_WARNINGS",
            "failure_reasons": [],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        t4_script.common,
        "run_preflight_check",
        lambda **kwargs: {
            "qualifies_for_training": False,
            "selection_mode": "none",
            "selected_threshold": None,
            "warnings": ["insufficient_positives"],
        },
    )

    payload = t4_script.run_t4_check(
        run_dir=run_dir,
        state_path=state_path,
        base_url="http://127.0.0.1:8000",
        restart_ts=t4_script.common.utc_now_ts(),
    )

    assert "preflight_not_qualified" in payload["failures"]
    assert (run_dir / "t4_quick_cert.json").is_file()
    assert (run_dir / "t4_preflight.json").is_file()
    assert (run_dir / "t4_summary.md").is_file()


def test_t8_stops_before_training_when_quick_cert_fails(tmp_path, monkeypatch):
    run_dir = tmp_path / "baseline_run"
    run_dir.mkdir()
    state_path = tmp_path / "tracker.sqlite"
    state_path.write_text("placeholder", encoding="utf-8")
    snapshot_path = run_dir / "snapshot.sqlite"

    monkeypatch.setattr(
        t8_script.common,
        "collect_runtime_probe",
        lambda base_url: {
            "health_ok": True,
            "perf_ok": True,
            "rejection_rate_pct": 0.0,
        },
    )
    def _snapshot_stub(**kwargs):
        snapshot_path.write_text("snapshot", encoding="utf-8")
        return snapshot_path

    monkeypatch.setattr(t8_script.common, "create_fresh_snapshot", _snapshot_stub)
    monkeypatch.setattr(
        t8_script.common,
        "run_quick_certification",
        lambda **kwargs: {
            "verdict": "FAILED",
            "failure_reasons": ["gate_06_failed"],
            "warnings": [],
        },
    )

    called = {"preflight": False, "training": False}

    def _preflight_stub(**kwargs):
        called["preflight"] = True
        return {"qualifies_for_training": True}

    def _training_stub(**kwargs):
        called["training"] = True
        return {"model_status": "trained"}

    monkeypatch.setattr(t8_script.common, "run_preflight_check", _preflight_stub)
    monkeypatch.setattr(t8_script.common, "run_full_training", _training_stub)

    payload = t8_script.run_t8_cycle(
        run_dir=run_dir,
        state_path=state_path,
        base_url="http://127.0.0.1:8000",
        restart_ts=t8_script.common.utc_now_ts() - (9 * 3600.0),
        fresh_hours=8.0,
        max_certification_duration_sec=1800,
    )

    assert "quick_certification_failed" in payload["failures"]
    assert called["preflight"] is False
    assert called["training"] is False
    assert (run_dir / "quick_certification.json").is_file()
    assert (run_dir / "summary.md").is_file()
