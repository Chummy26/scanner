import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from run_perf_investigation import (
    _summarize_server_log,
    browser_targets_for_scenario,
    build_browser_storage_state,
    resolve_tracker_db_path,
    summarize_probe_samples,
    summarize_server_perf,
)


def test_summarize_probe_samples_tracks_staleness_across_timeouts():
    samples = [
        {"timestamp": 1.0, "endpoint": "scanner_lite_summary", "ok": True, "latency_ms": 10.0, "payload_bytes": 120, "updated_at": "t1"},
        {"timestamp": 2.0, "endpoint": "scanner_lite_summary", "ok": False, "latency_ms": 5000.0, "payload_bytes": 0, "updated_at": ""},
        {"timestamp": 3.0, "endpoint": "scanner_lite_summary", "ok": True, "latency_ms": 12.0, "payload_bytes": 120, "updated_at": "t1"},
        {"timestamp": 4.0, "endpoint": "scanner_lite_summary", "ok": True, "latency_ms": 13.0, "payload_bytes": 120, "updated_at": "t1"},
        {"timestamp": 5.0, "endpoint": "scanner_lite_summary", "ok": True, "latency_ms": 14.0, "payload_bytes": 120, "updated_at": "t2"},
    ]

    summary = summarize_probe_samples(samples)

    assert summary["endpoints"]["scanner_lite_summary"]["error_count"] == 1
    assert summary["staleness"]["scanner_lite_summary"] == 2


def test_summarize_server_perf_extracts_hot_windows_from_debug_payloads():
    samples = [
        {
            "timestamp": 10.0,
            "payload": {
                "perf": {
                    "event_loop_lag_ms": {"p95": 300.0, "p99": 400.0},
                    "scanner_cycle": {"total_ms": {"p95": 1200.0}},
                    "routes": {"ml_dashboard_list": {"source_counts": {"cached_lite": 5}}},
                },
                "runtime": {"ml_cache_size": 8},
            },
        }
    ]

    summary = summarize_server_perf(samples)

    assert summary["runtime"]["ml_cache_size"] == 8
    assert summary["perf"]["routes"]["ml_dashboard_list"]["source_counts"]["cached_lite"] == 5
    assert len(summary["hot_windows"]) == 1


def test_browser_targets_use_real_scanner_route():
    targets = browser_targets_for_scenario("http://127.0.0.1:8000", "both_ui")

    assert ("scanner", "http://127.0.0.1:8000/dashboards/scanner") in targets
    assert ("dashboard", "http://127.0.0.1:8000/dashboard") in targets


def test_browser_storage_state_seeds_auth_token_for_origin():
    state = build_browser_storage_state("http://127.0.0.1:8000", "test-token")

    assert state["origins"][0]["origin"] == "http://127.0.0.1:8000"
    assert state["origins"][0]["localStorage"] == [{"name": "authToken", "value": "test-token"}]


def test_resolve_tracker_db_path_isolated_per_benchmark_output(tmp_path: Path):
    db_path = resolve_tracker_db_path(tmp_path / "scenario_a")

    assert db_path == (tmp_path / "scenario_a" / "tracker_history.sqlite")


def test_summarize_server_log_extracts_reconnect_counts(tmp_path: Path):
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "\n".join(
            [
                "2026 [INFO] spread.exchanges.base: [KUCOIN] spot#0 reconnecting (#1, 95 syms, lived 0s, wait 1.0s)",
                "2026 [INFO] spread.exchanges.base: [MEXC] futures#0 reconnecting (#2, 28 syms, lived 0s, wait 2.0s)",
                "2026 [INFO] spread.exchanges.base: [KUCOIN] futures#0 reconnecting (#3, 95 syms, lived 0s, wait 4.0s)",
            ]
        ),
        encoding="utf-8",
    )

    summary = _summarize_server_log(log_path)

    assert summary["reconnect_total"] == 3
    assert summary["reconnect_counts"]["KUCOIN"] == 2
    assert summary["reconnect_counts"]["MEXC"] == 1
