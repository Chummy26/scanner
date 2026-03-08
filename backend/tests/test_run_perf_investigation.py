import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from run_perf_investigation import summarize_probe_samples, summarize_server_perf


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
