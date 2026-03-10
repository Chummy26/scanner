from src.spread.perf_monitor import RuntimePerfMonitor


def test_runtime_perf_monitor_snapshot_aggregates_routes_cycles_broadcasts_and_lag():
    monitor = RuntimePerfMonitor(max_samples=16, max_events=16)

    monitor.record_route(
        "scanner_lite_list",
        latency_ms=25.0,
        status_code=200,
        payload_bytes=2048,
        rows=42,
        source="cached_lite",
        path="/api/spread/opportunities-lite",
    )
    monitor.record_route(
        "scanner_lite_list",
        latency_ms=1205.0,
        status_code=503,
        payload_bytes=0,
        rows=0,
        source="cached_lite",
        path="/api/spread/opportunities-lite",
    )
    monitor.record_scanner_cycle(
        {
            "total_ms": 85.0,
            "calculate_ms": 30.0,
            "batch_record_ms": 5.0,
            "market_enrich_ms": 10.0,
            "tracker_enrich_ms": 20.0,
            "history_fetch_ms": 3.0,
            "ml_analyze_ms": 7.0,
            "filter_ms": 4.0,
            "lite_refresh_ms": 6.0,
            "opportunities_before_filter": 100,
            "opportunities_after_filter": 80,
        }
    )
    monitor.record_scanner_cycle(
        {
            "kind": "ml_inference",
            "total_ms": 22.0,
            "history_fetch_ms": 8.0,
            "ml_analyze_ms": 6.0,
            "ml_render_ms": 4.0,
            "ml_predictions_drained": 3,
            "ml_refresh_queue_size": 9,
            "ml_cache_size": 12,
        }
    )
    monitor.record_broadcast(
        "scanner_lite",
        {
            "total_ms": 12.0,
            "payload_bytes": 8192,
            "client_count": 2,
        },
    )
    monitor.record_event_loop_lag(275.0)
    monitor.record_cache_state({"ml_cache_size": 9, "scanner_lite_rows": 80})

    snapshot = monitor.snapshot()

    assert snapshot["routes"]["scanner_lite_list"]["latency_ms"]["count"] == 2
    assert snapshot["routes"]["scanner_lite_list"]["error_count"] == 1
    assert snapshot["routes"]["scanner_lite_list"]["source_counts"]["cached_lite"] == 2
    assert snapshot["scanner_cycle"]["count"] == 1
    assert snapshot["scanner_cycle_by_kind"]["ws_manager"]["count"] == 1
    assert snapshot["scanner_cycle_by_kind"]["ml_inference"]["count"] == 1
    assert snapshot["scanner_cycle"]["total_ms"]["max"] == 85.0
    assert snapshot["scanner_cycle_by_kind"]["ml_inference"]["ml_render_ms"]["max"] == 4.0
    assert snapshot["broadcasts"]["scanner_lite"]["payload_bytes"]["max"] == 8192.0
    assert snapshot["event_loop_lag_ms"]["max"] == 275.0
    assert snapshot["cache_state"]["scanner_lite_rows"] == 80
    assert any(item["kind"] == "route" for item in snapshot["slow_events"])
    assert any(item["kind"] == "event_loop_lag" for item in snapshot["slow_events"])
