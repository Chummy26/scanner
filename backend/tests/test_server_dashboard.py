import asyncio
import json
from types import SimpleNamespace

from src.server import (
    handle_debug_perf,
    handle_ml_dashboard_api,
    handle_ml_dashboard_detail_api,
    handle_ml_dashboard_list_api,
    handle_ml_dashboard_summary_api,
    handle_ml_pipeline_status,
    handle_system_health_api,
)
from src.spread.perf_monitor import RuntimePerfMonitor


class _FakeRequest:
    def __init__(self, app: dict, *, query=None, match_info=None, path="/test"):
        self.app = app
        self.query = query or {}
        self.match_info = match_info or {}
        self.path = path


class _FakeWSManager:
    def __init__(self, opportunities, *, lite_rows=None):
        self._opportunities = opportunities
        self._lite_rows = lite_rows if lite_rows is not None else opportunities

    def get_current_opportunities(self):
        return list(self._opportunities)

    def get_current_opportunities_lite(self):
        return list(self._lite_rows)

    def get_scanner_opportunity_detail(self, pair_key):
        for row in self._opportunities:
            symbol = row.get("symbol")
            buy_from = row.get("buyFrom")
            buy_type = row.get("buyType")
            sell_to = row.get("sellTo")
            sell_type = row.get("sellType")
            current_key = f"{symbol}|{buy_from}|{buy_type}|{sell_to}|{sell_type}"
            if current_key == pair_key:
                return dict(row)
        return None

    def get_perf_state(self):
        return {"ml_cache_size": 3, "scanner_lite_rows": len(self._lite_rows)}

    def get_system_health(self):
        return {"hour_verdict": "healthy", "rejection_rate_pct": 1.5, "circuit_breakers": {}, "next_snapshot_sec": 600, "last_snapshot": "2026-03-10_08h", "last_snapshot_verdict": "PASS"}

    def get_pipeline_status(self):
        return {
            "snapshots": {"last": "2026-03-10_08h", "last_verdict": "PASS", "next_in_sec": 600, "total_7d": 3, "pass_7d": 3, "fail_7d": 0, "recent": []},
            "training": {"last_run": "run_1", "last_auc": 0.71, "model_version": "bundle-v2", "deployed_at": "2026-03-10T08:00:00Z", "retrain_trigger": None, "auto_state": {}, "run_count": 1},
            "hourly_health": {"last_24h": ["healthy"], "healthy_count": 1, "degraded_count": 0, "unhealthy_count": 0},
        }


def test_ml_dashboard_api_sorts_ready_rows_by_probability_and_returns_summary():
    opportunities = [
        {
            "symbol": "AAA",
            "mlScore": 91,
            "mlContext": {
                "signal_action": "WAIT",
                "model_status": "insufficient_history",
                "success_probability": 88.0,
                "ml_score": 91,
            },
        },
        {
            "symbol": "BBB",
            "mlScore": 20,
            "mlContext": {
                "signal_action": "EXECUTE",
                "model_status": "ready",
                "model_version": "bundle-v1",
                "success_probability": 71.0,
                "ml_score": 20,
                "drift_status": "stable",
                "inference_latency_ms": 12.5,
                "range_status": "ready_short",
                "entry_position_label": "inside_outer",
                "eta_alignment_status": "aligned",
            },
        },
        {
            "symbol": "CCC",
            "mlScore": 55,
            "mlContext": {
                "signal_action": "STRONG_EXECUTE",
                "model_status": "ready",
                "model_version": "bundle-v1",
                "success_probability": 71.0,
                "ml_score": 55,
                "drift_status": "drifted",
                "inference_latency_ms": 18.0,
                "range_status": "ready_short",
                "entry_position_label": "inside_core",
                "eta_alignment_status": "aligned",
            },
        },
        {
            "symbol": "DDD",
            "mlScore": 40,
            "mlContext": {
                "signal_action": "WAIT",
                "model_status": "ready",
                "model_version": "bundle-v2",
                "success_probability": 60.0,
                "ml_score": 40,
                "drift_status": "stable",
                "inference_latency_ms": 9.0,
                "range_status": "ready_short",
                "entry_position_label": "inside_outer",
                "eta_alignment_status": "aligned",
            },
        },
        {
            "symbol": "GGG",
            "mlScore": 84,
            "mlContext": {
                "signal_action": "WAIT",
                "model_status": "ready",
                "model_version": "bundle-v1",
                "success_probability": 88.0,
                "ml_score": 84,
                "drift_status": "stable",
                "inference_latency_ms": 14.0,
                "range_status": "insufficient_empirical_context",
                "entry_position_label": "unknown",
                "eta_alignment_status": "unknown",
            },
        },
        {
            "symbol": "HHH",
            "mlScore": 86,
            "mlContext": {
                "signal_action": "WAIT",
                "model_status": "ready",
                "model_version": "bundle-v1",
                "success_probability": 90.0,
                "ml_score": 86,
                "drift_status": "stable",
                "inference_latency_ms": 11.0,
                "range_status": "ready_short",
                "entry_position_label": "above_band",
                "eta_alignment_status": "aligned",
            },
        },
        {
            "symbol": "III",
            "mlScore": 74,
            "mlContext": {
                "signal_action": "EXECUTE",
                "model_status": "ready",
                "model_version": "bundle-v1",
                "success_probability": 78.0,
                "ml_score": 74,
                "drift_status": "stable",
                "inference_latency_ms": 15.0,
                "range_status": "ready_short",
                "entry_position_label": "inside_core",
                "eta_alignment_status": "divergent",
            },
        },
        {
            "symbol": "FFF",
            "mlScore": 10,
            "mlContext": {
                "signal_action": "WAIT",
                "model_status": "stale_artifact",
                "model_version": "bundle-v0",
                "success_probability": 0.0,
                "ml_score": 10,
                "drift_status": "unknown",
                "inference_latency_ms": 0.0,
            },
        },
        {
            "symbol": "EEE",
            "mlScore": 5,
        },
    ]
    request = _FakeRequest({"ws_manager": _FakeWSManager(opportunities)})

    response = asyncio.run(handle_ml_dashboard_api(request))
    payload = json.loads(response.text)

    assert [item["symbol"] for item in payload["data"]] == ["III", "BBB", "CCC", "DDD", "HHH", "GGG", "AAA", "FFF", "EEE"]
    assert payload["data"][0]["action_lane"] == "execute_now"
    assert payload["data"][0]["signal_reason_code"] == "eta_divergent"
    assert "eta_divergent" in payload["data"][0]["risk_flags"]
    assert payload["data"][1]["signal_reason_code"] == "execute_ready"
    assert payload["data"][2]["action_lane"] == "observe"
    assert payload["data"][2]["signal_reason_code"] == "drift_active"
    assert "drift" in payload["data"][2]["risk_flags"]
    assert payload["data"][3]["signal_reason_code"] == "probability_below_threshold"
    assert payload["data"][4]["action_lane"] == "blocked"
    assert payload["data"][4]["signal_reason_code"] == "entry_above_recurring_band"
    assert payload["data"][5]["signal_reason_code"] == "insufficient_empirical_context"
    assert payload["data"][6]["signal_reason_code"] == "insufficient_history"
    assert payload["data"][7]["signal_reason_code"] == "stale_artifact"
    assert payload["data"][8]["signal_reason_code"] == "missing_ml_context"
    assert isinstance(payload["data"][0]["operator_message"], str)
    assert payload["summary"]["total"] == 9
    assert payload["summary"]["ready"] == 6
    assert payload["summary"]["degraded"] == 3
    assert payload["summary"]["signals"]["STRONG_EXECUTE"] == 1
    assert payload["summary"]["signals"]["EXECUTE"] == 2
    assert payload["summary"]["signals"]["WAIT"] == 5
    assert payload["summary"]["signals"]["NONE"] == 1
    assert payload["summary"]["model_status"]["ready"] == 6
    assert payload["summary"]["model_status"]["insufficient_history"] == 1
    assert payload["summary"]["model_status"]["stale_artifact"] == 1
    assert payload["summary"]["model_status"]["missing_ml_context"] == 1
    assert payload["summary"]["drifted_ready"] == 1
    assert payload["summary"]["model_versions"]["bundle-v1"] == 5
    assert payload["summary"]["model_versions"]["bundle-v2"] == 1
    assert payload["summary"]["avg_inference_latency_ms_ready"] == 13.25
    assert payload["summary"]["p99_inference_latency_ms_ready"] == 17.85
    assert payload["summary"]["lane_counts"] == {"execute_now": 2, "observe": 2, "blocked": 5}
    assert payload["summary"]["reason_counts"]["execute_ready"] == 1
    assert payload["summary"]["reason_counts"]["drift_active"] == 1
    assert payload["summary"]["reason_counts"]["probability_below_threshold"] == 1
    assert payload["summary"]["reason_counts"]["insufficient_history"] == 1
    assert payload["summary"]["reason_counts"]["insufficient_empirical_context"] == 1
    assert payload["summary"]["reason_counts"]["entry_above_recurring_band"] == 1
    assert payload["summary"]["reason_counts"]["eta_divergent"] == 1
    assert payload["summary"]["reason_counts"]["stale_artifact"] == 1
    assert payload["summary"]["reason_counts"]["missing_ml_context"] == 1
    assert payload["summary"]["bundle_version"] == "bundle-v1"
    assert payload["summary"]["bundle_status"] == "ready_with_drift"
    assert payload["summary"]["blocked_by_empirical_context"] == 1
    assert payload["summary"]["entry_above_recurring_band_count"] == 1
    assert payload["summary"]["eta_divergent_ready"] == 1


def test_ml_dashboard_summary_endpoint_returns_only_summary_payload():
    opportunities = [
        {
            "symbol": "AAA",
            "mlScore": 20,
            "mlContext": {
                "signal_action": "EXECUTE",
                "model_status": "ready",
                "model_version": "bundle-v1",
                "success_probability": 70.0,
                "ml_score": 20,
                "drift_status": "stable",
                "inference_latency_ms": 10.0,
                "range_status": "ready_short",
                "entry_position_label": "inside_outer",
                "eta_alignment_status": "aligned",
            },
        },
        {
            "symbol": "BBB",
            "mlScore": 10,
            "mlContext": {
                "signal_action": "WAIT",
                "model_status": "insufficient_history",
                "success_probability": 0.0,
                "ml_score": 10,
            },
        },
    ]
    request = _FakeRequest({"ws_manager": _FakeWSManager(opportunities)})

    response = asyncio.run(handle_ml_dashboard_summary_api(request))
    payload = json.loads(response.text)

    assert set(payload.keys()) == {"summary"}
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["lane_counts"]["execute_now"] == 1
    assert payload["summary"]["lane_counts"]["blocked"] == 1


def test_ml_dashboard_list_endpoint_filters_by_lane_search_and_limit():
    opportunities = [
            {
                "symbol": "AAA",
                "buyFrom": "mexc",
                "buyType": "spot",
                "sellTo": "gate",
                "sellType": "futures",
                "entrySpread": 1.5,
                "exitSpread": 0.6,
                "mlScore": 66,
            "mlContext": {
                "signal_action": "EXECUTE",
                "model_status": "ready",
                "model_version": "bundle-v1",
                "success_probability": 70.0,
                "ml_score": 66,
                "drift_status": "stable",
                "inference_latency_ms": 12.0,
                "range_status": "ready_short",
                "entry_position_label": "inside_outer",
                "eta_alignment_status": "aligned",
                "display_eta_seconds": 1800,
                "recommended_entry_range": "1.20% à 1.80%",
                "recommended_exit_range": "0.40% à 0.70%",
            },
        },
            {
                "symbol": "BBB",
                "buyFrom": "kucoin",
                "buyType": "spot",
                "sellTo": "bingx",
                "sellType": "futures",
                "entrySpread": 0.7,
                "exitSpread": 0.2,
                "mlScore": 32,
            "mlContext": {
                "signal_action": "WAIT",
                "model_status": "ready",
                "model_version": "bundle-v1",
                "success_probability": 55.0,
                "ml_score": 32,
                "drift_status": "stable",
                "inference_latency_ms": 9.0,
                "range_status": "insufficient_empirical_context",
                "entry_position_label": "unknown",
                "eta_alignment_status": "unknown",
            },
        },
    ]
    request = _FakeRequest(
        {"ws_manager": _FakeWSManager(opportunities)},
        query={"lane": "execute_now", "search": "AA", "limit": "5", "offset": "0"},
    )

    response = asyncio.run(handle_ml_dashboard_list_api(request))
    payload = json.loads(response.text)

    assert payload["summary"]["total"] == 2
    assert payload["pagination"] == {"limit": 5, "offset": 0, "returned": 1}
    assert len(payload["data"]) == 1
    item = payload["data"][0]
    assert item["symbol"] == "AAA"
    assert item["pair_key"] == "AAA|mexc|spot|gate|futures"
    assert item["action_lane"] == "execute_now"
    assert item["signal_reason_code"] == "execute_ready"
    assert item["recommended_entry_range"] == "1.20% à 1.80%"
    assert "mlContext" not in item


def test_ml_dashboard_detail_endpoint_returns_full_enriched_row():
    opportunities = [
        {
            "symbol": "AAA",
            "buyFrom": "mexc",
            "buyType": "spot",
            "sellTo": "gate",
            "sellType": "futures",
            "mlScore": 66,
            "mlContext": {
                "signal_action": "EXECUTE",
                "model_status": "ready",
                "model_version": "bundle-v1",
                "success_probability": 70.0,
                "ml_score": 66,
                "drift_status": "stable",
                "inference_latency_ms": 12.0,
                "range_status": "ready_short",
                "entry_position_label": "inside_outer",
                "eta_alignment_status": "aligned",
            },
        },
    ]
    request = _FakeRequest(
        {"ws_manager": _FakeWSManager(opportunities)},
        match_info={"pair_key": "AAA|mexc|spot|gate|futures"},
    )

    response = asyncio.run(handle_ml_dashboard_detail_api(request))
    payload = json.loads(response.text)

    assert payload["data"]["symbol"] == "AAA"
    assert payload["data"]["action_lane"] == "execute_now"
    assert payload["data"]["signal_reason_code"] == "execute_ready"


def test_ml_dashboard_detail_endpoint_returns_404_for_unknown_pair():
    request = _FakeRequest(
        {"ws_manager": _FakeWSManager([])},
        match_info={"pair_key": "missing|pair"},
    )

    response = asyncio.run(handle_ml_dashboard_detail_api(request))

    assert response.status == 404


def test_ml_dashboard_summary_prefers_cached_lite_rows_and_records_perf_source():
    perf_monitor = RuntimePerfMonitor()
    lite_rows = [
        {
            "symbol": "AAA",
            "buyFrom": "mexc",
            "buyType": "spot",
            "sellTo": "gate",
            "sellType": "futures",
            "signalAction": "EXECUTE",
            "modelStatus": "ready",
            "modelVersion": "bundle-v1",
            "successProbability": 73.0,
            "mlScore": 61,
            "driftStatus": "stable",
            "inferenceLatencyMs": 11.0,
            "rangeStatus": "ready_short",
            "entryPositionLabel": "inside_outer",
            "etaAlignmentStatus": "aligned",
        }
    ]
    request = _FakeRequest(
        {"ws_manager": _FakeWSManager([], lite_rows=lite_rows), "perf_monitor": perf_monitor},
        path="/api/v1/ml/dashboard/summary",
    )

    response = asyncio.run(handle_ml_dashboard_summary_api(request))
    payload = json.loads(response.text)
    perf_snapshot = perf_monitor.snapshot()

    assert payload["summary"]["total"] == 1
    assert perf_snapshot["routes"]["ml_dashboard_summary"]["source_counts"]["cached_lite"] == 1


def test_debug_perf_endpoint_returns_perf_snapshot_and_runtime_state():
    perf_monitor = RuntimePerfMonitor()
    perf_monitor.record_route(
        "scanner_lite_list",
        latency_ms=12.5,
        status_code=200,
        payload_bytes=1234,
        rows=7,
        source="cached_lite",
        path="/api/spread/opportunities-lite",
    )
    request = _FakeRequest(
        {"perf_monitor": perf_monitor, "ws_manager": _FakeWSManager([])},
        path="/api/debug/perf",
    )

    response = asyncio.run(handle_debug_perf(request))
    payload = json.loads(response.text)

    assert "perf" in payload
    assert "runtime" in payload
    assert payload["runtime"]["scanner_lite_rows"] == 0
    assert payload["perf"]["routes"]["scanner_lite_list"]["source_counts"]["cached_lite"] == 1


def test_ml_dashboard_prefers_runtime_signal_reason_code_for_low_total_spread_blocks():
    opportunities = [
        {
            "symbol": "AAA",
            "buyFrom": "mexc",
            "buyType": "spot",
            "sellTo": "gate",
            "sellType": "futures",
            "mlScore": 81,
            "mlContext": {
                "signal_action": "WAIT",
                "signal_reason_code": "median_total_spread_below_threshold",
                "signal_reason": "Deslocamento total mediano abaixo do mínimo operacional.",
                "model_status": "ready",
                "model_version": "bundle-v1",
                "success_probability": 81.0,
                "ml_score": 81,
                "drift_status": "stable",
                "inference_latency_ms": 10.0,
                "range_status": "insufficient_empirical_context",
                "entry_position_label": "unknown",
                "eta_alignment_status": "unknown",
                "median_total_spread": 0.06,
                "min_total_spread_threshold": 1.0,
            },
        }
    ]
    request = _FakeRequest({"ws_manager": _FakeWSManager(opportunities)})

    response = asyncio.run(handle_ml_dashboard_api(request))
    payload = json.loads(response.text)

    assert payload["data"][0]["signal_reason_code"] == "median_total_spread_below_threshold"
    assert payload["data"][0]["action_lane"] == "blocked"
    assert "mínimo operacional" in payload["data"][0]["operator_message"]


def test_system_health_api_returns_runtime_health_payload():
    request = _FakeRequest({"ws_manager": _FakeWSManager([])})

    response = asyncio.run(handle_system_health_api(request))
    payload = json.loads(response.text)

    assert payload["hour_verdict"] == "healthy"
    assert payload["last_snapshot_verdict"] == "PASS"


def test_ml_pipeline_status_api_returns_pipeline_summary():
    request = _FakeRequest({"ws_manager": _FakeWSManager([])})

    response = asyncio.run(handle_ml_pipeline_status(request))
    payload = json.loads(response.text)

    assert payload["snapshots"]["last"] == "2026-03-10_08h"
    assert payload["training"]["model_version"] == "bundle-v2"
    assert payload["hourly_health"]["healthy_count"] == 1
