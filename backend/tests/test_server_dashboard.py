import asyncio
import json
from types import SimpleNamespace

from src.server import handle_ml_dashboard_api


class _FakeRequest:
    def __init__(self, app: dict):
        self.app = app


class _FakeWSManager:
    def __init__(self, opportunities):
        self._opportunities = opportunities

    def get_current_opportunities(self):
        return list(self._opportunities)


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
