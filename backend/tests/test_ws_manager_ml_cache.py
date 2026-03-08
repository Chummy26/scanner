from pathlib import Path

from src.spread.models import SpreadConfig, SpreadOpportunity
from src.spread.ws_manager import WSManager


class _FakeTracker:
    def __init__(self):
        self.history_calls = 0
        self.marker = {
            "inverted_counts": {"4h": 1},
            "inverted_count": 1,
            "total_entries": 2,
            "total_exits": 1,
            "last_crossover_ts": 123.0,
            "history_points": 24,
            "last_record_ts": 300.0,
            "current_block_id": 7,
            "history_enabled": True,
        }

    @staticmethod
    def _pair_key(symbol, buy_ex, buy_mt, sell_ex, sell_mt):
        return (symbol.upper(), buy_ex.lower(), buy_mt.lower(), sell_ex.lower(), sell_mt.lower())

    @staticmethod
    def _pair_id(key):
        return "|".join(key)

    def batch_enrich(self, keys, now_ts):
        return {key: dict(self.marker) for key in keys}

    def get_history(self, *args, **kwargs):
        self.history_calls += 1
        return [
            {"timestamp": float(100 + (index * 15)), "entry_spread": 0.4 + (index * 0.01), "exit_spread": -0.1}
            for index in range(20)
        ]


class _FakeAnalyzer:
    def __init__(self):
        self.predict_calls = 0
        self.render_calls = 0

    def predict_history(self, history):
        self.predict_calls += 1
        return {
            "prediction_status": "ready",
            "model_status": "ready",
            "signal_reason": "trained artifact ready",
            "history_points": len(history),
            "history_last_ts": float(history[-1]["timestamp"]),
            "canonical_history": list(history),
            "inversion_probability": 0.77,
            "model_eta_seconds": 1200,
            "inference_latency_ms": 3.5,
            "drift_status": "stable",
            "drifted_features": [],
            "artifact_feature_count": 10,
            "artifact_trained_at_utc": "2026-03-08T00:00:00+00:00",
            "artifact_dataset_samples": 100,
        }

    def render_prediction(self, current_entry, prediction, *, pair_key=None):
        self.render_calls += 1
        return {
            "ml_score": 77,
            "signal_action": "EXECUTE",
            "signal_reason_code": "execute_ready",
            "model_status": "ready",
            "model_version": "test-model",
            "success_probability": 77.0,
            "drift_status": "stable",
            "inference_latency_ms": 3.5,
            "context_strength": "normal",
            "range_status": "ready_short",
            "entry_position_label": "inside_core",
            "eta_alignment_status": "aligned",
            "display_eta_seconds": 1200,
            "recommended_entry_range": "0.50% à 0.80%",
            "recommended_exit_range": "0.10% à 0.20%",
        }


def _make_opportunity() -> SpreadOpportunity:
    return SpreadOpportunity(
        asset="BTC",
        arb_type="spot_futures",
        buy_exchange="mexc",
        sell_exchange="gate",
        buy_market_type="spot",
        sell_market_type="futures",
        buy_price=100.0,
        sell_price=101.0,
        entry_spread_pct=1.0,
        exit_spread_pct=-0.4,
    )


def test_ws_manager_reuses_prediction_when_tracker_marker_is_unchanged(tmp_path: Path):
    config = SpreadConfig(exchanges=[], symbols=[], tracker_db_path=str(tmp_path / "tracker.sqlite"))
    ws_mgr = WSManager(config)
    ws_mgr.tracker = _FakeTracker()
    ws_mgr.ml_analyzer = _FakeAnalyzer()
    ws_mgr.engine.calculate_all = lambda on_spread=None: [_make_opportunity()]

    ws_mgr._enrich_tracker_cycle = 15
    ws_mgr._do_calc_enrich(None)
    assert ws_mgr.tracker.history_calls == 1
    assert ws_mgr.ml_analyzer.predict_calls == 1
    assert ws_mgr.ml_analyzer.render_calls == 1

    ws_mgr._enrich_tracker_cycle = 15
    ws_mgr._do_calc_enrich(None)
    assert ws_mgr.tracker.history_calls == 1
    assert ws_mgr.ml_analyzer.predict_calls == 1
    assert ws_mgr.ml_analyzer.render_calls == 2


def test_ws_manager_refreshes_prediction_when_tracker_marker_changes(tmp_path: Path):
    config = SpreadConfig(exchanges=[], symbols=[], tracker_db_path=str(tmp_path / "tracker.sqlite"))
    ws_mgr = WSManager(config)
    fake_tracker = _FakeTracker()
    ws_mgr.tracker = fake_tracker
    ws_mgr.ml_analyzer = _FakeAnalyzer()
    ws_mgr.engine.calculate_all = lambda on_spread=None: [_make_opportunity()]

    ws_mgr._enrich_tracker_cycle = 15
    ws_mgr._do_calc_enrich(None)
    assert ws_mgr.ml_analyzer.predict_calls == 1

    fake_tracker.marker["last_record_ts"] = 315.0
    fake_tracker.marker["history_points"] = 25
    ws_mgr._enrich_tracker_cycle = 15
    ws_mgr._do_calc_enrich(None)

    assert ws_mgr.tracker.history_calls == 2
    assert ws_mgr.ml_analyzer.predict_calls == 2
