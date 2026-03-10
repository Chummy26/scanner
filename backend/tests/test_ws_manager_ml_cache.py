import asyncio
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
        self.model_status = "ready"
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


def test_ws_manager_uses_smoothed_tracker_cadence(tmp_path: Path):
    config = SpreadConfig(
        exchanges=[],
        symbols=[],
        tracker_db_path=str(tmp_path / "tracker.sqlite"),
        broadcast_interval_sec=0.15,
        tracker_record_interval_sec=15.0,
    )
    ws_mgr = WSManager(config)

    assert ws_mgr._tracker_cycle_every() == 5


def test_ws_manager_hot_path_enqueues_and_reuses_prediction_when_tracker_marker_is_unchanged(tmp_path: Path):
    config = SpreadConfig(exchanges=[], symbols=[], tracker_db_path=str(tmp_path / "tracker.sqlite"))
    ws_mgr = WSManager(config)
    ws_mgr.tracker = _FakeTracker()
    ws_mgr.ml_analyzer = _FakeAnalyzer()
    ws_mgr.engine.calculate_all = lambda on_spread=None, record_sink=None: [_make_opportunity()]

    ws_mgr._enrich_tracker_cycle = 15
    ws_mgr._do_calc_enrich(None)
    assert ws_mgr.tracker.history_calls == 0
    assert ws_mgr._ml_queue_size() == 1

    ws_mgr._drain_ml_inference_batch(now_ts=200.0)
    assert ws_mgr.tracker.history_calls == 1
    assert ws_mgr.ml_analyzer.predict_calls == 1
    assert ws_mgr.ml_analyzer.render_calls == 1

    ws_mgr._enrich_tracker_cycle = 15
    ws_mgr._do_calc_enrich(None)
    assert ws_mgr.tracker.history_calls == 1
    assert ws_mgr.ml_analyzer.predict_calls == 1
    assert ws_mgr.ml_analyzer.render_calls == 1


def test_ws_manager_hot_path_enqueues_refresh_when_tracker_marker_changes(tmp_path: Path):
    config = SpreadConfig(exchanges=[], symbols=[], tracker_db_path=str(tmp_path / "tracker.sqlite"))
    ws_mgr = WSManager(config)
    fake_tracker = _FakeTracker()
    ws_mgr.tracker = fake_tracker
    ws_mgr.ml_analyzer = _FakeAnalyzer()
    ws_mgr.engine.calculate_all = lambda on_spread=None, record_sink=None: [_make_opportunity()]

    ws_mgr._enrich_tracker_cycle = 15
    ws_mgr._do_calc_enrich(None)
    ws_mgr._drain_ml_inference_batch(now_ts=200.0)
    assert ws_mgr.ml_analyzer.predict_calls == 1

    fake_tracker.marker["last_record_ts"] = 315.0
    fake_tracker.marker["history_points"] = 25
    ws_mgr._enrich_tracker_cycle = 15
    ws_mgr._do_calc_enrich(None)

    assert ws_mgr.tracker.history_calls == 1
    assert ws_mgr._ml_queue_size() == 1

    ws_mgr._drain_ml_inference_batch(now_ts=215.0)
    assert ws_mgr.tracker.history_calls == 2
    assert ws_mgr.ml_analyzer.predict_calls == 2


def test_ws_manager_skips_ml_work_when_model_is_not_ready(tmp_path: Path):
    config = SpreadConfig(exchanges=[], symbols=[], tracker_db_path=str(tmp_path / "tracker.sqlite"))
    ws_mgr = WSManager(config)
    ws_mgr.tracker = _FakeTracker()
    analyzer = _FakeAnalyzer()
    analyzer.model_status = "missing_artifact"
    ws_mgr.ml_analyzer = analyzer
    ws_mgr.engine.calculate_all = lambda on_spread=None, record_sink=None: [_make_opportunity()]

    opportunities = ws_mgr._do_calc_enrich(None)

    assert ws_mgr.tracker.history_calls == 0
    assert analyzer.predict_calls == 0
    assert analyzer.render_calls == 0
    assert ws_mgr._ml_queue_size() == 0
    assert opportunities[0].ml_context is None
    assert opportunities[0].ml_score == 0


def test_ws_manager_background_ml_drain_renders_with_latest_current_entry(tmp_path: Path):
    config = SpreadConfig(exchanges=[], symbols=[], tracker_db_path=str(tmp_path / "tracker.sqlite"))
    ws_mgr = WSManager(config)
    ws_mgr.tracker = _FakeTracker()
    analyzer = _FakeAnalyzer()
    ws_mgr.ml_analyzer = analyzer

    key = ws_mgr.tracker._pair_key("BTC", "mexc", "spot", "gate", "futures")
    ws_mgr._queue_ml_refresh(
        key,
        {
            "symbol": "BTC",
            "buy_exchange": "mexc",
            "buy_market_type": "spot",
            "sell_exchange": "gate",
            "sell_market_type": "futures",
            "current_entry": 1.23,
            "tracker_marker": ws_mgr._tracker_marker_from_cache(ws_mgr.tracker.marker),
            "queued_at_perf": 100.0,
        },
    )

    ws_mgr._drain_ml_inference_batch(now_ts=200.0)

    cache_entry = ws_mgr._get_ml_cache_entry(key)
    assert cache_entry is not None
    assert cache_entry["rendered_entry_spread"] == 1.23
    assert cache_entry["context"]["ml_score"] == 77


def test_ws_manager_persist_loop_offloads_flush_to_thread(monkeypatch, tmp_path: Path):
    config = SpreadConfig(exchanges=[], symbols=[], tracker_db_path=str(tmp_path / "tracker.sqlite"))
    ws_mgr = WSManager(config)
    flush_calls: list[str] = []
    to_thread_calls: list[str] = []

    class _ThreadTracker:
        def flush_to_storage(self):
            flush_calls.append("flush")
            ws_mgr._running = False
            return True

    async def _fake_sleep(_seconds):
        return None

    async def _fake_to_thread(func, *args, **kwargs):
        to_thread_calls.append(getattr(func, "__name__", "unknown"))
        return func(*args, **kwargs)

    monkeypatch.setattr("src.spread.ws_manager.asyncio.sleep", _fake_sleep)
    monkeypatch.setattr("src.spread.ws_manager.asyncio.to_thread", _fake_to_thread)

    ws_mgr.tracker = _ThreadTracker()
    ws_mgr._running = True
    asyncio.run(ws_mgr._persist_loop())

    assert flush_calls == ["flush"]
    assert to_thread_calls == ["flush_to_storage"]


def test_ws_manager_perf_state_uses_tracker_storage_pair_counts(tmp_path: Path):
    config = SpreadConfig(exchanges=[], symbols=[], tracker_db_path=str(tmp_path / "tracker.sqlite"))
    ws_mgr = WSManager(config)

    class _TrackerWithStorageStats:
        @staticmethod
        def get_storage_stats():
            return {
                "pairs_in_memory": 17,
                "pairs_persisted": 9,
                "records_total": 123,
                "episodes_total": 7,
                "db_size_bytes": 4567,
            }

    ws_mgr.tracker = _TrackerWithStorageStats()
    state = ws_mgr.get_perf_state()

    assert state["tracker_pairs"] == 17
    assert state["tracker_records_total"] == 123
    assert state["tracker_episodes_total"] == 7
    assert state["tracker_db_size_bytes"] == 4567


def test_ws_manager_prunes_stale_ml_runtime_state_when_active_keys_shrink(tmp_path: Path):
    config = SpreadConfig(exchanges=[], symbols=[], tracker_db_path=str(tmp_path / "tracker.sqlite"))
    ws_mgr = WSManager(config)
    stale_key = "BTC|mexc|spot|gate|futures"
    active_key = "ETH|mexc|spot|gate|futures"
    old_ts = 100.0

    ws_mgr._ml_pending_refreshes[stale_key] = old_ts
    ws_mgr._ml_pending_refreshes[active_key] = old_ts
    ws_mgr._ml_cache[stale_key] = {
        "last_seen_at": old_ts - (ws_mgr._ML_PREDICTION_MAX_AGE_SEC * 5.0),
        "marker": {"history_points": 20},
        "prediction": {"prediction_status": "ready"},
    }
    ws_mgr._ml_cache[active_key] = {
        "last_seen_at": old_ts,
        "marker": {"history_points": 20},
        "prediction": {"prediction_status": "ready"},
    }

    ws_mgr._prune_ml_runtime_state({active_key}, now_ts=old_ts + 1.0)

    assert stale_key not in ws_mgr._ml_pending_refreshes
    assert active_key in ws_mgr._ml_pending_refreshes
    assert stale_key not in ws_mgr._ml_cache
    assert ws_mgr._ml_cache[active_key]["last_seen_at"] == old_ts + 1.0


def test_ws_manager_continuous_capture_uses_raw_records_not_filtered_opportunities(tmp_path: Path):
    config = SpreadConfig(exchanges=[], symbols=[], tracker_db_path=str(tmp_path / "tracker.sqlite"))
    ws_mgr = WSManager(config)
    captured_batches = []

    class _CaptureTracker(_FakeTracker):
        def batch_record(self, records, *, now_ts=None):
            captured_batches.append((list(records), now_ts))

    ws_mgr.tracker = _CaptureTracker()
    ws_mgr.ml_analyzer = _FakeAnalyzer()

    def _fake_calculate_all(on_spread=None, record_sink=None):
        if record_sink is not None:
            record_sink.append(("BTC", "mexc", "spot", "gate", "futures", 0.4, -0.2))
        return []

    ws_mgr.engine.calculate_all = _fake_calculate_all

    ws_mgr._do_calc_enrich(object())

    assert len(captured_batches) == 1
    assert captured_batches[0][0] == [("BTC", "mexc", "spot", "gate", "futures", 0.4, -0.2)]


def test_ws_manager_scanner_lite_refresh_requires_interest_or_clients(tmp_path: Path):
    config = SpreadConfig(exchanges=[], symbols=[], tracker_db_path=str(tmp_path / "tracker.sqlite"))
    ws_mgr = WSManager(config)
    ws_mgr.tracker = _FakeTracker()
    ws_mgr.ml_analyzer = _FakeAnalyzer()
    ws_mgr.engine.calculate_all = lambda on_spread=None, record_sink=None: [_make_opportunity()]

    ws_mgr._do_calc_enrich(None)
    assert ws_mgr.get_perf_state()["scanner_lite_rows"] == 0

    ws_mgr.mark_scanner_lite_interest()
    ws_mgr._do_calc_enrich(None)

    rows = ws_mgr.get_current_opportunities_lite()
    assert len(rows) == 1
    assert rows[0]["pairKey"] == "BTC|mexc|spot|gate|futures"
