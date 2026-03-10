import pytest

from src.spread.models import OrderBookSnapshot, SpreadConfig, SpreadOpportunity
from src.spread.spread_engine import SpreadEngine


def _snapshot(*, exchange: str, market_type: str, bid: float, ask: float, timestamp: float, connected: bool):
    return OrderBookSnapshot(
        exchange=exchange,
        symbol="BTC",
        market_type=market_type,
        bids=[(bid, 1.0)],
        asks=[(ask, 1.0)],
        timestamp=timestamp,
        connected=connected,
    )


def test_spread_engine_filters_connected_old_books(monkeypatch):
    monkeypatch.setattr("src.spread.spread_engine.time.time", lambda: 1_000.0)
    engine = SpreadEngine(SpreadConfig(min_spread_pct=0.1))
    engine._snapshots = {
        "BTC": {
            "mexc": {"spot": _snapshot(exchange="mexc", market_type="spot", bid=99.0, ask=100.0, timestamp=500.0, connected=True)},
            "gate": {"futures": _snapshot(exchange="gate", market_type="futures", bid=101.0, ask=102.0, timestamp=500.0, connected=True)},
        }
    }
    engine._dirty_symbols = {"BTC"}

    opps = engine.calculate_all()

    assert opps == []


def test_spread_engine_filters_disconnected_old_books(monkeypatch):
    monkeypatch.setattr("src.spread.spread_engine.time.time", lambda: 1_000.0)
    engine = SpreadEngine(SpreadConfig(min_spread_pct=0.1))
    engine._snapshots = {
        "BTC": {
            "mexc": {"spot": _snapshot(exchange="mexc", market_type="spot", bid=99.0, ask=100.0, timestamp=990.0, connected=True)},
            "gate": {"futures": _snapshot(exchange="gate", market_type="futures", bid=101.0, ask=102.0, timestamp=990.0, connected=False)},
        }
    }
    engine._dirty_symbols = {"BTC"}

    opps = engine.calculate_all()

    assert opps == []


def test_spread_engine_incremental_aging_prunes_cached_opportunities_once_books_turn_stale(monkeypatch):
    now_ref = {"value": 100.0}
    monkeypatch.setattr("src.spread.spread_engine.time.time", lambda: now_ref["value"])
    engine = SpreadEngine(SpreadConfig(min_spread_pct=0.1))
    engine._snapshots = {
        "BTC": {
            "mexc": {"spot": _snapshot(exchange="mexc", market_type="spot", bid=99.0, ask=100.0, timestamp=100.0, connected=True)},
            "gate": {"futures": _snapshot(exchange="gate", market_type="futures", bid=101.0, ask=102.0, timestamp=100.0, connected=True)},
        }
    }
    engine._dirty_symbols = {"BTC"}

    first = engine.calculate_all()
    assert len(first) == 1

    now_ref["value"] = 130.0
    second = engine.calculate_all()

    assert second == []


def test_spread_engine_collects_raw_records_separately_from_filtered_opportunities(monkeypatch):
    monkeypatch.setattr("src.spread.spread_engine.time.time", lambda: 1_000.0)
    engine = SpreadEngine(SpreadConfig(min_spread_pct=5.0))
    engine._snapshots = {
        "BTC": {
            "mexc": {"spot": _snapshot(exchange="mexc", market_type="spot", bid=99.0, ask=100.0, timestamp=1_000.0, connected=True)},
            "gate": {"futures": _snapshot(exchange="gate", market_type="futures", bid=101.0, ask=102.0, timestamp=1_000.0, connected=True)},
        }
    }
    engine._dirty_symbols = {"BTC"}
    raw_records = []

    opps = engine.calculate_all(record_sink=raw_records)

    assert opps == []
    assert len(raw_records) == 1
    assert raw_records[0][:5] == ("BTC", "mexc", "spot", "gate", "futures")
    assert raw_records[0][5] == pytest.approx(1.0)
    assert raw_records[0][6] == pytest.approx(-2.941176470588236)


def test_spread_engine_rejects_tracker_records_when_exit_spread_is_invalid(monkeypatch):
    monkeypatch.setattr("src.spread.spread_engine.time.time", lambda: 1_000.0)
    engine = SpreadEngine(SpreadConfig(min_spread_pct=0.1))
    engine._snapshots = {
        "BTC": {
            "mexc": {"spot": _snapshot(exchange="mexc", market_type="spot", bid=99.0, ask=100.0, timestamp=1_000.0, connected=True)},
            "gate": {"futures": _snapshot(exchange="gate", market_type="futures", bid=101.0, ask=float("nan"), timestamp=1_000.0, connected=True)},
        }
    }
    engine._dirty_symbols = {"BTC"}
    raw_records = []

    opps = engine.calculate_all(record_sink=raw_records)

    assert opps == []
    assert raw_records == []


def test_spread_engine_rejects_uncomputable_exit_spread_from_tracker_capture(monkeypatch):
    monkeypatch.setattr("src.spread.spread_engine.time.time", lambda: 1_000.0)
    engine = SpreadEngine(SpreadConfig(min_spread_pct=0.1))
    engine._snapshots = {
        "BTC": {
            "mexc": {"spot": _snapshot(exchange="mexc", market_type="spot", bid=99.0, ask=100.0, timestamp=1_000.0, connected=True)},
            "gate": {"futures": _snapshot(exchange="gate", market_type="futures", bid=101.0, ask=0.1, timestamp=1_000.0, connected=True)},
        }
    }
    engine._dirty_symbols = {"BTC"}
    raw_records = []
    rejections = []

    opps = engine.calculate_all(record_sink=raw_records, rejection_sink=rejections)

    assert opps == []
    assert raw_records == []
    assert len(rejections) == 1
    assert rejections[0]["invalid_fields"] == ["exit"]
    assert rejections[0]["reason"] == "exit_spread_invalid"


def test_spread_engine_discards_records_when_exit_spread_exceeds_max_bound(monkeypatch):
    monkeypatch.setattr("src.spread.spread_engine.time.time", lambda: 1_000.0)
    engine = SpreadEngine(SpreadConfig(min_spread_pct=0.1, max_spread_pct=50.0))
    engine._snapshots = {
        "BTC": {
            "mexc": {"spot": _snapshot(exchange="mexc", market_type="spot", bid=500.0, ask=100.0, timestamp=1_000.0, connected=True)},
            "gate": {"futures": _snapshot(exchange="gate", market_type="futures", bid=101.0, ask=102.0, timestamp=1_000.0, connected=True)},
        }
    }
    engine._dirty_symbols = {"BTC"}
    raw_records = []

    opps = engine.calculate_all(record_sink=raw_records)

    assert opps == []
    assert raw_records == []
