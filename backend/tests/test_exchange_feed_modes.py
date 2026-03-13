import asyncio
import logging

import pytest

from src.spread.exchanges.base import BaseExchangeWS
from src.spread.exchanges.bingx import BingxWS
from src.spread.exchanges.bitget import BitgetWS
from src.spread.exchanges.gate import GateWS
from src.spread.exchanges.kucoin import KucoinWS
from src.spread.exchanges.mexc import MexcWS
from src.spread.exchanges.xt import XtWS


def _on_book_update(*_args, **_kwargs):
    return None


class _DummyExchange(BaseExchangeWS):
    name = "dummy"

    def format_symbol(self, base: str, quote: str = "USDT"):
        return {"spot": f"{base}{quote}", "futures": f"{base}{quote}"}

    async def subscribe_symbol(self, base: str, quote: str = "USDT"):
        return None


@pytest.mark.parametrize(
    ("exchange_cls", "futures_fallback_name"),
    [
        (MexcWS, "_run_futures_ticker_fallback"),
        (BingxWS, "_run_futures_ticker_fallback"),
        (KucoinWS, "_run_futures_ticker_fallback"),
        (GateWS, "_run_futures_ticker_fallback"),
        (XtWS, "_run_agg_ticker_fallback"),
        (BitgetWS, "_run_futures_ticker_fallback"),
    ],
)
def test_exchange_ticker_first_mode_skips_depth_ws_and_uses_fallbacks(monkeypatch, exchange_cls, futures_fallback_name):
    exchange = exchange_cls(on_book_update=_on_book_update, depth_limit=5)
    exchange.spot_feed_mode = "ticker_first"
    exchange.futures_feed_mode = "ticker_first"

    calls = []

    async def _fake_run_batch_loop(*args, **kwargs):
        calls.append(("batch", args, kwargs))
        return None

    async def _fake_spot_fallback(symbols):
        calls.append(("spot_fallback", list(symbols)))
        return None

    async def _fake_futures_fallback(symbols):
        calls.append(("futures_fallback", list(symbols)))
        return None

    monkeypatch.setattr(exchange, "_run_batch_loop", _fake_run_batch_loop)
    monkeypatch.setattr(exchange, "_run_spot_ticker_fallback", _fake_spot_fallback)
    monkeypatch.setattr(exchange, futures_fallback_name, _fake_futures_fallback)

    symbols = ["BTC", "ETH"]
    exchange._spot_symbols = list(symbols)
    exchange._futures_symbols = list(symbols)

    asyncio.run(exchange.subscribe_symbol_batch(symbols))

    assert not any(kind == "batch" for kind, *_ in calls)
    assert ("spot_fallback", symbols) in calls
    assert ("futures_fallback", symbols) in calls
    assert exchange.has_book("BTC", "spot")
    assert exchange.has_book("BTC", "futures")


def test_has_book_bootstraps_declared_symbols_for_fallbacks():
    exchange = _DummyExchange(on_book_update=_on_book_update, depth_limit=5)
    exchange._spot_symbols = ["BTC", "ETH"]
    exchange._futures_symbols = ["BTC"]

    assert exchange.has_book("BTC", "spot")
    assert exchange.has_book("BTC", "futures")
    assert not exchange.has_book("SOL", "spot")

    spot_book = exchange.get_book("BTC", "spot")
    futures_book = exchange.get_book("BTC", "futures")
    assert spot_book.symbol == "BTC"
    assert futures_book.symbol == "BTC"


def test_fetch_json_fallback_logs_and_recovers(monkeypatch, caplog):
    exchange = _DummyExchange(on_book_update=_on_book_update, depth_limit=5)
    state = {"calls": 0}

    def _fake_fetch(_url, _timeout=10.0, _headers=None):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("boom")
        return {"ok": True}

    monkeypatch.setattr("src.spread.exchanges.base._fetch_json_sync", _fake_fetch)

    with caplog.at_level(logging.INFO):
        payload1 = asyncio.run(exchange._fetch_json_fallback("https://example.com", "spot_ticker_fallback"))
        payload2 = asyncio.run(exchange._fetch_json_fallback("https://example.com", "spot_ticker_fallback"))

    assert payload1 is None
    assert payload2 == {"ok": True}
    assert any("fetch failed" in record.message for record in caplog.records)
    assert any("fetch recovered" in record.message for record in caplog.records)


def test_fetch_json_sync_bypasses_environment_proxies(monkeypatch):
    captured = {}

    class _FakeOpener:
        def open(self, req, timeout=10.0):
            captured["url"] = req.full_url
            captured["timeout"] = timeout
            captured["headers"] = dict(req.header_items())

            class _Resp:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

                def read(self_inner):
                    return b'{"ok": true}'

            return _Resp()

    def _fake_build_opener(proxy_handler):
        captured["proxy_handler"] = proxy_handler
        return _FakeOpener()

    monkeypatch.setattr("src.spread.exchanges.base.urllib.request.build_opener", _fake_build_opener)

    from src.spread.exchanges.base import _fetch_json_sync

    payload = _fetch_json_sync("https://example.com/feed", headers={"X-Test": "1"})

    assert payload == {"ok": True}
    assert captured["url"] == "https://example.com/feed"
    assert captured["timeout"] == 10.0
    assert captured["headers"]["X-test"] == "1"
    assert captured["proxy_handler"].proxies == {}
