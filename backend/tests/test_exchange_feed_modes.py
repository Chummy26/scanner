import asyncio

import pytest

from src.spread.exchanges.bingx import BingxWS
from src.spread.exchanges.bitget import BitgetWS
from src.spread.exchanges.gate import GateWS
from src.spread.exchanges.kucoin import KucoinWS
from src.spread.exchanges.mexc import MexcWS
from src.spread.exchanges.xt import XtWS


def _on_book_update(*_args, **_kwargs):
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
