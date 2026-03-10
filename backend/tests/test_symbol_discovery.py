import asyncio

from aiohttp.resolver import ThreadedResolver

from src.spread.symbol_discovery import _build_discovery_connector, _xt_symbols


def test_symbol_discovery_uses_threaded_resolver():
    async def _run():
        connector = _build_discovery_connector()
        try:
            assert isinstance(connector._resolver, ThreadedResolver)
        finally:
            await connector.close()

    asyncio.run(_run())


def test_xt_symbol_discovery_filters_non_tradeable_hidden_futures(monkeypatch):
    responses = iter(
        [
            {
                "result": {
                    "symbols": [
                        {"baseCurrency": "AERGO", "quoteCurrency": "usdt", "state": "ONLINE"},
                        {"baseCurrency": "IDEX", "quoteCurrency": "usdt", "state": "ONLINE"},
                    ]
                }
            },
            {
                "result": [
                    {
                        "symbol": "aergo_usdt",
                        "tradeSwitch": False,
                        "openSwitch": False,
                        "isDisplay": False,
                        "displaySwitch": False,
                        "isOpenApi": False,
                    },
                    {
                        "symbol": "idex_usdt",
                        "tradeSwitch": False,
                        "openSwitch": False,
                        "isDisplay": False,
                        "displaySwitch": False,
                        "isOpenApi": False,
                    },
                    {
                        "symbol": "btc_usdt",
                        "tradeSwitch": True,
                        "openSwitch": True,
                        "isDisplay": True,
                        "displaySwitch": True,
                        "isOpenApi": True,
                    },
                ]
            },
        ]
    )

    async def _fake_fetch_json(_session, _url, timeout=15):
        return next(responses)

    monkeypatch.setattr("src.spread.symbol_discovery._fetch_json", _fake_fetch_json)

    async def _run():
        result = await _xt_symbols(None)
        assert "AERGO" in result["spot"]
        assert "IDEX" in result["spot"]
        assert "BTC" in result["futures"]
        assert "AERGO" not in result["futures"]
        assert "IDEX" not in result["futures"]

    asyncio.run(_run())
