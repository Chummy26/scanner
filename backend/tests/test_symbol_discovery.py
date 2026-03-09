import asyncio

from aiohttp.resolver import ThreadedResolver

from src.spread.symbol_discovery import _build_discovery_connector


def test_symbol_discovery_uses_threaded_resolver():
    async def _run():
        connector = _build_discovery_connector()
        try:
            assert isinstance(connector._resolver, ThreadedResolver)
        finally:
            await connector.close()

    asyncio.run(_run())
