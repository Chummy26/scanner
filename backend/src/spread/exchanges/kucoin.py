"""KuCoin exchange WebSocket adapter.

KuCoin Spot: bullet-public token + /spotMarket/level2Depth50
KuCoin Futures: bullet-public token + /contractMarket/level2Depth50

Uses shared connections: one WS per market type per batch.
KuCoin allows ~100 subscriptions per connection.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List

import aiohttp
import websockets

from .base import BaseExchangeWS, json_loads_safe, chunk_list, _fetch_json_sync

logger = logging.getLogger(__name__)

KUCOIN_BULLET_URL = "https://api.kucoin.com/api/v1/bullet-public"
KUCOIN_FUTURES_BULLET_URL = "https://api-futures.kucoin.com/api/v1/bullet-public"
KUCOIN_SPOT_TICKERS = "https://api.kucoin.com/api/v1/market/allTickers"
KUCOIN_FUTURES_TICKERS = "https://api-futures.kucoin.com/api/v1/allTickers"


class KucoinWS(BaseExchangeWS):
    name = "kucoin"
    batch_size = 95  # KuCoin limits ~100 subs per connection
    max_connections_per_market = 10  # KuCoin allows 800 conns/UID; 10 spot + 10 fut = 20 total

    def format_symbol(self, base: str, quote: str = "USDT") -> Dict[str, str]:
        kucoin_base = "XBT" if base == "BTC" else base
        return {
            "spot": f"{base}-{quote}",
            "futures": f"{kucoin_base}{quote}M",
        }

    async def subscribe_symbol(self, base: str, quote: str = "USDT"):
        await asyncio.gather(
            self._batch_session([base], "spot"),
            self._batch_session([base], "futures"),
            return_exceptions=True,
        )

    async def _batch_session(self, symbols: List[str], market: str):
        if market == "spot":
            bullet_url = KUCOIN_BULLET_URL
            topic_prefix = "/spotMarket/level2Depth50"
        else:
            bullet_url = KUCOIN_FUTURES_BULLET_URL
            topic_prefix = "/contractMarket/level2Depth50"

        # Build symbol -> base mapping
        sym_map = {}
        topic_syms = []
        for base in symbols:
            s = self.format_symbol(base)[market]
            sym_map[s] = base
            topic_syms.append(s)

        async with aiohttp.ClientSession() as session:
            ws_url, ping_interval = await self._get_ws_url(session, bullet_url)

        # KuCoin allows comma-separated multi-topic subscriptions
        # /spotMarket/level2Depth50:BTC-USDT,ETH-USDT,...
        topic = f"{topic_prefix}:{','.join(topic_syms)}"

        async with websockets.connect(
            ws_url, ping_interval=None, ping_timeout=None,
            proxy=None,
            max_size=16 * 1024 * 1024,
        ) as ws:
            # Subscribe in sub-batches of 50 symbols per subscribe message
            for i in range(0, len(topic_syms), 50):
                batch = topic_syms[i:i+50]
                sub_topic = f"{topic_prefix}:{','.join(batch)}"
                await ws.send(json.dumps({
                    "id": uuid.uuid4().hex[:12],
                    "type": "subscribe",
                    "topic": sub_topic,
                    "privateChannel": False,
                    "response": True,
                }))
                await asyncio.sleep(0.1)

            async def _ping():
                while not self.shutdown.is_set():
                    try:
                        await asyncio.sleep(ping_interval)
                        await ws.send(json.dumps({
                            "id": uuid.uuid4().hex[:8], "type": "ping"
                        }))
                    except Exception:
                        return

            ping_task = asyncio.create_task(_ping())
            try:
                async for raw in ws:
                    if self.shutdown.is_set():
                        return

                    msg = json_loads_safe(raw)
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("type") != "message":
                        continue

                    topic_str = msg.get("topic", "")
                    # Topic format: /spotMarket/level2Depth50:BTC-USDT
                    parts = topic_str.split(":")
                    if len(parts) < 2:
                        continue
                    symbol_part = parts[-1]
                    base = sym_map.get(symbol_part)
                    if not base:
                        continue

                    data = msg.get("data") or {}
                    bids = data.get("bids") or []
                    asks = data.get("asks") or []

                    book = self.get_book(base, market)
                    if book.apply_snapshot(bids, asks):
                        self.on_book_update(base, "kucoin", market, book)
            finally:
                ping_task.cancel()

    async def subscribe_symbol_batch(self, symbols):
        """Override to add spot+futures ticker fallbacks."""
        spot_syms = list(getattr(self, "_spot_symbols", symbols))
        fut_syms = list(getattr(self, "_futures_symbols", symbols))
        spot_mode = self.get_feed_mode("spot")
        futures_mode = self.get_feed_mode("futures")
        tasks = []
        if self.spot_enabled:
            if spot_mode == "depth_ws":
                for i, chunk in enumerate(chunk_list(spot_syms, self.batch_size)):
                    tasks.append(self._run_batch_loop(chunk, "spot", i, delay=i * 3.0))
            else:
                for base in spot_syms:
                    self.get_book(base, "spot")
            tasks.append(self._run_spot_ticker_fallback(spot_syms))
        if self.futures_enabled:
            if futures_mode == "depth_ws":
                for i, chunk in enumerate(chunk_list(fut_syms, self.batch_size)):
                    tasks.append(self._run_batch_loop(chunk, "futures", i, delay=i * 3.0 + 1.5))
            else:
                for base in fut_syms:
                    self.get_book(base, "futures")
            tasks.append(self._run_futures_ticker_fallback(fut_syms))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _get_ws_url(self, session: aiohttp.ClientSession, bullet_url: str):
        async with session.post(bullet_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
        token = data.get("data", {}).get("token")
        servers = data.get("data", {}).get("instanceServers") or []
        if not token or not servers:
            raise RuntimeError("KuCoin bullet-public missing token/servers")
        endpoint = servers[0].get("endpoint")
        ping_ms = servers[0].get("pingInterval", 18000)
        connect_id = uuid.uuid4().hex
        return f"{endpoint}?token={token}&connectId={connect_id}", float(ping_ms) / 1000.0

    async def _run_spot_ticker_fallback(self, symbols: List[str]):
        """Poll GET /api/v1/market/allTickers every 3s for stale spot books.
        Uses a fresh session per request.
        """
        if not symbols:
            return

        sym_map = {}
        for base in symbols:
            s = self.format_symbol(base)["spot"]
            sym_map[s] = base

        await asyncio.sleep(3)
        logger.info(f"[KUCOIN] spot ticker fallback started for {len(sym_map)} symbols")

        _cycle = 0
        while not self.shutdown.is_set():
            payload = await self._fetch_json_fallback(
                KUCOIN_SPOT_TICKERS,
                "spot_ticker_fallback",
            )

            if isinstance(payload, dict):
                tickers = (payload.get("data") or {}).get("ticker") or []
                if tickers:
                    _updated = 0
                    for t in tickers:
                        sym = t.get("symbol", "")
                        base = sym_map.get(sym)
                        if not base:
                            continue
                        if not self.has_book(base, "spot"):
                            continue
                        book = self.get_book(base, "spot")
                        if not book.is_stale(1.5):
                            continue
                        bid = t.get("buy")
                        ask = t.get("sell")
                        if bid and ask:
                            try:
                                bf, af = float(bid), float(ask)
                            except (ValueError, TypeError):
                                continue
                            if bf > 0 and af > 0:
                                book.apply_snapshot(
                                    [[bid, "1"]], [[ask, "1"]])
                                self.on_book_update(base, "kucoin", "spot", book)
                                _updated += 1

                    _cycle += 1
                    if _cycle <= 5 or _cycle % 50 == 0:
                        logger.info(
                            f"[KUCOIN] spot ticker fallback cycle #{_cycle}: "
                            f"updated {_updated} stale books")

            await asyncio.sleep(1.5)

    async def _run_futures_ticker_fallback(self, symbols: List[str]):
        """Poll GET /api/v1/allTickers every 3s for stale futures books.
        Uses a fresh session per request.
        """
        if not symbols:
            return

        sym_map = {}
        for base in symbols:
            s = self.format_symbol(base)["futures"]
            sym_map[s] = base

        await asyncio.sleep(3)
        logger.info(f"[KUCOIN] futures ticker fallback started for {len(sym_map)} symbols")

        _cycle = 0
        while not self.shutdown.is_set():
            payload = await self._fetch_json_fallback(
                KUCOIN_FUTURES_TICKERS,
                "futures_ticker_fallback",
            )

            if isinstance(payload, dict):
                tickers = payload.get("data") or []
                if tickers:
                    _updated = 0
                    for t in tickers:
                        sym = t.get("symbol", "")
                        base = sym_map.get(sym)
                        if not base:
                            continue
                        if not self.has_book(base, "futures"):
                            continue
                        book = self.get_book(base, "futures")
                        if not book.is_stale(1.5):
                            continue
                        bid = t.get("bestBidPrice")
                        ask = t.get("bestAskPrice")
                        if bid and ask:
                            try:
                                bf, af = float(bid), float(ask)
                            except (ValueError, TypeError):
                                continue
                            if bf > 0 and af > 0:
                                book.apply_snapshot(
                                    [[str(bid), "1"]], [[str(ask), "1"]])
                                self.on_book_update(base, "kucoin", "futures", book)
                                _updated += 1

                    _cycle += 1
                    if _cycle <= 5 or _cycle % 50 == 0:
                        logger.info(
                            f"[KUCOIN] futures ticker fallback cycle #{_cycle}: "
                            f"updated {_updated} stale books")

            await asyncio.sleep(1.5)
