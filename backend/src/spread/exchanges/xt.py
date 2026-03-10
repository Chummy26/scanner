"""XT.com exchange WebSocket adapter.

Spot:    wss://stream.xt.com/public
Futures: wss://fstream.xt.com/ws/market  (dedicated futures WS, V2 API)

Both use the same depth topic format: depth@{symbol},{levels}[,{interval}]
Response format is identical: {topic: "depth", data: {s, a, b}}

A REST fallback polls GET /future/market/v1/public/q/agg-tickers every 3s
to fill in any futures symbols that don't receive WS updates (low-volume).
"""

import asyncio
import json
import logging
import time
from typing import Dict, List

import aiohttp
import websockets

from .base import BaseExchangeWS, chunk_list, json_loads_safe, periodic_send_text, _fetch_json_sync

logger = logging.getLogger(__name__)

XT_SPOT_WS = "wss://stream.xt.com/public"
XT_FUTURES_WS = "wss://fstream.xt.com/ws/market"
XT_FUTURES_AGG_TICKERS = "https://fapi.xt.com/future/market/v1/public/q/agg-tickers"
XT_SPOT_BOOK_TICKERS = "https://sapi.xt.com/v4/public/ticker/book"


class XtWS(BaseExchangeWS):
    name = "xt"
    batch_size = 200

    def format_symbol(self, base: str, quote: str = "USDT") -> Dict[str, str]:
        sym = f"{base.lower()}_{quote.lower()}"
        return {"spot": sym, "futures": sym}

    async def subscribe_symbol_batch(self, symbols: List[str]):
        """Subscribe spot and futures via WebSocket, with agg-ticker fallback."""
        symbols = list(symbols or [])
        spot_mode = self.get_feed_mode("spot")
        futures_mode = self.get_feed_mode("futures")

        tasks = []
        if self.spot_enabled:
            spot_syms = getattr(self, "_spot_symbols", symbols)
            if spot_mode == "depth_ws":
                for i, chunk in enumerate(chunk_list(spot_syms, self.batch_size)):
                    tasks.append(self._run_batch_loop(chunk, "spot", i, delay=i * 2.0))
            else:
                for base in spot_syms:
                    self.get_book(base, "spot")
            tasks.append(self._run_spot_ticker_fallback(spot_syms))

        if self.futures_enabled:
            fut_syms = getattr(self, "_futures_symbols", symbols)
            if futures_mode == "depth_ws":
                for i, chunk in enumerate(chunk_list(fut_syms, self.batch_size)):
                    tasks.append(self._run_batch_loop(chunk, "futures", i, delay=i * 2.0 + 1.0))
            else:
                for base in fut_syms:
                    self.get_book(base, "futures")
            # Fallback: poll agg-tickers to catch symbols without WS updates
            tasks.append(self._run_agg_ticker_fallback(fut_syms))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def subscribe_symbol(self, base: str, quote: str = "USDT"):
        await self._batch_session([base], "spot")

    async def _batch_session(self, symbols: List[str], market: str):
        ws_url = XT_SPOT_WS if market == "spot" else XT_FUTURES_WS

        # Build symbol mapping: exchange_symbol -> base
        sym_map = {}
        params = []
        for base in symbols:
            s = self.format_symbol(base)[market]
            sym_map[s] = base
            if market == "futures":
                params.append(f"depth@{s},{self.depth_limit},1000ms")
            else:
                params.append(f"depth@{s},{self.depth_limit}")

        logger.info(f"[XT] {market} connecting ({len(symbols)} symbols) -> {ws_url}")

        async with websockets.connect(
            ws_url, ping_interval=None, ping_timeout=None,
            max_size=16 * 1024 * 1024,
        ) as ws:
            # XT futures WS silently drops ALL subscriptions if >10 topics per
            # message. Spot tolerates 20. Use 10 for both to be safe.
            sub_method = "SUBSCRIBE" if market == "futures" else "subscribe"
            chunk_size = 10
            for i in range(0, len(params), chunk_size):
                batch = params[i:i+chunk_size]
                await ws.send(json.dumps({
                    "method": sub_method,
                    "params": batch,
                    "id": str(i),
                }))
                await asyncio.sleep(0.1)

            logger.info(f"[XT] {market} subscribed {len(params)} channels")

            # Futures WS disconnects after 30s without ping; use 15s interval
            ping_interval = 15.0 if market == "futures" else 10.0
            ping_task = asyncio.create_task(
                periodic_send_text(ws, "ping", ping_interval, self.shutdown))

            _update_count = 0
            try:
                async for raw in ws:
                    if self.shutdown.is_set():
                        return

                    msg = json_loads_safe(raw)
                    if not isinstance(msg, dict):
                        continue

                    topic = msg.get("topic")
                    data = msg.get("data") or {}
                    if not isinstance(data, dict):
                        continue

                    # Extract symbol from data.s or event field
                    sym_part = data.get("s", "")
                    if not sym_part:
                        event = msg.get("event", "")
                        if "@" in event:
                            sym_part = event.split("@")[1].split(",")[0]
                    base = sym_map.get(sym_part)
                    if not base:
                        continue

                    book = self.get_book(base, market)

                    if topic == "depth":
                        asks = data.get("a") or []
                        bids = data.get("b") or []
                        book.apply_snapshot(bids, asks)
                        _update_count += 1
                        if _update_count <= 3 or _update_count % 2000 == 0:
                            logger.info(
                                f"[XT] {market} update #{_update_count}: {base}")
                        self.on_book_update(base, "xt", market, book)
                    elif topic == "depth_update":
                        asks = data.get("a") or []
                        bids = data.get("b") or []
                        if book.apply_delta(bids, asks):
                            _update_count += 1
                            self.on_book_update(base, "xt", market, book)
            finally:
                ping_task.cancel()

    async def _run_agg_ticker_fallback(self, symbols: List[str]):
        """Poll agg-tickers every 3s as fallback for futures symbols
        not receiving WS depth updates (low-volume or unsubscribed).
        Uses a fresh session per request to avoid session corruption.
        """
        if not symbols:
            return

        sym_map = {}
        for base in symbols:
            s = self.format_symbol(base)["futures"]
            sym_map[s] = base

        await asyncio.sleep(3)
        logger.info(f"[XT] agg-ticker fallback started for {len(sym_map)} futures symbols")

        _fallback_count = 0
        while not self.shutdown.is_set():
            data = None
            try:
                data = await asyncio.to_thread(
                    _fetch_json_sync, XT_FUTURES_AGG_TICKERS)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

            if isinstance(data, dict) and data.get("returnCode") == 0:
                _cycle_updated = 0
                for ticker in (data.get("result") or []):
                    sym = ticker.get("s", "")
                    base = sym_map.get(sym)
                    if not base:
                        continue

                    if not self.has_book(base, "futures"):
                        continue
                    book = self.get_book(base, "futures")
                    if not book.is_stale(1.5):
                        continue

                    bp = ticker.get("bp")
                    ap = ticker.get("ap")
                    if bp and ap:
                        try:
                            bp_f = float(bp)
                            ap_f = float(ap)
                        except (ValueError, TypeError):
                            continue
                        if bp_f > 0 and ap_f > 0:
                            book.apply_snapshot(
                                [[str(bp), "1"]], [[str(ap), "1"]])
                            self.on_book_update(base, "xt", "futures", book)
                            _cycle_updated += 1

                _fallback_count += 1
                if _fallback_count <= 5 or _fallback_count % 50 == 0:
                    logger.info(
                        f"[XT] agg-ticker fallback cycle #{_fallback_count}: "
                        f"updated {_cycle_updated} stale books")

            await asyncio.sleep(1.5)

    async def _run_spot_ticker_fallback(self, symbols: List[str]):
        """Poll GET /v4/public/ticker/book every 3s for stale spot books.

        One request returns bid/ask for ALL XT spot pairs.
        Only updates books that are stale (>8s without WS update).
        """
        if not symbols:
            return

        sym_map = {}
        for base in symbols:
            s = self.format_symbol(base)["spot"]
            sym_map[s] = base

        await asyncio.sleep(3)
        logger.info(f"[XT] spot ticker fallback started for {len(sym_map)} symbols")

        _cycle = 0
        while not self.shutdown.is_set():
            data = None
            try:
                data = await asyncio.to_thread(
                    _fetch_json_sync, XT_SPOT_BOOK_TICKERS)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

            if isinstance(data, dict) and data.get("rc") == 0:
                _updated = 0
                for t in (data.get("result") or []):
                    sym = t.get("s", "")
                    base = sym_map.get(sym)
                    if not base:
                        continue
                    if not self.has_book(base, "spot"):
                        continue
                    book = self.get_book(base, "spot")
                    if not book.is_stale(1.5):
                        continue
                    bid = t.get("bp")
                    ask = t.get("ap")
                    if bid and ask:
                        try:
                            bf, af = float(bid), float(ask)
                        except (ValueError, TypeError):
                            continue
                        if bf > 0 and af > 0:
                            book.apply_snapshot(
                                [[str(bid), t.get("bq", "1")]],
                                [[str(ask), t.get("aq", "1")]])
                            self.on_book_update(base, "xt", "spot", book)
                            _updated += 1

                _cycle += 1
                if _cycle <= 5 or _cycle % 50 == 0:
                    logger.info(
                        f"[XT] spot ticker fallback cycle #{_cycle}: "
                        f"updated {_updated} stale books")

            await asyncio.sleep(1.5)
