"""Bitget exchange WebSocket adapter.

Bitget v2 Public: wss://ws.bitget.com/v2/ws/public (books15)

A REST fallback polls /api/v2/mix/market/tickers every 3s to keep
futures books fresh when WS connections drop.

Uses shared connections: one WS per market type per batch.
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

BITGET_WS_URL = "wss://ws.bitget.com/v2/ws/public"
BITGET_FUTURES_TICKERS = "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES"
BITGET_SPOT_TICKERS = "https://api.bitget.com/api/v2/spot/market/tickers"


class BitgetWS(BaseExchangeWS):
    name = "bitget"
    batch_size = 240  # Bitget allows 1000 subs/conn, 100 conns/IP

    def format_symbol(self, base: str, quote: str = "USDT") -> Dict[str, str]:
        return {
            "spot": f"{base}{quote}",
            "futures": f"{base}{quote}",
        }

    async def subscribe_symbol(self, base: str, quote: str = "USDT"):
        await asyncio.gather(
            self._batch_session([base], "spot"),
            self._batch_session([base], "futures"),
            return_exceptions=True,
        )

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

    async def _batch_session(self, symbols: List[str], market: str):
        inst_type = "SPOT" if market == "spot" else "USDT-FUTURES"

        # Build instId -> base mapping
        inst_map = {}
        args = []
        for base in symbols:
            s = self.format_symbol(base)[market]
            inst_map[s] = base
            args.append({
                "instType": inst_type,
                "channel": "books15",
                "instId": s,
            })

        logger.info(f"[BITGET] {market} connecting ({len(symbols)} symbols)...")

        async with websockets.connect(
            BITGET_WS_URL, ping_interval=None, ping_timeout=None,
            max_size=16 * 1024 * 1024,
        ) as ws:
            # Subscribe in chunks of 30 args per message
            for i in range(0, len(args), 30):
                batch = args[i:i+30]
                await ws.send(json.dumps({"op": "subscribe", "args": batch}))
                await asyncio.sleep(0.1)

            logger.info(f"[BITGET] {market} subscribed {len(args)} channels")

            ping_task = asyncio.create_task(
                periodic_send_text(ws, "ping", 20.0, self.shutdown))
            _update_count = 0
            # Per-symbol throttle to reduce event-loop pressure.
            _last_ts: Dict[str, float] = {}
            _THROTTLE_SEC = 0.05  # max 20 updates/symbol/second
            try:
                async for raw in ws:
                    if self.shutdown.is_set():
                        return

                    if raw == "pong":
                        continue

                    msg = json_loads_safe(raw)
                    if not isinstance(msg, dict):
                        continue

                    action = msg.get("action")
                    arg = msg.get("arg") or {}
                    data_list = msg.get("data") or []
                    if not isinstance(data_list, list) or not data_list:
                        continue

                    inst_id = arg.get("instId", "")
                    base = inst_map.get(inst_id)
                    if not base:
                        continue

                    # Always accept snapshots; throttle deltas
                    if action != "snapshot":
                        now_m = time.monotonic()
                        if (now_m - _last_ts.get(base, 0.0)) < _THROTTLE_SEC:
                            continue
                        _last_ts[base] = now_m

                    book = self.get_book(base, market)
                    data = data_list[0]

                    bids = data.get("bids") or []
                    asks = data.get("asks") or []

                    if action == "snapshot":
                        if book.apply_snapshot(bids, asks):
                            _update_count += 1
                            if _update_count <= 3 or _update_count % 2000 == 0:
                                logger.info(f"[BITGET] {market} update #{_update_count}: {base}")
                            self.on_book_update(base, "bitget", market, book)
                    elif action == "update":
                        if book.apply_delta(bids, asks):
                            _update_count += 1
                            if _update_count % 2000 == 0:
                                logger.info(f"[BITGET] {market} update #{_update_count}: {base}")
                            self.on_book_update(base, "bitget", market, book)
            finally:
                ping_task.cancel()

    async def _run_spot_ticker_fallback(self, symbols: List[str]):
        """Poll GET /api/v2/spot/market/tickers every 3s for stale spot books.
        Uses a fresh session per request to avoid session corruption.
        """
        if not symbols:
            return

        sym_map = {}
        for base in symbols:
            s = self.format_symbol(base)["spot"]
            sym_map[s] = base

        await asyncio.sleep(3)
        logger.info(f"[BITGET] spot ticker fallback started for {len(sym_map)} symbols")

        _cycle = 0
        while not self.shutdown.is_set():
            payload = None
            try:
                payload = await asyncio.to_thread(
                    _fetch_json_sync, BITGET_SPOT_TICKERS)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

            if isinstance(payload, dict) and payload.get("data"):
                _updated = 0
                for t in (payload.get("data") or []):
                    sym = t.get("symbol", "")
                    base = sym_map.get(sym)
                    if not base:
                        continue
                    if not self.has_book(base, "spot"):
                        continue
                    book = self.get_book(base, "spot")
                    if not book.is_stale(1.5):
                        continue
                    bid = t.get("bidPr")
                    ask = t.get("askPr")
                    if bid and ask:
                        try:
                            bf, af = float(bid), float(ask)
                        except (ValueError, TypeError):
                            continue
                        if bf > 0 and af > 0:
                            book.apply_snapshot(
                                [[bid, "1"]], [[ask, "1"]])
                            self.on_book_update(base, "bitget", "spot", book)
                            _updated += 1

                _cycle += 1
                if _cycle <= 5 or _cycle % 50 == 0:
                    logger.info(
                        f"[BITGET] spot ticker fallback cycle #{_cycle}: "
                        f"updated {_updated} stale books")

            await asyncio.sleep(1.5)

    async def _run_futures_ticker_fallback(self, symbols: List[str]):
        """Poll GET /api/v2/mix/market/tickers every 3s.

        One request returns bid/ask for ALL ~335 Bitget futures contracts.
        Only updates books that are stale (>8s without WS update).
        Uses a fresh session per request to avoid session corruption.
        """
        if not symbols:
            return

        sym_map = {}
        for base in symbols:
            s = self.format_symbol(base)["futures"]
            sym_map[s] = base

        await asyncio.sleep(3)
        logger.info(f"[BITGET] futures ticker fallback started for {len(sym_map)} symbols")

        _cycle = 0
        while not self.shutdown.is_set():
            payload = None
            try:
                payload = await asyncio.to_thread(
                    _fetch_json_sync, BITGET_FUTURES_TICKERS)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

            if isinstance(payload, dict) and payload.get("data"):
                _updated = 0
                for t in (payload.get("data") or []):
                    sym = t.get("symbol", "")
                    base = sym_map.get(sym)
                    if not base:
                        continue
                    if not self.has_book(base, "futures"):
                        continue
                    book = self.get_book(base, "futures")
                    if not book.is_stale(1.5):
                        continue
                    bid = t.get("bidPr")
                    ask = t.get("askPr")
                    if bid and ask:
                        try:
                            bf, af = float(bid), float(ask)
                        except (ValueError, TypeError):
                            continue
                        if bf > 0 and af > 0:
                            book.apply_snapshot(
                                [[bid, "1"]], [[ask, "1"]])
                            self.on_book_update(base, "bitget", "futures", book)
                            _updated += 1

                _cycle += 1
                if _cycle <= 5 or _cycle % 50 == 0:
                    logger.info(
                        f"[BITGET] futures ticker fallback cycle #{_cycle}: "
                        f"updated {_updated} stale books")

            await asyncio.sleep(1.5)
