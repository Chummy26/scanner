"""Gate.io exchange WebSocket adapter.

Gate Spot: wss://api.gateio.ws/ws/v4/ (spot.order_book)
Gate Futures: wss://fx-ws.gateio.ws/v4/ws/usdt (futures.order_book)

Futures WS connections are unstable (~60-90s lifespan) so a REST
bulk ticker fallback polls /api/v4/futures/usdt/tickers every 3s
to keep stale books fresh (one request returns ALL contracts).

Uses shared connections: one WS per market type per batch.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List

import aiohttp
import websockets

from .base import BaseExchangeWS, json_loads_safe, chunk_list, _fetch_json_sync

logger = logging.getLogger(__name__)

GATE_SPOT_WS = "wss://api.gateio.ws/ws/v4/"
GATE_FUTURES_WS = "wss://fx-ws.gateio.ws/v4/ws/usdt"
GATE_FUTURES_TICKERS = "https://api.gateio.ws/api/v4/futures/usdt/tickers"
GATE_SPOT_TICKERS = "https://api.gateio.ws/api/v4/spot/tickers"


class GateWS(BaseExchangeWS):
    name = "gate"
    batch_size = 200  # 200 symbols per connection — keeps per-connection data volume manageable

    _futures_batch_size = 200  # Same size for futures

    def format_symbol(self, base: str, quote: str = "USDT") -> Dict[str, str]:
        return {
            "spot": f"{base}_{quote}",
            "futures": f"{base}_{quote}",
        }

    async def subscribe_symbol(self, base: str, quote: str = "USDT"):
        syms = self.format_symbol(base, quote)
        await asyncio.gather(
            self._single_spot(base, syms["spot"]),
            self._single_futures(base, syms["futures"]),
            return_exceptions=True,
        )

    async def subscribe_symbol_batch(self, symbols):
        """Override: split into 200-symbol chunks for spot, 100 for futures.

        Gate futures WS connections are unstable (~60-90s lifespan), so we
        also run a REST bulk ticker fallback that keeps stale books fresh.
        """
        spot_syms = getattr(self, "_spot_symbols", symbols)
        fut_syms = getattr(self, "_futures_symbols", symbols)
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
                for i, chunk in enumerate(chunk_list(fut_syms, self._futures_batch_size)):
                    tasks.append(self._run_batch_loop(chunk, "futures", i, delay=i * 3.0 + 1.5))
            else:
                for base in fut_syms:
                    self.get_book(base, "futures")
            # REST fallback: poll all futures tickers every 3s
            tasks.append(self._run_futures_ticker_fallback(fut_syms))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _batch_session(self, symbols: List[str], market: str):
        if market == "spot":
            await self._batch_spot(symbols)
        else:
            await self._batch_futures(symbols)

    async def _batch_spot(self, symbols: List[str]):
        # Build lookup: exchange_symbol -> base
        sym_map = {}
        for base in symbols:
            s = self.format_symbol(base)["spot"]
            sym_map[s] = base

        logger.info(f"[GATE] Spot connecting ({len(symbols)} symbols)...")

        async with websockets.connect(
            GATE_SPOT_WS, ping_interval=20, ping_timeout=30,
            max_size=16 * 1024 * 1024,
        ) as ws:
            # Subscribe in a background task while simultaneously reading
            # messages. This keeps the connection alive — Gate disconnects
            # idle connections, so we must be reading while subscribing.
            sub_done = asyncio.Event()

            async def _subscribe():
                sent = 0
                for ex_sym in sym_map:
                    if self.shutdown.is_set():
                        break
                    await ws.send(json.dumps({
                        "time": int(time.time()),
                        "channel": "spot.order_book",
                        "event": "subscribe",
                        "payload": [ex_sym, str(self.depth_limit), "1000ms"],
                    }))
                    sent += 1
                    await asyncio.sleep(0.03)
                sub_done.set()
                logger.info(f"[GATE] Spot subscribed {sent} symbols")

            sub_task = asyncio.create_task(_subscribe())
            _update_count = 0

            try:
                async for raw in ws:
                    if self.shutdown.is_set():
                        return

                    msg = json_loads_safe(raw)
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("channel") != "spot.order_book" or msg.get("event") != "update":
                        continue

                    res = msg.get("result") or {}
                    s_field = res.get("s", "")
                    base = sym_map.get(s_field)
                    if not base:
                        continue

                    bids = res.get("bids") or []
                    asks = res.get("asks") or []

                    book = self.get_book(base, "spot")
                    if book.apply_snapshot(bids, asks):
                        _update_count += 1
                        if _update_count <= 3 or _update_count % 2000 == 0:
                            logger.info(f"[GATE] Spot update #{_update_count}: {base}")
                        self.on_book_update(base, "gate", "spot", book)
            finally:
                sub_task.cancel()

    async def _batch_futures(self, symbols: List[str]):
        sym_map = {}
        for base in symbols:
            s = self.format_symbol(base)["futures"]
            sym_map[s] = base

        # Gate futures `futures.order_book` uses an "accuracy" parameter, not an
        # update interval like spot. Using "100ms" causes `invalid accuracy` and
        # results in no book updates (missing FUTURO-FUTURO opportunities).
        #
        # Tested values: "0" and "0.1" work; "100ms"/"1000ms" do not.
        accuracy = "0"

        logger.info(f"[GATE] Futures connecting ({len(symbols)} symbols)...")

        async with websockets.connect(
            GATE_FUTURES_WS, ping_interval=20, ping_timeout=30,
            max_size=16 * 1024 * 1024,
        ) as ws:
            # Subscribe in background while reading messages (same pattern
            # as spot) to keep the connection alive during long subscription.
            sub_done = asyncio.Event()

            async def _subscribe():
                sent = 0
                for ex_sym in sym_map:
                    if self.shutdown.is_set():
                        break
                    await ws.send(json.dumps({
                        "time": int(time.time()),
                        "channel": "futures.order_book",
                        "event": "subscribe",
                        "payload": [ex_sym, str(self.depth_limit), accuracy],
                    }))
                    sent += 1
                    await asyncio.sleep(0.03)
                sub_done.set()
                logger.info(f"[GATE] Futures subscribed {sent} symbols")

            sub_task = asyncio.create_task(_subscribe())
            _update_count = 0

            try:
                async for raw in ws:
                    if self.shutdown.is_set():
                        return

                    msg = json_loads_safe(raw)
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("channel") != "futures.order_book":
                        continue
                    if msg.get("event") not in ("all", "update"):
                        continue

                    res = msg.get("result") or {}
                    contract = res.get("contract", "") or res.get("s", "")
                    base = sym_map.get(contract)
                    if not base:
                        continue

                    bids_raw = res.get("bids") or []
                    asks_raw = res.get("asks") or []

                    bids = []
                    for b in bids_raw[:self.depth_limit]:
                        if isinstance(b, dict) and "p" in b and "s" in b:
                            bids.append([b["p"], b["s"]])
                        elif isinstance(b, (list, tuple)) and len(b) >= 2:
                            bids.append(b)
                    asks = []
                    for a in asks_raw[:self.depth_limit]:
                        if isinstance(a, dict) and "p" in a and "s" in a:
                            asks.append([a["p"], a["s"]])
                        elif isinstance(a, (list, tuple)) and len(a) >= 2:
                            asks.append(a)

                    book = self.get_book(base, "futures")
                    if book.apply_snapshot(bids, asks):
                        _update_count += 1
                        if _update_count <= 3 or _update_count % 2000 == 0:
                            logger.info(f"[GATE] Futures update #{_update_count}: {base}")
                        self.on_book_update(base, "gate", "futures", book)
            finally:
                sub_task.cancel()

    async def _run_futures_ticker_fallback(self, symbols: List[str]):
        """Poll GET /api/v4/futures/usdt/tickers every 3s.

        One request returns bid/ask for ALL ~534 Gate futures contracts.
        Only updates books that are stale (>8s without WS update).
        Uses asyncio.to_thread() to avoid event-loop starvation.
        """
        if not symbols:
            return

        sym_map = {}
        for base in symbols:
            s = self.format_symbol(base)["futures"]
            sym_map[s] = base

        await asyncio.sleep(3)
        logger.info(f"[GATE] futures ticker fallback started for {len(sym_map)} symbols")

        _cycle = 0
        while not self.shutdown.is_set():
            tickers = None
            try:
                tickers = await asyncio.to_thread(
                    _fetch_json_sync, GATE_FUTURES_TICKERS)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

            if isinstance(tickers, list):
                _updated = 0
                for t in tickers:
                    contract = t.get("contract", "")
                    base = sym_map.get(contract)
                    if not base:
                        continue
                    if not self.has_book(base, "futures"):
                        continue
                    book = self.get_book(base, "futures")
                    if not book.is_stale(1.5):
                        continue
                    bid = t.get("highest_bid")
                    ask = t.get("lowest_ask")
                    if bid and ask:
                        try:
                            bf, af = float(bid), float(ask)
                        except (ValueError, TypeError):
                            continue
                        if bf > 0 and af > 0:
                            book.apply_snapshot([[bid, "1"]], [[ask, "1"]])
                            self.on_book_update(base, "gate", "futures", book)
                            _updated += 1

                _cycle += 1
                if _cycle <= 5 or _cycle % 50 == 0:
                    logger.info(
                        f"[GATE] futures ticker fallback cycle #{_cycle}: "
                        f"updated {_updated} stale books")

            await asyncio.sleep(1.5)

    async def _run_spot_ticker_fallback(self, symbols: List[str]):
        """Poll GET /api/v4/spot/tickers every 3s for stale spot books.

        One request returns bid/ask for ALL Gate spot pairs.
        Only updates books that are stale (>8s without WS update).
        """
        if not symbols:
            return

        sym_map = {}
        for base in symbols:
            s = self.format_symbol(base)["spot"]
            sym_map[s] = base

        await asyncio.sleep(3)
        logger.info(f"[GATE] spot ticker fallback started for {len(sym_map)} symbols")

        _cycle = 0
        while not self.shutdown.is_set():
            tickers = None
            try:
                tickers = await asyncio.to_thread(
                    _fetch_json_sync, GATE_SPOT_TICKERS)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

            if isinstance(tickers, list):
                _updated = 0
                for t in tickers:
                    pair = t.get("currency_pair", "")
                    base = sym_map.get(pair)
                    if not base:
                        continue
                    if not self.has_book(base, "spot"):
                        continue
                    book = self.get_book(base, "spot")
                    if not book.is_stale(1.5):
                        continue
                    bid = t.get("highest_bid")
                    ask = t.get("lowest_ask")
                    if bid and ask:
                        try:
                            bf, af = float(bid), float(ask)
                        except (ValueError, TypeError):
                            continue
                        if bf > 0 and af > 0:
                            book.apply_snapshot([[bid, "1"]], [[ask, "1"]])
                            self.on_book_update(base, "gate", "spot", book)
                            _updated += 1

                _cycle += 1
                if _cycle <= 5 or _cycle % 50 == 0:
                    logger.info(
                        f"[GATE] spot ticker fallback cycle #{_cycle}: "
                        f"updated {_updated} stale books")

            await asyncio.sleep(1.5)

    # Single-symbol fallback methods (used by subscribe_symbol)
    async def _single_spot(self, base, symbol):
        await self._batch_spot([base])

    async def _single_futures(self, base, contract):
        await self._batch_futures([base])
