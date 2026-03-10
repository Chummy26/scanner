"""MEXC exchange WebSocket adapter for order book data.

MEXC Spot: wss://wbs-api.mexc.com/ws (protobuf depth channel)
MEXC Futures: wss://contract.mexc.com/edge (sub.depth, incremental JSON)

Uses shared connections: one WS per market type per batch.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import aiohttp
import websockets

from .base import BaseExchangeWS, json_loads_safe, periodic_send_json, chunk_list, _fetch_json_sync

logger = logging.getLogger(__name__)

MEXC_SPOT_WS_URL = "wss://wbs-api.mexc.com/ws"
MEXC_FUTURES_WS_URL = "wss://contract.mexc.com/edge"
MEXC_FUTURES_DEPTH_REST = "https://futures.mexc.com/api/v1/contract/depth/{symbol}?limit={limit}"
MEXC_FUTURES_TICKERS = "https://contract.mexc.com/api/v1/contract/ticker"

# Lazy protobuf loader
_PROTO_READY = False
_WRAPPER_CLS = None


def _init_protobuf() -> bool:
    global _PROTO_READY, _WRAPPER_CLS
    if _PROTO_READY:
        return _WRAPPER_CLS is not None

    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "TEAM OP (leitura) bot", "mock-server", "mexc_ws_proto"),
    ]
    # Also search in any installed ccxt package (pip-installed protobuf stubs)
    try:
        import ccxt
        ccxt_proto = os.path.join(os.path.dirname(ccxt.__file__), "protobuf", "mexc")
        if os.path.isdir(ccxt_proto):
            candidates.append(ccxt_proto)
    except Exception:
        pass
    for proto_dir in candidates:
        proto_dir = os.path.abspath(proto_dir)
        if os.path.isdir(proto_dir) and proto_dir not in sys.path:
            sys.path.insert(0, proto_dir)

    try:
        import PushDataV3ApiWrapper_pb2 as pb2
        _WRAPPER_CLS = pb2.PushDataV3ApiWrapper
        _PROTO_READY = True
        return True
    except Exception as e:
        logger.warning(f"[MEXC] Protobuf not available ({e}); spot depth disabled")
        _PROTO_READY = True
        return False


class MexcWS(BaseExchangeWS):
    name = "mexc"
    batch_size = 28  # MEXC limits ~30 subs per WS connection
    max_connections_per_market = 30  # 28 x 30 = 840 symbols max per market

    def format_symbol(self, base: str, quote: str = "USDT") -> Dict[str, str]:
        return {
            "spot": f"{base}{quote}",        # BTCUSDT
            "futures": f"{base}_{quote}",     # BTC_USDT
        }

    async def subscribe_symbol(self, base: str, quote: str = "USDT"):
        syms = self.format_symbol(base, quote)
        await asyncio.gather(
            self._batch_session([base], "spot"),
            self._batch_session([base], "futures"),
            return_exceptions=True,
        )

    async def subscribe_symbol_batch(self, symbols):
        """Override to add futures ticker fallback."""
        spot_syms = list(getattr(self, "_spot_symbols", symbols))
        fut_syms = list(getattr(self, "_futures_symbols", symbols))
        spot_mode = self.get_feed_mode("spot")
        futures_mode = self.get_feed_mode("futures")

        # Truncate to respect max_connections_per_market
        self._overflow_spot = []
        self._overflow_futures = []
        if self.max_connections_per_market > 0 and spot_mode == "depth_ws":
            max_syms = self.max_connections_per_market * self.batch_size
            if len(spot_syms) > max_syms:
                self._overflow_spot = spot_syms[max_syms:]
                spot_syms = spot_syms[:max_syms]
                logger.warning(
                    f"[MEXC] Spot: WS={len(spot_syms)}, "
                    f"REST overflow={len(self._overflow_spot)}")
        if self.max_connections_per_market > 0 and futures_mode == "depth_ws":
            max_syms = self.max_connections_per_market * self.batch_size
            if len(fut_syms) > max_syms:
                self._overflow_futures = fut_syms[max_syms:]
                fut_syms = fut_syms[:max_syms]
                logger.warning(
                    f"[MEXC] Futures: WS={len(fut_syms)}, "
                    f"REST overflow={len(self._overflow_futures)}")

        tasks = []
        if self.spot_enabled:
            all_spot = list(spot_syms) + list(self._overflow_spot)
            if spot_mode == "depth_ws":
                for i, chunk in enumerate(chunk_list(spot_syms, self.batch_size)):
                    tasks.append(self._run_batch_loop(chunk, "spot", i, delay=i * 3.0))
            else:
                for base in all_spot:
                    self.get_book(base, "spot")
            tasks.append(self._run_spot_ticker_fallback(all_spot))
        if self.futures_enabled:
            all_fut = list(fut_syms) + list(self._overflow_futures)
            if futures_mode == "depth_ws":
                for i, chunk in enumerate(chunk_list(fut_syms, self.batch_size)):
                    tasks.append(self._run_batch_loop(chunk, "futures", i, delay=i * 3.0 + 1.5))
            else:
                for base in all_fut:
                    self.get_book(base, "futures")
            tasks.append(self._run_futures_ticker_fallback(all_fut))
        if (self._overflow_spot or self._overflow_futures) and (spot_mode == "depth_ws" or futures_mode == "depth_ws"):
            tasks.append(self._rest_overflow_loop())
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _batch_session(self, symbols: List[str], market: str):
        if market == "spot":
            await self._batch_spot(symbols)
        else:
            await self._batch_futures(symbols)

    async def _batch_spot(self, symbols: List[str]):
        """MEXC spot: subscribe multiple protobuf depth channels on one WS."""
        if not _init_protobuf():
            await asyncio.sleep(60)
            return

        # Build channel -> base mapping
        channel_map = {}
        channels = []
        for base in symbols:
            sym = self.format_symbol(base)["spot"]
            ch = f"spot@public.limit.depth.v3.api.pb@{sym}@{self.depth_limit}"
            channel_map[ch] = base
            channels.append(ch)

        logger.info(f"[MEXC] Connecting spot WS ({len(symbols)} symbols)...")

        async with websockets.connect(
            MEXC_SPOT_WS_URL,
            ping_interval=20, ping_timeout=10,
            max_size=16 * 1024 * 1024,
        ) as ws:
            # Subscribe in chunks to avoid too-large messages
            for i in range(0, len(channels), 20):
                batch = channels[i:i+20]
                await ws.send(json.dumps({"method": "SUBSCRIPTION", "params": batch}))
                await asyncio.sleep(0.1)

            logger.info(f"[MEXC] spot subscribed {len(channels)} channels")
            _update_count = 0

            async for raw in ws:
                if self.shutdown.is_set():
                    return
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    msg = _WRAPPER_CLS.FromString(raw)
                except Exception:
                    continue

                ch = getattr(msg, "channel", "")
                base = channel_map.get(ch)
                if not base:
                    continue
                if not msg.HasField("publicLimitDepths"):
                    continue

                d = msg.publicLimitDepths
                bids = [[str(i.price), str(i.quantity)] for i in d.bids[:self.depth_limit]]
                asks = [[str(i.price), str(i.quantity)] for i in d.asks[:self.depth_limit]]

                book = self.get_book(base, "spot")
                if book.apply_snapshot(bids, asks):
                    _update_count += 1
                    if _update_count <= 3 or _update_count % 2000 == 0:
                        logger.info(
                            f"[MEXC] spot book update #{_update_count}: "
                            f"{base} bids={len(bids)} asks={len(asks)}"
                        )
                    self.on_book_update(base, "mexc", "spot", book)

    async def _batch_futures(self, symbols: List[str]):
        """MEXC futures: subscribe multiple sub.depth on one WS connection."""
        # Build symbol mapping
        sym_map = {}
        for base in symbols:
            s = self.format_symbol(base)["futures"]
            sym_map[s.upper()] = base

        # REST-seed ALL symbols so that WS deltas have a base snapshot.
        # Without seeding, sparse deltas (only bids or only asks) leave books
        # invalid and symbols never appear in opportunities.
        sem = asyncio.Semaphore(30)
        connector = aiohttp.TCPConnector(limit=30, ssl=False)
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=4), connector=connector,
            headers={"User-Agent": "Mozilla/5.0"})
        async def _seed(base, symbol):
            async with sem:
                book = self.get_book(base, "futures")
                url = MEXC_FUTURES_DEPTH_REST.format(symbol=symbol, limit=self.depth_limit)
                try:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            return
                        payload = await resp.json(content_type=None)
                except Exception:
                    return
                if not isinstance(payload, dict) or not payload.get("success"):
                    return
                data = payload.get("data") or {}
                bids = data.get("bids") or []
                asks = data.get("asks") or []
                book.apply_snapshot(bids, asks)

        try:
            seed_tasks = [_seed(base, self.format_symbol(base)["futures"]) for base in symbols]
            await asyncio.gather(*seed_tasks, return_exceptions=True)
        finally:
            await session.close()
        seeded = sum(1 for base in symbols if self.get_book(base, "futures").connected)
        logger.info(f"[MEXC] Futures seeded {seeded}/{len(symbols)} symbols via REST")

        async with websockets.connect(
            MEXC_FUTURES_WS_URL,
            ping_interval=None, ping_timeout=None,
            max_size=16 * 1024 * 1024,
        ) as ws:
            # Subscribe all symbols
            for base in symbols:
                s = self.format_symbol(base)["futures"]
                await ws.send(json.dumps({"method": "sub.depth", "param": {"symbol": s}}))
                await asyncio.sleep(0.05)

            ping_task = asyncio.create_task(
                periodic_send_json(ws, {"method": "ping"}, 15.0, self.shutdown))
            try:
                # Per-symbol throttle to reduce event-loop pressure.
                _last_ts: Dict[str, float] = {}
                _THROTTLE_SEC = 0.05  # max 20 updates/symbol/second

                async for raw in ws:
                    if self.shutdown.is_set():
                        return

                    msg = json_loads_safe(raw)
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("channel") != "push.depth":
                        continue

                    sym_raw = str(msg.get("symbol") or "").upper()
                    base = sym_map.get(sym_raw)
                    if not base:
                        continue

                    now_m = time.monotonic()
                    if (now_m - _last_ts.get(base, 0.0)) < _THROTTLE_SEC:
                        continue
                    _last_ts[base] = now_m

                    data = msg.get("data") or {}
                    bids_raw = data.get("bids") or []
                    asks_raw = data.get("asks") or []

                    book = self.get_book(base, "futures")
                    if book.apply_delta(bids_raw, asks_raw):
                        self.on_book_update(base, "mexc", "futures", book)
            finally:
                ping_task.cancel()

    async def _rest_overflow_loop(self):
        """Poll MEXC REST endpoints for overflow symbols not covered by WS."""
        overflow_spot = list(self._overflow_spot)
        overflow_fut = list(self._overflow_futures)
        if not overflow_spot and not overflow_fut:
            return

        # Build exchange-symbol -> base mappings
        spot_map = {}
        for base in overflow_spot:
            spot_map[self.format_symbol(base)["spot"]] = base
        fut_map = {}
        for base in overflow_fut:
            fut_map[self.format_symbol(base)["futures"].upper()] = base

        logger.info(
            f"[MEXC] REST overflow polling: {len(overflow_spot)} spot, "
            f"{len(overflow_fut)} futures"
        )

        await asyncio.sleep(10.0)  # Let WS connections establish first

        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"User-Agent": "Mozilla/5.0"},
        )
        try:
            while not self.shutdown.is_set():
                try:
                    # Spot: /api/v3/ticker/bookTicker returns ALL best bid/ask
                    if spot_map:
                        try:
                            async with session.get(
                                "https://api.mexc.com/api/v3/ticker/bookTicker"
                            ) as resp:
                                if resp.status == 200:
                                    data = await resp.json(content_type=None)
                                    if isinstance(data, list):
                                        _count = 0
                                        for t in data:
                                            sym = t.get("symbol", "")
                                            base = spot_map.get(sym)
                                            if not base:
                                                continue
                                            bp = t.get("bidPrice")
                                            ap = t.get("askPrice")
                                            if bp and ap:
                                                self._apply_ticker_to_book(
                                                    base, "spot", bp,
                                                    t.get("bidQty", "1"),
                                                    ap, t.get("askQty", "1"))
                                                _count += 1
                                        if _count:
                                            logger.debug(
                                                f"[MEXC] REST spot overflow: {_count} books updated"
                                            )
                        except Exception as e:
                            logger.debug(f"[MEXC] REST spot overflow error: {e}")

                    # Futures: /api/v1/contract/ticker returns ALL bid1/ask1
                    if fut_map:
                        try:
                            async with session.get(
                                "https://contract.mexc.com/api/v1/contract/ticker"
                            ) as resp:
                                if resp.status == 200:
                                    payload = await resp.json(content_type=None)
                                    tickers = []
                                    if isinstance(payload, dict):
                                        tickers = payload.get("data") or []
                                    elif isinstance(payload, list):
                                        tickers = payload
                                    _count = 0
                                    for t in tickers:
                                        sym = str(t.get("symbol", "")).upper()
                                        base = fut_map.get(sym)
                                        if not base:
                                            continue
                                        bid = t.get("bid1")
                                        ask = t.get("ask1")
                                        if bid and ask:
                                            self._apply_ticker_to_book(
                                                base, "futures",
                                                bid, "1", ask, "1")
                                            _count += 1
                                    if _count:
                                        logger.debug(
                                            f"[MEXC] REST futures overflow: {_count} books updated"
                                        )
                        except Exception as e:
                            logger.debug(f"[MEXC] REST futures overflow error: {e}")
                except Exception as e:
                    logger.debug(f"[MEXC] REST overflow session error: {e}")

                await asyncio.sleep(2.0)
        finally:
            await session.close()

    async def _seed_futures_rest(self, symbol: str, book):
        url = MEXC_FUTURES_DEPTH_REST.format(symbol=symbol, limit=self.depth_limit)
        try:
            payload = await asyncio.to_thread(
                _fetch_json_sync, url, 4.0,
                {"User-Agent": "Mozilla/5.0"})
        except Exception:
            return

        if not isinstance(payload, dict) or not payload.get("success"):
            return
        data = payload.get("data") or {}
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        book.apply_snapshot(bids, asks)

    async def _run_futures_ticker_fallback(self, symbols: List[str]):
        """Poll GET /api/v1/contract/ticker every 3s.

        One request returns bid/ask for ALL ~120+ MEXC futures contracts.
        Only updates books that are stale (>8s without WS update).
        Uses a fresh session per request to avoid session corruption.
        """
        if not symbols:
            return

        sym_map = {}
        for base in symbols:
            s = self.format_symbol(base)["futures"].upper()
            sym_map[s] = base

        await asyncio.sleep(3)
        logger.info(f"[MEXC] futures ticker fallback started for {len(sym_map)} symbols")

        _cycle = 0
        while not self.shutdown.is_set():
            tickers = None
            try:
                payload = await asyncio.to_thread(
                    _fetch_json_sync, MEXC_FUTURES_TICKERS, 10.0,
                    {"User-Agent": "Mozilla/5.0"})
                if isinstance(payload, dict):
                    tickers = payload.get("data") or []
                elif isinstance(payload, list):
                    tickers = payload
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

            if tickers:
                _updated = 0
                for t in tickers:
                    sym = str(t.get("symbol", "")).upper()
                    base = sym_map.get(sym)
                    if not base:
                        continue
                    if not self.has_book(base, "futures"):
                        continue
                    book = self.get_book(base, "futures")
                    if not book.is_stale(1.5):
                        continue
                    bid = t.get("bid1")
                    ask = t.get("ask1")
                    if bid and ask:
                        try:
                            bf, af = float(bid), float(ask)
                        except (ValueError, TypeError):
                            continue
                        if bf > 0 and af > 0:
                            book.apply_snapshot(
                                [[str(bid), "1"]], [[str(ask), "1"]])
                            self.on_book_update(base, "mexc", "futures", book)
                            _updated += 1

                _cycle += 1
                if _cycle <= 5 or _cycle % 50 == 0:
                    logger.info(
                        f"[MEXC] futures ticker fallback cycle #{_cycle}: "
                        f"updated {_updated} stale books")

            await asyncio.sleep(1.5)

    async def _run_spot_ticker_fallback(self, symbols: List[str]):
        """Poll GET /api/v3/ticker/bookTicker every 3s for stale spot books.

        One request returns bid/ask for ALL MEXC spot pairs.
        Only updates books that are stale (>8s without WS update).
        Uses asyncio.to_thread() to avoid event-loop starvation.
        """
        if not symbols:
            return

        sym_map = {}
        for base in symbols:
            s = self.format_symbol(base)["spot"]
            sym_map[s] = base

        await asyncio.sleep(3)
        logger.info(f"[MEXC] spot ticker fallback started for {len(sym_map)} symbols")

        _cycle = 0
        while not self.shutdown.is_set():
            tickers = None
            try:
                tickers = await asyncio.to_thread(
                    _fetch_json_sync,
                    "https://api.mexc.com/api/v3/ticker/bookTicker",
                    10.0, {"User-Agent": "Mozilla/5.0"})
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

            if isinstance(tickers, list):
                _updated = 0
                for t in tickers:
                    sym = t.get("symbol", "")
                    base = sym_map.get(sym)
                    if not base:
                        continue
                    book = self.get_book(base, "spot")
                    if not book.is_stale(1.5):
                        continue
                    bid = t.get("bidPrice")
                    ask = t.get("askPrice")
                    if bid and ask:
                        try:
                            bf, af = float(bid), float(ask)
                        except (ValueError, TypeError):
                            continue
                        if bf > 0 and af > 0:
                            book.apply_snapshot(
                                [[bid, t.get("bidQty", "1")]],
                                [[ask, t.get("askQty", "1")]])
                            self.on_book_update(base, "mexc", "spot", book)
                            _updated += 1

                _cycle += 1
                if _cycle <= 5 or _cycle % 50 == 0:
                    logger.info(
                        f"[MEXC] spot ticker fallback cycle #{_cycle}: "
                        f"updated {_updated} stale books")

            await asyncio.sleep(1.5)
