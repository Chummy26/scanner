"""BingX exchange WebSocket adapter.

BingX Spot: wss://open-api-ws.bingx.com/market (gzip frames)
BingX Swap: wss://open-api-swap.bingx.com/swap-market (gzip frames)

A REST fallback polls /openApi/swap/v2/quote/ticker every 3s to keep
futures books fresh when WS connections cycle.

Uses shared connections: one WS per market type per batch.
"""

import asyncio
import gzip
import json
import logging
import uuid
from typing import Dict, List

import aiohttp
import websockets

from .base import BaseExchangeWS, json_loads_safe, chunk_list, _fetch_json_sync

logger = logging.getLogger(__name__)

BINGX_SPOT_WS = "wss://open-api-ws.bingx.com/market"
BINGX_SWAP_WS = "wss://open-api-swap.bingx.com/swap-market"
BINGX_FUTURES_TICKERS = "https://open-api.bingx.com/openApi/swap/v2/quote/ticker"
BINGX_SPOT_BOOK_TICKERS = "https://open-api.bingx.com/openApi/spot/v1/ticker/bookTicker"


def _decode_gzip(raw) -> str | None:
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, (bytes, bytearray)):
        return None
    try:
        return gzip.decompress(raw).decode("utf-8", errors="ignore")
    except Exception:
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return None


class BingxWS(BaseExchangeWS):
    name = "bingx"
    batch_size = 100  # Smaller batches — BingX disconnects when 150 subs sent too fast

    def format_symbol(self, base: str, quote: str = "USDT") -> Dict[str, str]:
        return {
            "spot": f"{base}-{quote}",
            "futures": f"{base}-{quote}",
        }

    async def subscribe_symbol(self, base: str, quote: str = "USDT"):
        await asyncio.gather(
            self._batch_session([base], "spot"),
            self._batch_session([base], "futures"),
            return_exceptions=True,
        )

    async def subscribe_symbol_batch(self, symbols):
        """Override to add futures ticker fallback."""
        spot_syms = list(getattr(self, "_spot_symbols", symbols))
        fut_syms = list(getattr(self, "_futures_symbols", symbols))
        tasks = []
        if self.spot_enabled:
            for i, chunk in enumerate(chunk_list(spot_syms, self.batch_size)):
                tasks.append(self._run_batch_loop(chunk, "spot", i, delay=i * 3.0))
            tasks.append(self._run_spot_ticker_fallback(spot_syms))
        if self.futures_enabled:
            for i, chunk in enumerate(chunk_list(fut_syms, self.batch_size)):
                tasks.append(self._run_batch_loop(chunk, "futures", i, delay=i * 3.0 + 1.5))
            tasks.append(self._run_futures_ticker_fallback(fut_syms))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _batch_session(self, symbols: List[str], market: str):
        ws_url = BINGX_SPOT_WS if market == "spot" else BINGX_SWAP_WS

        # BingX Spot only supports depth levels 20 and 100;
        # Futures supports 5, 10, 20, 50, 100.
        if market == "spot":
            level = 20 if self.depth_limit <= 20 else 100
        else:
            level = max(self.depth_limit, 10)

        # Build data_type -> base mapping
        dt_map = {}
        for base in symbols:
            sym = self.format_symbol(base)[market]
            if market == "futures":
                dt = f"{sym}@depth{level}@500ms"
            else:
                dt = f"{sym}@depth{level}"
            dt_map[dt] = base

        logger.info(f"[BINGX] Connecting {market} WS ({len(symbols)} symbols, depth{level})...")

        async with websockets.connect(
            ws_url, ping_interval=None, ping_timeout=None,
            max_size=16 * 1024 * 1024,
        ) as ws:
            # Subscribe all in one batch or small groups
            for dt in dt_map:
                sub_msg = {
                    "id": uuid.uuid4().hex[:12],
                    "dataType": dt,
                }
                # BingX Spot does NOT use reqType; only futures/swap needs it.
                if market == "futures":
                    sub_msg["reqType"] = "sub"
                await ws.send(json.dumps(sub_msg))
                await asyncio.sleep(0.05)  # Slower subscribe to avoid rate-limit

            logger.info(f"[BINGX] {market} subscribed {len(dt_map)} channels")
            _update_count = 0

            async for raw in ws:
                if self.shutdown.is_set():
                    return

                txt = _decode_gzip(raw)
                if not txt:
                    continue
                if txt.strip().lower() in ("ping", "pong"):
                    await ws.send("Pong")
                    continue

                msg = json_loads_safe(txt)
                if not isinstance(msg, dict):
                    continue

                # Identify which symbol this update is for
                data_type = msg.get("dataType", "")
                base = dt_map.get(data_type)
                if not base:
                    # Try matching by symbol prefix
                    for dt, b in dt_map.items():
                        if data_type.startswith(self.format_symbol(b)[market]):
                            base = b
                            break
                if not base:
                    continue

                # BingX may wrap data as a list (spot) or dict (futures).
                data = msg.get("data")
                if isinstance(data, list) and data:
                    data = data[0]
                if not isinstance(data, dict):
                    continue

                bids = data.get("bids") or []
                asks = data.get("asks") or []

                book = self.get_book(base, market)
                if book.apply_snapshot(bids, asks):
                    _update_count += 1
                    if _update_count <= 3 or _update_count % 500 == 0:
                        logger.info(
                            f"[BINGX] {market} book update #{_update_count}: "
                            f"{base} bids={len(bids)} asks={len(asks)}"
                        )
                    self.on_book_update(base, "bingx", market, book)

    async def _run_futures_ticker_fallback(self, symbols: List[str]):
        """Poll GET /openApi/swap/v2/quote/ticker every 3s.

        One request returns bid/ask for ALL ~500+ BingX futures contracts.
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
        logger.info(f"[BINGX] futures ticker fallback started for {len(sym_map)} symbols")

        _cycle = 0
        while not self.shutdown.is_set():
            payload = None
            try:
                payload = await asyncio.to_thread(
                    _fetch_json_sync, BINGX_FUTURES_TICKERS)
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
                    bid = t.get("bidPrice")
                    ask = t.get("askPrice")
                    if bid and ask:
                        try:
                            bf, af = float(bid), float(ask)
                        except (ValueError, TypeError):
                            continue
                        if bf > 0 and af > 0:
                            book.apply_snapshot(
                                [[bid, "1"]], [[ask, "1"]])
                            self.on_book_update(base, "bingx", "futures", book)
                            _updated += 1

                _cycle += 1
                if _cycle <= 5 or _cycle % 50 == 0:
                    logger.info(
                        f"[BINGX] futures ticker fallback cycle #{_cycle}: "
                        f"updated {_updated} stale books")

            await asyncio.sleep(1.5)

    async def _run_spot_ticker_fallback(self, symbols: List[str]):
        """Poll GET /openApi/spot/v1/ticker/bookTicker every 3s for stale spot books.

        One request returns bid/ask for ALL BingX spot pairs.
        Only updates books that are stale (>8s without WS update).
        """
        if not symbols:
            return

        sym_map = {}
        for base in symbols:
            s = self.format_symbol(base)["spot"]
            sym_map[s] = base

        await asyncio.sleep(3)
        logger.info(f"[BINGX] spot ticker fallback started for {len(sym_map)} symbols")

        _cycle = 0
        while not self.shutdown.is_set():
            payload = None
            try:
                payload = await asyncio.to_thread(
                    _fetch_json_sync, BINGX_SPOT_BOOK_TICKERS)
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
                    bid = t.get("bidPrice")
                    ask = t.get("askPrice")
                    if bid and ask:
                        try:
                            bf, af = float(bid), float(ask)
                        except (ValueError, TypeError):
                            continue
                        if bf > 0 and af > 0:
                            book.apply_snapshot(
                                [[bid, t.get("bidVolume", "1")]],
                                [[ask, t.get("askVolume", "1")]])
                            self.on_book_update(base, "bingx", "spot", book)
                            _updated += 1

                _cycle += 1
                if _cycle <= 5 or _cycle % 50 == 0:
                    logger.info(
                        f"[BINGX] spot ticker fallback cycle #{_cycle}: "
                        f"updated {_updated} stale books")

            await asyncio.sleep(1.5)
