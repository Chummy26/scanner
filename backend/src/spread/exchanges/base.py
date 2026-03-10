"""Base class for exchange WebSocket connections."""

import asyncio
import json
import gzip
import logging
import threading
import time
import urllib.request
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Callable, Any

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

from ..orderbook import OrderBook

logger = logging.getLogger(__name__)


def _fetch_json_sync(url: str, timeout: float = 10.0,
                     headers: Optional[Dict[str, str]] = None):
    """Synchronous HTTP GET returning parsed JSON.

    Designed to be called via ``asyncio.to_thread()`` so it runs in a
    worker thread and never blocks the event loop.  This avoids the
    event-loop starvation problem where thousands of WS messages prevent
    aiohttp coroutines from completing within timeout.
    """
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())

# Max symbols per single WS connection (conservative default)
DEFAULT_BATCH_SIZE = 100


def json_loads_safe(raw) -> Optional[dict]:
    """Safely parse a WS message to JSON dict."""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8", errors="ignore")
        except Exception:
            return None
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


async def periodic_send_json(ws, payload: dict, interval: float,
                              shutdown):
    """Send a JSON payload periodically (for keepalive pings)."""
    while not shutdown.is_set():
        try:
            await asyncio.sleep(interval)
            await ws.send(json.dumps(payload))
        except Exception:
            return


async def periodic_send_text(ws, text: str, interval: float,
                              shutdown):
    """Send a text payload periodically."""
    while not shutdown.is_set():
        try:
            await asyncio.sleep(interval)
            await ws.send(text)
        except Exception:
            return


def chunk_list(lst: list, size: int) -> list:
    """Split a list into chunks of given size."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


class BaseExchangeWS(ABC):
    """Base class for exchange WebSocket order book feeds.

    Subclasses implement spot/futures sessions for each exchange.
    The manager calls run() which loops reconnecting sessions.
    """

    name: str = ""
    batch_size: int = DEFAULT_BATCH_SIZE
    max_connections_per_market: int = 10  # Max WS connections per market type (spot/futures)

    def __init__(
        self,
        on_book_update: Callable[[str, str, str, OrderBook], None],
        depth_limit: int = 10,
        spot_enabled: bool = True,
        futures_enabled: bool = True,
    ):
        self.on_book_update = on_book_update
        self.depth_limit = depth_limit
        self.spot_enabled = bool(spot_enabled)
        self.futures_enabled = bool(futures_enabled)
        self.shutdown = threading.Event()
        # symbol -> OrderBook (keyed as "EXCHANGE:SYMBOL:MARKET")
        self._books: Dict[str, OrderBook] = {}
        self._connected_symbols: Set[str] = set()
        # Overflow symbols that didn't fit in WS capacity (filled by subscribe_symbol_batch)
        self._overflow_spot: List[str] = []
        self._overflow_futures: List[str] = []
        self.spot_feed_mode: str = "depth_ws"
        self.futures_feed_mode: str = "depth_ws"
        self._reconnect_counts: Dict[str, int] = {"spot": 0, "futures": 0}

    def get_feed_mode(self, market: str) -> str:
        if str(market).lower() == "futures":
            return str(getattr(self, "futures_feed_mode", "depth_ws") or "depth_ws")
        return str(getattr(self, "spot_feed_mode", "depth_ws") or "depth_ws")

    def get_reconnect_counts(self) -> Dict[str, int]:
        return {
            "spot": int(self._reconnect_counts.get("spot", 0) or 0),
            "futures": int(self._reconnect_counts.get("futures", 0) or 0),
        }

    def has_book(self, symbol: str, market_type: str) -> bool:
        """Return True if a book has been created (subscribed).

        Used by REST ticker fallback to decide which symbols to refresh.
        Does NOT require a prior WS update — books created via get_book()
        during WS subscription are eligible for REST refresh even if they
        never received a WS message (no trading activity).
        """
        key = f"{self.name}:{symbol}:{market_type}"
        return key in self._books

    def get_book(self, symbol: str, market_type: str) -> OrderBook:
        key = f"{self.name}:{symbol}:{market_type}"
        if key not in self._books:
            self._books[key] = OrderBook(
                exchange=self.name,
                symbol=symbol,
                market_type=market_type,
                depth_limit=self.depth_limit,
            )
        return self._books[key]

    @abstractmethod
    def format_symbol(self, base: str, quote: str = "USDT") -> Dict[str, str]:
        """Return exchange-specific symbol formats.

        Returns dict with keys like 'spot', 'futures' mapping to
        the exchange-specific symbol string.
        """
        ...

    @abstractmethod
    async def subscribe_symbol(self, base: str, quote: str = "USDT"):
        """Subscribe to order book updates for a symbol (both spot + futures)."""
        ...

    async def run(self, symbols: List[str],
                  spot_symbols: Optional[List[str]] = None,
                  futures_symbols: Optional[List[str]] = None):
        """Main loop: subscribe to all symbols and reconnect on failure.

        If spot_symbols / futures_symbols are given, only those bases are
        subscribed on the respective market type.  Otherwise falls back to
        the full *symbols* list for both.
        """
        self._spot_symbols = spot_symbols or symbols
        self._futures_symbols = futures_symbols or symbols
        while not self.shutdown.is_set():
            try:
                await self.subscribe_symbol_batch(symbols)
            except Exception as e:
                if not self.shutdown.is_set():
                    logger.warning(f"[{self.name.upper()}] Error: {e}")
                for sym in symbols:
                    for mt in ("spot", "futures"):
                        book = self.get_book(sym, mt)
                        book._connected = False
                        book._timestamp = 0.0
                await asyncio.sleep(2.0)

    async def subscribe_symbol_batch(self, symbols: List[str]):
        """Subscribe to multiple symbols using shared connections.

        Default: split into chunks and run batch sessions.
        Override for exchange-specific batch logic.
        """
        spot_syms = list(getattr(self, "_spot_symbols", symbols))
        fut_syms = list(getattr(self, "_futures_symbols", symbols))

        # Truncate symbols to respect max_connections_per_market;
        # overflow symbols are served by REST batch polling instead.
        self._overflow_spot = []
        self._overflow_futures = []
        if self.max_connections_per_market > 0:
            max_syms = self.max_connections_per_market * self.batch_size
            if len(spot_syms) > max_syms:
                self._overflow_spot = spot_syms[max_syms:]
                spot_syms = spot_syms[:max_syms]
                logger.warning(
                    f"[{self.name.upper()}] Spot: WS={len(spot_syms)}, "
                    f"REST overflow={len(self._overflow_spot)} "
                    f"({self.max_connections_per_market} conns x {self.batch_size} batch)"
                )
            if len(fut_syms) > max_syms:
                self._overflow_futures = fut_syms[max_syms:]
                fut_syms = fut_syms[:max_syms]
                logger.warning(
                    f"[{self.name.upper()}] Futures: WS={len(fut_syms)}, "
                    f"REST overflow={len(self._overflow_futures)} "
                    f"({self.max_connections_per_market} conns x {self.batch_size} batch)"
                )

        tasks = []
        if self.spot_enabled:
            for i, chunk in enumerate(chunk_list(spot_syms, self.batch_size)):
                tasks.append(self._run_batch_loop(chunk, "spot", i, delay=i * 3.0))
        if self.futures_enabled:
            for i, chunk in enumerate(chunk_list(fut_syms, self.batch_size)):
                tasks.append(self._run_batch_loop(chunk, "futures", i, delay=i * 3.0 + 1.5))

        # Start REST polling for overflow symbols (if subclass implements it)
        if self._overflow_spot or self._overflow_futures:
            tasks.append(self._rest_overflow_loop())

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_batch_loop(self, symbols: List[str], market: str, batch_id: int, delay: float = 0):
        """Reconnecting loop for a batch of symbols on one market type.

        Uses exponential backoff (1s -> 2s -> 4s -> max 15s) to avoid
        rate-limiting on repeated reconnections.  Backoff resets when
        a session survives for >30 s (healthy connection).
        """
        if delay > 0:
            await asyncio.sleep(delay)
        backoff = 1.0
        max_backoff = 15.0
        _reconn_count = 0
        while not self.shutdown.is_set():
            t0 = time.time()
            try:
                await self._batch_session(symbols, market)
            except Exception as e:
                if not self.shutdown.is_set():
                    logger.debug(f"[{self.name.upper()}] Batch {market}#{batch_id} error: {e}")
            # WS session ended — mark all batch books as disconnected so
            # the spread engine knows these books are no longer live.
            for sym in symbols:
                book = self.get_book(sym, market)
                if book._connected:
                    book._connected = False
                    self.on_book_update(sym, self.name, market, book)
            elapsed = time.time() - t0
            if elapsed > 30:
                backoff = 1.0
                _reconn_count = 0
            else:
                _reconn_count += 1
                self._reconnect_counts[market] = int(self._reconnect_counts.get(market, 0) or 0) + 1
                if not self.shutdown.is_set():
                    logger.info(
                        f"[{self.name.upper()}] {market}#{batch_id} reconnecting "
                        f"(#{_reconn_count}, {len(symbols)} syms, "
                        f"lived {elapsed:.0f}s, wait {backoff:.1f}s)")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    async def _batch_session(self, symbols: List[str], market: str):
        """Override this to implement shared-connection batch subscription."""
        # Fallback: individual subscriptions (not recommended for large lists)
        tasks = []
        for sym in symbols:
            tasks.append(self.subscribe_symbol(sym))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _rest_overflow_loop(self):
        """Poll REST batch endpoint for overflow symbols. Override per exchange."""
        # Default: no-op. Subclasses with batch ticker endpoints should override.
        pass

    def _apply_ticker_to_book(self, base: str, market: str,
                               bid_price, bid_qty, ask_price, ask_qty):
        """Create/update a book from ticker data and notify the engine."""
        book = self.get_book(base, market)
        bids = [[str(bid_price), str(bid_qty)]]
        asks = [[str(ask_price), str(ask_qty)]]
        if book.apply_snapshot(bids, asks):
            self.on_book_update(base, self.name, market, book)

    def stop(self):
        self.shutdown.set()
        loop = getattr(self, '_loop', None)
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._cancel_all_tasks)

    def _cancel_all_tasks(self):
        for task in asyncio.all_tasks(self._loop):
            task.cancel()

    def run_in_thread(self, symbols, spot_symbols=None, futures_symbols=None):
        """Run this exchange in a dedicated thread with its own event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(
                self.run(symbols, spot_symbols=spot_symbols,
                         futures_symbols=futures_symbols)
            )
        except Exception as e:
            if not self.shutdown.is_set():
                logger.error(f"[{self.name.upper()}] Thread crashed: {e}")
        finally:
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            self._loop.close()
