"""Order book data structure with incremental update support."""

import time
import logging
from typing import Dict, List, Tuple, Optional

from .models import OrderBookSnapshot

logger = logging.getLogger(__name__)


def safe_float(v, default=None) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def is_zero_qty(q) -> bool:
    try:
        return float(q) == 0.0
    except (TypeError, ValueError):
        return str(q).strip() in {"0", "0.0", "0.00", "0.000"}


class OrderBook:
    """Maintains an order book with incremental update support.

    Stores bids/asks as price->qty maps and materializes sorted snapshots.
    Handles both full snapshots and incremental (delta) updates where qty=0
    means deletion.
    """

    def __init__(self, exchange: str, symbol: str, market_type: str,
                 depth_limit: int = 10, max_levels: int = 300):
        self.exchange = exchange
        self.symbol = symbol
        self.market_type = market_type
        self.depth_limit = depth_limit
        self.max_levels = max_levels
        self._bids: Dict[str, str] = {}  # price_str -> qty_str
        self._asks: Dict[str, str] = {}
        self._timestamp: float = 0.0
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def is_stale(self, max_age: float = 15.0) -> bool:
        """Check if the book hasn't been updated in max_age seconds."""
        if self._timestamp <= 0:
            return True
        return (time.time() - self._timestamp) > max_age

    def apply_snapshot(self, bids: list, asks: list) -> bool:
        """Replace the entire book with a full snapshot.

        Args:
            bids: List of [price, qty] or (price, qty)
            asks: List of [price, qty] or (price, qty)

        Returns:
            True if the resulting book is valid (has both sides).
        """
        new_bids: Dict[str, str] = {}
        new_asks: Dict[str, str] = {}

        for entry in bids:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            p, q = str(entry[0]).strip(), str(entry[1]).strip()
            if p and not is_zero_qty(q):
                new_bids[p] = q

        for entry in asks:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            p, q = str(entry[0]).strip(), str(entry[1]).strip()
            if p and not is_zero_qty(q):
                new_asks[p] = q

        if new_bids and new_asks:
            self._bids = new_bids
            self._asks = new_asks
            self._timestamp = time.time()
            self._connected = True
            return True

        self._connected = False
        return False

    def apply_delta(self, bids: list, asks: list) -> bool:
        """Apply incremental updates. qty=0 means delete that level.

        Returns:
            True if the resulting book is valid.
        """
        for entry in bids:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            p, q = str(entry[0]).strip(), str(entry[1]).strip()
            if not p:
                continue
            if is_zero_qty(q):
                self._bids.pop(p, None)
            else:
                self._bids[p] = q

        for entry in asks:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            p, q = str(entry[0]).strip(), str(entry[1]).strip()
            if not p:
                continue
            if is_zero_qty(q):
                self._asks.pop(p, None)
            else:
                self._asks[p] = q

        self._prune()

        if self._bids and self._asks:
            self._timestamp = time.time()
            self._connected = True
            return True

        self._connected = False
        return False

    def _prune(self):
        """Prune maps to max_levels to prevent unbounded growth."""
        if len(self._asks) > self.max_levels:
            items = sorted(self._asks.items(),
                           key=lambda kv: safe_float(kv[0], float("inf")))
            self._asks = dict(items[:self.max_levels])
        if len(self._bids) > self.max_levels:
            items = sorted(self._bids.items(),
                           key=lambda kv: safe_float(kv[0], 0.0), reverse=True)
            self._bids = dict(items[:self.max_levels])

    def snapshot(self) -> Optional[OrderBookSnapshot]:
        """Return an OrderBookSnapshot of current top-of-book levels."""
        if not self._bids or not self._asks:
            return None

        bid_items = sorted(self._bids.items(),
                           key=lambda kv: safe_float(kv[0], 0.0), reverse=True)
        ask_items = sorted(self._asks.items(),
                           key=lambda kv: safe_float(kv[0], float("inf")))

        bids = []
        for p, q in bid_items[:self.depth_limit]:
            pf, qf = safe_float(p), safe_float(q)
            if pf is not None and qf is not None and qf > 0:
                bids.append((pf, qf))

        asks = []
        for p, q in ask_items[:self.depth_limit]:
            pf, qf = safe_float(p), safe_float(q)
            if pf is not None and qf is not None and qf > 0:
                asks.append((pf, qf))

        if not bids or not asks:
            return None

        # Crossed book detection: best_bid >= best_ask means corrupted data
        if bids[0][0] >= asks[0][0]:
            return None

        return OrderBookSnapshot(
            exchange=self.exchange,
            symbol=self.symbol,
            market_type=self.market_type,
            bids=bids,
            asks=asks,
            timestamp=self._timestamp,
            connected=self._connected,
        )

    def clear(self):
        self._bids.clear()
        self._asks.clear()
        self._connected = False
        self._timestamp = 0.0
