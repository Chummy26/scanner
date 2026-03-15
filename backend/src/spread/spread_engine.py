"""High-performance spread calculation engine.

Receives order book snapshots from all exchanges in real-time and calculates
spread opportunities between all possible pairs of exchanges.

Key optimisations over the naive O(N²) approach:
  1. Incremental calculation — only recalculates symbols that received book
     updates since the last cycle (dirty-symbol tracking).
  2. Pre-grouped market types — separates spot/futures sources so the inner
     loop never needs to check or skip invalid combinations.
  3. Inline top-of-book — when min_volume_usd=0 (default), reads bids[0][0]
     directly instead of calling a function.
  4. Deferred tracker callbacks — collects spread records and batch-calls
     on_spread after the hot loop.
"""

import logging
import math
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .models import OrderBookSnapshot, SpreadOpportunity, SpreadConfig
from .orderbook import OrderBook

logger = logging.getLogger(__name__)

# How often to force a full recalculation (catches stale books, symbol removal, etc.)
# Higher values improve average cycle time by keeping most cycles incremental.
# Dirty-symbol tracking handles >90% of updates; full-recalc is only needed to
# expire stale books and detect symbol removals, which change slowly.
_FULL_RECALC_EVERY = 80  # ~12s at 0.15s broadcast interval

# Maximum dirty symbols to recalculate per incremental cycle.
# Caps per-cycle work so broadcast stays near real-time.
# Remaining dirty symbols carry over to the next cycle.
# With thread-per-exchange, calc ~25ms for 500 symbols (within 150ms budget).
_MAX_DIRTY_PER_CYCLE = 500


def _valid_tracker_spread(value: float, *, max_spread: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and abs(numeric) <= float(max_spread)


def _top_level_qty(levels: list) -> float:
    if not levels:
        return 0.0
    try:
        return float(levels[0][1])
    except (IndexError, TypeError, ValueError):
        return 0.0


class SpreadEngine:
    """Calculates spread arbitrage opportunities from order book data.

    For each symbol, compares best ask (buy price) on one exchange against
    best bid (sell price) on another exchange, for all valid pair combinations.

    Entry spread: (sell_bid / buy_ask - 1) * 100
    Exit spread:  (buy_bid / sell_ask - 1) * 100 (reverse direction)
    """

    def __init__(self, config: Optional[SpreadConfig] = None):
        self.config = config or SpreadConfig()
        self.perf_monitor = None
        self._lock = threading.Lock()
        # symbol -> exchange -> market_type -> OrderBookSnapshot
        self._snapshots: Dict[str, Dict[str, Dict[str, OrderBookSnapshot]]] = {}
        # Current calculated opportunities (sorted, final output)
        self._opportunities: List[SpreadOpportunity] = []
        self._last_calc_time: float = 0.0
        self._last_calc_ms: float = 0.0
        # Update counting for health monitoring
        self._update_counts: Dict[str, int] = {}
        # --- Incremental calc state ---
        self._dirty_symbols: Set[str] = set()
        # Per-symbol cached opportunity lists — key = symbol
        self._opps_by_symbol: Dict[str, List[SpreadOpportunity]] = {}
        self._cycle_count: int = 0
        self._last_stale_count: int = 0

    def set_perf_monitor(self, monitor):
        self.perf_monitor = monitor
        return monitor

    # ------------------------------------------------------------------
    # Public: book updates
    # ------------------------------------------------------------------

    def update_book(self, symbol: str, exchange: str, market_type: str,
                    book: OrderBook):
        """Called when an exchange sends a new order book update."""
        snap = book.snapshot()
        if not snap:
            return

        with self._lock:
            if symbol not in self._snapshots:
                self._snapshots[symbol] = {}
            if exchange not in self._snapshots[symbol]:
                self._snapshots[symbol][exchange] = {}
            self._snapshots[symbol][exchange][market_type] = snap
            # Track update counts for health monitoring
            key = f"{exchange}:{market_type}"
            self._update_counts[key] = self._update_counts.get(key, 0) + 1
            # Mark symbol as dirty for incremental recalc
            self._dirty_symbols.add(symbol)

    def get_and_reset_update_counts(self) -> Dict[str, int]:
        """Return update counts per exchange:market and reset counters."""
        with self._lock:
            counts = dict(self._update_counts)
            self._update_counts.clear()
        return counts

    # ------------------------------------------------------------------
    # Public: snapshot queries (used by export API)
    # ------------------------------------------------------------------

    def get_snapshot(
        self,
        symbol: str,
        exchange: str,
        market_type: str,
    ) -> Optional[OrderBookSnapshot]:
        """Return the latest snapshot for a given (symbol, exchange, market)."""
        sym = (symbol or "").upper()
        ex = (exchange or "").lower()
        mt = (market_type or "").lower()
        with self._lock:
            snap = (
                self._snapshots.get(sym, {})
                .get(ex, {})
                .get(mt)
            )
            if not snap:
                return None
            return OrderBookSnapshot(
                exchange=snap.exchange,
                symbol=snap.symbol,
                market_type=snap.market_type,
                bids=list(snap.bids),
                asks=list(snap.asks),
                timestamp=snap.timestamp,
                connected=snap.connected,
            )

    # ------------------------------------------------------------------
    # Core: spread calculation
    # ------------------------------------------------------------------

    def calculate_all(
        self,
        on_spread: Optional[Callable[[str, str, str, str, str, float, float], None]] = None,
        *,
        record_sink: Optional[list] = None,
        rejection_sink: Optional[list] = None,
    ) -> List[SpreadOpportunity]:
        """Calculate spreads across all symbol/exchange/market combinations.

        Uses incremental calculation: only recalculates symbols whose books
        changed since the last cycle.  Every ``_FULL_RECALC_EVERY`` cycles a
        full recalculation is performed to catch stale-book expiry and symbol
        removals.

        Returns a sorted list of SpreadOpportunity objects.
        """
        now = time.time()
        perf_started = time.perf_counter()
        stale_threshold = 8.0  # Disconnected books older than 8s are unreliable for arbitrage
        notional_usd = float(getattr(self.config, "min_volume_usd", 0.0) or 0.0)
        min_spread = self.config.min_spread_pct
        max_spread = self.config.max_spread_pct
        use_top_of_book = (notional_usd <= 0)

        self._cycle_count += 1
        force_full = (self._cycle_count % _FULL_RECALC_EVERY == 0)

        # ------ snapshot copy under lock (minimal time) ------
        lock_started = time.perf_counter()
        with self._lock:
            if force_full:
                self._dirty_symbols.clear()
                snapshot_copy = {
                    sym: {ex: dict(mkts) for ex, mkts in exs.items()}
                    for sym, exs in self._snapshots.items()
                }
                symbols_to_calc = set(snapshot_copy.keys())
            else:
                dirty = self._dirty_symbols
                if len(dirty) > _MAX_DIRTY_PER_CYCLE:
                    # Take a subset; leave the rest for next cycle
                    symbols_to_calc = set()
                    for sym in dirty:
                        symbols_to_calc.add(sym)
                        if len(symbols_to_calc) >= _MAX_DIRTY_PER_CYCLE:
                            break
                    dirty -= symbols_to_calc
                else:
                    symbols_to_calc = dirty.copy()
                    dirty.clear()

                snapshot_copy = {}
                for sym in symbols_to_calc:
                    exs = self._snapshots.get(sym)
                    if exs:
                        snapshot_copy[sym] = {
                            ex: dict(mkts) for ex, mkts in exs.items()
                        }
        lock_finished = time.perf_counter()

        # ------ per-symbol calculation (outside lock) ------
        _stale_count = 0
        spread_records: list = []  # deferred on_spread payloads

        calc_started = time.perf_counter()
        for symbol in symbols_to_calc:
            exchanges = snapshot_copy.get(symbol)
            if not exchanges:
                self._opps_by_symbol.pop(symbol, None)
                continue

            opps, stale_n, records, rejections = self._calc_symbol(
                symbol, exchanges, now, stale_threshold,
                use_top_of_book, notional_usd,
                min_spread,
                max_spread,
                (on_spread is not None) or (record_sink is not None) or (rejection_sink is not None),
            )
            _stale_count += stale_n
            self._opps_by_symbol[symbol] = opps
            if records:
                spread_records.extend(records)
            if rejection_sink is not None and rejections:
                rejection_sink.extend(rejections)
        calc_finished = time.perf_counter()

        # ------ update cached (non-dirty) opps in incremental mode ------
        aging_started = time.perf_counter()
        if not force_full and self._last_calc_time > 0:
            elapsed = now - self._last_calc_time
            # Snapshot pending dirty symbols — these have fresh WS data
            # waiting for a recalc slot.  Do NOT age them: their underlying
            # books are being updated, they just haven't been processed yet.
            with self._lock:
                pending_dirty = set(self._dirty_symbols)
            for sym, opps in list(self._opps_by_symbol.items()):
                if sym in symbols_to_calc:
                    continue
                if sym in pending_dirty:
                    # Fresh data pending — keep opps as-is, don't age
                    continue
                symbol_snapshots = self._snapshots.get(sym, {})
                # Only age symbols that truly have no new data
                kept: List[SpreadOpportunity] = []
                for opp in opps:
                    opp.buy_book_age += elapsed
                    opp.sell_book_age += elapsed
                    opp.timestamp = now
                    buy_snap = (
                        symbol_snapshots.get(opp.buy_exchange, {})
                        .get(opp.buy_market_type)
                    )
                    sell_snap = (
                        symbol_snapshots.get(opp.sell_exchange, {})
                        .get(opp.sell_market_type)
                    )
                    if opp.buy_book_age < stale_threshold and opp.sell_book_age < stale_threshold:
                        kept.append(opp)
                if kept:
                    self._opps_by_symbol[sym] = kept
                else:
                    self._opps_by_symbol.pop(sym, None)

        # On full recalc, prune symbols no longer in snapshots
        if force_full:
            live = set(snapshot_copy.keys())
            for sym in list(self._opps_by_symbol):
                if sym not in live:
                    del self._opps_by_symbol[sym]
        aging_finished = time.perf_counter()

        if record_sink is not None and spread_records:
            record_sink.extend(spread_records)

        # ------ batch on_spread callbacks ------
        callbacks_started = time.perf_counter()
        if on_spread and spread_records:
            for rec in spread_records:
                try:
                    on_spread(*rec)
                except Exception:
                    pass
        callbacks_finished = time.perf_counter()

        # ------ assemble final sorted list ------
        sort_started = time.perf_counter()
        all_opps: List[SpreadOpportunity] = []
        for opps in self._opps_by_symbol.values():
            all_opps.extend(opps)
        all_opps.sort(key=lambda o: o.entry_spread_pct, reverse=True)
        sort_finished = time.perf_counter()

        self._opportunities = all_opps
        self._last_calc_ms = round((sort_finished - perf_started) * 1000.0, 6)
        self._last_calc_time = now
        self._last_stale_count = _stale_count
        self._last_dirty_count = len(symbols_to_calc)
        if self.perf_monitor is not None:
            self.perf_monitor.record_scanner_cycle(
                {
                    "kind": "spread_engine",
                    "total_ms": round((sort_finished - perf_started) * 1000.0, 6),
                    "snapshot_lock_ms": round((lock_finished - lock_started) * 1000.0, 6),
                    "symbol_calc_ms": round((calc_finished - calc_started) * 1000.0, 6),
                    "aging_ms": round((aging_finished - aging_started) * 1000.0, 6),
                    "callback_ms": round((callbacks_finished - callbacks_started) * 1000.0, 6),
                    "sort_ms": round((sort_finished - sort_started) * 1000.0, 6),
                    "symbols_to_calc_count": len(symbols_to_calc),
                    "dirty_symbols_pending": len(self._dirty_symbols),
                    "force_full": bool(force_full),
                    "stale_count": int(_stale_count),
                    "spread_record_count": len(spread_records),
                    "opportunity_count": len(all_opps),
                }
            )
        return all_opps

    # ------------------------------------------------------------------

    @staticmethod
    def _validate_cross_exchange_price(
        symbol: str,
        exchanges: Dict[str, Dict[str, OrderBookSnapshot]],
        *,
        max_divergence_pct: float = 30.0,
    ) -> dict[str, set[str]]:
        mids: dict[str, float] = {}
        for exchange, markets in exchanges.items():
            for market_type, snap in markets.items():
                best_bid = float(getattr(snap, "best_bid", 0.0) or 0.0)
                best_ask = float(getattr(snap, "best_ask", 0.0) or 0.0)
                if best_bid > 0.0 and best_ask > 0.0:
                    mids[f"{exchange}:{market_type}"] = (best_bid + best_ask) / 2.0
        if len(mids) < 3:
            return {"flagged_sources": set(), "flagged_exchanges": set()}
        ordered = sorted(mids.values())
        median_price = ordered[len(ordered) // 2]
        if median_price <= 0.0:
            return {"flagged_sources": set(), "flagged_exchanges": set()}
        flagged_sources = {
            key
            for key, mid in mids.items()
            if abs(float(mid) - float(median_price)) / float(median_price) * 100.0 > float(max_divergence_pct)
        }
        flagged_exchanges = {str(key.split(":", 1)[0]) for key in flagged_sources}
        return {
            "flagged_sources": flagged_sources,
            "flagged_exchanges": flagged_exchanges,
        }

    @staticmethod
    def _calc_symbol(
        symbol: str,
        exchanges: Dict[str, Dict[str, OrderBookSnapshot]],
        now: float,
        stale_threshold: float,
        use_top_of_book: bool,
        notional_usd: float,
        min_spread: float,
        max_spread: float,
        collect_records: bool,
    ) -> Tuple[List[SpreadOpportunity], int, list, list]:
        """Calculate all opportunities for a single symbol.

        Returns (opportunities, stale_count, spread_records).
        """
        stale_n = 0
        # Pre-group sources by market type
        # Each entry: (exchange, snap, bid_px, ask_px, bid_qty, ask_qty)
        spot_src: List[Tuple[str, OrderBookSnapshot, float, float, float, float]] = []
        fut_src: List[Tuple[str, OrderBookSnapshot, float, float, float, float]] = []

        opps: List[SpreadOpportunity] = []
        records: list = []
        rejections: list = []

        def _record_rejection(
            buy_ex: str,
            buy_market_type: str,
            sell_ex: str,
            sell_market_type: str,
            *,
            invalid_fields: list[str],
            reason: str,
            affected_exchanges: list[str] | None = None,
        ) -> None:
            if not collect_records:
                return
            rejections.append(
                {
                    "symbol": symbol,
                    "buy_ex": buy_ex,
                    "buy_mt": buy_market_type,
                    "sell_ex": sell_ex,
                    "sell_mt": sell_market_type,
                    "invalid_fields": list(invalid_fields),
                    "reason": str(reason),
                    "affected_exchanges": list(affected_exchanges or []),
                }
            )

        def _record_source_rejection(exchange: str, market_type: str, reason: str) -> None:
            _record_rejection(
                exchange,
                market_type,
                exchange,
                market_type,
                invalid_fields=["entry", "exit"],
                reason=reason,
                affected_exchanges=[exchange],
            )

        def _extract_book_view(snap: OrderBookSnapshot) -> tuple[tuple[float, float, float, float] | None, str | None]:
            bids = list(getattr(snap, "bids", []) or [])
            asks = list(getattr(snap, "asks", []) or [])
            if not bids or not asks:
                return None, "missing_top_of_book"
            bid_qty = _top_level_qty(bids)
            ask_qty = _top_level_qty(asks)
            if use_top_of_book:
                try:
                    bid_px = float(bids[0][0])
                    ask_px = float(asks[0][0])
                except (IndexError, TypeError, ValueError):
                    return None, "missing_top_of_book"
            else:
                bid_px = _effective_price(bids, notional_usd)
                ask_px = _effective_price(asks, notional_usd)
            if bid_px <= 0.0 or ask_px <= 0.0:
                return None, "non_positive_book_px"
            if bid_px > ask_px:
                return None, "crossed_book"
            return (bid_px, ask_px, bid_qty, ask_qty), None

        def _quality_hints(
            *,
            buy_snap: OrderBookSnapshot,
            sell_snap: OrderBookSnapshot,
            buy_price: float,
            buy_qty: float,
            sell_price: float,
            sell_qty: float,
        ) -> dict[str, Any]:
            buy_notional = max(0.0, float(buy_price) * max(float(buy_qty), 0.0))
            sell_notional = max(0.0, float(sell_price) * max(float(sell_qty), 0.0))
            depth_floor = max(float(notional_usd or 0.0), 25.0)
            age_asymmetry = abs((now - float(buy_snap.timestamp)) - (now - float(sell_snap.timestamp)))
            return {
                "buy_top_of_book_notional_usd": round(buy_notional, 6),
                "sell_top_of_book_notional_usd": round(sell_notional, 6),
                "thin_book_warning": bool(buy_notional < depth_floor or sell_notional < depth_floor),
                "book_age_asymmetry_sec": round(age_asymmetry, 6),
                "book_age_asymmetry_warning": bool(age_asymmetry > max(1.0, float(stale_threshold) * 0.25)),
            }

        for exchange, markets in exchanges.items():
            for market_type, snap in markets.items():
                age = now - snap.timestamp
                # Keep quiet books visible while the feed itself is still
                # Do not surface opportunities backed by very old books even
                # if the feed still appears logically connected.
                if age >= stale_threshold:
                    stale_n += 1
                    continue

                book_view, rejection_reason = _extract_book_view(snap)
                if book_view is None:
                    _record_source_rejection(exchange, market_type, str(rejection_reason or "missing_top_of_book"))
                    continue
                bid_px, ask_px, bid_qty, ask_qty = book_view

                entry = (exchange, snap, bid_px, ask_px, bid_qty, ask_qty)
                if market_type == "spot":
                    spot_src.append(entry)
                elif market_type == "futures":
                    fut_src.append(entry)
        cross_exchange_validation = SpreadEngine._validate_cross_exchange_price(symbol, exchanges)
        flagged_sources = set(cross_exchange_validation.get("flagged_sources") or set())

        # ---------- spot_futures: buy spot, sell futures ----------
        for buy_ex, buy_snap, buy_bid, buy_ask, buy_bid_qty, buy_ask_qty in spot_src:
            for sell_ex, sell_snap, sell_bid, sell_ask, sell_bid_qty, sell_ask_qty in fut_src:
                if f"{buy_ex}:spot" in flagged_sources or f"{sell_ex}:futures" in flagged_sources:
                    _record_rejection(
                        buy_ex,
                        "spot",
                        sell_ex,
                        "futures",
                        invalid_fields=[],
                        reason="cross_exchange_price_divergence",
                        affected_exchanges=[
                            ex
                            for ex in (buy_ex if f"{buy_ex}:spot" in flagged_sources else "", sell_ex if f"{sell_ex}:futures" in flagged_sources else "")
                            if ex
                        ],
                    )
                    continue
                if buy_ask <= 0:
                    continue
                entry_spread = (sell_bid / buy_ask - 1.0) * 100.0

                if buy_bid <= 0 or sell_ask <= 0:
                    _record_rejection(
                        buy_ex,
                        "spot",
                        sell_ex,
                        "futures",
                        invalid_fields=["exit"],
                        reason="exit_spread_uncomputable",
                    )
                    continue
                exit_spread = (buy_bid / sell_ask - 1.0) * 100.0
                if not _valid_tracker_spread(entry_spread, max_spread=max_spread):
                    _record_rejection(
                        buy_ex,
                        "spot",
                        sell_ex,
                        "futures",
                        invalid_fields=["entry"],
                        reason="entry_spread_invalid",
                    )
                    continue
                if not _valid_tracker_spread(exit_spread, max_spread=max_spread):
                    _record_rejection(
                        buy_ex,
                        "spot",
                        sell_ex,
                        "futures",
                        invalid_fields=["exit"],
                        reason="exit_spread_invalid",
                    )
                    continue

                if collect_records:
                    records.append((
                        symbol, buy_ex, "spot",
                        sell_ex, "futures",
                        entry_spread, exit_spread,
                    ))

                if entry_spread >= min_spread:
                    opp = SpreadOpportunity(
                        asset=symbol, arb_type="spot_futures",
                        buy_exchange=buy_ex, sell_exchange=sell_ex,
                        buy_market_type="spot", sell_market_type="futures",
                        buy_price=buy_ask, sell_price=sell_bid,
                        entry_spread_pct=entry_spread,
                        exit_spread_pct=exit_spread,
                        timestamp=now,
                        buy_symbol=buy_snap.symbol,
                        sell_symbol=sell_snap.symbol,
                        buy_book_age=now - buy_snap.timestamp,
                        sell_book_age=now - sell_snap.timestamp,
                    )
                    opp.quality_hints = _quality_hints(
                        buy_snap=buy_snap,
                        sell_snap=sell_snap,
                        buy_price=buy_ask,
                        buy_qty=buy_ask_qty,
                        sell_price=sell_bid,
                        sell_qty=sell_bid_qty,
                    )
                    opps.append(opp)

        # ---------- futures_futures: different exchanges ----------
        n_fut = len(fut_src)
        for i in range(n_fut):
            buy_ex, buy_snap, buy_bid, buy_ask, buy_bid_qty, buy_ask_qty = fut_src[i]
            if buy_ask <= 0:
                continue
            for j in range(n_fut):
                if i == j:
                    continue
                sell_ex, sell_snap, sell_bid, sell_ask, sell_bid_qty, sell_ask_qty = fut_src[j]
                if buy_ex == sell_ex:
                    continue
                if f"{buy_ex}:futures" in flagged_sources or f"{sell_ex}:futures" in flagged_sources:
                    _record_rejection(
                        buy_ex,
                        "futures",
                        sell_ex,
                        "futures",
                        invalid_fields=[],
                        reason="cross_exchange_price_divergence",
                        affected_exchanges=[
                            ex
                            for ex in (buy_ex if f"{buy_ex}:futures" in flagged_sources else "", sell_ex if f"{sell_ex}:futures" in flagged_sources else "")
                            if ex
                        ],
                    )
                    continue

                entry_spread = (sell_bid / buy_ask - 1.0) * 100.0

                if buy_bid <= 0 or sell_ask <= 0:
                    _record_rejection(
                        buy_ex,
                        "futures",
                        sell_ex,
                        "futures",
                        invalid_fields=["exit"],
                        reason="exit_spread_uncomputable",
                    )
                    continue
                exit_spread = (buy_bid / sell_ask - 1.0) * 100.0
                if not _valid_tracker_spread(entry_spread, max_spread=max_spread):
                    _record_rejection(
                        buy_ex,
                        "futures",
                        sell_ex,
                        "futures",
                        invalid_fields=["entry"],
                        reason="entry_spread_invalid",
                    )
                    continue
                if not _valid_tracker_spread(exit_spread, max_spread=max_spread):
                    _record_rejection(
                        buy_ex,
                        "futures",
                        sell_ex,
                        "futures",
                        invalid_fields=["exit"],
                        reason="exit_spread_invalid",
                    )
                    continue

                if collect_records:
                    records.append((
                        symbol, buy_ex, "futures",
                        sell_ex, "futures",
                        entry_spread, exit_spread,
                    ))

                if entry_spread >= min_spread:
                    opp = SpreadOpportunity(
                        asset=symbol, arb_type="futures_futures",
                        buy_exchange=buy_ex, sell_exchange=sell_ex,
                        buy_market_type="futures", sell_market_type="futures",
                        buy_price=buy_ask, sell_price=sell_bid,
                        entry_spread_pct=entry_spread,
                        exit_spread_pct=exit_spread,
                        timestamp=now,
                        buy_symbol=buy_snap.symbol,
                        sell_symbol=sell_snap.symbol,
                        buy_book_age=now - buy_snap.timestamp,
                        sell_book_age=now - sell_snap.timestamp,
                    )
                    opp.quality_hints = _quality_hints(
                        buy_snap=buy_snap,
                        sell_snap=sell_snap,
                        buy_price=buy_ask,
                        buy_qty=buy_ask_qty,
                        sell_price=sell_bid,
                        sell_qty=sell_bid_qty,
                    )
                    opps.append(opp)

        return opps, stale_n, records, rejections

    # ------------------------------------------------------------------
    # Public: opportunity queries
    # ------------------------------------------------------------------

    def get_opportunities(self) -> List[SpreadOpportunity]:
        """Return the last calculated opportunities."""
        return list(self._opportunities)

    def get_opportunities_for_symbol(self, symbol: str) -> List[SpreadOpportunity]:
        """Return opportunities for a specific symbol."""
        return [o for o in self._opportunities if o.asset == symbol]

    def get_connected_exchanges(self) -> Dict[str, Dict[str, bool]]:
        """Return which exchanges have active order book data."""
        result: Dict[str, Dict[str, bool]] = {}
        now = time.time()
        with self._lock:
            for symbol, exchanges in self._snapshots.items():
                for exchange, markets in exchanges.items():
                    if exchange not in result:
                        result[exchange] = {"spot": False, "futures": False}
                    for market_type, snap in markets.items():
                        if snap.is_valid() and (now - snap.timestamp) < 120.0:
                            result[exchange][market_type] = True
        return result

    def get_snapshot_summary(self) -> Dict:
        """Return a summary of current state for debugging."""
        with self._lock:
            symbols_count = len(self._snapshots)
            total_books = sum(
                len(markets)
                for exchanges in self._snapshots.values()
                for markets in exchanges.values()
            )
        return {
            "symbols_tracked": symbols_count,
            "total_books": total_books,
            "opportunities": len(self._opportunities),
            "last_calc": self._last_calc_time,
        }


# ------------------------------------------------------------------
# Module-level helper (kept outside class to avoid method-dispatch overhead)
# ------------------------------------------------------------------

def _effective_price(levels: list, notional: float) -> float:
    """Return volume-weighted average price for the given notional depth.

    Used only when min_volume_usd > 0 (L2 book-walk mode).
    Returns 0.0 on empty/invalid book.
    """
    if not levels:
        return 0.0
    try:
        top_price = float(levels[0][0])
        top_qty = float(levels[0][1])
    except Exception:
        return 0.0

    if notional <= 0:
        return top_price

    total_quote = 0.0
    total_base = 0.0
    for p, q in levels:
        try:
            pf = float(p)
            qf = float(q)
        except Exception:
            continue
        if pf <= 0 or qf <= 0:
            continue
        total_quote += pf * qf
        total_base += qf
        if total_quote >= notional:
            break
    if total_quote > 0 and total_base > 0:
        return total_quote / total_base
    return top_price
