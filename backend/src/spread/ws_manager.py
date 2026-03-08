"""WebSocket connection manager for the spread engine.

Orchestrates exchange WS connections, feeds data to the spread engine,
and broadcasts opportunities to frontend clients.
Fetches volume and funding rate data periodically from REST APIs.
"""

import asyncio
import json
import logging
import os
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Set, Optional, Any

from .models import SpreadConfig, SpreadOpportunity
from .orderbook import OrderBook
from .spread_engine import SpreadEngine
from .spread_tracker import SpreadTracker
from .ml_analyzer import SpreadMLAnalyzer
from .exchanges import ALL_EXCHANGES
from .exchanges.base import BaseExchangeWS
from . import market_data

logger = logging.getLogger(__name__)


class WSManager:
    """Manages all exchange WS connections and the spread calculation loop."""

    def __init__(self, config: Optional[SpreadConfig] = None):
        self.config = config or SpreadConfig()
        self.engine = SpreadEngine(self.config)
        tracker_db_path = (
            Path(getattr(self.config, "tracker_db_path", "")).expanduser()
            if getattr(self.config, "tracker_db_path", "")
            else Path(__file__).resolve().parent.parent.parent / "out" / "config" / "tracker_history.sqlite"
        )
        self.config.tracker_db_path = str(tracker_db_path)
        self.tracker = SpreadTracker(
            window_sec=max(int(getattr(self.config, "tracking_window_sec", 604800) or 604800), 1),
            record_interval_sec=getattr(self.config, "tracker_record_interval_sec", 15.0),
            max_records_per_pair=getattr(self.config, "tracker_max_records_per_pair", 0),
            epsilon_pct=getattr(self.config, "tracker_epsilon_pct", 0.02),
            history_enable_entry_spread_pct=getattr(self.config, "min_spread_pct", 0.1),
            track_enable_entry_spread_pct=getattr(self.config, "min_spread_pct", 0.1),
            max_pairs=10_000,
            db_path=tracker_db_path,
            gap_threshold_sec=getattr(self.config, "tracker_gap_threshold_sec", 0.0),
            min_total_spread_pct=getattr(self.config, "min_total_spread_pct", 1.0),
        )
        self.ml_analyzer = SpreadMLAnalyzer(
            sequence_length=15,
            min_total_spread_pct=getattr(self.config, "min_total_spread_pct", 1.0),
        )
        self.ml_analyzer.attach_tracker(self.tracker)
        self.runtime_audit = None
        self.perf_monitor = None
        self._ml_cache: OrderedDict = OrderedDict()
        self._ML_CACHE_MAX = 2000
        self._ML_PREDICTION_MAX_AGE_SEC = 30.0
        self._exchange_instances: Dict[str, BaseExchangeWS] = {}
        self._scanner_clients: Set[Any] = set()
        self._scanner_lite_clients: Set[Any] = set()
        self._lock = threading.Lock()
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._threads: List[threading.Thread] = []
        self._last_filtered_opps: List[SpreadOpportunity] = []  # Last broadcast batch (filtered)
        self._last_scanner_lite_rows: List[Dict[str, Any]] = []
        self._last_scanner_lite_rows_by_key: Dict[str, Dict[str, Any]] = {}
        self._last_scanner_lite_snapshot: Dict[str, Any] = {"type": "arbitrage_data_lite_snapshot", "data": [], "count": 0, "total_count": 0}
        self._last_scanner_lite_delta: Dict[str, Any] = {"type": "arbitrage_data_lite_delta", "upserts": [], "removes": [], "count": 0, "remove_count": 0, "total_count": 0}
        self._last_scanner_lite_summary: Dict[str, Any] = {"total": 0, "spot_future": 0, "future_future": 0, "updated_at": None}

        self._tracker_db_path = tracker_db_path
        self._tracker_db_path.parent.mkdir(parents=True, exist_ok=True)
        restored = self.tracker.load_from_storage()
        if restored:
            logger.info("[WSManager] Restored tracker SQLite state (%s pairs)", restored)

    @staticmethod
    def _tracker_marker_from_cache(cached: Optional[Dict[str, Any]]) -> tuple[int, float, int]:
        if not cached:
            return (0, 0.0, 0)
        return (
            int(cached.get("history_points", 0) or 0),
            float(cached.get("last_record_ts", 0.0) or 0.0),
            int(cached.get("current_block_id", 0) or 0),
        )

    def _cache_prediction_is_stale(
        self,
        cache_entry: Optional[Dict[str, Any]],
        tracker_row: Optional[Dict[str, Any]],
        *,
        now_ts: float,
    ) -> bool:
        if not cache_entry:
            return True
        cached_marker = tuple(cache_entry.get("tracker_marker") or (0, 0.0, 0))
        current_marker = self._tracker_marker_from_cache(tracker_row)
        if cached_marker != current_marker:
            return True
        computed_at = float(cache_entry.get("computed_at", 0.0) or 0.0)
        return (now_ts - computed_at) >= self._ML_PREDICTION_MAX_AGE_SEC

    def _build_scanner_lite_summary(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        spot_future = 0
        future_future = 0
        for row in rows:
            buy_type = str(row.get("buyType") or "").upper()
            sell_type = str(row.get("sellType") or "").upper()
            if buy_type == "SPOT" and sell_type == "FUTURES":
                spot_future += 1
            elif buy_type == "FUTURES" and sell_type == "FUTURES":
                future_future += 1
        return {
            "total": len(rows),
            "spot_future": spot_future,
            "future_future": future_future,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _refresh_scanner_lite_state(self, opportunities: List[SpreadOpportunity]) -> None:
        rows = [opp.to_scanner_lite_dict() for opp in opportunities[:500]]
        rows_by_key = {
            str(row.get("pairKey") or row.get("code") or f"row:{index}"): row
            for index, row in enumerate(rows)
        }
        previous_by_key = self._last_scanner_lite_rows_by_key
        removes = [key for key in previous_by_key if key not in rows_by_key]
        upserts = [
            row for key, row in rows_by_key.items()
            if previous_by_key.get(key) != row
        ]
        now_iso = datetime.now(timezone.utc).isoformat()
        self._last_scanner_lite_rows = rows
        self._last_scanner_lite_rows_by_key = rows_by_key
        self._last_scanner_lite_summary = self._build_scanner_lite_summary(rows)
        self._last_scanner_lite_snapshot = {
            "timestamp": now_iso,
            "type": "arbitrage_data_lite_snapshot",
            "source": "spread_engine",
            "count": len(rows),
            "total_count": len(opportunities),
            "summary": self._last_scanner_lite_summary,
            "data": rows,
        }
        self._last_scanner_lite_delta = {
            "timestamp": now_iso,
            "type": "arbitrage_data_lite_delta",
            "source": "spread_engine",
            "count": len(upserts),
            "remove_count": len(removes),
            "total_count": len(opportunities),
            "summary": self._last_scanner_lite_summary,
            "upserts": upserts,
            "removes": removes,
        }

    def _on_book_update(self, symbol: str, exchange: str, market_type: str,
                         book: OrderBook):
        """Callback when any exchange sends an order book update."""
        start = time.perf_counter()
        self.engine.update_book(symbol, exchange, market_type, book)
        if self.runtime_audit is not None:
            self.runtime_audit.record_ws_ingest(
                exchange=exchange,
                symbol=symbol,
                market_type=market_type,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

    def attach_runtime_audit(self, collector):
        self.runtime_audit = collector
        self.tracker.audit_collector = collector
        return collector

    def set_perf_monitor(self, monitor):
        self.perf_monitor = monitor
        if hasattr(self.engine, "set_perf_monitor"):
            self.engine.set_perf_monitor(monitor)
        return monitor

    async def start(self):
        """Start all exchange connections and the calculation broadcast loop."""
        if self._running:
            return
        self._running = True

        # Auto-discover symbols (bounded) when config has a default/small list.
        exchange_symbols: Dict[str, Dict[str, set]] = {}
        if getattr(self.config, "symbol_discovery_enabled", True) and len(self.config.symbols) <= 50:
            try:
                from .symbol_discovery import discover_all_symbols, get_all_tradeable_bases
                exchange_names = [ec.name for ec in self.config.exchanges if ec.enabled]
                logger.info(f"[WSManager] Discovering symbols from {exchange_names}...")
                exchange_symbols = await discover_all_symbols(exchange_names)
                all_bases = get_all_tradeable_bases(exchange_symbols)
                max_discovered = getattr(self.config, "symbol_discovery_max_symbols", 1500)
                if max_discovered and len(all_bases) > max_discovered:
                    all_bases = all_bases[:max_discovered]
                if all_bases:
                    # Pre-fetch detailed volume (per exchange + market) to filter
                    # low-liquidity symbols. Only tokens with >=min_vol on at least
                    # one exchange+market are kept in the global list. Per-exchange
                    # filtering happens later when building _ex_spot_sets/_ex_fut_sets.
                    min_vol = getattr(self.config, "min_discovery_volume_usd", 0)
                    _detailed_volumes: dict = {}
                    if min_vol > 0:
                        try:
                            _detailed_volumes = await market_data.fetch_detailed_volumes(exchange_names)
                            # Build max-per-base for global filter
                            max_per_base: dict = {}
                            for _ex, mkts in _detailed_volumes.items():
                                for _mkt, base_vols in mkts.items():
                                    for base, vol in base_vols.items():
                                        cur = max_per_base.get(base)
                                        if cur is None or vol > cur:
                                            max_per_base[base] = vol

                            before = len(all_bases)
                            kept_unknown = 0
                            filtered = []
                            for b in all_bases:
                                vol = max_per_base.get(b)
                                if vol is None or vol >= min_vol:
                                    filtered.append(b)
                                    if vol is None:
                                        kept_unknown += 1
                            all_bases = filtered
                            logger.info(
                                f"[WSManager] Volume filter: {before} -> {len(all_bases)} symbols "
                                f"(min ${min_vol:,.0f} USDT 24h, {kept_unknown} unknown kept)"
                            )
                        except Exception as ve:
                            logger.warning(f"[WSManager] Volume pre-filter failed: {ve}, using all symbols")

                    self.config.symbols = all_bases
                    # Store detailed volumes for per-exchange filtering below
                    self._detailed_volumes = _detailed_volumes
                    logger.info(f"[WSManager] Discovered {len(all_bases)} tradeable symbols")
            except Exception as e:
                logger.warning(f"[WSManager] Symbol discovery failed: {e}, using config symbols")

        # Hard cap to avoid WS overload / slow updates.
        max_symbols = getattr(self.config, "max_symbols", 1500)
        if max_symbols and len(self.config.symbols) > max_symbols:
            logger.warning(
                f"[WSManager] Too many symbols ({len(self.config.symbols)}). "
                f"Capping to {max_symbols} for stability."
            )
            self.config.symbols = list(self.config.symbols)[:max_symbols]

        # Build per-exchange per-market symbol sets from discovery data.
        # This lets us send each exchange ONLY the symbols it actually lists,
        # dramatically reducing WS connections and subscription failures.
        #
        # Per-exchange volume filter: only subscribe to books where that
        # specific exchange+market has >= min_discovery_volume_usd.
        # This avoids tracking low-volume pairs that create noise.
        _ex_spot_sets: Dict[str, set] = {}
        _ex_fut_sets: Dict[str, set] = {}
        _ex_all_sets: Dict[str, set] = {}
        min_vol = getattr(self.config, "min_discovery_volume_usd", 0)
        _dv = getattr(self, "_detailed_volumes", {})
        if exchange_symbols:
            total_removed = 0
            for ex_name, markets in exchange_symbols.items():
                raw_spot = markets.get("spot", set())
                raw_fut = markets.get("futures", set())
                if min_vol > 0 and ex_name in _dv:
                    ex_vols = _dv[ex_name]
                    spot_vols = ex_vols.get("spot", {})
                    fut_vols = ex_vols.get("futures", {})
                    # Keep symbol if:
                    #   - volume on THIS exchange+market >= min_vol, OR
                    #   - volume is unknown (symbol not in ticker response = keep)
                    filtered_spot = {s for s in raw_spot if s not in spot_vols or spot_vols[s] >= min_vol}
                    filtered_fut = {s for s in raw_fut if s not in fut_vols or fut_vols[s] >= min_vol}
                    removed = (len(raw_spot) - len(filtered_spot)) + (len(raw_fut) - len(filtered_fut))
                    total_removed += removed
                    _ex_spot_sets[ex_name] = filtered_spot
                    _ex_fut_sets[ex_name] = filtered_fut
                else:
                    _ex_spot_sets[ex_name] = raw_spot
                    _ex_fut_sets[ex_name] = raw_fut
                _ex_all_sets[ex_name] = _ex_spot_sets[ex_name] | _ex_fut_sets[ex_name]
            if total_removed > 0:
                logger.info(
                    f"[WSManager] Per-exchange volume filter: removed {total_removed} "
                    f"low-volume exchange+market pairs (min ${min_vol:,.0f})"
                )

        logger.info(f"[WSManager] Starting with {len(self.config.symbols)} symbols, "
                     f"{len(self.config.exchanges)} exchanges")

        # Create exchange instances
        for ex_config in self.config.exchanges:
            if not ex_config.enabled:
                continue
            cls = ALL_EXCHANGES.get(ex_config.name)
            if not cls:
                logger.warning(f"[WSManager] Unknown exchange: {ex_config.name}")
                continue
            instance = cls(
                on_book_update=self._on_book_update,
                depth_limit=self.config.depth_limit,
                spot_enabled=getattr(ex_config, "spot_enabled", True),
                futures_enabled=getattr(ex_config, "futures_enabled", True),
            )
            self._exchange_instances[ex_config.name] = instance

        # Start exchange WS tasks — filter symbols per exchange per market
        for name, instance in self._exchange_instances.items():
            all_syms = list(self.config.symbols)
            if name in _ex_all_sets:
                ex_syms = [s for s in all_syms if s in _ex_all_sets[name]]
                spot_syms = [s for s in all_syms if s in _ex_spot_sets.get(name, set())]
                fut_syms = [s for s in all_syms if s in _ex_fut_sets.get(name, set())]
            else:
                ex_syms = all_syms
                spot_syms = None
                fut_syms = None
            logger.info(
                f"[WSManager] {name}: {len(ex_syms)} symbols "
                f"(spot={len(spot_syms) if spot_syms else '?'}, "
                f"fut={len(fut_syms) if fut_syms else '?'})"
            )
            t = threading.Thread(
                target=instance.run_in_thread,
                args=(ex_syms,),
                kwargs={"spot_symbols": spot_syms, "futures_symbols": fut_syms},
                name=f"exchange_{name}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
            logger.info(f"[WSManager] Started {name} feed (thread)")

        # Start calculation + broadcast loop
        calc_task = asyncio.create_task(
            self._calc_and_broadcast_loop(),
            name="calc_broadcast",
        )
        self._tasks.append(calc_task)

        # Start market data fetcher loop (volume + funding)
        mdata_task = asyncio.create_task(
            self._market_data_loop(),
            name="market_data",
        )
        self._tasks.append(mdata_task)

        # Start tracker prune loop
        prune_task = asyncio.create_task(
            self._prune_loop(),
            name="tracker_prune",
        )
        self._tasks.append(prune_task)

        # Start tracker persistence loop (save every 60s)
        persist_task = asyncio.create_task(
            self._persist_loop(),
            name="tracker_persist",
        )
        self._tasks.append(persist_task)

        if self.runtime_audit is not None:
            audit_task = asyncio.create_task(
                self._runtime_audit_loop(),
                name="runtime_audit",
            )
            self._tasks.append(audit_task)

        # Start withdraw status loop (every 10 min, separate from market data)
        withdraw_task = asyncio.create_task(
            self._withdraw_status_loop(),
            name="withdraw_status",
        )
        self._tasks.append(withdraw_task)

        # Start futures meta loop (OI, funding interval, next settlement, limits)
        fmeta_task = asyncio.create_task(
            self._futures_meta_loop(),
            name="futures_meta",
        )
        self._tasks.append(fmeta_task)

        # Start bulk funding history loop (last 24 cycles per exchange:symbol)
        fhist_task = asyncio.create_task(
            self._funding_history_loop(),
            name="funding_history",
        )
        self._tasks.append(fhist_task)

        logger.info("[WSManager] All tasks started")

    async def stop(self):
        """Stop all connections and tasks."""
        self._running = False
        for instance in self._exchange_instances.values():
            instance.stop()  # sets shutdown + cancels tasks in exchange loop
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads.clear()
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        # Flush tracker state on shutdown so invertidas survive restarts.
        self.tracker.flush_to_storage(force=True)
        self.tracker.close_active_session()
        if self.runtime_audit is not None:
            self.runtime_audit.finalize()
        logger.info("[WSManager] Stopped (tracker state flushed)")

    def get_perf_state(self) -> Dict[str, Any]:
        tracker_stats = {}
        process_rss_mb = 0.0
        process_threads = 0
        try:
            tracker_stats = self.tracker.get_storage_stats()
        except Exception:
            tracker_stats = {}
        try:
            import psutil

            proc = psutil.Process(os.getpid())
            process_rss_mb = round(proc.memory_info().rss / (1024 * 1024), 3)
            process_threads = int(proc.num_threads())
        except Exception:
            pass
        return {
            "ml_cache_size": len(self._ml_cache),
            "scanner_clients": len(self._scanner_clients),
            "scanner_lite_clients": len(self._scanner_lite_clients),
            "opportunities_cached": len(self._last_filtered_opps),
            "scanner_lite_rows": len(self._last_scanner_lite_rows),
            "tracker_pairs": int(tracker_stats.get("pairs_tracked", 0) or 0),
            "tracker_records_total": int(tracker_stats.get("records_total", 0) or 0),
            "tracker_episodes_total": int(tracker_stats.get("episodes_total", 0) or 0),
            "tracker_db_size_bytes": int(tracker_stats.get("db_size_bytes", 0) or 0),
            "process_rss_mb": process_rss_mb,
            "process_threads": process_threads,
        }

    async def _market_data_loop(self):
        """Periodically fetch volume and funding rate data from REST APIs."""
        # Initial delay to let WS connections establish
        await asyncio.sleep(10)
        exchange_names = list(self._exchange_instances.keys())
        while self._running:
            try:
                await market_data.fetch_all_market_data(
                    exchange_names, self.config.symbols
                )
                logger.debug(f"[WSManager] Market data updated ({len(market_data._cache)} entries)")
            except Exception as e:
                logger.debug(f"[WSManager] Market data fetch error: {e}")
            await asyncio.sleep(60)  # Refresh every 60 seconds

    async def _withdraw_status_loop(self):
        """Periodically fetch bulk withdrawal status (every 10 min).

        Separated from _market_data_loop to avoid hitting rate limits —
        network/withdrawal data changes infrequently.
        """
        await asyncio.sleep(15)  # Initial delay: let WS connections establish first
        while self._running:
            try:
                await market_data.fetch_bulk_withdraw_status()
            except Exception as e:
                logger.debug(f"[WSManager] Withdraw status fetch error: {e}")
            await asyncio.sleep(600)  # 10 minutes

    async def _futures_meta_loop(self):
        """Periodically fetch futures contract metadata (OI, funding interval,
        next settlement, position limits). Every 60s."""
        await asyncio.sleep(12)  # Initial delay: let WS connections establish first
        exchange_names = list(self._exchange_instances.keys())
        while self._running:
            try:
                await market_data.fetch_all_futures_meta(
                    exchange_names, self.config.symbols
                )
            except Exception as e:
                logger.debug(f"[WSManager] Futures meta fetch error: {e}")
            await asyncio.sleep(60)

    async def _funding_history_loop(self):
        """Periodically fetch bulk funding rate history (last 24 cycles).
        Every 120s, only for symbols currently in opportunities."""
        await asyncio.sleep(20)  # Initial delay
        exchange_names = list(self._exchange_instances.keys())
        while self._running:
            try:
                # Only fetch for symbols currently showing opportunities
                active_symbols = list(set(
                    opp.asset for opp in self._last_filtered_opps
                ))
                if active_symbols:
                    await market_data.fetch_all_funding_history(
                        exchange_names, active_symbols
                    )
            except Exception as e:
                logger.debug(f"[WSManager] Funding history fetch error: {e}")
            await asyncio.sleep(120)

    # How often to refresh tracker enrichment (batch_enrich).
    # 16 cycles × 0.25s = ~4s.  Inversion counts change on the order of
    # minutes, so 4s staleness is imperceptible in the UI.
    _ENRICH_TRACKER_EVERY = 16

    def _do_calc_enrich(self, on_spread_cb):
        """Synchronous calc + enrichment — runs in a thread pool so the event
        loop stays free for WS heartbeats and message delivery.

        Performance-critical path.  Key optimisations vs the original:
        1. Tracker recording uses batch_record (one lock instead of N locks).
        2. Tracker enrichment uses batch_enrich (one lock, no deque copies).
        3. Tracker enrichment is cached and only refreshed every ~4s.
        """
        _t0 = time.time()

        # --- 1. Spread calculation ----------------------------------------
        _perf_started = time.perf_counter()
        _records_batch = []
        if on_spread_cb:
            record_started_at_by_key = {}
            def _collect_record(*args):
                _records_batch.append(args)
                key = self.tracker._pair_key(args[0], args[1], args[2], args[3], args[4])
                record_started_at_by_key[key] = time.perf_counter()
            calc_cb = _collect_record
        else:
            calc_cb = None
            record_started_at_by_key = {}

        opportunities = self.engine.calculate_all(on_spread=calc_cb)
        _t_calc = time.time()
        _perf_after_calc = time.perf_counter()

        # Batch-record all tracker data in one lock acquisition.
        if _records_batch:
            self.tracker.batch_record(_records_batch)
        _perf_after_batch_record = time.perf_counter()

        now_ts = time.time()

        # --- 2. Volume / funding / futures-meta / deposit-withdraw enrichment -----
        _get_vol = market_data.get_volume
        _get_fr = market_data.get_funding_rate
        _get_dw = market_data.get_deposit_withdraw_status
        _get_meta = market_data.get_futures_meta
        for opp in opportunities:
            opp.buy_volume_24h = _get_vol(opp.buy_exchange, opp.asset, opp.buy_market_type)
            opp.sell_volume_24h = _get_vol(opp.sell_exchange, opp.asset, opp.sell_market_type)
            opp._raw_buy_vol = opp.buy_volume_24h
            opp._raw_sell_vol = opp.sell_volume_24h

            if opp.buy_market_type == "futures":
                opp.funding_rate_buy = _get_fr(opp.buy_exchange, opp.asset)
            if opp.sell_market_type == "futures":
                opp.funding_rate_sell = _get_fr(opp.sell_exchange, opp.asset)

            buy_dw = _get_dw(opp.buy_exchange, opp.asset)
            sell_dw = _get_dw(opp.sell_exchange, opp.asset)
            if buy_dw is not None:
                opp.buy_withdraw_status = buy_dw["withdraw_ok"]
                opp.buy_deposit_status = buy_dw["deposit_ok"]
            if sell_dw is not None:
                opp.sell_withdraw_status = sell_dw["withdraw_ok"]
                opp.sell_deposit_status = sell_dw["deposit_ok"]

            # Futures metadata: OI, funding interval, next settlement, position limits
            if opp.buy_market_type == "futures":
                meta = _get_meta(opp.buy_exchange, opp.asset)
                if meta:
                    opp.funding_interval_buy = meta.get("funding_interval")
                    opp.next_settlement_buy = meta.get("next_settlement")
                    opp.open_interest_buy = meta.get("open_interest")
                    opp.position_limit_buy = meta.get("position_limit")
            if opp.sell_market_type == "futures":
                meta = _get_meta(opp.sell_exchange, opp.asset)
                if meta:
                    opp.funding_interval_sell = meta.get("funding_interval")
                    opp.next_settlement_sell = meta.get("next_settlement")
                    opp.open_interest_sell = meta.get("open_interest")
                    opp.position_limit_sell = meta.get("position_limit")
        _perf_after_market_enrich = time.perf_counter()

        # --- 3. Tracker enrichment (batched + cached) ---------------------
        self._enrich_tracker_cycle = getattr(self, '_enrich_tracker_cycle', 0) + 1
        tracker_cache = getattr(self, '_tracker_enrich_cache', None)
        refresh_tracker = (
            self._enrich_tracker_cycle % self._ENRICH_TRACKER_EVERY == 0
            or tracker_cache is None
        )

        if refresh_tracker:
            pair_key = self.tracker._pair_key
            keys = [
                pair_key(opp.asset, opp.buy_exchange, opp.buy_market_type,
                         opp.sell_exchange, opp.sell_market_type)
                for opp in opportunities
            ]
            tracker_cache = self.tracker.batch_enrich(keys, now_ts)
            self._tracker_enrich_cache = tracker_cache
        _perf_after_tracker_enrich = time.perf_counter()

        _history_fetch_ms = 0.0
        _ml_analyze_ms = 0.0
        _ml_render_ms = 0.0
        if tracker_cache:
            pair_key = self.tracker._pair_key
            for opp in opportunities:
                key = pair_key(opp.asset, opp.buy_exchange, opp.buy_market_type,
                               opp.sell_exchange, opp.sell_market_type)
                cached = tracker_cache.get(key)
                if cached:
                    opp.inverted_counts = cached['inverted_counts']
                    opp.inverted_count = cached['inverted_count']
                    opp.total_entries = cached['total_entries']
                    opp.total_exits = cached['total_exits']
                    opp.last_crossover_ts = cached['last_crossover_ts']

                cache_entry = self._ml_cache.get(key)
                if self._cache_prediction_is_stale(cache_entry, cached, now_ts=now_ts):
                    _history_started = time.perf_counter()
                    history = self.tracker.get_history(
                        opp.asset, opp.buy_exchange, opp.buy_market_type,
                        opp.sell_exchange, opp.sell_market_type, limit=100
                    )
                    _history_fetch_ms += (time.perf_counter() - _history_started) * 1000.0
                    _ml_started = time.perf_counter()
                    prediction = self.ml_analyzer.predict_history(history)
                    _ml_analyze_ms += (time.perf_counter() - _ml_started) * 1000.0
                    cache_entry = {
                        "prediction": prediction,
                        "tracker_marker": self._tracker_marker_from_cache(cached),
                        "computed_at": now_ts,
                    }
                    self._ml_cache[key] = cache_entry
                    self._ml_cache.move_to_end(key)
                    if len(self._ml_cache) > self._ML_CACHE_MAX:
                        self._ml_cache.popitem(last=False)

                if refresh_tracker and cache_entry and cache_entry.get("prediction") is not None:
                    _render_started = time.perf_counter()
                    rendered_ctx = self.ml_analyzer.render_prediction(
                        opp.entry_spread_pct,
                        cache_entry["prediction"],
                        pair_key=key,
                    )
                    _ml_render_ms += (time.perf_counter() - _render_started) * 1000.0
                    cache_entry["context"] = rendered_ctx
                    cache_entry["rendered_at"] = now_ts
                    cache_entry["rendered_entry_spread"] = float(opp.entry_spread_pct)
                    if self.runtime_audit is not None and cache_entry.get("computed_at") == now_ts:
                        end_to_end_ms = None
                        started_at = record_started_at_by_key.get(key)
                        if started_at is not None:
                            end_to_end_ms = (time.perf_counter() - started_at) * 1000.0
                        self.runtime_audit.record_inference(
                            pair_id=self.tracker._pair_id(key),
                            result=rendered_ctx,
                            end_to_end_ms=end_to_end_ms,
                        )

                if cache_entry and cache_entry.get("context") is not None:
                    ctx = cache_entry["context"]
                    opp.ml_context = ctx
                    opp.ml_score = ctx.get("ml_score", 0)

        # --- 4. Clamp + filter --------------------------------------------
        for opp in opportunities:
            if opp.buy_volume_24h < 0:
                opp.buy_volume_24h = 0.0
            if opp.sell_volume_24h < 0:
                opp.sell_volume_24h = 0.0

        min_opp_vol = getattr(self.config, "min_opportunity_volume_usd", 500)
        if min_opp_vol > 0:
            md_ready = market_data._market_data_ready
            def _vol_ok(raw_vol: float) -> bool:
                if raw_vol < 0:
                    return not md_ready
                return raw_vol >= min_opp_vol
            opportunities = [
                opp for opp in opportunities
                if _vol_ok(getattr(opp, "_raw_buy_vol", -1.0))
                and _vol_ok(getattr(opp, "_raw_sell_vol", -1.0))
            ]
        _perf_after_filter = time.perf_counter()

        self._refresh_scanner_lite_state(opportunities)
        _perf_after_lite_refresh = time.perf_counter()

        _t_end = time.time()
        # All timing measured inside thread (no event-loop scheduling noise).
        self._last_timing = (
            int((_t_calc - _t0) * 1000),       # calc_ms
            int((_t_end - _t_calc) * 1000),     # enrich_ms
            int((_t_end - _t0) * 1000),         # thread_total_ms
        )
        if self.perf_monitor is not None:
            self.perf_monitor.record_scanner_cycle(
                {
                    "total_ms": round((_perf_after_lite_refresh - _perf_started) * 1000.0, 6),
                    "calculate_ms": round((_perf_after_calc - _perf_started) * 1000.0, 6),
                    "batch_record_ms": round((_perf_after_batch_record - _perf_after_calc) * 1000.0, 6),
                    "market_enrich_ms": round((_perf_after_market_enrich - _perf_after_batch_record) * 1000.0, 6),
                    "tracker_enrich_ms": round((_perf_after_tracker_enrich - _perf_after_market_enrich) * 1000.0, 6),
                    "history_fetch_ms": round(_history_fetch_ms, 6),
                    "ml_analyze_ms": round(_ml_analyze_ms, 6),
                    "ml_render_ms": round(_ml_render_ms, 6),
                    "filter_ms": round((_perf_after_filter - _perf_after_tracker_enrich) * 1000.0, 6),
                    "lite_refresh_ms": round((_perf_after_lite_refresh - _perf_after_filter) * 1000.0, 6),
                    "opportunities_before_filter": len(self.engine._opportunities),
                    "opportunities_after_filter": len(opportunities),
                    "refresh_tracker": bool(refresh_tracker),
                    "tracker_cache_size": len(tracker_cache or {}),
                    "ml_cache_size": len(self._ml_cache),
                }
            )
        return opportunities

    async def _calc_and_broadcast_loop(self):
        """Periodically calculate spreads and broadcast to scanner clients."""
        _cycle = 0
        _TRACKER_EVERY = 5  # Run tracker recording every 5th cycle
        # Offset tracker recording by +2 so it never coincides with
        # full-recalc cycles (engine._FULL_RECALC_EVERY = 80).
        # Full-recalc: cycle 80, 160, 240 …  Tracker: cycle 3, 8, 13, 18 …
        # Since (80+2)%5 = 2 ≠ 0 they never overlap, avoiding worst-case
        # "full-recalc + massive batch_record" in the same cycle.
        _TRACKER_OFFSET = 2
        _LOG_EVERY = 21  # coprime with 5 → shows both tracker=ON and off
        _max_calc_ms = 0  # track worst cycle for health monitoring
        while self._running:
            try:
                _cycle += 1
                t0 = time.time()

                # Tracker recording on offset cycles (avoids full-recalc overlap).
                is_tracker_cycle = ((_cycle + _TRACKER_OFFSET) % _TRACKER_EVERY == 0)
                on_spread_cb = self.tracker.record_spread if is_tracker_cycle else None

                # Run CPU-heavy calc+enrichment in a thread so the event loop
                # stays free for WS heartbeats, pings, and message delivery.
                opportunities = await asyncio.to_thread(
                    self._do_calc_enrich, on_spread_cb
                )
                self._last_filtered_opps = opportunities

                t1 = time.time()
                calc_ms = int((t1 - t0) * 1000)
                _max_calc_ms = max(_max_calc_ms, calc_ms)

                # Broadcast to scanner clients (skip if previous broadcast still in progress)
                if not getattr(self, '_broadcasting', False):
                    asyncio.create_task(self._broadcast_scanner(opportunities))
                if not getattr(self, '_broadcasting_lite', False):
                    asyncio.create_task(self._broadcast_scanner_lite())

                t2 = time.time()

                # Log timing every _LOG_EVERY cycles (21 is coprime with 5,
                # so we see a mix of tracker=ON and tracker=off cycles).
                if _cycle % _LOG_EVERY == 0:
                    stale = getattr(self.engine, '_last_stale_count', 0)
                    dirty_n = getattr(self.engine, '_last_dirty_count', 0)
                    is_full = (self.engine._cycle_count % 80 == 0)
                    # Timing from inside the thread (no event-loop noise)
                    timing = getattr(self, '_last_timing', (0, 0, 0))
                    t_calc, t_enrich, t_thread = timing
                    # Event-loop overhead = wall-clock total - thread total
                    t_loop = max(0, calc_ms - t_thread)
                    logger.info(
                        f"[WSManager] Cycle #{_cycle}: "
                        f"calc={t_calc}ms enrich={t_enrich}ms "
                        f"loop_wait={t_loop}ms wall={calc_ms}ms "
                        f"opps={len(opportunities)} dirty={dirty_n} stale={stale} "
                        f"clients={len(self._scanner_clients)} "
                        f"tracker={'ON' if is_tracker_cycle else 'off'} "
                        f"full={'Y' if is_full else 'n'} peak={_max_calc_ms}ms"
                    )
                    _max_calc_ms = 0  # reset peak after logging

                # Health log: update rates per exchange (every ~30s at 0.15s interval)
                if _cycle % 200 == 0:
                    counts = self.engine.get_and_reset_update_counts()
                    if counts:
                        parts = [f"{k}={v}" for k, v in sorted(counts.items())]
                        logger.info(f"[WSManager] Updates/30s: {' | '.join(parts)}")
                    else:
                        logger.warning("[WSManager] Updates/30s: NONE — no exchange data received!")
                    # Resource monitoring (requires psutil)
                    try:
                        import os as _os
                        _proc = __import__('psutil').Process(_os.getpid())
                        _mem = _proc.memory_info()
                        logger.info(
                            f"[WSManager] Resources: RSS={_mem.rss // (1024*1024)}MB "
                            f"threads={_proc.num_threads()}"
                        )
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"[WSManager] Calc/broadcast error: {e}")
                if self.runtime_audit is not None:
                    self.runtime_audit.record_error("calc_broadcast", str(e))

            await asyncio.sleep(self.config.broadcast_interval_sec)

    async def _broadcast_scanner(self, opportunities: List[SpreadOpportunity]):
        """Send opportunities to all connected scanner WebSocket clients."""
        self._broadcasting = True
        try:
            await self._broadcast_scanner_inner(opportunities)
        finally:
            self._broadcasting = False

    async def _broadcast_scanner_inner(self, opportunities: List[SpreadOpportunity]):
        with self._lock:
            clients = list(self._scanner_clients)

        if not clients:
            return

        # When there are no opportunities (e.g. startup warmup), send a
        # lightweight keepalive so the frontend watchdog doesn't reconnect.
        if not opportunities:
            ping_payload = '{"type":"ping"}'
            async def _ping(ws):
                try:
                    await asyncio.wait_for(ws.send_str(ping_payload), timeout=3.0)
                    return None
                except Exception:
                    return ws
            results = await asyncio.gather(*[_ping(ws) for ws in clients])
            disconnected = [ws for ws in results if ws is not None]
            if disconnected:
                with self._lock:
                    for ws in disconnected:
                        self._scanner_clients.discard(ws)
            return

        # Cap at 500 to keep payload manageable
        broadcast_opps = opportunities[:500]

        _t0 = time.time()
        now = datetime.now(timezone.utc).isoformat()
        data_list = [opp.to_scanner_dict() for opp in broadcast_opps]
        _t1 = time.time()
        message = {
            "timestamp": now,
            "type": "arbitrage_data",
            "source": "spread_engine",
            "count": len(broadcast_opps),
            "total_count": len(opportunities),
            "data": data_list,
        }
        payload = json.dumps(message)
        _t2 = time.time()

        # Send to all clients in parallel with timeout
        async def _send(ws):
            try:
                await asyncio.wait_for(ws.send_str(payload), timeout=5.0)
                return None
            except Exception as exc:
                logger.warning(f"[WSManager] WS send failed ({type(exc).__name__}): {exc}")
                return ws

        results = await asyncio.gather(*[_send(ws) for ws in clients])
        _t3 = time.time()

        # Log broadcast breakdown periodically
        if hasattr(self, '_bc_log_count'):
            self._bc_log_count += 1
        else:
            self._bc_log_count = 0
        if self._bc_log_count % 60 == 0:
            logger.info(
                f"[WSManager] Broadcast: dict={int((_t1-_t0)*1000)}ms "
                f"json={int((_t2-_t1)*1000)}ms send={int((_t3-_t2)*1000)}ms "
                f"payload={len(payload)//1024}KB clients={len(clients)}"
            )
        disconnected = [ws for ws in results if ws is not None]
        if self.perf_monitor is not None:
            self.perf_monitor.record_broadcast(
                "scanner_full",
                {
                    "dict_ms": round((_t1 - _t0) * 1000.0, 6),
                    "json_ms": round((_t2 - _t1) * 1000.0, 6),
                    "send_ms": round((_t3 - _t2) * 1000.0, 6),
                    "total_ms": round((_t3 - _t0) * 1000.0, 6),
                    "payload_bytes": len(payload.encode("utf-8")),
                    "client_count": len(clients),
                },
            )

        if disconnected:
            with self._lock:
                for ws in disconnected:
                    self._scanner_clients.discard(ws)

    async def _broadcast_scanner_lite(self):
        """Send compact scanner deltas to migrated scanner clients."""
        self._broadcasting_lite = True
        try:
            await self._broadcast_scanner_lite_inner()
        finally:
            self._broadcasting_lite = False

    async def _broadcast_scanner_lite_inner(self):
        with self._lock:
            clients = list(self._scanner_lite_clients)

        if not clients:
            return

        delta = self._last_scanner_lite_delta
        snapshot = self._last_scanner_lite_snapshot
        if not snapshot.get("data"):
            ping_payload = '{"type":"ping"}'

            async def _ping(ws):
                try:
                    await asyncio.wait_for(ws.send_str(ping_payload), timeout=3.0)
                    return None
                except Exception:
                    return ws

            results = await asyncio.gather(*[_ping(ws) for ws in clients])
            disconnected = [ws for ws in results if ws is not None]
            if disconnected:
                with self._lock:
                    for ws in disconnected:
                        self._scanner_lite_clients.discard(ws)
            return

        payload_obj = delta if delta.get("upserts") or delta.get("removes") else {"type": "ping"}
        _t0 = time.perf_counter()
        payload = json.dumps(payload_obj)
        _t1 = time.perf_counter()

        async def _send(ws):
            try:
                await asyncio.wait_for(ws.send_str(payload), timeout=5.0)
                return None
            except Exception as exc:
                logger.warning(f"[WSManager] Scanner-lite WS send failed ({type(exc).__name__}): {exc}")
                return ws

        results = await asyncio.gather(*[_send(ws) for ws in clients])
        _t2 = time.perf_counter()
        disconnected = [ws for ws in results if ws is not None]
        if self.perf_monitor is not None:
            self.perf_monitor.record_broadcast(
                "scanner_lite",
                {
                    "json_ms": round((_t1 - _t0) * 1000.0, 6),
                    "send_ms": round((_t2 - _t1) * 1000.0, 6),
                    "total_ms": round((_t2 - _t0) * 1000.0, 6),
                    "payload_bytes": len(payload.encode("utf-8")),
                    "client_count": len(clients),
                    "upsert_count": int(delta.get("count") or 0),
                    "remove_count": int(delta.get("remove_count") or 0),
                },
            )
        if disconnected:
            with self._lock:
                for ws in disconnected:
                    self._scanner_lite_clients.discard(ws)

    async def _prune_loop(self):
        """Periodically prune stale tracker data."""
        while self._running:
            await asyncio.sleep(60)
            try:
                self.tracker.prune()
            except Exception as e:
                logger.error(f"[WSManager] Prune error: {e}")

    async def _persist_loop(self):
        """Periodically flush tracker state to SQLite."""
        while self._running:
            await asyncio.sleep(60)
            try:
                started = time.perf_counter()
                self.tracker.flush_to_storage()
                logger.debug("[WSManager] Tracker SQLite flushed")
                if self.perf_monitor is not None:
                    self.perf_monitor.record_cache_state(
                        {
                            **self.get_perf_state(),
                            "tracker_persist_ms": round((time.perf_counter() - started) * 1000.0, 6),
                        }
                    )
                if self.runtime_audit is not None:
                    tracker_stats = self.tracker.get_storage_stats()
                    self.runtime_audit.event(
                        "tracker_persist",
                        flush_ms=round((time.perf_counter() - started) * 1000.0, 3),
                        db_size_bytes=int(tracker_stats.get("db_size_bytes", 0)),
                        last_flush_at=float(tracker_stats.get("last_flush_at", 0.0) or 0.0),
                        storage_stats=tracker_stats,
                    )
            except Exception as e:
                logger.error(f"[WSManager] Persist error: {e}")
                if self.runtime_audit is not None:
                    self.runtime_audit.record_error("tracker_persist", str(e))

    def _runtime_audit_exchange_statuses(self) -> list[dict]:
        statuses: list[dict] = []
        for ex_name, instance in self._exchange_instances.items():
            try:
                statuses.append(
                    {
                        "name": ex_name,
                        "ws_running": not instance.shutdown.is_set(),
                        "book_count": len(getattr(instance, "_books", {})),
                    }
                )
            except Exception:
                statuses.append({"name": ex_name, "ws_running": False, "book_count": 0})
        return statuses

    async def _runtime_audit_loop(self):
        sample_index = 0
        while self._running and self.runtime_audit is not None:
            try:
                exchange_statuses = self._runtime_audit_exchange_statuses()
                self.runtime_audit.record_exchange_status(exchange_statuses)
                if sample_index % 12 == 0:
                    if self.perf_monitor is not None:
                        self.perf_monitor.record_cache_state(self.get_perf_state())
                    self.runtime_audit.sample_system(
                        pid=os.getpid(),
                        db_path=self._tracker_db_path,
                        tracker_stats=self.tracker.get_storage_stats(),
                        exchange_statuses=exchange_statuses,
                        extra={
                            "pairs_active": len(self._ml_cache),
                            "scanner_clients": len(self._scanner_clients),
                            "opportunities": len(self._last_filtered_opps),
                        },
                    )
                sample_index += 1
            except Exception as exc:
                logger.error("[WSManager] Runtime audit loop error: %s", exc)
                if self.runtime_audit is not None:
                    self.runtime_audit.record_error("runtime_audit_loop", str(exc))
            await asyncio.sleep(5)

    def add_scanner_client(self, ws):
        """Register a frontend scanner WebSocket connection."""
        with self._lock:
            self._scanner_clients.add(ws)
        logger.info(f"[WSManager] Scanner client connected (total: {len(self._scanner_clients)})")

    def remove_scanner_client(self, ws):
        """Unregister a frontend scanner WebSocket connection."""
        with self._lock:
            self._scanner_clients.discard(ws)
        logger.info(f"[WSManager] Scanner client disconnected (total: {len(self._scanner_clients)})")

    def add_scanner_lite_client(self, ws):
        """Register a compact scanner WebSocket connection."""
        with self._lock:
            self._scanner_lite_clients.add(ws)
        logger.info(f"[WSManager] Scanner-lite client connected (total: {len(self._scanner_lite_clients)})")

    def remove_scanner_lite_client(self, ws):
        """Unregister a compact scanner WebSocket connection."""
        with self._lock:
            self._scanner_lite_clients.discard(ws)
        logger.info(f"[WSManager] Scanner-lite client disconnected (total: {len(self._scanner_lite_clients)})")

    def get_current_opportunities(self) -> List[Dict]:
        """Get current opportunities as dicts (for REST API).

        Returns the same filtered set that is broadcast to WS scanner clients,
        so low-volume tokens are excluded from both channels.
        """
        return [opp.to_scanner_dict() for opp in self._last_filtered_opps]

    def get_current_opportunities_lite(self) -> List[Dict]:
        """Get cached compact opportunity rows for the scanner list."""
        return list(self._last_scanner_lite_rows)

    def get_current_opportunities_lite_summary(self) -> Dict[str, Any]:
        return dict(self._last_scanner_lite_summary)

    def get_current_scanner_lite_snapshot(self) -> Dict[str, Any]:
        return dict(self._last_scanner_lite_snapshot)

    def get_scanner_opportunity_detail(self, pair_key: str) -> Optional[Dict]:
        target = str(pair_key or "").strip()
        if not target:
            return None
        for opp in self._last_filtered_opps:
            if opp.pair_key() == target:
                return opp.to_scanner_dict()
        return None

    def get_status(self) -> Dict:
        """Get manager status for monitoring."""
        return {
            "running": self._running,
            "exchanges": self.engine.get_connected_exchanges(),
            "summary": self.engine.get_snapshot_summary(),
            "scanner_clients": len(self._scanner_clients),
        }
