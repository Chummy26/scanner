"""WebSocket connection manager for the spread engine.

Orchestrates exchange WS connections, feeds data to the spread engine,
and broadcasts opportunities to frontend clients.
Fetches volume and funding rate data periodically from REST APIs.
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
import threading
import zlib
from datetime import datetime, timezone
from pathlib import Path
from collections import OrderedDict, deque
from typing import Dict, List, Set, Optional, Any

from .models import SpreadConfig, SpreadOpportunity
from .orderbook import OrderBook
from .spread_engine import SpreadEngine
from .spread_tracker import SpreadTracker
from .ml_analyzer import SpreadMLAnalyzer
from .ingest_health import ExchangeCircuitBreaker, HourlyHealthDigest
from .auto_retrain import (
    archive_expired_snapshots,
    backend_out_root,
    current_snapshot_slot,
    latest_snapshot_entry,
    latest_snapshot_timestamp,
    latest_training_run,
    launch_auto_retrain_worker,
    load_snapshot_manifest,
    load_training_manifest,
    next_snapshot_slot,
    seconds_until_next_snapshot_slot,
    should_retrain,
    snapshots_dir,
    state_file_path,
    training_dir,
    update_snapshot_manifest,
)
from .exchanges import ALL_EXCHANGES
from .exchanges.base import BaseExchangeWS
from . import market_data
from .runtime_audit import create_sqlite_snapshot
from .train_model import certify_data_for_training

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
            memory_window_sec=max(int(getattr(self.config, "tracker_memory_window_sec", 43200) or 43200), 1),
            record_interval_sec=getattr(self.config, "tracker_record_interval_sec", 15.0),
            max_records_per_pair=getattr(self.config, "tracker_max_records_per_pair", 0),
            epsilon_pct=getattr(self.config, "tracker_epsilon_pct", 0.02),
            history_enable_entry_spread_pct=getattr(self.config, "tracker_min_spread_pct", 0.05),
            track_enable_entry_spread_pct=getattr(self.config, "tracker_min_spread_pct", 0.05),
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
        self._ML_REFRESHES_PER_CYCLE = 12
        self._ML_REFRESH_BUDGET_MS = 75.0
        self._ml_pending_refreshes: OrderedDict = OrderedDict()
        self._ml_state_lock = threading.Lock()
        self._tracker_ingest_lock = threading.Lock()
        self._tracker_pending_records: OrderedDict = OrderedDict()
        self._tracker_pending_rejections: OrderedDict = OrderedDict()
        self._tracker_history_mirror: Dict[Any, deque] = {}
        self._tracker_history_last_ts: Dict[Any, float] = {}
        self._TRACKER_HISTORY_MIRROR_MAX = 128
        self._TRACKER_INGEST_LOOP_SEC = 0.5
        self._TRACKER_INGEST_BATCH_MAX = 2048
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
        self._SCANNER_LITE_INTEREST_TTL_SEC = 30.0
        self._last_scanner_lite_interest_perf = 0.0
        self._force_scanner_lite_refresh = False
        self._startup_started_at = 0.0
        self._startup_filtered_opportunities = 0
        self._last_startup_gate_state: Dict[str, Any] = {
            "warmup_elapsed_sec": 0.0,
            "startup_warmup_sec": float(getattr(self.config, "startup_warmup_sec", 20.0) or 20.0),
            "market_data_ready": False,
            "allow_publication": False,
            "warmup_remaining_sec": float(getattr(self.config, "startup_warmup_sec", 20.0) or 20.0),
        }

        self._tracker_db_path = tracker_db_path
        self._tracker_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifact_root = backend_out_root(self._tracker_db_path.parent.parent)
        self._snapshots_dir = snapshots_dir(self._artifact_root)
        self._training_dir = training_dir(self._artifact_root)
        self._archive_dir = self._artifact_root / "archive"
        self._auto_retrain_state_path = state_file_path(self._artifact_root)
        self._circuit_breaker = ExchangeCircuitBreaker()
        self._hourly_health_state: dict[str, Any] = {}
        self._current_hour_bucket = int(time.time() // 3600)
        self._last_health_digest: dict[str, Any] | None = None
        self._latest_tracker_episode_total = 0
        self._auto_retrain_process = None
        self._last_reloaded_model_version = self.ml_analyzer.model_version
        self.tracker.add_event_listener(self._on_tracker_event)
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

    def _tracker_cycle_every(self) -> int:
        # Smoothed cadence: a small fixed stride keeps tracker work spread
        # across time instead of creating 15s bursts that stall the scanner.
        return 5

    def _startup_gate_state(self, *, now_ts: float | None = None) -> Dict[str, Any]:
        current_ts = float(now_ts) if now_ts is not None else time.time()
        started_at = float(self._startup_started_at or current_ts)
        elapsed = max(0.0, current_ts - started_at)
        warmup_sec = max(float(getattr(self.config, "startup_warmup_sec", 20.0) or 20.0), 0.0)
        market_data_ready = bool(getattr(market_data, "_market_data_ready", False))
        allow_publication = bool(market_data_ready and elapsed >= warmup_sec)
        return {
            "warmup_elapsed_sec": round(elapsed, 3),
            "startup_warmup_sec": round(warmup_sec, 3),
            "market_data_ready": market_data_ready,
            "allow_publication": allow_publication,
            "warmup_remaining_sec": round(max(warmup_sec - elapsed, 0.0), 3),
        }

    def _tracker_capture_shards(self) -> int:
        capture_period_sec = self._tracker_cycle_every() * max(float(getattr(self.config, "broadcast_interval_sec", 0.15) or 0.15), 0.001)
        record_interval_sec = max(float(getattr(self.config, "tracker_record_interval_sec", 15.0) or 15.0), capture_period_sec)
        return max(1, int(round(record_interval_sec / capture_period_sec)))

    def _get_ml_cache_entry(self, key: Any) -> Optional[Dict[str, Any]]:
        with self._ml_state_lock:
            cache_entry = self._ml_cache.get(key)
            return dict(cache_entry) if isinstance(cache_entry, dict) else None

    def _queue_ml_refresh(self, key: Any, payload: Dict[str, Any]) -> None:
        with self._ml_state_lock:
            existing = self._ml_pending_refreshes.get(key)
            merged_payload = dict(existing) if isinstance(existing, dict) else {}
            merged_payload.update(payload)
            self._ml_pending_refreshes[key] = merged_payload
            self._ml_pending_refreshes.move_to_end(key)

    def _ml_queue_size(self) -> int:
        with self._ml_state_lock:
            return len(self._ml_pending_refreshes)

    def _ml_cache_size(self) -> int:
        with self._ml_state_lock:
            return len(self._ml_cache)

    def _tracker_pending_record_count(self) -> int:
        with self._tracker_ingest_lock:
            return len(self._tracker_pending_records)

    def _tracker_pending_rejection_count(self) -> int:
        with self._tracker_ingest_lock:
            return len(self._tracker_pending_rejections)

    def _history_mirror_size(self) -> int:
        with self._tracker_ingest_lock:
            return len(self._tracker_history_mirror)

    def _coalesce_tracker_records(self, records: List[tuple]) -> Dict[Any, tuple]:
        latest: Dict[Any, tuple] = {}
        pair_key = self.tracker._pair_key
        for record in records:
            if not isinstance(record, tuple) or len(record) < 7:
                continue
            latest[pair_key(*record[:5])] = record
        return latest

    def _pair_in_capture_shard(self, key: Any, *, capture_shard: int, shard_count: int) -> bool:
        if shard_count <= 1:
            return True
        if isinstance(key, tuple):
            payload = "|".join(str(part) for part in key)
        else:
            payload = str(key)
        return (zlib.crc32(payload.encode("utf-8")) % shard_count) == capture_shard

    def _queue_tracker_records(self, records: List[tuple], *, now_ts: float, capture_shard: int = 0, shard_count: int = 1) -> int:
        latest = self._coalesce_tracker_records(records)
        if not latest:
            return 0
        with self._tracker_ingest_lock:
            for key, record in latest.items():
                if not self._pair_in_capture_shard(key, capture_shard=capture_shard, shard_count=shard_count):
                    continue
                self._tracker_pending_records[key] = record
                self._tracker_pending_records.move_to_end(key)
                last_ts = float(self._tracker_history_last_ts.get(key, 0.0) or 0.0)
                if last_ts > 0.0 and (now_ts - last_ts) < max(float(getattr(self.config, "tracker_record_interval_sec", 15.0) or 15.0), 0.0):
                    continue
                history = self._tracker_history_mirror.get(key)
                if history is None:
                    history = deque(maxlen=self._TRACKER_HISTORY_MIRROR_MAX)
                    self._tracker_history_mirror[key] = history
                history.append(
                    {
                        "timestamp": float(now_ts),
                        "entry_spread": float(record[5]),
                        "exit_spread": float(record[6]),
                    }
                )
                self._tracker_history_last_ts[key] = float(now_ts)
        return sum(
            1 for key in latest.keys()
            if self._pair_in_capture_shard(key, capture_shard=capture_shard, shard_count=shard_count)
        )

    def _queue_tracker_rejections(self, rejections: List[dict[str, Any]], *, capture_shard: int = 0, shard_count: int = 1) -> int:
        if not rejections:
            return 0
        aggregated: Dict[Any, dict[str, Any]] = {}
        for payload in rejections:
            if not isinstance(payload, dict):
                continue
            invalid_fields = tuple(sorted(str(field) for field in (payload.get("invalid_fields") or [])))
            queue_key = (
                str(payload.get("symbol") or ""),
                str(payload.get("buy_ex") or ""),
                str(payload.get("buy_mt") or ""),
                str(payload.get("sell_ex") or ""),
                str(payload.get("sell_mt") or ""),
                invalid_fields,
                str(payload.get("reason") or ""),
            )
            existing = aggregated.get(queue_key)
            if existing is None:
                aggregated[queue_key] = {
                    "symbol": str(payload.get("symbol") or ""),
                    "buy_ex": str(payload.get("buy_ex") or ""),
                    "buy_mt": str(payload.get("buy_mt") or ""),
                    "sell_ex": str(payload.get("sell_ex") or ""),
                    "sell_mt": str(payload.get("sell_mt") or ""),
                    "invalid_fields": list(payload.get("invalid_fields") or []),
                    "reason": str(payload.get("reason") or ""),
                    "count": 1,
                }
            else:
                existing["count"] = int(existing.get("count", 0) or 0) + 1
        if not aggregated:
            return 0
        with self._tracker_ingest_lock:
            for key, payload in aggregated.items():
                pair_key = self.tracker._pair_key(
                    str(payload.get("symbol") or ""),
                    str(payload.get("buy_ex") or ""),
                    str(payload.get("buy_mt") or ""),
                    str(payload.get("sell_ex") or ""),
                    str(payload.get("sell_mt") or ""),
                )
                if not self._pair_in_capture_shard(pair_key, capture_shard=capture_shard, shard_count=shard_count):
                    continue
                existing = self._tracker_pending_rejections.get(key)
                if isinstance(existing, dict):
                    payload["count"] = int(payload.get("count", 0) or 0) + int(existing.get("count", 0) or 0)
                self._tracker_pending_rejections[key] = payload
                self._tracker_pending_rejections.move_to_end(key)
        return sum(
            1
            for payload in aggregated.values()
            if self._pair_in_capture_shard(
                self.tracker._pair_key(
                    str(payload.get("symbol") or ""),
                    str(payload.get("buy_ex") or ""),
                    str(payload.get("buy_mt") or ""),
                    str(payload.get("sell_ex") or ""),
                    str(payload.get("sell_mt") or ""),
                ),
                capture_shard=capture_shard,
                shard_count=shard_count,
            )
        )

    def _reset_hourly_health_state(self, bucket: int, *, now_ts: float) -> None:
        tracker_stats = {}
        if hasattr(self.tracker, "get_storage_stats"):
            try:
                tracker_stats = dict(self.tracker.get_storage_stats() or {})
            except Exception:
                tracker_stats = {}
        self._current_hour_bucket = int(bucket)
        self._hourly_health_state = {
            "hour_start_ts": float(bucket * 3600),
            "records_total": 0,
            "records_rejected": 0,
            "pairs_with_records": set(),
            "pairs_with_gaps": set(),
            "cross_exchange_flags": 0,
            "book_age_sum": {},
            "book_age_count": {},
            "last_age_sample_ts": 0.0,
            "episode_total_start": int(tracker_stats.get("episodes_total", 0) or 0),
            "created_at": float(now_ts),
        }

    def _ensure_hourly_health_state(self, *, now_ts: float) -> None:
        bucket = int(float(now_ts) // 3600)
        if not self._hourly_health_state:
            self._reset_hourly_health_state(bucket, now_ts=float(now_ts))
            return
        if bucket == self._current_hour_bucket:
            return
        digest = self._compute_hourly_digest((self._current_hour_bucket + 1) * 3600)
        self._last_health_digest = digest.to_dict()
        self.tracker.save_hourly_health_digest(self._last_health_digest)
        if str(digest.quality_verdict) == "unhealthy":
            self.tracker.disable_open_blocks(reason="unhealthy_ingest_period")
        self._reset_hourly_health_state(bucket, now_ts=float(now_ts))

    def _sample_book_ages(self, *, now_ts: float) -> None:
        state = self._hourly_health_state
        if (float(now_ts) - float(state.get("last_age_sample_ts", 0.0) or 0.0)) < 5.0:
            return
        for status in self._runtime_audit_exchange_statuses():
            exchange = str(status.get("name") or "").strip().lower()
            if not exchange:
                continue
            age = float(status.get("median_book_age_sec", 0.0) or 0.0)
            sums = dict(state.get("book_age_sum") or {})
            counts = dict(state.get("book_age_count") or {})
            sums[exchange] = float(sums.get(exchange, 0.0) or 0.0) + age
            counts[exchange] = int(counts.get(exchange, 0) or 0) + 1
            state["book_age_sum"] = sums
            state["book_age_count"] = counts
        state["last_age_sample_ts"] = float(now_ts)

    def _record_ingest_health(
        self,
        records: List[tuple],
        rejections: List[dict[str, Any]],
        *,
        now_ts: float,
    ) -> None:
        self._ensure_hourly_health_state(now_ts=float(now_ts))
        state = self._hourly_health_state
        state["records_total"] = int(state.get("records_total", 0) or 0) + len(records) + len(rejections)
        state["records_rejected"] = int(state.get("records_rejected", 0) or 0) + len(rejections)
        pairs_with_records = state.get("pairs_with_records")
        if isinstance(pairs_with_records, set):
            for record in records:
                pairs_with_records.add(self.tracker._pair_key(*record[:5]))
        for rejection in rejections:
            if str(rejection.get("reason") or "") == "cross_exchange_price_divergence":
                state["cross_exchange_flags"] = int(state.get("cross_exchange_flags", 0) or 0) + 1
        self._sample_book_ages(now_ts=float(now_ts))

    def _compute_hourly_digest(self, hour_end_ts: float | None = None) -> HourlyHealthDigest:
        state = dict(self._hourly_health_state)
        end_ts = float(hour_end_ts if hour_end_ts is not None else ((self._current_hour_bucket + 1) * 3600))
        records_total = int(state.get("records_total", 0) or 0)
        records_rejected = int(state.get("records_rejected", 0) or 0)
        rejection_rate_pct = (float(records_rejected) / float(records_total) * 100.0) if records_total > 0 else 0.0
        breaker_snapshot = self._circuit_breaker.snapshot(now_ts=end_ts)
        open_breakers = sorted(
            exchange
            for exchange, payload in breaker_snapshot.items()
            if str(payload.get("state") or "") == "OPEN"
        )
        pairs_with_records = len(state.get("pairs_with_records") or set())
        pairs_with_gaps = len(state.get("pairs_with_gaps") or set())
        book_age_sum = dict(state.get("book_age_sum") or {})
        book_age_count = dict(state.get("book_age_count") or {})
        avg_book_age_sec = {
            exchange: round(float(book_age_sum.get(exchange, 0.0) or 0.0) / max(int(book_age_count.get(exchange, 0) or 0), 1), 6)
            for exchange in sorted(book_age_count)
        }
        exchanges_active = sum(
            1
            for status in self._runtime_audit_exchange_statuses()
            if bool(status.get("ws_running", False)) and str(status.get("name") or "").strip().lower() not in set(open_breakers)
        )
        tracker_stats = {}
        if hasattr(self.tracker, "get_storage_stats"):
            try:
                tracker_stats = dict(self.tracker.get_storage_stats() or {})
            except Exception:
                tracker_stats = {}
        latest_episode_total = int(tracker_stats.get("episodes_total", 0) or 0)
        episode_count = max(latest_episode_total - int(state.get("episode_total_start", 0) or 0), 0)
        if rejection_rate_pct > 50.0 or len(open_breakers) >= 2 or pairs_with_records < max(1, int(len(self._last_filtered_opps) * 0.5)):
            verdict = "unhealthy"
        elif rejection_rate_pct >= 20.0 or len(open_breakers) == 1:
            verdict = "degraded"
        else:
            verdict = "healthy"
        return HourlyHealthDigest(
            hour_start_ts=float(state.get("hour_start_ts", self._current_hour_bucket * 3600) or (self._current_hour_bucket * 3600)),
            hour_end_ts=float(end_ts),
            records_total=int(records_total),
            records_rejected=int(records_rejected),
            rejection_rate_pct=float(round(rejection_rate_pct, 6)),
            exchanges_active=int(exchanges_active),
            exchanges_circuit_open=open_breakers,
            pairs_with_records=int(pairs_with_records),
            pairs_with_gaps=int(pairs_with_gaps),
            cross_exchange_flags=int(state.get("cross_exchange_flags", 0) or 0),
            avg_book_age_sec=avg_book_age_sec,
            episode_count=int(episode_count),
            quality_verdict=str(verdict),
        )

    def _on_tracker_event(self, payload: dict[str, Any]) -> None:
        if str(payload.get("kind") or "") != "tracker_record":
            return
        self._ensure_hourly_health_state(now_ts=float(payload.get("record_ts", time.time()) or time.time()))
        if bool(payload.get("gap_detected")):
            pairs_with_gaps = self._hourly_health_state.get("pairs_with_gaps")
            if isinstance(pairs_with_gaps, set):
                pairs_with_gaps.add(str(payload.get("pair_key") or ""))

    def _apply_ingest_filters(
        self,
        opportunities: List[SpreadOpportunity],
        records: List[tuple],
        rejections: List[dict[str, Any]],
        *,
        now_ts: float,
    ) -> tuple[List[SpreadOpportunity], List[tuple], List[dict[str, Any]]]:
        def _filter_opportunities_for_breakers(rows: List[SpreadOpportunity]) -> List[SpreadOpportunity]:
            if not rows:
                return rows
            active_cache: dict[str, bool] = {}

            def _is_active(exchange: str) -> bool:
                key = str(exchange or "").strip().lower()
                if not key:
                    return True
                cached = active_cache.get(key)
                if cached is None:
                    cached = self._circuit_breaker.is_active(key, now_ts=float(now_ts))
                    active_cache[key] = cached
                return bool(cached)

            return [
                opp
                for opp in rows
                if _is_active(opp.buy_exchange) and _is_active(opp.sell_exchange)
            ]

        if not records and not rejections:
            self._record_ingest_health([], [], now_ts=float(now_ts))
            return _filter_opportunities_for_breakers(opportunities), records, rejections
        filtered_records: List[tuple] = []
        filtered_rejections = list(rejections)
        for rejection in filtered_rejections:
            affected = list(rejection.get("affected_exchanges") or [])
            if not affected:
                affected = [
                    str(rejection.get("buy_ex") or "").strip().lower(),
                    str(rejection.get("sell_ex") or "").strip().lower(),
                ]
            for exchange in {item for item in affected if item}:
                self._circuit_breaker.record(exchange, accepted=False, now_ts=float(now_ts))
        for record in records:
            key = self.tracker._pair_key(*record[:5])
            blocked = [
                exchange
                for exchange in {str(record[1] or "").strip().lower(), str(record[3] or "").strip().lower()}
                if exchange and not self._circuit_breaker.is_active(exchange, now_ts=float(now_ts))
            ]
            if blocked:
                filtered_rejections.append(
                    {
                        "symbol": str(record[0] or ""),
                        "buy_ex": str(record[1] or ""),
                        "buy_mt": str(record[2] or ""),
                        "sell_ex": str(record[3] or ""),
                        "sell_mt": str(record[4] or ""),
                        "invalid_fields": [],
                        "reason": "exchange_circuit_open",
                        "affected_exchanges": blocked,
                    }
                )
                for exchange in blocked:
                    self._circuit_breaker.record(exchange, accepted=False, now_ts=float(now_ts))
                continue
            filtered_records.append(record)
            for exchange in {str(record[1] or "").strip().lower(), str(record[3] or "").strip().lower()}:
                if exchange:
                    self._circuit_breaker.record(exchange, accepted=True, now_ts=float(now_ts))
        filtered_opportunities = _filter_opportunities_for_breakers(opportunities)
        self._record_ingest_health(filtered_records, filtered_rejections, now_ts=float(now_ts))
        return filtered_opportunities, filtered_records, filtered_rejections

    def _get_history_mirror(self, key: Any, *, limit: int = 100) -> List[Dict[str, float]]:
        with self._tracker_ingest_lock:
            history = self._tracker_history_mirror.get(key)
            if not history:
                return []
            window = list(history)[-max(int(limit), 0):]
        return [
            {
                "timestamp": float(item["timestamp"]),
                "entry_spread": float(item["entry_spread"]),
                "exit_spread": float(item["exit_spread"]),
            }
            for item in window
        ]

    def _drain_tracker_ingest_batch(
        self,
        *,
        now_ts: float,
        drain_all: bool = False,
    ) -> Dict[str, int | float]:
        started = time.perf_counter()
        with self._tracker_ingest_lock:
            record_keys = list(self._tracker_pending_records.keys())
            rejection_keys = list(self._tracker_pending_rejections.keys())
            if not drain_all:
                record_keys = record_keys[: self._TRACKER_INGEST_BATCH_MAX]
                rejection_keys = rejection_keys[: self._TRACKER_INGEST_BATCH_MAX]
            records = [self._tracker_pending_records.pop(key) for key in record_keys]
            rejections = [self._tracker_pending_rejections.pop(key) for key in rejection_keys]
            pending_records_after = len(self._tracker_pending_records)
            pending_rejections_after = len(self._tracker_pending_rejections)
        if records:
            self.tracker.batch_record(records, now_ts=now_ts)
        if rejections:
            self.tracker.batch_record_rejections(rejections)
        total_ms = (time.perf_counter() - started) * 1000.0
        if self.perf_monitor is not None:
            self.perf_monitor.record_scanner_cycle(
                {
                    "kind": "tracker_ingest",
                    "total_ms": round(total_ms, 6),
                    "tracker_records_drained": len(records),
                    "tracker_rejections_drained": len(rejections),
                    "tracker_pending_records": pending_records_after,
                    "tracker_pending_rejections": pending_rejections_after,
                }
            )
        return {
            "records_drained": len(records),
            "rejections_drained": len(rejections),
            "pending_records": pending_records_after,
            "pending_rejections": pending_rejections_after,
            "total_ms": round(total_ms, 6),
        }

    def _drain_ml_inference_batch(
        self,
        *,
        now_ts: float,
    ) -> tuple[float, float, float, int]:
        history_fetch_ms = 0.0
        ml_analyze_ms = 0.0
        ml_render_ms = 0.0
        processed = 0
        started = time.perf_counter()
        queue_size_before = self._ml_queue_size()
        while processed < self._ML_REFRESHES_PER_CYCLE:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if processed > 0 and elapsed_ms >= self._ML_REFRESH_BUDGET_MS:
                break
            with self._ml_state_lock:
                if not self._ml_pending_refreshes:
                    break
                key, refresh_payload = self._ml_pending_refreshes.popitem(last=False)
            _history_started = time.perf_counter()
            history = self._get_history_mirror(key, limit=100)
            history_fetch_ms += (time.perf_counter() - _history_started) * 1000.0
            _ml_started = time.perf_counter()
            try:
                prediction = self.ml_analyzer.predict_history(history, pair_key=key)
            except TypeError:
                prediction = self.ml_analyzer.predict_history(history)
            ml_analyze_ms += (time.perf_counter() - _ml_started) * 1000.0
            _render_started = time.perf_counter()
            rendered_ctx = self.ml_analyzer.render_prediction(
                float(refresh_payload.get("current_entry", 0.0)),
                prediction,
                pair_key=key,
            )
            ml_render_ms += (time.perf_counter() - _render_started) * 1000.0
            with self._ml_state_lock:
                cache_entry = self._ml_cache.get(key) or {}
                cache_entry["prediction"] = prediction
                cache_entry["tracker_marker"] = refresh_payload["tracker_marker"]
                cache_entry["computed_at"] = now_ts
                cache_entry["context"] = rendered_ctx
                cache_entry["rendered_at"] = now_ts
                cache_entry["rendered_entry_spread"] = float(refresh_payload.get("current_entry", 0.0))
                self._ml_cache[key] = cache_entry
                self._ml_cache.move_to_end(key)
                if len(self._ml_cache) > self._ML_CACHE_MAX:
                    self._ml_cache.popitem(last=False)
                queue_size_after = len(self._ml_pending_refreshes)
                cache_size_after = len(self._ml_cache)
            if self.runtime_audit is not None and str(prediction.get("prediction_status") or "") == "ready":
                end_to_end_ms = None
                started_at_perf = float(refresh_payload.get("queued_at_perf", 0.0) or 0.0)
                if started_at_perf > 0.0:
                    end_to_end_ms = (time.perf_counter() - started_at_perf) * 1000.0
                self.runtime_audit.record_inference(
                    pair_id=self.tracker._pair_id(key),
                    result=rendered_ctx,
                    end_to_end_ms=end_to_end_ms,
                )
            processed += 1
        total_ms = (time.perf_counter() - started) * 1000.0
        if self.perf_monitor is not None:
            self.perf_monitor.record_scanner_cycle(
                {
                    "kind": "ml_inference",
                    "total_ms": round(total_ms, 6),
                    "history_fetch_ms": round(history_fetch_ms, 6),
                    "ml_analyze_ms": round(ml_analyze_ms, 6),
                    "ml_render_ms": round(ml_render_ms, 6),
                    "ml_predictions_drained": int(processed),
                    "ml_refresh_queue_size": int(queue_size_after if processed else queue_size_before),
                    "ml_cache_size": int(cache_size_after if processed else self._ml_cache_size()),
                }
            )
        return history_fetch_ms, ml_analyze_ms, ml_render_ms, processed

    def _prune_ml_runtime_state(self, active_keys: Set[Any], *, now_ts: float) -> None:
        with self._ml_state_lock:
            if self._ml_pending_refreshes:
                stale_pending = [key for key in self._ml_pending_refreshes.keys() if key not in active_keys]
                for stale_key in stale_pending:
                    self._ml_pending_refreshes.pop(stale_key, None)
            if not self._ml_cache:
                return
            stale_cache: List[Any] = []
            for key, cache_entry in list(self._ml_cache.items()):
                if key in active_keys:
                    cache_entry["last_seen_at"] = now_ts
                    continue
                last_seen_at = float(cache_entry.get("last_seen_at", 0.0) or 0.0)
                if last_seen_at <= 0.0 or (now_ts - last_seen_at) >= (self._ML_PREDICTION_MAX_AGE_SEC * 4.0):
                    stale_cache.append(key)
            for stale_key in stale_cache:
                self._ml_cache.pop(stale_key, None)
        with self._tracker_ingest_lock:
            stale_history = [key for key in self._tracker_history_mirror.keys() if key not in active_keys]
            for stale_key in stale_history:
                self._tracker_history_mirror.pop(stale_key, None)
                self._tracker_history_last_ts.pop(stale_key, None)

    def mark_scanner_lite_interest(self) -> None:
        with self._lock:
            self._last_scanner_lite_interest_perf = time.perf_counter()
            self._force_scanner_lite_refresh = True

    def _should_refresh_scanner_lite_state(self, *, now_perf: float | None = None) -> bool:
        with self._lock:
            now_value = float(now_perf) if now_perf is not None else time.perf_counter()
            has_clients = bool(self._scanner_lite_clients)
            recent_interest = (
                self._last_scanner_lite_interest_perf > 0.0
                and (now_value - self._last_scanner_lite_interest_perf) <= self._SCANNER_LITE_INTEREST_TTL_SEC
            )
            return bool(self._force_scanner_lite_refresh or has_clients or recent_interest)

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
        with self._lock:
            previous_by_key = dict(self._last_scanner_lite_rows_by_key)
        removes = [key for key in previous_by_key if key not in rows_by_key]
        upserts = [
            row for key, row in rows_by_key.items()
            if previous_by_key.get(key) != row
        ]
        now_iso = datetime.now(timezone.utc).isoformat()
        summary = self._build_scanner_lite_summary(rows)
        snapshot = {
            "timestamp": now_iso,
            "type": "arbitrage_data_lite_snapshot",
            "source": "spread_engine",
            "count": len(rows),
            "total_count": len(opportunities),
            "summary": summary,
            "data": rows,
        }
        delta = {
            "timestamp": now_iso,
            "type": "arbitrage_data_lite_delta",
            "source": "spread_engine",
            "count": len(upserts),
            "remove_count": len(removes),
            "total_count": len(opportunities),
            "summary": summary,
            "upserts": upserts,
            "removes": removes,
        }
        with self._lock:
            self._last_scanner_lite_rows = rows
            self._last_scanner_lite_rows_by_key = rows_by_key
            self._last_scanner_lite_summary = summary
            self._last_scanner_lite_snapshot = snapshot
            self._last_scanner_lite_delta = delta
            self._force_scanner_lite_refresh = False

    async def _ml_inference_loop(self):
        """Drain ML predictions off the calc/broadcast hot path."""
        while self._running:
            try:
                if self.ml_analyzer.model_status != "ready":
                    await asyncio.sleep(5.0)
                    continue
                if self._ml_queue_size() <= 0:
                    await asyncio.sleep(1.0)
                    continue
                await asyncio.to_thread(
                    self._drain_ml_inference_batch,
                    now_ts=time.time(),
                )
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[WSManager] ML inference loop error: {exc}")
                if self.runtime_audit is not None:
                    self.runtime_audit.record_error("ml_inference_loop", str(exc))
                await asyncio.sleep(5.0)

    async def _tracker_ingest_loop(self):
        """Drain tracker ingest queue outside the calc hot path."""
        while self._running:
            try:
                if self._tracker_pending_record_count() <= 0 and self._tracker_pending_rejection_count() <= 0:
                    await asyncio.sleep(self._TRACKER_INGEST_LOOP_SEC)
                    continue
                await asyncio.to_thread(
                    self._drain_tracker_ingest_batch,
                    now_ts=time.time(),
                )
                await asyncio.sleep(self._TRACKER_INGEST_LOOP_SEC)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[WSManager] Tracker ingest loop error: {exc}")
                if self.runtime_audit is not None:
                    self.runtime_audit.record_error("tracker_ingest_loop", str(exc))
                await asyncio.sleep(1.0)

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
        self._startup_started_at = time.time()
        self._startup_filtered_opportunities = 0
        self._last_startup_gate_state = self._startup_gate_state(now_ts=self._startup_started_at)

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
            instance.spot_feed_mode = str(getattr(ex_config, "spot_feed_mode", "depth_ws") or "depth_ws")
            instance.futures_feed_mode = str(getattr(ex_config, "futures_feed_mode", "depth_ws") or "depth_ws")
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

        tracker_ingest_task = asyncio.create_task(
            self._tracker_ingest_loop(),
            name="tracker_ingest",
        )
        self._tasks.append(tracker_ingest_task)

        ml_task = asyncio.create_task(
            self._ml_inference_loop(),
            name="ml_inference",
        )
        self._tasks.append(ml_task)

        # Start tracker persistence loop (save every 60s)
        persist_task = asyncio.create_task(
            self._persist_loop(),
            name="tracker_persist",
        )
        self._tasks.append(persist_task)

        snapshot_task = asyncio.create_task(
            self._snapshot_scheduler(),
            name="snapshot_scheduler",
        )
        self._tasks.append(snapshot_task)

        maintenance_task = asyncio.create_task(
            self._db_maintenance(),
            name="db_maintenance",
        )
        self._tasks.append(maintenance_task)

        archive_task = asyncio.create_task(
            self._archive_old_snapshots(),
            name="archive_scheduler",
        )
        self._tasks.append(archive_task)

        auto_retrain_task = asyncio.create_task(
            self._auto_retrain_loop(),
            name="auto_retrain",
        )
        self._tasks.append(auto_retrain_task)

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

        tracker_every = self._tracker_cycle_every()
        logger.info(
            "[WSManager] Tracker cadence: every %s calc cycles (~%.2fs)",
            tracker_every,
            tracker_every * max(float(getattr(self.config, "broadcast_interval_sec", 0.15) or 0.15), 0.001),
        )
        logger.info("[WSManager] All tasks started")

    async def stop(self):
        """Stop all connections and tasks."""
        self._running = False
        try:
            if self._hourly_health_state:
                digest = self._compute_hourly_digest(time.time())
                self._last_health_digest = digest.to_dict()
                self.tracker.save_hourly_health_digest(self._last_health_digest)
        except Exception:
            pass
        for instance in self._exchange_instances.values():
            instance.stop()  # sets shutdown + cancels tasks in exchange loop
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads.clear()
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        if self._auto_retrain_process is not None and self._auto_retrain_process.poll() is None:
            self._auto_retrain_process.terminate()
            self._auto_retrain_process = None

        await asyncio.to_thread(
            self._drain_tracker_ingest_batch,
            now_ts=time.time(),
            drain_all=True,
        )

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
            "ml_cache_size": self._ml_cache_size(),
            "ml_refresh_queue_size": self._ml_queue_size(),
            "scanner_clients": len(self._scanner_clients),
            "scanner_lite_clients": len(self._scanner_lite_clients),
            "opportunities_cached": len(self._last_filtered_opps),
            "scanner_lite_rows": len(self._last_scanner_lite_rows),
            "tracker_cycle_every": self._tracker_cycle_every(),
            "tracker_cycle_offset": 2,
            "tracker_capture_shards": self._tracker_capture_shards(),
            "startup_gate": dict(self._last_startup_gate_state),
            "startup_filtered_opportunities": int(self._startup_filtered_opportunities),
            "scanner_lite_interest_age_sec": round(
                max(0.0, time.perf_counter() - self._last_scanner_lite_interest_perf),
                3,
            ) if self._last_scanner_lite_interest_perf > 0.0 else None,
            "scanner_lite_interest_active": self._should_refresh_scanner_lite_state(),
            "tracker_ingest_queue_records": self._tracker_pending_record_count(),
            "tracker_ingest_queue_rejections": self._tracker_pending_rejection_count(),
            "tracker_history_pairs": self._history_mirror_size(),
            "tracker_pairs": int(
                tracker_stats.get(
                    "pairs_in_memory",
                    tracker_stats.get("pairs_persisted", tracker_stats.get("pairs_tracked", 0)),
                )
                or 0
            ),
            "tracker_records_total": int(tracker_stats.get("records_total", 0) or 0),
            "tracker_episodes_total": int(tracker_stats.get("episodes_total", 0) or 0),
            "tracker_db_size_bytes": int(tracker_stats.get("db_size_bytes", 0) or 0),
            "process_rss_mb": process_rss_mb,
            "process_threads": process_threads,
        }

    def get_system_health(self) -> dict[str, Any]:
        latest_hour = self.tracker.get_latest_hourly_health() or self._last_health_digest or (
            self._compute_hourly_digest(time.time()).to_dict() if self._hourly_health_state else {}
        )
        circuit_snapshot = self._circuit_breaker.snapshot()
        latest_snapshot = latest_snapshot_entry(base_path=self._artifact_root) or {}
        return {
            "hour_verdict": str(latest_hour.get("quality_verdict") or "unknown"),
            "rejection_rate_pct": float(latest_hour.get("rejection_rate_pct", 0.0) or 0.0),
            "circuit_breakers": {
                exchange: str(payload.get("state") or "CLOSED")
                for exchange, payload in circuit_snapshot.items()
                if str(payload.get("state") or "CLOSED") != "CLOSED"
            },
            "next_snapshot_sec": int(round(seconds_until_next_snapshot_slot())),
            "last_snapshot": str(latest_snapshot.get("filename") or "").replace("snapshot_", "").replace(".sqlite", ""),
            "last_snapshot_verdict": str(latest_snapshot.get("certification_verdict") or "unknown"),
        }

    def get_pipeline_status(self) -> dict[str, Any]:
        snapshot_manifest = load_snapshot_manifest(base_path=self._artifact_root)
        training_manifest = load_training_manifest(base_path=self._artifact_root)
        now_ts = time.time()
        recent_snapshots = list(snapshot_manifest.get("snapshots") or [])
        last_7d = [
            item
            for item in recent_snapshots
            if float(item.get("created_at_utc_ts", 0.0) or 0.0) >= (now_ts - (7 * 86400))
        ]
        latest_snapshot = latest_snapshot_entry(base_path=self._artifact_root) or {}
        latest_run = latest_training_run(base_path=self._artifact_root) or {}
        hourly = self.tracker.list_hourly_health(limit=24)
        auto_state = self._read_auto_retrain_state()
        return {
            "snapshots": {
                "last": str(latest_snapshot.get("filename") or "").replace("snapshot_", "").replace(".sqlite", ""),
                "last_verdict": str(latest_snapshot.get("certification_verdict") or "unknown"),
                "next_in_sec": int(round(seconds_until_next_snapshot_slot())),
                "total_7d": len(last_7d),
                "pass_7d": sum(1 for item in last_7d if str(item.get("certification_verdict") or "") == "PASS"),
                "warn_7d": sum(1 for item in last_7d if str(item.get("certification_verdict") or "") == "WARN"),
                "certified_7d": sum(1 for item in last_7d if str(item.get("certification_verdict") or "") in {"PASS", "WARN"}),
                "fail_7d": sum(1 for item in last_7d if str(item.get("certification_verdict") or "") == "FAIL"),
                "recent": list(reversed(recent_snapshots[-10:])),
            },
            "training": {
                "last_run": str(latest_run.get("run_id") or ""),
                "last_auc": float(latest_run.get("challenger_auc", 0.0) or 0.0) if latest_run else 0.0,
                "model_version": self.ml_analyzer.model_version,
                "deployed_at": str(auto_state.get("updated_at_utc") or ""),
                "retrain_trigger": str(auto_state.get("reason") or "") if str(auto_state.get("status") or "") in {"queued", "running", "failed"} else None,
                "auto_state": auto_state,
                "run_count": len(list(training_manifest.get("runs") or [])),
            },
            "hourly_health": {
                "last_24h": [str(item.get("quality_verdict") or "unknown") for item in reversed(hourly)],
                "healthy_count": sum(1 for item in hourly if str(item.get("quality_verdict") or "") == "healthy"),
                "degraded_count": sum(1 for item in hourly if str(item.get("quality_verdict") or "") == "degraded"),
                "unhealthy_count": sum(1 for item in hourly if str(item.get("quality_verdict") or "") == "unhealthy"),
            },
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

    def _do_calc_enrich(self, capture_tracker: bool, capture_shard: int = 0):
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
        capture_mode = str(getattr(self.config, "tracker_capture_mode", "continuous_all_pairs") or "continuous_all_pairs").strip().lower()
        capture_tracker_records = bool(capture_tracker and capture_mode == "continuous_all_pairs")
        tracker_shard_count = self._tracker_capture_shards()
        raw_capture_records: List[tuple] | None = [] if capture_tracker_records else None
        raw_capture_rejections: List[dict[str, Any]] | None = [] if capture_tracker_records else None
        try:
            opportunities = self.engine.calculate_all(
                on_spread=None,
                record_sink=raw_capture_records,
                rejection_sink=raw_capture_rejections,
            )
        except TypeError:
            try:
                opportunities = self.engine.calculate_all(
                    on_spread=None,
                    record_sink=raw_capture_records,
                )
            except TypeError:
                opportunities = self.engine.calculate_all(on_spread=None)
        _t_calc = time.time()
        _perf_after_calc = time.perf_counter()

        now_ts = time.time()
        opportunities, raw_capture_records, raw_capture_rejections = self._apply_ingest_filters(
            opportunities,
            list(raw_capture_records or []),
            list(raw_capture_rejections or []),
            now_ts=float(now_ts),
        )
        _perf_after_ingest_filters = time.perf_counter()

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
        pair_key = self.tracker._pair_key
        opportunity_keys = [
            pair_key(opp.asset, opp.buy_exchange, opp.buy_market_type,
                     opp.sell_exchange, opp.sell_market_type)
            for opp in opportunities
        ]
        active_keys = set(opportunity_keys)

        if refresh_tracker:
            tracker_cache = self.tracker.batch_enrich(opportunity_keys, now_ts)
            self._tracker_enrich_cache = tracker_cache
        elif tracker_cache:
            tracker_cache = {key: tracker_cache.get(key) for key in opportunity_keys}
            self._tracker_enrich_cache = tracker_cache
        _perf_after_tracker_enrich = time.perf_counter()

        _history_fetch_ms = 0.0
        _ml_analyze_ms = 0.0
        _ml_render_ms = 0.0
        drained_predictions = 0
        ml_model_ready = self.ml_analyzer.model_status == "ready"
        if tracker_cache:
            pending_refreshes: List[tuple[Any, Dict[str, Any]]] = []
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

                if not ml_model_ready:
                    continue

                cache_entry = self._get_ml_cache_entry(key)
                if cache_entry and cache_entry.get("context") is not None:
                    ctx = cache_entry["context"]
                    opp.ml_context = ctx
                    opp.ml_score = ctx.get("ml_score", 0)
                if self._cache_prediction_is_stale(cache_entry, cached, now_ts=now_ts):
                    pending_refreshes.append(
                        (
                            key,
                            {
                                "symbol": opp.asset,
                                "buy_exchange": opp.buy_exchange,
                                "buy_market_type": opp.buy_market_type,
                                "sell_exchange": opp.sell_exchange,
                                "sell_market_type": opp.sell_market_type,
                                "current_entry": float(opp.entry_spread_pct),
                                "tracker_marker": self._tracker_marker_from_cache(cached),
                                "queued_at_perf": time.perf_counter(),
                            },
                        )
                    )
            if ml_model_ready and pending_refreshes:
                for key, payload in pending_refreshes:
                    self._queue_ml_refresh(key, payload)
        self._prune_ml_runtime_state(active_keys, now_ts=now_ts)
        _perf_after_ml = time.perf_counter()

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
        record_source_opportunities = list(opportunities)
        startup_gate = self._startup_gate_state(now_ts=now_ts)
        self._last_startup_gate_state = dict(startup_gate)
        published_opportunities = opportunities
        if not bool(startup_gate.get("allow_publication")):
            self._startup_filtered_opportunities = len(opportunities)
            published_opportunities = []
        else:
            self._startup_filtered_opportunities = 0

        _records_batch = []
        queued_records = 0
        queued_rejections = 0
        if capture_tracker_records:
            _records_batch = list(raw_capture_records or [])
        elif capture_tracker:
            for opp in record_source_opportunities:
                record = (
                    opp.asset,
                    opp.buy_exchange,
                    opp.buy_market_type,
                    opp.sell_exchange,
                    opp.sell_market_type,
                    opp.entry_spread_pct,
                    opp.exit_spread_pct,
                )
                _records_batch.append(record)
        if _records_batch:
            queued_records = self._queue_tracker_records(
                _records_batch,
                now_ts=now_ts,
                capture_shard=int(capture_shard),
                shard_count=tracker_shard_count,
            )
        if raw_capture_rejections and capture_tracker_records:
            queued_rejections = self._queue_tracker_rejections(
                list(raw_capture_rejections),
                capture_shard=int(capture_shard),
                shard_count=tracker_shard_count,
            )
        _perf_after_batch_record = time.perf_counter()

        if self._should_refresh_scanner_lite_state():
            self._refresh_scanner_lite_state(published_opportunities)
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
                    "ingest_filter_ms": round((_perf_after_ingest_filters - _perf_after_calc) * 1000.0, 6),
                    "market_enrich_ms": round((_perf_after_market_enrich - _perf_after_ingest_filters) * 1000.0, 6),
                    "tracker_enrich_ms": round((_perf_after_tracker_enrich - _perf_after_market_enrich) * 1000.0, 6),
                    "filter_ms": round((_perf_after_filter - _perf_after_ml) * 1000.0, 6),
                    "batch_record_ms": round((_perf_after_batch_record - _perf_after_filter) * 1000.0, 6),
                    "history_fetch_ms": round(_history_fetch_ms, 6),
                    "ml_analyze_ms": round(_ml_analyze_ms, 6),
                    "ml_render_ms": round(_ml_render_ms, 6),
                    "lite_refresh_ms": round((_perf_after_lite_refresh - _perf_after_batch_record) * 1000.0, 6),
                    "opportunities_before_filter": len(self.engine._opportunities),
                    "opportunities_after_filter": len(published_opportunities),
                    "refresh_tracker": bool(refresh_tracker),
                    "tracker_cache_size": len(tracker_cache or {}),
                    "ml_cache_size": self._ml_cache_size(),
                    "ml_refresh_queue_size": self._ml_queue_size(),
                    "ml_predictions_drained": int(drained_predictions),
                    "tracker_records_enqueued": int(queued_records),
                    "tracker_rejections_enqueued": int(queued_rejections),
                    "tracker_pending_records": self._tracker_pending_record_count(),
                    "tracker_pending_rejections": self._tracker_pending_rejection_count(),
                    "tracker_capture_shards": tracker_shard_count,
                    "startup_gate_open": 1 if startup_gate.get("allow_publication") else 0,
                    "startup_filtered_opportunities": int(self._startup_filtered_opportunities),
                    "market_data_ready": 1 if startup_gate.get("market_data_ready") else 0,
                }
            )
        return published_opportunities

    async def _calc_and_broadcast_loop(self):
        """Periodically calculate spreads and broadcast to scanner clients."""
        _cycle = 0
        _TRACKER_EVERY = self._tracker_cycle_every()
        # Offset tracker recording by +2 so it never coincides with
        # full-recalc cycles (engine._FULL_RECALC_EVERY = 80).
        # Full-recalc: cycle 80, 160, 240 …  Tracker: cycle 3, 8, 13, 18 …
        # Since (80+2)%tracker_every should remain non-zero for the current
        # cadence, they avoid worst-case overlap of
        # "full-recalc + massive batch_record" in the same cycle.
        _TRACKER_OFFSET = 2
        _LOG_EVERY = 21
        _max_calc_ms = 0  # track worst cycle for health monitoring
        _capture_round = 0
        while self._running:
            try:
                _cycle += 1
                t0 = time.time()

                # Tracker recording on offset cycles (avoids full-recalc overlap).
                capture_tracker = ((_cycle + _TRACKER_OFFSET) % _TRACKER_EVERY == 0)
                capture_shard = _capture_round % self._tracker_capture_shards() if capture_tracker else 0
                if capture_tracker:
                    _capture_round += 1

                # Run CPU-heavy calc+enrichment in a thread so the event loop
                # stays free for WS heartbeats, pings, and message delivery.
                opportunities = await asyncio.to_thread(
                    self._do_calc_enrich, capture_tracker, capture_shard
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
                        f"tracker={'ON' if capture_tracker else 'off'} "
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
            delta = dict(self._last_scanner_lite_delta)
            snapshot = dict(self._last_scanner_lite_snapshot)

        if not clients:
            return

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
                await asyncio.to_thread(self.tracker.prune)
            except Exception as e:
                logger.error(f"[WSManager] Prune error: {e}")

    async def _persist_loop(self):
        """Periodically flush tracker state to SQLite."""
        while self._running:
            await asyncio.sleep(60)
            try:
                started = time.perf_counter()
                flushed = await asyncio.to_thread(self.tracker.flush_to_storage)
                if not flushed:
                    logger.warning("[WSManager] Tracker SQLite flush returned False")
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

    def _snapshot_path_for_slot(self, slot_time: datetime) -> Path:
        label = slot_time.astimezone(timezone.utc).strftime("%Y-%m-%d_%Hh")
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        return self._snapshots_dir / f"snapshot_{label}.sqlite"

    async def _create_snapshot_for_slot(self, slot_time: datetime) -> Path | None:
        target = self._snapshot_path_for_slot(slot_time)
        if target.exists():
            return target
        await asyncio.to_thread(self._drain_tracker_ingest_batch, now_ts=time.time(), drain_all=True)
        await asyncio.to_thread(self.tracker.flush_to_storage, force=True)
        await asyncio.to_thread(create_sqlite_snapshot, self.tracker.db_path, target)
        certification = await asyncio.to_thread(
            certify_data_for_training,
            state_file=target,
            certification_mode="quick",
        )
        cert_path = target.with_suffix(".cert.json")
        cert_path.write_text(json.dumps(certification, indent=2, sort_keys=True), encoding="utf-8")
        update_snapshot_manifest(target, certification, base_path=self._artifact_root)
        return target

    async def _snapshot_scheduler(self):
        while self._running:
            try:
                await asyncio.sleep(seconds_until_next_snapshot_slot())
                slot_time = current_snapshot_slot(datetime.now(timezone.utc))
                snapshot_path = await self._create_snapshot_for_slot(slot_time)
                if snapshot_path is not None:
                    logger.info("[Snapshot] Created %s", snapshot_path.name)
                await self._evaluate_auto_retrain(force_check=True)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[Snapshot] Failed: %s", exc)
                if self.runtime_audit is not None:
                    self.runtime_audit.record_error("snapshot_scheduler", str(exc))
                await asyncio.sleep(30.0)

    def _run_db_cleanup(self, cutoff: float) -> None:
        with self.tracker._connect() as conn:
            conn.execute("DELETE FROM tracker_records WHERE ts < ?", (float(cutoff),))
            conn.execute("DELETE FROM tracker_events WHERE ts < ?", (float(cutoff),))
            conn.execute("DELETE FROM tracker_pair_episodes WHERE end_ts < ?", (float(cutoff),))
            conn.execute(
                """
                DELETE FROM tracker_pair_blocks
                WHERE end_ts < ? AND is_open = 0
                """,
                (float(cutoff),),
            )
            if self.tracker.db_path is not None:
                conn.execute("DELETE FROM tracker_hourly_health WHERE hour_end_ts < ?", (float(cutoff),))
        with sqlite3.connect(self.tracker.db_path, timeout=30.0) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")

    async def _db_maintenance(self):
        while self._running:
            try:
                await asyncio.sleep(24 * 60 * 60)
                latest_snapshot = latest_snapshot_entry(base_path=self._artifact_root)
                if latest_snapshot is None:
                    continue
                cutoff = time.time() - (8 * 86400)
                snapshot_range = list(latest_snapshot.get("record_time_range") or [0.0, 0.0])
                if float(snapshot_range[0] or 0.0) > cutoff:
                    continue
                await asyncio.to_thread(self._run_db_cleanup, cutoff)
                logger.info("[Maintenance] DB cleanup completed")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[Maintenance] Failed: %s", exc)
                if self.runtime_audit is not None:
                    self.runtime_audit.record_error("db_maintenance", str(exc))

    async def _archive_old_snapshots(self):
        while self._running:
            try:
                await asyncio.sleep(7 * 24 * 60 * 60)
                moved = await asyncio.to_thread(archive_expired_snapshots, base_path=self._artifact_root)
                if moved:
                    logger.info("[Archive] Moved %s snapshots", len(moved))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[Archive] Failed: %s", exc)
                if self.runtime_audit is not None:
                    self.runtime_audit.record_error("archive_scheduler", str(exc))

    def _read_auto_retrain_state(self) -> dict[str, Any]:
        if not self._auto_retrain_state_path.is_file():
            return {}
        try:
            return json.loads(self._auto_retrain_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _maybe_reload_deployed_model(self) -> None:
        deployed_version = ""
        metadata_path = self._artifact_root / "config" / "best_lstm_model.meta.json"
        if metadata_path.is_file():
            try:
                deployed_version = str(json.loads(metadata_path.read_text(encoding="utf-8")).get("version") or "")
            except Exception:
                deployed_version = ""
        if deployed_version and deployed_version != self._last_reloaded_model_version:
            self.ml_analyzer.reload_artifact()
            self._last_reloaded_model_version = self.ml_analyzer.model_version

    async def _evaluate_auto_retrain(self, *, force_check: bool = False) -> None:
        if self._auto_retrain_process is not None and self._auto_retrain_process.poll() is None:
            return
        if self._auto_retrain_process is not None and self._auto_retrain_process.poll() is not None:
            self._auto_retrain_process = None
            self._maybe_reload_deployed_model()
        reason = should_retrain(base_path=self._artifact_root, model_dir=self._artifact_root / "config")
        if reason is None and not force_check:
            return
        if reason is None:
            return
        self._auto_retrain_process = launch_auto_retrain_worker(
            reason=reason,
            tracker_db_path=self._tracker_db_path,
            base_path=self._artifact_root,
        )
        logger.info("[Retrain] Launched auto retrain worker (%s)", reason)

    async def _auto_retrain_loop(self):
        while self._running:
            try:
                await asyncio.sleep(60 * 60)
                await self._evaluate_auto_retrain()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[Retrain] Scheduler failed: %s", exc)
                if self.runtime_audit is not None:
                    self.runtime_audit.record_error("auto_retrain_loop", str(exc))

    def _runtime_audit_exchange_statuses(self) -> list[dict]:
        statuses: list[dict] = []
        for ex_name, instance in self._exchange_instances.items():
            try:
                ages: list[float] = []
                invalid_books = 0
                for book in getattr(instance, "_books", {}).values():
                    book_ts = float(getattr(book, "_timestamp", 0.0) or 0.0)
                    if book_ts <= 0.0:
                        invalid_books += 1
                        continue
                    ages.append(max(0.0, time.time() - book_ts))
                ordered_ages = sorted(ages)
                median_age = ordered_ages[len(ordered_ages) // 2] if ordered_ages else 0.0
                p95_age = ordered_ages[min(len(ordered_ages) - 1, int(len(ordered_ages) * 0.95))] if ordered_ages else 0.0
                statuses.append(
                    {
                        "name": ex_name,
                        "ws_running": not instance.shutdown.is_set(),
                        "spot_feed_mode": instance.get_feed_mode("spot"),
                        "futures_feed_mode": instance.get_feed_mode("futures"),
                        "spot_reconnects": int(instance.get_reconnect_counts().get("spot", 0) or 0),
                        "futures_reconnects": int(instance.get_reconnect_counts().get("futures", 0) or 0),
                        "book_count": len(getattr(instance, "_books", {})),
                        "median_book_age_sec": round(float(median_age), 6),
                        "p95_book_age_sec": round(float(p95_age), 6),
                        "max_book_age_sec": round(float(max(ages, default=0.0)), 6),
                        "invalid_book_count": int(invalid_books),
                    }
                )
            except Exception:
                statuses.append({"name": ex_name, "ws_running": False, "book_count": 0})
        return statuses

    def _runtime_audit_book_health(self) -> dict[str, float | int]:
        buy_ages: list[float] = []
        sell_ages: list[float] = []
        asymmetries: list[float] = []
        invalid_quote_count = 0
        opportunities = list(getattr(self, "_last_filtered_opps", []) or [])
        for opp in opportunities:
            buy_age = float(getattr(opp, "buy_book_age", -1.0) or -1.0)
            sell_age = float(getattr(opp, "sell_book_age", -1.0) or -1.0)
            if buy_age >= 0.0:
                buy_ages.append(buy_age)
            if sell_age >= 0.0:
                sell_ages.append(sell_age)
            if buy_age >= 0.0 and sell_age >= 0.0:
                asymmetries.append(abs(buy_age - sell_age))
            if (
                float(getattr(opp, "buy_price", 0.0) or 0.0) <= 0.0
                or float(getattr(opp, "sell_price", 0.0) or 0.0) <= 0.0
            ):
                invalid_quote_count += 1

        def _percentile(values: list[float], pct: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
            return float(ordered[index])

        total_quotes = len(opportunities)
        combined_ages = buy_ages + sell_ages
        return {
            "sample_count": int(total_quotes),
            "median_book_age_sec": round(_percentile(combined_ages, 50.0), 6),
            "p95_book_age_sec": round(_percentile(combined_ages, 95.0), 6),
            "max_book_age_sec": round(max(combined_ages, default=0.0), 6),
            "p95_book_asymmetry_sec": round(_percentile(asymmetries, 95.0), 6),
            "invalid_quote_rate": round(float(invalid_quote_count / total_quotes), 6) if total_quotes else 0.0,
        }

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
                            "book_health": self._runtime_audit_book_health(),
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
        self.mark_scanner_lite_interest()
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
        self.mark_scanner_lite_interest()
        with self._lock:
            return list(self._last_scanner_lite_rows)

    def get_current_opportunities_lite_summary(self) -> Dict[str, Any]:
        self.mark_scanner_lite_interest()
        with self._lock:
            return dict(self._last_scanner_lite_summary)

    def get_current_scanner_lite_snapshot(self) -> Dict[str, Any]:
        self.mark_scanner_lite_interest()
        with self._lock:
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
            "startup_gate": dict(self._last_startup_gate_state),
            "startup_filtered_opportunities": int(self._startup_filtered_opportunities),
            "feed_modes": {
                name: {
                    "spot": instance.get_feed_mode("spot"),
                    "futures": instance.get_feed_mode("futures"),
                }
                for name, instance in self._exchange_instances.items()
            },
            "reconnect_counts": {
                name: instance.get_reconnect_counts()
                for name, instance in self._exchange_instances.items()
            },
        }
