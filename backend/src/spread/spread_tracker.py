"""Spread tracker: history, invertidas, capture sessions, and training blocks."""

from __future__ import annotations

import json
import logging
import math
import shutil
import sqlite3
import threading
import time
from bisect import bisect_left
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from .feature_contracts import build_feature_rows

logger = logging.getLogger(__name__)

PairKey = Tuple[str, str, str, str, str]

_SCHEMA_VERSION = "tracker_sqlite_v3"
_EPISODE_SOURCE_VERSION = "recurring_v1"
_EPISODE_BASELINE_WINDOW = 32
_SHORT_RANGE_WINDOW_SEC = 2 * 60 * 60
_LONG_RANGE_WINDOW_SEC = 24 * 60 * 60
_DEFAULT_MIN_TOTAL_SPREAD_PCT = 0.0
_HOT_PATH_REJECTION_META_KEY = "hot_path_rejection_stats"
_TRACKER_META_KEYS = (
    "schema_version",
    "record_interval_sec",
    "tracking_window_sec",
    "tracker_memory_window_sec",
    "gap_threshold_sec",
    "min_total_spread_pct",
    "quality_fix_activated_at",
    _HOT_PATH_REJECTION_META_KEY,
    "created_at",
    "last_flush_at",
)
_BLOCK_BOUNDARY_REASONS = {
    "initial",
    "session_boundary",
    "auto_gap",
    "manual_split",
    "manual_merge",
    "legacy_import",
}
_DEFAULT_CLEAN_SEQUENCE_LENGTH = 15
_DEFAULT_CLEAN_PREDICTION_HORIZON_SEC = 14_400
_BURN_IN_CHECKPOINT_SEC = 30 * 60
_BURN_IN_MIN_SEC = 60 * 60
_FLUSH_PAIR_BATCH_MAX = 512
_TIMESTAMP_RETROGRADE_TOLERANCE_SEC = 0.5
_RECENT_RECORD_FINGERPRINT_MAXLEN = 64
_RECENT_SPREAD_HISTORY_MAXLEN = 64
_SPREAD_OUTLIER_MIN_HISTORY = 10
_SPREAD_OUTLIER_MULTIPLIER = 10.0
_DEFAULT_EPISODE_ABANDONED_CUTOFF_SEC = 2 * _DEFAULT_CLEAN_PREDICTION_HORIZON_SEC


@dataclass(slots=True)
class SpreadRecord:
    timestamp: float
    entry_spread_pct: float
    exit_spread_pct: float
    session_id: int = 0
    block_id: int = 0


@dataclass(slots=True)
class TrackerEvent:
    timestamp: float
    event_type: str
    session_id: int = 0
    block_id: Optional[int] = None


@dataclass(slots=True)
class TrackerEpisode:
    start_ts: float
    peak_ts: float
    end_ts: float
    duration_sec: float
    peak_entry_spread: float
    exit_spread_at_close: float
    baseline_median: float
    baseline_mad: float
    activation_threshold: float
    release_threshold: float
    session_id: int = 0
    block_id: int = 0
    source_version: str = _EPISODE_SOURCE_VERSION
    is_closed: bool = True

    @property
    def total_spread(self) -> float:
        return float(self.peak_entry_spread + self.exit_spread_at_close)


@dataclass(slots=True)
class TrackerBlockMeta:
    session_id: int
    start_ts: float
    end_ts: float
    record_count: int
    max_gap_sec: float
    boundary_reason: str
    is_open: bool = True


def _linear_percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    pct = max(0.0, min(100.0, float(percentile)))
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + ((ordered[upper] - ordered[lower]) * (rank - lower)))


def _median(values: Iterable[float]) -> float:
    return _linear_percentile(list(values), 50.0)


def _mad(values: Iterable[float], center: float | None = None) -> float:
    ordered = [float(value) for value in values]
    if not ordered:
        return 0.0
    baseline = float(center) if center is not None else _median(ordered)
    deviations = [abs(value - baseline) for value in ordered]
    return _median(deviations)


def _iqr(values: Iterable[float]) -> float:
    ordered = [float(value) for value in values]
    if not ordered:
        return 0.0
    return _linear_percentile(ordered, 75.0) - _linear_percentile(ordered, 25.0)


def _episode_stats(episodes: list[TrackerEpisode]) -> dict[str, float]:
    if not episodes:
        return {
            "support": 0.0,
            "entry_outer_range_min": 0.0,
            "entry_outer_range_max": 0.0,
            "entry_core_range_min": 0.0,
            "entry_core_range_max": 0.0,
            "entry_median": 0.0,
            "entry_mad": 0.0,
            "entry_iqr": 0.0,
            "exit_outer_range_min": 0.0,
            "exit_outer_range_max": 0.0,
            "exit_core_range_min": 0.0,
            "exit_core_range_max": 0.0,
            "exit_median": 0.0,
            "exit_mad": 0.0,
            "exit_iqr": 0.0,
            "eta_q10": 0.0,
            "eta_q25": 0.0,
            "eta_q50": 0.0,
            "eta_q75": 0.0,
            "eta_q90": 0.0,
            "context_percentile_excursions": 0.0,
        }
    peaks = [episode.peak_entry_spread for episode in episodes]
    exits = [episode.exit_spread_at_close for episode in episodes]
    durations = [episode.duration_sec for episode in episodes]
    entry_median = _median(peaks)
    exit_median = _median(exits)
    return {
        "support": float(len(episodes)),
        "entry_outer_range_min": _linear_percentile(peaks, 10.0),
        "entry_outer_range_max": _linear_percentile(peaks, 90.0),
        "entry_core_range_min": _linear_percentile(peaks, 25.0),
        "entry_core_range_max": _linear_percentile(peaks, 75.0),
        "entry_median": entry_median,
        "entry_mad": _mad(peaks, entry_median),
        "entry_iqr": _iqr(peaks),
        "exit_outer_range_min": _linear_percentile(exits, 10.0),
        "exit_outer_range_max": _linear_percentile(exits, 90.0),
        "exit_core_range_min": _linear_percentile(exits, 25.0),
        "exit_core_range_max": _linear_percentile(exits, 75.0),
        "exit_median": exit_median,
        "exit_mad": _mad(exits, exit_median),
        "exit_iqr": _iqr(exits),
        "eta_q10": _linear_percentile(durations, 10.0),
        "eta_q25": _linear_percentile(durations, 25.0),
        "eta_q50": _linear_percentile(durations, 50.0),
        "eta_q75": _linear_percentile(durations, 75.0),
        "eta_q90": _linear_percentile(durations, 90.0),
        "context_percentile_excursions": 0.0,
    }


def _episode_total_spread(episode: TrackerEpisode) -> float:
    return float(episode.peak_entry_spread) + float(episode.exit_spread_at_close)


def compute_closed_episodes(records: Iterable[SpreadRecord]) -> list[TrackerEpisode]:
    closed_episodes, _ = compute_closed_episodes_with_diagnostics(records)
    return closed_episodes


def compute_closed_episodes_with_diagnostics(
    records: Iterable[SpreadRecord],
    *,
    abandoned_cutoff_sec: float = _DEFAULT_EPISODE_ABANDONED_CUTOFF_SEC,
) -> tuple[list[TrackerEpisode], dict[str, int]]:
    ordered = sorted(
        (
            record
            for record in records
            if int(getattr(record, "block_id", 0) or 0) != 0
        ),
        key=lambda item: (int(item.block_id), float(item.timestamp)),
    )
    episodes: list[TrackerEpisode] = []
    baseline_entries: deque[float] = deque(maxlen=_EPISODE_BASELINE_WINDOW)
    last_entry: float | None = None
    current_block_id = 0
    active: TrackerEpisode | None = None
    last_record_ts = 0.0
    abandoned_episode_count = 0

    def _close_abandoned_if_needed(current_ts: float) -> None:
        nonlocal active, abandoned_episode_count
        if active is None:
            return
        if max(0.0, float(current_ts) - float(active.start_ts)) > float(abandoned_cutoff_sec):
            abandoned_episode_count += 1
            active = None

    for record in ordered:
        block_id = int(record.block_id or 0)
        if block_id != current_block_id:
            _close_abandoned_if_needed(last_record_ts)
            baseline_entries.clear()
            last_entry = None
            current_block_id = block_id
            active = None

        current_entry = float(record.entry_spread_pct)
        last_record_ts = float(record.timestamp)
        n = len(baseline_entries)
        if n > 0:
            # Sort once, compute median and MAD inline — avoids redundant
            # list copies and re-sorts that _median/_mad would do
            history_sorted = sorted(baseline_entries)
            mid = n >> 1
            baseline_median = history_sorted[mid] if (n & 1) else (history_sorted[mid - 1] + history_sorted[mid]) * 0.5
            deviations_sorted = sorted(abs(v - baseline_median) for v in baseline_entries)
            baseline_mad = max(
                deviations_sorted[mid] if (n & 1) else (deviations_sorted[mid - 1] + deviations_sorted[mid]) * 0.5,
                0.0,
            )
        else:
            baseline_median = current_entry
            baseline_mad = 0.0
        activation_threshold = baseline_median + max(1.0 * baseline_mad, 0.05)
        release_threshold = baseline_median + max(0.25 * baseline_mad, 0.02)

        if active is None:
            if last_entry is not None and last_entry <= activation_threshold and current_entry > activation_threshold:
                active = TrackerEpisode(
                    start_ts=float(record.timestamp),
                    peak_ts=float(record.timestamp),
                    end_ts=0.0,
                    duration_sec=0.0,
                    peak_entry_spread=current_entry,
                    exit_spread_at_close=float(record.exit_spread_pct),
                    baseline_median=float(baseline_median),
                    baseline_mad=float(baseline_mad),
                    activation_threshold=float(activation_threshold),
                    release_threshold=float(release_threshold),
                    session_id=int(record.session_id or 0),
                    block_id=block_id,
                    source_version=_EPISODE_SOURCE_VERSION,
                    is_closed=False,
                )
        else:
            if current_entry >= active.peak_entry_spread:
                active.peak_entry_spread = current_entry
                active.peak_ts = float(record.timestamp)
            if current_entry <= active.release_threshold:
                active.end_ts = float(record.timestamp)
                active.duration_sec = max(0.0, active.end_ts - active.start_ts)
                active.exit_spread_at_close = float(record.exit_spread_pct)
                active.is_closed = True
                if (
                    math.isfinite(active.duration_sec)
                    and math.isfinite(active.peak_entry_spread)
                    and math.isfinite(active.exit_spread_at_close)
                    and active.duration_sec >= 0.0
                    and active.peak_entry_spread >= active.activation_threshold
                ):
                    episodes.append(active)
                active = None

        baseline_entries.append(current_entry)
        last_entry = current_entry

    _close_abandoned_if_needed(last_record_ts)
    return episodes, {
        "abandoned_episode_count": int(abandoned_episode_count),
        "closed_episode_count": int(len(episodes)),
    }


def build_recurring_context_from_episodes(
    episodes: Iterable[TrackerEpisode],
    *,
    current_entry: float,
    now_ts: float,
    short_window_sec: float = _SHORT_RANGE_WINDOW_SEC,
    long_window_sec: float = _LONG_RANGE_WINDOW_SEC,
    min_total_spread_pct: float = _DEFAULT_MIN_TOTAL_SPREAD_PCT,
) -> dict[str, object]:
    closed = [episode for episode in episodes if episode.is_closed and episode.end_ts > 0.0 and episode.end_ts <= float(now_ts)]
    short_raw_episodes = [episode for episode in closed if episode.end_ts >= float(now_ts) - float(short_window_sec)]
    long_raw_episodes = [episode for episode in closed if episode.end_ts >= float(now_ts) - float(long_window_sec)]
    min_total_threshold = float(min_total_spread_pct or 0.0)
    short_episodes = [episode for episode in short_raw_episodes if _episode_total_spread(episode) >= min_total_threshold]
    long_episodes = [episode for episode in long_raw_episodes if _episode_total_spread(episode) >= min_total_threshold]

    short_stats = _episode_stats(short_episodes)
    long_stats = _episode_stats(long_episodes)

    short_ready = int(short_stats["support"]) >= 2
    long_ready = int(long_stats["support"]) >= 3
    strong_short_ready = int(short_stats["support"]) >= 3
    strong_long_ready = int(long_stats["support"]) >= 5

    raw_short_ready = len(short_raw_episodes) >= 2
    raw_long_ready = len(long_raw_episodes) >= 3

    if raw_short_ready:
        active_raw_episodes = short_raw_episodes
    elif raw_long_ready:
        active_raw_episodes = long_raw_episodes
    else:
        active_raw_episodes = short_raw_episodes or long_raw_episodes

    if short_ready:
        active_stats = short_stats
        active_episodes = short_episodes
        range_status = "ready_short"
        range_window = "2h"
    elif long_ready:
        active_stats = long_stats
        active_episodes = long_episodes
        range_status = "ready_long_fallback"
        range_window = "24h_fallback"
    else:
        active_stats = _episode_stats([])
        active_episodes = []
        range_status = "insufficient_empirical_context"
        range_window = "none"

    total_spreads = [_episode_total_spread(episode) for episode in active_raw_episodes]
    median_total_spread = _median(total_spreads) if total_spreads else 0.0

    current_value = float(current_entry)
    entry_position_label = "unknown"
    context_strength = "weak"
    is_entry_inside_range = False
    if range_status != "insufficient_empirical_context":
        outer_min = float(active_stats["entry_outer_range_min"])
        outer_max = float(active_stats["entry_outer_range_max"])
        core_min = float(active_stats["entry_core_range_min"])
        core_max = float(active_stats["entry_core_range_max"])
        if current_value < outer_min:
            entry_position_label = "below_band"
        elif current_value > outer_max:
            entry_position_label = "above_band"
        elif core_min <= current_value <= core_max:
            entry_position_label = "inside_core"
            is_entry_inside_range = True
        else:
            entry_position_label = "inside_outer"
            is_entry_inside_range = True

        if entry_position_label in {"inside_outer", "inside_core"}:
            context_strength = "normal"

    entry_coherent = False
    exit_coherent = False
    if strong_short_ready and strong_long_ready:
        center_short = float(short_stats["entry_median"])
        center_long = float(long_stats["entry_median"])
        scale_short = max(float(short_stats["entry_iqr"]) / 1.349, float(short_stats["entry_mad"]) * 1.4826, 0.05)
        scale_long = max(float(long_stats["entry_iqr"]) / 1.349, float(long_stats["entry_mad"]) * 1.4826, 0.05)
        entry_coherent = abs(center_short - center_long) <= max(scale_short, scale_long)

        exit_center_short = float(short_stats["exit_median"])
        exit_center_long = float(long_stats["exit_median"])
        exit_scale_short = max(float(short_stats["exit_iqr"]) / 1.349, float(short_stats["exit_mad"]) * 1.4826, 0.05)
        exit_scale_long = max(float(long_stats["exit_iqr"]) / 1.349, float(long_stats["exit_mad"]) * 1.4826, 0.05)
        exit_coherent = abs(exit_center_short - exit_center_long) <= max(exit_scale_short, exit_scale_long)

    if context_strength == "normal" and entry_position_label == "inside_core" and entry_coherent and exit_coherent:
        context_strength = "strong"

    peaks = [episode.peak_entry_spread for episode in active_episodes]
    if peaks:
        context_percentile_excursions = float(sum(1.0 for value in peaks if value <= current_value) / len(peaks))
    else:
        context_percentile_excursions = 0.0

    return {
        "range_status": range_status,
        "range_window": range_window,
        "empirical_support": int(active_stats["support"]),
        "empirical_support_short": int(short_stats["support"]),
        "empirical_support_long": int(long_stats["support"]),
        "raw_empirical_support": len(active_raw_episodes),
        "raw_empirical_support_short": len(short_raw_episodes),
        "raw_empirical_support_long": len(long_raw_episodes),
        "median_total_spread": float(median_total_spread),
        "min_total_spread_threshold": float(min_total_threshold),
        "entry_outer_range_min": float(active_stats["entry_outer_range_min"]),
        "entry_outer_range_max": float(active_stats["entry_outer_range_max"]),
        "entry_core_range_min": float(active_stats["entry_core_range_min"]),
        "entry_core_range_max": float(active_stats["entry_core_range_max"]),
        "exit_outer_range_min": float(active_stats["exit_outer_range_min"]),
        "exit_outer_range_max": float(active_stats["exit_outer_range_max"]),
        "exit_core_range_min": float(active_stats["exit_core_range_min"]),
        "exit_core_range_max": float(active_stats["exit_core_range_max"]),
        "recommended_entry_range": (
            f"{active_stats['entry_core_range_min']:.2f}% à {active_stats['entry_core_range_max']:.2f}%"
            if range_status != "insufficient_empirical_context"
            else "--"
        ),
        "recommended_exit_range": (
            f"{active_stats['exit_core_range_min']:.2f}% à {active_stats['exit_core_range_max']:.2f}%"
            if range_status != "insufficient_empirical_context"
            else "--"
        ),
        "entry_position_label": entry_position_label,
        "context_strength": context_strength,
        "is_entry_inside_range": is_entry_inside_range,
        "context_percentile_excursions": round(context_percentile_excursions, 4),
        "entry_coherent_short_long": bool(entry_coherent),
        "exit_coherent_short_long": bool(exit_coherent),
        "raw_short_ready": bool(raw_short_ready),
        "raw_long_ready": bool(raw_long_ready),
        "short_ready": bool(short_ready),
        "long_ready": bool(long_ready),
        "strong_short_ready": bool(strong_short_ready),
        "strong_long_ready": bool(strong_long_ready),
        "entry_median": float(active_stats["entry_median"]),
        "exit_median": float(active_stats["exit_median"]),
        "empirical_eta_median_seconds": float(active_stats["eta_q50"]),
        "empirical_eta_q10_seconds": float(active_stats["eta_q10"]),
        "empirical_eta_q25_seconds": float(active_stats["eta_q25"]),
        "empirical_eta_q75_seconds": float(active_stats["eta_q75"]),
        "empirical_eta_q90_seconds": float(active_stats["eta_q90"]),
    }


@dataclass
class PairStats:
    storage_pair_id: int = 0
    last_state: int = 0
    last_seen_ts: float = 0.0
    last_crossover_ts: float = 0.0
    inverted_events: Deque[TrackerEvent] = field(default_factory=deque)
    entry_events: Deque[TrackerEvent] = field(default_factory=deque)
    exit_events: Deque[TrackerEvent] = field(default_factory=deque)
    history_enabled: bool = False
    records: Deque[SpreadRecord] = field(default_factory=deque)
    _last_record_ts: float = 0.0
    current_session_id: int = 0
    current_block_id: int = 0
    current_block_start_ts: float = 0.0
    current_block_end_ts: float = 0.0
    current_block_record_count: int = 0
    current_block_max_gap_sec: float = 0.0
    current_block_boundary_reason: str = ""
    min_spread: float = float("inf")
    max_spread: float = float("-inf")
    avg_spread: float = 0.0
    _stats_dirty: bool = True
    block_meta: Dict[int, TrackerBlockMeta] = field(default_factory=dict)
    episodes: Deque[TrackerEpisode] = field(default_factory=deque)
    _episodes_cache_valid: bool = False
    exact_duplicate_ts_rejections: int = 0
    retrograde_ts_rejections: int = 0
    clock_jitter_events: int = 0
    spread_outlier_warnings: int = 0
    abandoned_episode_count: int = 0
    recent_record_fingerprints: Deque[Tuple[int, float, float]] = field(
        default_factory=lambda: deque(maxlen=_RECENT_RECORD_FINGERPRINT_MAXLEN)
    )
    recent_abs_entry_spreads: Deque[float] = field(
        default_factory=lambda: deque(maxlen=_RECENT_SPREAD_HISTORY_MAXLEN)
    )

    def _prune_events(self, cutoff: float) -> bool:
        pruned = False
        while self.inverted_events and self.inverted_events[0].timestamp < cutoff:
            self.inverted_events.popleft()
            pruned = True
        while self.entry_events and self.entry_events[0].timestamp < cutoff:
            self.entry_events.popleft()
            pruned = True
        while self.exit_events and self.exit_events[0].timestamp < cutoff:
            self.exit_events.popleft()
            pruned = True
        return pruned

    def _prune_records(self, cutoff: float, *, preserve_after_ts: float | None = None) -> bool:
        pruned = False
        while self.records and self.records[0].timestamp < cutoff:
            if preserve_after_ts is not None and self.records[0].timestamp >= float(preserve_after_ts):
                break
            self.records.popleft()
            self._stats_dirty = True
            self._episodes_cache_valid = False
            pruned = True
        return pruned

    def _prune_episodes(self, cutoff: float):
        while self.episodes and self.episodes[0].end_ts < cutoff:
            self.episodes.popleft()

    def _prune_block_meta(self, cutoff: float):
        stale_block_ids = [
            int(block_id)
            for block_id, meta in self.block_meta.items()
            if (not meta.is_open) and float(meta.end_ts) < cutoff
        ]
        for block_id in stale_block_ids:
            self.block_meta.pop(block_id, None)

    def _recompute_stats(self):
        if not self.records:
            self.min_spread = float("inf")
            self.max_spread = float("-inf")
            self.avg_spread = 0.0
            self._stats_dirty = False
            return
        spreads = [record.entry_spread_pct for record in self.records]
        self.min_spread = min(spreads)
        self.max_spread = max(spreads)
        self.avg_spread = sum(spreads) / len(spreads)
        self._stats_dirty = False

    def ensure_stats(self):
        if self._stats_dirty:
            self._recompute_stats()

    @property
    def inverted_count(self) -> int:
        return len(self.inverted_events)

    @property
    def total_entries(self) -> int:
        return len(self.entry_events)

    @property
    def total_exits(self) -> int:
        return len(self.exit_events)


class SpreadTracker:
    _WINDOWS = {
        "30m": 30 * 60,
        "1h": 60 * 60,
        "2h": 2 * 60 * 60,
        "4h": 4 * 60 * 60,
        "6h": 6 * 60 * 60,
        "8h": 8 * 60 * 60,
        "24h": 24 * 60 * 60,
    }
    _ZERO_COUNTS: Dict[str, int] = {key: 0 for key in _WINDOWS}

    def __init__(
        self,
        window_sec: int = 604800,
        memory_window_sec: int | None = None,
        record_interval_sec: float = 15.0,
        max_records_per_pair: int = 0,
        epsilon_pct: float = 0.02,
        history_enable_entry_spread_pct: float = 0.1,
        track_enable_entry_spread_pct: float = 0.0,
        max_pairs: int = 10_000,
        db_path: str | Path | None = None,
        gap_threshold_sec: float = 0.0,
        min_total_spread_pct: float = _DEFAULT_MIN_TOTAL_SPREAD_PCT,
        audit_collector: Any | None = None,
    ):
        self.storage_window_sec = max(int(window_sec), 1)
        effective_memory_window = self.storage_window_sec if memory_window_sec is None else max(int(memory_window_sec), 1)
        self.memory_window_sec = int(effective_memory_window)
        self.window_sec = int(self.storage_window_sec)
        self.record_interval_sec = float(record_interval_sec)
        self.max_records_per_pair = self._resolve_max_records_per_pair(
            max_records_per_pair,
            self.memory_window_sec,
            self.record_interval_sec,
        )
        self.epsilon_pct = float(epsilon_pct)
        self.max_pairs = int(max_pairs)
        self.history_enable_entry_spread_pct = float(history_enable_entry_spread_pct)
        self.track_enable_entry_spread_pct = float(track_enable_entry_spread_pct)
        self.db_path = Path(db_path) if db_path is not None else None
        self.gap_threshold_sec = self._resolve_gap_threshold_sec(gap_threshold_sec, self.record_interval_sec)
        self.min_total_spread_pct = self._resolve_min_total_spread_pct(min_total_spread_pct)
        self.episode_abandoned_cutoff_sec = float(_DEFAULT_EPISODE_ABANDONED_CUTOFF_SEC)
        self.audit_collector = audit_collector
        self._lock = threading.Lock()
        self._pairs: Dict[PairKey, PairStats] = defaultdict(PairStats)
        self._dirty_pairs: set[PairKey] = set()
        self._deleted_pairs: set[PairKey] = set()
        self._last_flush_at: float = 0.0
        self._active_session_id: int = 0
        self._ephemeral_session_id = 1
        self._ephemeral_block_id = 1
        self._event_listeners: list[Any] = []
        self.quality_fix_activated_at: float = 0.0
        self._hot_path_rejection_stats = self._empty_hot_path_rejection_stats()
        self._hot_path_rejection_stats_dirty = False
        self._feature_history_cache: dict[tuple[PairKey, tuple[str, ...], int], dict[str, Any]] = {}
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize_storage()
            self._open_runtime_session()

    def add_event_listener(self, listener: Any):
        if listener in self._event_listeners:
            return
        self._event_listeners.append(listener)

    def remove_event_listener(self, listener: Any):
        self._event_listeners = [existing for existing in self._event_listeners if existing is not listener]

    def _emit_event(self, payload: dict[str, Any]):
        if not self._event_listeners:
            return
        for listener in list(self._event_listeners):
            try:
                listener(dict(payload))
            except Exception as exc:
                logger.debug("[Tracker] Event listener failed: %s", exc)

    @staticmethod
    def _resolve_max_records_per_pair(
        max_records_per_pair: int,
        window_sec: int,
        record_interval_sec: float,
    ) -> int:
        if int(max_records_per_pair) > 0:
            return int(max_records_per_pair)
        if record_interval_sec <= 0:
            return 0
        return max(1, int(math.ceil(float(window_sec) / float(record_interval_sec))))

    @staticmethod
    def _resolve_gap_threshold_sec(gap_threshold_sec: float, record_interval_sec: float) -> float:
        if float(gap_threshold_sec or 0.0) > 0.0:
            return float(gap_threshold_sec)
        return max(60.0, 4.0 * max(float(record_interval_sec), 0.0))

    def _memory_cutoff(self, ts: float) -> float:
        return float(ts) - float(self.memory_window_sec)

    def _storage_cutoff(self, ts: float) -> float:
        return float(ts) - float(self.storage_window_sec)

    @staticmethod
    def _resolve_min_total_spread_pct(min_total_spread_pct: float) -> float:
        value = float(min_total_spread_pct or 0.0)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("min_total_spread_pct must be a finite value >= 0")
        return value

    @staticmethod
    def _pair_key(symbol: str, buy_ex: str, buy_mt: str, sell_ex: str, sell_mt: str) -> PairKey:
        return (
            (symbol or "").upper(),
            (buy_ex or "").lower(),
            (buy_mt or "").lower(),
            (sell_ex or "").lower(),
            (sell_mt or "").lower(),
        )

    @staticmethod
    def _pair_id(key: PairKey) -> str:
        symbol, buy_ex, buy_mt, sell_ex, sell_mt = key
        return f"{symbol}|{buy_ex}|{buy_mt}|{sell_ex}|{sell_mt}"

    @staticmethod
    def _pair_id_from_row(row: sqlite3.Row) -> str:
        return f"{row['symbol']}|{row['buy_ex']}|{row['buy_mt']}|{row['sell_ex']}|{row['sell_mt']}"

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _empty_hot_path_rejection_stats() -> dict[str, Any]:
        return {
            "attempted_records_total": 0,
            "invalid_record_rejections_total": 0,
            "invalid_entry_rejections_total": 0,
            "invalid_exit_rejections_total": 0,
            "rejection_rate_total": 0.0,
            "rejection_rate_by_pair": {},
            "rejection_rate_by_exchange": {},
            "pair_attempts": {},
            "exchange_attempts": {},
            "pair_rejections": {},
            "exchange_rejections": {},
        }

    @staticmethod
    def _ratio_map(rejections: dict[str, int], attempts: dict[str, int]) -> dict[str, float]:
        ratios: dict[str, float] = {}
        for key, attempted in attempts.items():
            attempted_count = int(attempted or 0)
            rejected_count = int(rejections.get(key, 0) or 0)
            ratios[str(key)] = float(rejected_count / attempted_count) if attempted_count > 0 else 0.0
        return dict(sorted(ratios.items()))

    @classmethod
    def _normalized_hot_path_rejection_stats(cls, payload: dict[str, Any] | None) -> dict[str, Any]:
        stats = cls._empty_hot_path_rejection_stats()
        if not isinstance(payload, dict):
            return stats
        stats["attempted_records_total"] = int(payload.get("attempted_records_total", 0) or 0)
        stats["invalid_record_rejections_total"] = int(payload.get("invalid_record_rejections_total", 0) or 0)
        stats["invalid_entry_rejections_total"] = int(payload.get("invalid_entry_rejections_total", 0) or 0)
        stats["invalid_exit_rejections_total"] = int(payload.get("invalid_exit_rejections_total", 0) or 0)
        stats["pair_attempts"] = {
            str(key): int(value or 0)
            for key, value in dict(payload.get("pair_attempts") or {}).items()
            if str(key)
        }
        stats["exchange_attempts"] = {
            str(key): int(value or 0)
            for key, value in dict(payload.get("exchange_attempts") or {}).items()
            if str(key)
        }
        stats["pair_rejections"] = {
            str(key): int(value or 0)
            for key, value in dict(payload.get("pair_rejections") or {}).items()
            if str(key)
        }
        stats["exchange_rejections"] = {
            str(key): int(value or 0)
            for key, value in dict(payload.get("exchange_rejections") or {}).items()
            if str(key)
        }
        stats["rejection_rate_by_pair"] = cls._ratio_map(
            stats["pair_rejections"],
            stats["pair_attempts"],
        )
        stats["rejection_rate_by_exchange"] = cls._ratio_map(
            stats["exchange_rejections"],
            stats["exchange_attempts"],
        )
        attempted_total = int(stats["attempted_records_total"])
        rejected_total = int(stats["invalid_record_rejections_total"])
        stats["rejection_rate_total"] = float(rejected_total / attempted_total) if attempted_total > 0 else 0.0
        return stats

    def _hot_path_rejection_snapshot_locked(self) -> dict[str, Any]:
        stats = self._normalized_hot_path_rejection_stats(self._hot_path_rejection_stats)
        return {
            "attempted_records_total": int(stats["attempted_records_total"]),
            "invalid_record_rejections_total": int(stats["invalid_record_rejections_total"]),
            "invalid_entry_rejections_total": int(stats["invalid_entry_rejections_total"]),
            "invalid_exit_rejections_total": int(stats["invalid_exit_rejections_total"]),
            "rejection_rate_total": float(stats["rejection_rate_total"]),
            "rejection_rate_by_pair": dict(stats["rejection_rate_by_pair"]),
            "rejection_rate_by_exchange": dict(stats["rejection_rate_by_exchange"]),
            "pair_attempts": dict(stats["pair_attempts"]),
            "exchange_attempts": dict(stats["exchange_attempts"]),
            "pair_rejections": dict(stats["pair_rejections"]),
            "exchange_rejections": dict(stats["exchange_rejections"]),
        }

    def _record_hot_path_attempt_locked(self, key: PairKey, *, count: int = 1) -> None:
        increment = max(int(count), 1)
        pair_id = self._pair_id(key)
        stats = self._hot_path_rejection_stats
        if not isinstance(stats, dict):
            stats = self._empty_hot_path_rejection_stats()
            self._hot_path_rejection_stats = stats
        stats["attempted_records_total"] = int(stats.get("attempted_records_total", 0) or 0) + increment
        pair_attempts = stats.setdefault("pair_attempts", {})
        pair_attempts[pair_id] = int(pair_attempts.get(pair_id, 0) or 0) + increment
        exchange_attempts = stats.setdefault("exchange_attempts", {})
        for exchange in {str(key[1] or ""), str(key[3] or "")}:
            if not exchange:
                continue
            exchange_attempts[exchange] = int(exchange_attempts.get(exchange, 0) or 0) + increment
        pair_rates = stats.setdefault("rejection_rate_by_pair", {})
        pair_rejections = stats.setdefault("pair_rejections", {})
        pair_attempt_total = int(pair_attempts.get(pair_id, 0) or 0)
        pair_rates[pair_id] = float(int(pair_rejections.get(pair_id, 0) or 0) / pair_attempt_total) if pair_attempt_total > 0 else 0.0
        exchange_rates = stats.setdefault("rejection_rate_by_exchange", {})
        exchange_rejections = stats.setdefault("exchange_rejections", {})
        for exchange in {str(key[1] or ""), str(key[3] or "")}:
            if not exchange:
                continue
            exchange_attempt_total = int(exchange_attempts.get(exchange, 0) or 0)
            exchange_rates[exchange] = float(int(exchange_rejections.get(exchange, 0) or 0) / exchange_attempt_total) if exchange_attempt_total > 0 else 0.0
        attempted_total = int(stats.get("attempted_records_total", 0) or 0)
        rejected_total = int(stats.get("invalid_record_rejections_total", 0) or 0)
        stats["rejection_rate_total"] = float(rejected_total / attempted_total) if attempted_total > 0 else 0.0
        self._hot_path_rejection_stats_dirty = True

    def _record_hot_path_rejection_locked(
        self,
        key: PairKey,
        *,
        invalid_fields: Iterable[str],
        count: int = 1,
    ) -> None:
        increment = max(int(count), 1)
        stats = self._hot_path_rejection_stats
        if not isinstance(stats, dict):
            stats = self._empty_hot_path_rejection_stats()
            self._hot_path_rejection_stats = stats
        fields = {str(field) for field in invalid_fields}
        pair_id = self._pair_id(key)
        stats["invalid_record_rejections_total"] = int(stats.get("invalid_record_rejections_total", 0) or 0) + increment
        if "entry" in fields:
            stats["invalid_entry_rejections_total"] = int(stats.get("invalid_entry_rejections_total", 0) or 0) + increment
        if "exit" in fields:
            stats["invalid_exit_rejections_total"] = int(stats.get("invalid_exit_rejections_total", 0) or 0) + increment
        pair_rejections = stats.setdefault("pair_rejections", {})
        pair_rejections[pair_id] = int(pair_rejections.get(pair_id, 0) or 0) + increment
        exchange_rejections = stats.setdefault("exchange_rejections", {})
        for exchange in {str(key[1] or ""), str(key[3] or "")}:
            if not exchange:
                continue
            exchange_rejections[exchange] = int(exchange_rejections.get(exchange, 0) or 0) + increment
        pair_rates = stats.setdefault("rejection_rate_by_pair", {})
        pair_attempts = stats.setdefault("pair_attempts", {})
        pair_attempt_total = int(pair_attempts.get(pair_id, 0) or 0)
        pair_rates[pair_id] = float(int(pair_rejections.get(pair_id, 0) or 0) / pair_attempt_total) if pair_attempt_total > 0 else 0.0
        exchange_rates = stats.setdefault("rejection_rate_by_exchange", {})
        exchange_attempts = stats.setdefault("exchange_attempts", {})
        for exchange in {str(key[1] or ""), str(key[3] or "")}:
            if not exchange:
                continue
            exchange_attempt_total = int(exchange_attempts.get(exchange, 0) or 0)
            exchange_rates[exchange] = float(int(exchange_rejections.get(exchange, 0) or 0) / exchange_attempt_total) if exchange_attempt_total > 0 else 0.0
        attempted_total = int(stats.get("attempted_records_total", 0) or 0)
        rejected_total = int(stats.get("invalid_record_rejections_total", 0) or 0)
        stats["rejection_rate_total"] = float(rejected_total / attempted_total) if attempted_total > 0 else 0.0
        self._hot_path_rejection_stats_dirty = True

    def _load_hot_path_rejection_stats(self, meta_rows: dict[str, str]) -> None:
        raw_payload = str(meta_rows.get(_HOT_PATH_REJECTION_META_KEY, "") or "").strip()
        if not raw_payload:
            self._hot_path_rejection_stats = self._empty_hot_path_rejection_stats()
            return
        try:
            parsed = json.loads(raw_payload)
        except json.JSONDecodeError:
            logger.warning("[Tracker] Could not parse hot-path rejection stats from tracker_meta")
            self._hot_path_rejection_stats = self._empty_hot_path_rejection_stats()
            return
        self._hot_path_rejection_stats = self._normalized_hot_path_rejection_stats(parsed if isinstance(parsed, dict) else {})

    def record_hot_path_rejection(
        self,
        symbol: str,
        buy_ex: str,
        buy_mt: str,
        sell_ex: str,
        sell_mt: str,
        *,
        invalid_fields: list[str] | None = None,
    ) -> None:
        key = self._pair_key(symbol, buy_ex, buy_mt, sell_ex, sell_mt)
        with self._lock:
            self._record_hot_path_attempt_locked(key)
            self._record_hot_path_rejection_locked(key, invalid_fields=list(invalid_fields or []))

    def batch_record_rejections(self, rejections: list[dict[str, Any]]) -> None:
        if not rejections:
            return
        with self._lock:
            for payload in rejections:
                if not isinstance(payload, dict):
                    continue
                key = self._pair_key(
                    str(payload.get("symbol") or ""),
                    str(payload.get("buy_ex") or ""),
                    str(payload.get("buy_mt") or ""),
                    str(payload.get("sell_ex") or ""),
                    str(payload.get("sell_mt") or ""),
                )
                count = max(int(payload.get("count", 1) or 1), 1)
                self._record_hot_path_attempt_locked(key, count=count)
                self._record_hot_path_rejection_locked(
                    key,
                    invalid_fields=list(payload.get("invalid_fields") or []),
                    count=count,
                )

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str):
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _sign(self, x: float) -> int:
        try:
            value = float(x)
        except Exception:
            return 0
        if abs(value) <= self.epsilon_pct:
            return 0
        return 1 if value > 0 else -1

    def _state(self, entry_spread: float, exit_spread: float) -> int:
        entry_sign = self._sign(entry_spread)
        exit_sign = self._sign(exit_spread)
        if entry_sign == 1 and exit_sign == -1:
            return 1
        if entry_sign == -1 and exit_sign == 1:
            return -1
        return 0

    @staticmethod
    def _invalid_numeric_fields(ts: Any, entry_spread: Any, exit_spread: Any) -> list[str]:
        invalid_fields: list[str] = []
        for name, value in (("ts", ts), ("entry", entry_spread), ("exit", exit_spread)):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                invalid_fields.append(name)
                continue
            if not math.isfinite(numeric):
                invalid_fields.append(name)
        return invalid_fields

    @staticmethod
    def _record_fingerprint(ts: float, entry_spread: float, exit_spread: float, interval_sec: float) -> tuple[int, float, float]:
        bucket_size = max(float(interval_sec or 0.0), 1.0)
        return (
            int(round(float(ts) / bucket_size)),
            round(float(entry_spread), 6),
            round(float(exit_spread), 6),
        )

    @staticmethod
    def _timestamp_status_locked(ps: PairStats, ts: float) -> str:
        last_ts = float(ps._last_record_ts or 0.0)
        candidate_ts = float(ts)
        if last_ts <= 0.0:
            return "ok"
        if candidate_ts == last_ts:
            return "exact_duplicate"
        if candidate_ts < (last_ts - _TIMESTAMP_RETROGRADE_TOLERANCE_SEC):
            return "retrograde"
        if candidate_ts < last_ts:
            return "clock_jitter"
        return "ok"

    @staticmethod
    def _adjust_record_ts(last_ts: float, ts: float) -> float:
        candidate_ts = float(ts)
        if candidate_ts < float(last_ts):
            return float(last_ts) + 1e-6
        return candidate_ts

    @staticmethod
    def _is_light_duplicate_locked(ps: PairStats, fingerprint: tuple[int, float, float]) -> bool:
        return fingerprint in ps.recent_record_fingerprints

    @staticmethod
    def _remember_record_locked(ps: PairStats, *, fingerprint: tuple[int, float, float], entry_spread: float) -> None:
        ps.recent_record_fingerprints.append(fingerprint)
        ps.recent_abs_entry_spreads.append(abs(float(entry_spread)))

    @staticmethod
    def _is_pair_spread_outlier_locked(ps: PairStats, entry_spread: float) -> bool:
        history = list(ps.recent_abs_entry_spreads)
        if len(history) < _SPREAD_OUTLIER_MIN_HISTORY:
            return False
        baseline = max(_median(history), 0.05)
        return abs(float(entry_spread)) > max(baseline * _SPREAD_OUTLIER_MULTIPLIER, 0.50)

    def _emit_tracker_record_rejected(
        self,
        *,
        key: PairKey,
        ts: float,
        entry_spread: float,
        exit_spread: float,
        invalid_fields: list[str],
        reason: str,
        last_record_ts: float,
    ) -> None:
        self._emit_event(
            {
                "kind": "tracker_record_rejected",
                "pair_key": self._pair_id(key),
                "record_ts": float(ts),
                "entry": float(entry_spread),
                "exit": float(exit_spread),
                "invalid_fields": list(invalid_fields),
                "reason": str(reason),
                "last_record_ts": float(last_record_ts),
            }
        )

    def _connect(self) -> sqlite3.Connection:
        if self.db_path is None:
            raise RuntimeError("Tracker storage is not configured")
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _write_meta(self, conn: sqlite3.Connection, updates: dict[str, Any]):
        rows = [(str(key), str(value)) for key, value in updates.items()]
        conn.executemany(
            "INSERT INTO tracker_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            rows,
        )

    def _initialize_storage(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tracker_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    buy_ex TEXT NOT NULL,
                    buy_mt TEXT NOT NULL,
                    sell_ex TEXT NOT NULL,
                    sell_mt TEXT NOT NULL,
                    last_state INTEGER NOT NULL,
                    last_seen_ts REAL NOT NULL,
                    last_crossover_ts REAL NOT NULL,
                    history_enabled INTEGER NOT NULL,
                    UNIQUE(symbol, buy_ex, buy_mt, sell_ex, sell_mt)
                );
                CREATE TABLE IF NOT EXISTS tracker_events (
                    pair_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL CHECK(event_type IN ('inverted', 'entry', 'exit')),
                    ts REAL NOT NULL,
                    session_id INTEGER,
                    block_id INTEGER,
                    PRIMARY KEY(pair_id, event_type, ts),
                    FOREIGN KEY(pair_id) REFERENCES tracker_pairs(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS tracker_records (
                    pair_id INTEGER NOT NULL,
                    ts REAL NOT NULL,
                    entry_spread_pct REAL NOT NULL,
                    exit_spread_pct REAL NOT NULL,
                    session_id INTEGER,
                    block_id INTEGER,
                    PRIMARY KEY(pair_id, ts),
                    FOREIGN KEY(pair_id) REFERENCES tracker_pairs(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS tracker_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tracker_capture_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    status TEXT NOT NULL,
                    record_interval_sec REAL NOT NULL,
                    tracking_window_sec REAL NOT NULL,
                    gap_threshold_sec REAL NOT NULL,
                    created_by TEXT NOT NULL DEFAULT 'runtime',
                    approved_for_training INTEGER NOT NULL DEFAULT 1,
                    excluded_reason TEXT NOT NULL DEFAULT '',
                    manual_override INTEGER NOT NULL DEFAULT 0,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tracker_pair_blocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair_id INTEGER NOT NULL,
                    session_id INTEGER NOT NULL,
                    start_ts REAL NOT NULL,
                    end_ts REAL NOT NULL,
                    record_count INTEGER NOT NULL DEFAULT 0,
                    max_gap_sec REAL NOT NULL DEFAULT 0.0,
                    boundary_reason TEXT NOT NULL,
                    selected_for_training INTEGER NOT NULL DEFAULT 1,
                    disabled_reason TEXT NOT NULL DEFAULT '',
                    manual_override INTEGER NOT NULL DEFAULT 0,
                    notes TEXT NOT NULL DEFAULT '',
                    is_open INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(pair_id) REFERENCES tracker_pairs(id) ON DELETE CASCADE,
                    FOREIGN KEY(session_id) REFERENCES tracker_capture_sessions(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS tracker_pair_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair_id INTEGER NOT NULL,
                    session_id INTEGER NOT NULL,
                    block_id INTEGER NOT NULL,
                    start_ts REAL NOT NULL,
                    peak_ts REAL NOT NULL,
                    end_ts REAL NOT NULL,
                    duration_sec REAL NOT NULL,
                    peak_entry_spread REAL NOT NULL,
                    exit_spread_at_close REAL NOT NULL,
                    baseline_median REAL NOT NULL,
                    baseline_mad REAL NOT NULL,
                    activation_threshold REAL NOT NULL,
                    release_threshold REAL NOT NULL,
                    source_version TEXT NOT NULL DEFAULT 'recurring_v1',
                    is_closed INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(pair_id, block_id, start_ts),
                    FOREIGN KEY(pair_id) REFERENCES tracker_pairs(id) ON DELETE CASCADE,
                    FOREIGN KEY(session_id) REFERENCES tracker_capture_sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY(block_id) REFERENCES tracker_pair_blocks(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS tracker_hourly_health (
                    hour_start_ts REAL PRIMARY KEY,
                    hour_end_ts REAL NOT NULL,
                    records_total INTEGER NOT NULL DEFAULT 0,
                    records_rejected INTEGER NOT NULL DEFAULT 0,
                    rejection_rate_pct REAL NOT NULL DEFAULT 0.0,
                    exchanges_active INTEGER NOT NULL DEFAULT 0,
                    exchanges_circuit_open_json TEXT NOT NULL DEFAULT '[]',
                    pairs_with_records INTEGER NOT NULL DEFAULT 0,
                    pairs_with_gaps INTEGER NOT NULL DEFAULT 0,
                    cross_exchange_flags INTEGER NOT NULL DEFAULT 0,
                    avg_book_age_json TEXT NOT NULL DEFAULT '{}',
                    episode_count INTEGER NOT NULL DEFAULT 0,
                    quality_verdict TEXT NOT NULL DEFAULT 'healthy',
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ml_training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    status TEXT NOT NULL,
                    state_path TEXT NOT NULL,
                    artifact_dir TEXT NOT NULL DEFAULT '',
                    audit_path TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    threshold_sec REAL NOT NULL,
                    sequence_length INTEGER NOT NULL DEFAULT 15,
                    prediction_horizon_sec INTEGER NOT NULL DEFAULT 14400,
                    selected_block_count INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS ml_training_run_blocks (
                    run_id INTEGER NOT NULL,
                    block_id INTEGER NOT NULL,
                    session_id INTEGER NOT NULL,
                    pair_key TEXT NOT NULL,
                    start_ts REAL NOT NULL,
                    end_ts REAL NOT NULL,
                    record_count INTEGER NOT NULL,
                    PRIMARY KEY(run_id, block_id),
                    FOREIGN KEY(run_id) REFERENCES ml_training_runs(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS ml_training_run_sessions (
                    run_id INTEGER NOT NULL,
                    session_id INTEGER NOT NULL,
                    position INTEGER NOT NULL DEFAULT 0,
                    started_at REAL NOT NULL,
                    ended_at REAL NOT NULL,
                    data_start_ts REAL NOT NULL,
                    data_end_ts REAL NOT NULL,
                    status TEXT NOT NULL,
                    approved_for_training INTEGER NOT NULL,
                    excluded_reason TEXT NOT NULL DEFAULT '',
                    block_count INTEGER NOT NULL DEFAULT 0,
                    trainable_block_count INTEGER NOT NULL DEFAULT 0,
                    exception_block_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(run_id, session_id),
                    FOREIGN KEY(run_id) REFERENCES ml_training_runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_tracker_pairs_lookup
                    ON tracker_pairs(symbol, buy_ex, buy_mt, sell_ex, sell_mt);
                CREATE INDEX IF NOT EXISTS idx_tracker_events_pair_type_ts
                    ON tracker_events(pair_id, event_type, ts);
                CREATE INDEX IF NOT EXISTS idx_tracker_records_pair_ts
                    ON tracker_records(pair_id, ts);
                CREATE INDEX IF NOT EXISTS idx_tracker_sessions_status_started
                    ON tracker_capture_sessions(status, started_at);
                CREATE INDEX IF NOT EXISTS idx_tracker_blocks_session_pair
                    ON tracker_pair_blocks(session_id, pair_id, start_ts);
                CREATE INDEX IF NOT EXISTS idx_tracker_blocks_training
                    ON tracker_pair_blocks(selected_for_training, is_open, start_ts);
                CREATE INDEX IF NOT EXISTS idx_tracker_episodes_pair_end
                    ON tracker_pair_episodes(pair_id, end_ts);
                CREATE INDEX IF NOT EXISTS idx_tracker_episodes_block
                    ON tracker_pair_episodes(block_id, end_ts);
                CREATE INDEX IF NOT EXISTS idx_tracker_hourly_health_verdict
                    ON tracker_hourly_health(quality_verdict, hour_start_ts);
                CREATE INDEX IF NOT EXISTS idx_ml_training_runs_status_created
                    ON ml_training_runs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_ml_training_run_sessions_run
                    ON ml_training_run_sessions(run_id, session_id);
                """
            )
            self._ensure_column(conn, "tracker_records", "session_id", "INTEGER")
            self._ensure_column(conn, "tracker_records", "block_id", "INTEGER")
            self._ensure_column(conn, "tracker_events", "session_id", "INTEGER")
            self._ensure_column(conn, "tracker_events", "block_id", "INTEGER")
            self._ensure_column(conn, "tracker_capture_sessions", "approved_for_training", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column(conn, "tracker_capture_sessions", "excluded_reason", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "tracker_capture_sessions", "manual_override", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "tracker_capture_sessions", "notes", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "ml_training_run_sessions", "position", "INTEGER NOT NULL DEFAULT 0")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracker_records_block_ts ON tracker_records(block_id, ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracker_events_block_ts ON tracker_events(block_id, ts)")
            conn.execute(
                """
                UPDATE tracker_capture_sessions
                SET approved_for_training = COALESCE(approved_for_training, CASE WHEN status = 'open' THEN 0 ELSE 1 END),
                    excluded_reason = COALESCE(excluded_reason, ''),
                    manual_override = COALESCE(manual_override, 0),
                    notes = COALESCE(notes, '')
                """
            )
            now = time.time()
            existing_meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM tracker_meta")}
            if "last_flush_at" in existing_meta:
                self._last_flush_at = self._coerce_float(existing_meta.get("last_flush_at"), self._last_flush_at)
            meta_updates = {
                "record_interval_sec": self.record_interval_sec,
                "tracking_window_sec": self.storage_window_sec,
                "tracker_memory_window_sec": self.memory_window_sec,
                "gap_threshold_sec": self.gap_threshold_sec,
                "min_total_spread_pct": self.min_total_spread_pct,
            }
            if "last_flush_at" not in existing_meta:
                meta_updates["last_flush_at"] = self._last_flush_at
            self._write_meta(
                conn,
                meta_updates,
            )
            conn.execute(
                "INSERT INTO tracker_meta(key, value) VALUES('created_at', ?) "
                "ON CONFLICT(key) DO NOTHING",
                (str(now),),
            )
            conn.execute(
                "INSERT INTO tracker_meta(key, value) VALUES('quality_fix_activated_at', ?) "
                "ON CONFLICT(key) DO NOTHING",
                (str(now),),
            )
            meta_rows = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM tracker_meta")}
            self.quality_fix_activated_at = self._coerce_float(meta_rows.get("quality_fix_activated_at"), now)
            self._load_hot_path_rejection_stats(meta_rows)
            if meta_rows.get("schema_version") != _SCHEMA_VERSION:
                self._migrate_to_v2(conn, meta_rows)
            self._write_meta(
                conn,
                {
                    "schema_version": _SCHEMA_VERSION,
                    "min_total_spread_pct": self.min_total_spread_pct,
                    "quality_fix_activated_at": self.quality_fix_activated_at,
                    _HOT_PATH_REJECTION_META_KEY: json.dumps(self._hot_path_rejection_snapshot_locked(), sort_keys=True),
                },
            )
            self._hot_path_rejection_stats_dirty = False

    def _migrate_to_v2(self, conn: sqlite3.Connection, meta_rows: dict[str, str]):
        records_exist = int(conn.execute("SELECT COUNT(*) FROM tracker_records").fetchone()[0]) > 0
        sessions_exist = int(conn.execute("SELECT COUNT(*) FROM tracker_capture_sessions").fetchone()[0]) > 0
        if not records_exist or sessions_exist:
            self._write_meta(
                conn,
                {
                    "schema_version": _SCHEMA_VERSION,
                    "gap_threshold_sec": self.gap_threshold_sec,
                    "min_total_spread_pct": self.min_total_spread_pct,
                },
            )
            return

        now = time.time()
        min_record_ts = self._coerce_float(conn.execute("SELECT MIN(ts) FROM tracker_records").fetchone()[0], now)
        max_record_ts = self._coerce_float(conn.execute("SELECT MAX(ts) FROM tracker_records").fetchone()[0], min_record_ts)
        created_at = self._coerce_float(meta_rows.get("created_at"), min_record_ts)
        ended_at = self._coerce_float(meta_rows.get("last_flush_at"), max_record_ts)
        cursor = conn.execute(
            """
            INSERT INTO tracker_capture_sessions(
                started_at, ended_at, status, record_interval_sec, tracking_window_sec,
                gap_threshold_sec, created_by, created_at, updated_at
            )
            VALUES(?, ?, 'legacy_imported', ?, ?, ?, 'migration', ?, ?)
            """,
            (
                min_record_ts,
                ended_at if ended_at >= min_record_ts else max_record_ts,
                self.record_interval_sec,
                self.storage_window_sec,
                self.gap_threshold_sec,
                created_at,
                now,
            ),
        )
        legacy_session_id = int(cursor.lastrowid)
        pair_rows = list(conn.execute("SELECT id FROM tracker_pairs ORDER BY symbol, buy_ex, buy_mt, sell_ex, sell_mt"))
        for row in pair_rows:
            pair_id = int(row["id"])
            record_rows = list(conn.execute("SELECT ts FROM tracker_records WHERE pair_id = ? ORDER BY ts ASC", (pair_id,)))
            blocks: list[tuple[int, float, float]] = []
            last_ts = 0.0
            current_block_id = 0
            block_start = 0.0
            block_end = 0.0
            block_count = 0
            block_max_gap = 0.0
            for index, record_row in enumerate(record_rows):
                ts_value = float(record_row["ts"])
                needs_new_block = index == 0 or (last_ts > 0.0 and (ts_value - last_ts) > self.gap_threshold_sec)
                if needs_new_block:
                    if current_block_id:
                        conn.execute(
                            """
                            UPDATE tracker_pair_blocks
                            SET start_ts = ?, end_ts = ?, record_count = ?, max_gap_sec = ?, is_open = 0, updated_at = ?
                            WHERE id = ?
                            """,
                            (block_start, block_end, block_count, block_max_gap, now, current_block_id),
                        )
                        blocks.append((current_block_id, block_start, block_end))
                    boundary_reason = "legacy_import" if index == 0 else "auto_gap"
                    cursor = conn.execute(
                        """
                        INSERT INTO tracker_pair_blocks(
                            pair_id, session_id, start_ts, end_ts, record_count, max_gap_sec, boundary_reason,
                            selected_for_training, disabled_reason, manual_override, notes, is_open, created_at, updated_at
                        )
                        VALUES(?, ?, ?, ?, 0, 0.0, ?, 1, '', 0, '', 0, ?, ?)
                        """,
                        (pair_id, legacy_session_id, ts_value, ts_value, boundary_reason, now, now),
                    )
                    current_block_id = int(cursor.lastrowid)
                    block_start = ts_value
                    block_end = ts_value
                    block_count = 0
                    block_max_gap = 0.0
                elif last_ts > 0.0:
                    block_max_gap = max(block_max_gap, ts_value - last_ts)
                    block_end = ts_value
                block_count += 1
                conn.execute(
                    "UPDATE tracker_records SET session_id = ?, block_id = ? WHERE pair_id = ? AND ts = ?",
                    (legacy_session_id, current_block_id, pair_id, ts_value),
                )
                last_ts = ts_value
            if current_block_id:
                conn.execute(
                    """
                    UPDATE tracker_pair_blocks
                    SET start_ts = ?, end_ts = ?, record_count = ?, max_gap_sec = ?, is_open = 0, updated_at = ?
                    WHERE id = ?
                    """,
                    (block_start, block_end, block_count, block_max_gap, now, current_block_id),
                )
                blocks.append((current_block_id, block_start, block_end))
            event_rows = list(conn.execute("SELECT event_type, ts FROM tracker_events WHERE pair_id = ? ORDER BY ts ASC", (pair_id,)))
            for event_row in event_rows:
                event_ts = float(event_row["ts"])
                event_block_id: Optional[int] = None
                for block_id, block_start_ts, block_end_ts in blocks:
                    if block_start_ts <= event_ts <= block_end_ts:
                        event_block_id = block_id
                        break
                conn.execute(
                    """
                    UPDATE tracker_events
                    SET session_id = ?, block_id = ?
                    WHERE pair_id = ? AND event_type = ? AND ts = ?
                    """,
                    (legacy_session_id, event_block_id, pair_id, str(event_row["event_type"]), event_ts),
                )
            self._rebuild_episodes_for_pair_conn(conn, pair_id)
        self._write_meta(
            conn,
            {
                "schema_version": _SCHEMA_VERSION,
                "gap_threshold_sec": self.gap_threshold_sec,
                "min_total_spread_pct": self.min_total_spread_pct,
            },
        )

    def _open_runtime_session(self):
        if self.db_path is None:
            self._active_session_id = self._ephemeral_session_id
            self._ephemeral_session_id += 1
            return
        now = time.time()
        with self._connect() as conn:
            open_sessions = list(conn.execute("SELECT id FROM tracker_capture_sessions WHERE status = 'open' ORDER BY id ASC"))
            for row in open_sessions:
                session_id = int(row["id"])
                conn.execute(
                    """
                    UPDATE tracker_capture_sessions
                    SET status = 'interrupted', ended_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (now, now, session_id),
                )
                conn.execute(
                    """
                    UPDATE tracker_pair_blocks
                    SET is_open = 0,
                        end_ts = CASE WHEN end_ts <= 0 THEN start_ts ELSE end_ts END,
                        updated_at = ?
                    WHERE session_id = ? AND is_open = 1
                    """,
                    (now, session_id),
                )
            cursor = conn.execute(
                """
                INSERT INTO tracker_capture_sessions(
                    started_at, ended_at, status, record_interval_sec, tracking_window_sec,
                    gap_threshold_sec, created_by, created_at, updated_at
                )
                VALUES(?, NULL, 'open', ?, ?, ?, 'runtime', ?, ?)
                """,
                (now, self.record_interval_sec, self.storage_window_sec, self.gap_threshold_sec, now, now),
            )
            self._active_session_id = int(cursor.lastrowid)
            self._write_meta(
                conn,
                {
                    "record_interval_sec": self.record_interval_sec,
                    "tracking_window_sec": self.storage_window_sec,
                    "tracker_memory_window_sec": self.memory_window_sec,
                    "gap_threshold_sec": self.gap_threshold_sec,
                },
            )

    def close_active_session(
        self,
        *,
        status: str = "closed",
        ended_at: float | None = None,
    ) -> None:
        finished_at = float(ended_at) if ended_at is not None else time.time()
        active_session_id = 0
        restore_payload: list[tuple[PairStats, int, int, float, float, int, float, str, TrackerBlockMeta | None]] = []
        with self._lock:
            current_blocks = []
            for stats in self._pairs.values():
                if stats.current_session_id == self._active_session_id and stats.current_block_id:
                    current_block_id = int(stats.current_block_id)
                    previous_meta = stats.block_meta.get(current_block_id)
                    restore_payload.append(
                        (
                            stats,
                            int(stats.current_session_id),
                            current_block_id,
                            float(stats.current_block_start_ts),
                            float(stats.current_block_end_ts),
                            int(stats.current_block_record_count),
                            float(stats.current_block_max_gap_sec),
                            str(stats.current_block_boundary_reason),
                            None
                            if previous_meta is None
                            else TrackerBlockMeta(
                                session_id=int(previous_meta.session_id),
                                start_ts=float(previous_meta.start_ts),
                                end_ts=float(previous_meta.end_ts),
                                record_count=int(previous_meta.record_count),
                                max_gap_sec=float(previous_meta.max_gap_sec),
                                boundary_reason=str(previous_meta.boundary_reason),
                                is_open=bool(previous_meta.is_open),
                            ),
                        )
                    )
                    self._close_current_block_locked(stats, finished_at)
                    closed_meta = stats.block_meta.get(current_block_id)
                    current_blocks.append(
                        (
                            current_block_id,
                            float(closed_meta.end_ts if closed_meta is not None else 0.0),
                            int(closed_meta.record_count if closed_meta is not None else 0),
                            float(closed_meta.max_gap_sec if closed_meta is not None else 0.0),
                        )
                    )
                    stats.current_session_id = 0
                elif stats.current_session_id == self._active_session_id:
                    restore_payload.append((stats, int(stats.current_session_id), 0, 0.0, 0.0, 0, 0.0, "", None))
                    stats.current_session_id = 0
            active_session_id = self._active_session_id
            self._active_session_id = 0

        if not active_session_id or self.db_path is None:
            return
        try:
            with self._connect() as conn:
                session_row = conn.execute(
                    "SELECT started_at FROM tracker_capture_sessions WHERE id = ?",
                    (active_session_id,),
                ).fetchone()
                session_started_at = self._coerce_float(session_row["started_at"] if session_row is not None else 0.0, 0.0)
                normalized_finished_at = max(
                    finished_at,
                    session_started_at,
                    max((float(block_end_ts) for _, block_end_ts, _, _ in current_blocks), default=0.0),
                )
                for block_id, end_ts, record_count, max_gap_sec in current_blocks:
                    if int(block_id) <= 0:
                        continue
                    conn.execute(
                        """
                        UPDATE tracker_pair_blocks
                        SET is_open = 0, end_ts = ?, record_count = ?, max_gap_sec = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (max(end_ts, normalized_finished_at if end_ts <= 0.0 else end_ts), record_count, max_gap_sec, normalized_finished_at, block_id),
                    )
                conn.execute(
                    """
                    UPDATE tracker_pair_blocks
                    SET is_open = 0,
                        end_ts = CASE WHEN end_ts <= 0 THEN start_ts ELSE end_ts END,
                        updated_at = ?
                    WHERE session_id = ? AND is_open = 1
                    """,
                    (normalized_finished_at, active_session_id),
                )
                conn.execute(
                    """
                    UPDATE tracker_capture_sessions
                    SET status = ?, ended_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, normalized_finished_at, normalized_finished_at, active_session_id),
                )
                self._reconcile_storage_conn(conn, session_ids=[active_session_id])
        except Exception:
            with self._lock:
                self._active_session_id = int(active_session_id)
                for (
                    stats,
                    previous_session_id,
                    previous_block_id,
                    previous_start_ts,
                    previous_end_ts,
                    previous_record_count,
                    previous_max_gap_sec,
                    previous_boundary_reason,
                    previous_meta,
                ) in restore_payload:
                    stats.current_session_id = int(previous_session_id)
                    stats.current_block_id = int(previous_block_id)
                    stats.current_block_start_ts = float(previous_start_ts)
                    stats.current_block_end_ts = float(previous_end_ts)
                    stats.current_block_record_count = int(previous_record_count)
                    stats.current_block_max_gap_sec = float(previous_max_gap_sec)
                    stats.current_block_boundary_reason = str(previous_boundary_reason)
                    if previous_block_id and previous_meta is not None:
                        stats.block_meta[int(previous_block_id)] = previous_meta
            raise
        self._run_wal_checkpoint()

    def _mark_dirty(self, key: PairKey):
        self._dirty_pairs.add(key)
        self._deleted_pairs.discard(key)

    def _drop_pair_locked(self, key: PairKey, *, delete_storage: bool = True):
        if key in self._pairs:
            del self._pairs[key]
            if delete_storage:
                self._deleted_pairs.add(key)
            self._dirty_pairs.discard(key)

    @staticmethod
    def _normalize_boundary_reason(boundary_reason: str) -> str:
        normalized = str(boundary_reason or "initial")
        return normalized if normalized in _BLOCK_BOUNDARY_REASONS else "initial"

    def _set_current_block_locked(self, ps: PairStats, block_id: int, ts: float, boundary_reason: str, session_id: int):
        normalized_reason = self._normalize_boundary_reason(boundary_reason)
        ps.current_session_id = int(session_id)
        ps.current_block_id = int(block_id)
        ps.current_block_start_ts = float(ts)
        ps.current_block_end_ts = float(ts)
        ps.current_block_record_count = 0
        ps.current_block_max_gap_sec = 0.0
        ps.current_block_boundary_reason = normalized_reason
        ps.block_meta[int(block_id)] = TrackerBlockMeta(
            session_id=int(session_id),
            start_ts=float(ts),
            end_ts=float(ts),
            record_count=0,
            max_gap_sec=0.0,
            boundary_reason=normalized_reason,
            is_open=True,
        )

    @staticmethod
    def _update_open_block_meta_locked(ps: PairStats):
        if not ps.current_block_id:
            return
        meta = ps.block_meta.get(int(ps.current_block_id))
        if meta is None:
            return
        meta.start_ts = float(ps.current_block_start_ts)
        meta.end_ts = float(ps.current_block_end_ts)
        meta.record_count = int(ps.current_block_record_count)
        meta.max_gap_sec = float(ps.current_block_max_gap_sec)
        meta.boundary_reason = str(ps.current_block_boundary_reason or meta.boundary_reason or "initial")
        meta.session_id = int(ps.current_session_id or meta.session_id)
        meta.is_open = True

    def _ensure_storage_pair_id_locked(self, key: PairKey, ps: PairStats) -> int:
        if ps.storage_pair_id:
            return ps.storage_pair_id
        if self.db_path is None:
            ps.storage_pair_id = max(1, len(self._pairs))
            return ps.storage_pair_id
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tracker_pairs(
                    symbol, buy_ex, buy_mt, sell_ex, sell_mt,
                    last_state, last_seen_ts, last_crossover_ts, history_enabled
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, buy_ex, buy_mt, sell_ex, sell_mt) DO UPDATE SET
                    last_state = excluded.last_state,
                    last_seen_ts = excluded.last_seen_ts,
                    last_crossover_ts = excluded.last_crossover_ts,
                    history_enabled = excluded.history_enabled
                """,
                (
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    key[4],
                    int(ps.last_state),
                    float(ps.last_seen_ts),
                    float(ps.last_crossover_ts),
                    1 if ps.history_enabled else 0,
                ),
            )
            row = conn.execute(
                """
                SELECT id
                FROM tracker_pairs
                WHERE symbol = ? AND buy_ex = ? AND buy_mt = ? AND sell_ex = ? AND sell_mt = ?
                """,
                key,
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Tracker pair upsert failed for {self._pair_id(key)}")
        ps.storage_pair_id = int(row["id"])
        return ps.storage_pair_id

    def _create_block_locked(self, key: PairKey, ps: PairStats, ts: float, boundary_reason: str) -> int:
        session_id = self._active_session_id or self._ephemeral_session_id
        if self.db_path is None:
            block_id = int(self._ephemeral_block_id)
        else:
            block_id = -int(self._ephemeral_block_id)
        self._ephemeral_block_id += 1
        self._set_current_block_locked(ps, block_id, ts, boundary_reason, session_id)
        return int(block_id)

    def _close_current_block_locked(self, ps: PairStats, finished_at: float):
        if not ps.current_block_id:
            return
        end_ts = float(ps.current_block_end_ts or ps.current_block_start_ts or finished_at)
        meta = ps.block_meta.get(int(ps.current_block_id))
        if meta is not None:
            meta.end_ts = end_ts
            meta.record_count = int(ps.current_block_record_count)
            meta.max_gap_sec = float(ps.current_block_max_gap_sec)
            meta.is_open = False
        ps.current_block_id = 0
        ps.current_block_start_ts = 0.0
        ps.current_block_end_ts = 0.0
        ps.current_block_record_count = 0
        ps.current_block_max_gap_sec = 0.0
        ps.current_block_boundary_reason = ""
        ps.current_session_id = self._active_session_id

    def _ensure_block_locked(self, key: PairKey, ps: PairStats, ts: float) -> int:
        active_session_id = self._active_session_id or self._ephemeral_session_id
        if not ps.current_block_id:
            boundary_reason = "session_boundary" if ps.current_session_id and ps.current_session_id != active_session_id else "initial"
            return self._create_block_locked(key, ps, ts, boundary_reason)
        if ps.current_session_id != active_session_id:
            self._close_current_block_locked(ps, ts)
            return self._create_block_locked(key, ps, ts, "session_boundary")
        if ps._last_record_ts > 0.0 and (ts - ps._last_record_ts) > self.gap_threshold_sec:
            self._close_current_block_locked(ps, ps._last_record_ts)
            return self._create_block_locked(key, ps, ts, "auto_gap")
        return ps.current_block_id

    def _pair_snapshot(
        self,
        key: PairKey,
        ps: PairStats,
        *,
        state_cutoff: float,
        record_cutoff: float,
    ) -> dict[str, object]:
        inv = [
            (float(event.timestamp), int(event.session_id), int(event.block_id) if event.block_id is not None else None)
            for event in ps.inverted_events
            if event.timestamp >= state_cutoff
        ]
        inv = list(dict.fromkeys(inv))
        ent = [
            (float(event.timestamp), int(event.session_id), int(event.block_id) if event.block_id is not None else None)
            for event in ps.entry_events
            if event.timestamp >= state_cutoff
        ]
        ent = list(dict.fromkeys(ent))
        ext = [
            (float(event.timestamp), int(event.session_id), int(event.block_id) if event.block_id is not None else None)
            for event in ps.exit_events
            if event.timestamp >= state_cutoff
        ]
        ext = list(dict.fromkeys(ext))
        recs = [
            (
                float(record.timestamp),
                float(record.entry_spread_pct),
                float(record.exit_spread_pct),
                int(record.session_id),
                int(record.block_id),
            )
            for record in ps.records
            if record.timestamp >= record_cutoff
        ]
        recs = list(dict.fromkeys(recs))
        blocks = [
            {
                "runtime_block_id": int(block_id),
                "session_id": int(meta.session_id),
                "start_ts": float(meta.start_ts),
                "end_ts": float(meta.end_ts),
                "record_count": int(meta.record_count),
                "max_gap_sec": float(meta.max_gap_sec),
                "boundary_reason": str(meta.boundary_reason or "initial"),
                "is_open": bool(meta.is_open),
            }
            for block_id, meta in ps.block_meta.items()
            if bool(meta.is_open) or float(meta.end_ts) >= state_cutoff
        ]
        episodes = [
            self._episode_payload(episode)
            for episode in ps.episodes
            if bool(episode.is_closed) and float(episode.end_ts) >= state_cutoff
        ]
        return {
            "key": key,
            "storage_pair_id": int(ps.storage_pair_id),
            "last_state": int(ps.last_state),
            "last_seen_ts": float(ps.last_seen_ts),
            "last_crossover_ts": float(ps.last_crossover_ts),
            "history_enabled": bool(ps.history_enabled),
            "inverted_events": inv,
            "entry_events": ent,
            "exit_events": ext,
            "records": recs,
            "blocks": blocks,
            "episodes": episodes,
            "abandoned_episode_count": int(ps.abandoned_episode_count),
        }

    @staticmethod
    def _delete_pair_storage_range(
        conn: sqlite3.Connection,
        *,
        pair_id: int,
        storage_cutoff: float,
        record_rewrite_cutoff: float,
    ) -> None:
        conn.execute("DELETE FROM tracker_events WHERE pair_id = ?", (pair_id,))
        conn.execute(
            "DELETE FROM tracker_records WHERE pair_id = ? AND (ts < ? OR ts >= ?)",
            (pair_id, float(storage_cutoff), float(record_rewrite_cutoff)),
        )
        conn.execute(
            "DELETE FROM tracker_pair_episodes WHERE pair_id = ? AND end_ts < ?",
            (pair_id, float(storage_cutoff)),
        )

    @staticmethod
    def _group_records_by_block(records: list[tuple[float, float, float, int, int]]) -> dict[int, dict[str, float]]:
        grouped: dict[int, dict[str, float]] = {}
        last_ts_by_block: dict[int, float] = {}
        for ts_value, _, _, session_id, block_id in records:
            if not block_id:
                continue
            meta = grouped.setdefault(
                int(block_id),
                {
                    "session_id": float(session_id),
                    "start_ts": float(ts_value),
                    "end_ts": float(ts_value),
                    "record_count": 0.0,
                    "max_gap_sec": 0.0,
                },
            )
            meta["record_count"] += 1.0
            meta["end_ts"] = float(ts_value)
            last_ts = last_ts_by_block.get(int(block_id))
            if last_ts is not None:
                meta["max_gap_sec"] = max(float(meta["max_gap_sec"]), float(ts_value - last_ts))
            last_ts_by_block[int(block_id)] = float(ts_value)
        return grouped

    @staticmethod
    def _episode_payload(episode: TrackerEpisode) -> dict[str, object]:
        return {
            "start_ts": float(episode.start_ts),
            "peak_ts": float(episode.peak_ts),
            "end_ts": float(episode.end_ts),
            "duration_sec": float(episode.duration_sec),
            "peak_entry_spread": float(episode.peak_entry_spread),
            "exit_spread_at_close": float(episode.exit_spread_at_close),
            "baseline_median": float(episode.baseline_median),
            "baseline_mad": float(episode.baseline_mad),
            "activation_threshold": float(episode.activation_threshold),
            "release_threshold": float(episode.release_threshold),
            "total_spread": float(_episode_total_spread(episode)),
            "session_id": int(episode.session_id),
            "block_id": int(episode.block_id),
            "source_version": str(episode.source_version),
            "is_closed": bool(episode.is_closed),
        }

    @staticmethod
    def _episodes_from_rows(rows: Iterable[sqlite3.Row]) -> list[TrackerEpisode]:
        return [
            TrackerEpisode(
                start_ts=float(row["start_ts"]),
                peak_ts=float(row["peak_ts"]),
                end_ts=float(row["end_ts"]),
                duration_sec=float(row["duration_sec"]),
                peak_entry_spread=float(row["peak_entry_spread"]),
                exit_spread_at_close=float(row["exit_spread_at_close"]),
                baseline_median=float(row["baseline_median"]),
                baseline_mad=float(row["baseline_mad"]),
                activation_threshold=float(row["activation_threshold"]),
                release_threshold=float(row["release_threshold"]),
                session_id=int(row["session_id"]),
                block_id=int(row["block_id"]),
                source_version=str(row["source_version"] or _EPISODE_SOURCE_VERSION),
                is_closed=bool(row["is_closed"]),
            )
            for row in rows
        ]

    def _ensure_episode_cache_locked(self, ps: PairStats):
        if ps._episodes_cache_valid:
            return
        episodes, diagnostics = compute_closed_episodes_with_diagnostics(
            list(ps.records),
            abandoned_cutoff_sec=self.episode_abandoned_cutoff_sec,
        )
        ps.episodes = deque(episodes)
        ps.abandoned_episode_count = int(diagnostics.get("abandoned_episode_count", 0) or 0)
        ps._episodes_cache_valid = True

    def _rebuild_episodes_for_pair_conn(self, conn: sqlite3.Connection, pair_id: int):
        records = [
            SpreadRecord(
                timestamp=float(row["ts"]),
                entry_spread_pct=float(row["entry_spread_pct"]),
                exit_spread_pct=float(row["exit_spread_pct"]),
                session_id=int(row["session_id"] or 0),
                block_id=int(row["block_id"] or 0),
            )
            for row in conn.execute(
                """
                SELECT ts, entry_spread_pct, exit_spread_pct, session_id, block_id
                FROM tracker_records
                WHERE pair_id = ?
                ORDER BY block_id ASC, ts ASC
                """,
                (int(pair_id),),
            )
        ]
        episodes, diagnostics = compute_closed_episodes_with_diagnostics(
            records,
            abandoned_cutoff_sec=self.episode_abandoned_cutoff_sec,
        )
        conn.execute("DELETE FROM tracker_pair_episodes WHERE pair_id = ?", (int(pair_id),))
        if episodes:
            conn.executemany(
                """
                INSERT INTO tracker_pair_episodes(
                    pair_id, session_id, block_id, start_ts, peak_ts, end_ts, duration_sec,
                    peak_entry_spread, exit_spread_at_close, baseline_median, baseline_mad,
                    activation_threshold, release_threshold, source_version, is_closed
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(pair_id),
                        int(episode.session_id),
                        int(episode.block_id),
                        float(episode.start_ts),
                        float(episode.peak_ts),
                        float(episode.end_ts),
                        float(episode.duration_sec),
                        float(episode.peak_entry_spread),
                        float(episode.exit_spread_at_close),
                        float(episode.baseline_median),
                        float(episode.baseline_mad),
                        float(episode.activation_threshold),
                        float(episode.release_threshold),
                        str(episode.source_version),
                        1 if episode.is_closed else 0,
                    )
                    for episode in episodes
                ],
            )
        with self._lock:
            for _, ps in self._pairs.items():
                if int(ps.storage_pair_id) == int(pair_id):
                    ps.episodes = deque(episodes)
                    ps.abandoned_episode_count = int(diagnostics.get("abandoned_episode_count", 0) or 0)
                    ps._episodes_cache_valid = True
                    break

    def _refresh_pair_records_from_storage(self, pair_id: int):
        if self.db_path is None:
            return
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT symbol, buy_ex, buy_mt, sell_ex, sell_mt, last_seen_ts
                FROM tracker_pairs
                WHERE id = ?
                """,
                (int(pair_id),),
            ).fetchone()
            if row is None:
                return
            reference_now = float(row["last_seen_ts"] or 0.0)
            if reference_now <= 0.0:
                reference_now = time.time()
            record_cutoff = self._memory_cutoff(reference_now)
            state_cutoff = self._storage_cutoff(reference_now)
            key = self._pair_key(row["symbol"], row["buy_ex"], row["buy_mt"], row["sell_ex"], row["sell_mt"])
            records = [
                SpreadRecord(
                    timestamp=float(record_row["ts"]),
                    entry_spread_pct=float(record_row["entry_spread_pct"]),
                    exit_spread_pct=float(record_row["exit_spread_pct"]),
                    session_id=int(record_row["session_id"] or 0),
                    block_id=int(record_row["block_id"] or 0),
                )
                for record_row in conn.execute(
                    """
                    SELECT ts, entry_spread_pct, exit_spread_pct, session_id, block_id
                    FROM tracker_records
                    WHERE pair_id = ? AND ts >= ?
                    ORDER BY ts ASC
                    """,
                    (int(pair_id), float(record_cutoff)),
                )
            ]
            episode_rows = list(
                conn.execute(
                    """
                    SELECT start_ts, peak_ts, end_ts, duration_sec, peak_entry_spread, exit_spread_at_close,
                           baseline_median, baseline_mad, activation_threshold, release_threshold,
                           session_id, block_id, source_version, is_closed
                    FROM tracker_pair_episodes
                    WHERE pair_id = ? AND end_ts >= ?
                    ORDER BY end_ts ASC, start_ts ASC
                    """,
                    (int(pair_id), float(state_cutoff)),
                )
            )
        with self._lock:
            ps = self._pairs.get(key)
            if ps is None:
                return
            ps.records = deque(records)
            ps._last_record_ts = max((record.timestamp for record in records), default=0.0)
            ps._stats_dirty = True
            ps.episodes = deque(self._episodes_from_rows(episode_rows))
            ps.abandoned_episode_count = 0
            ps._episodes_cache_valid = True

    def _delete_pairs_from_storage(self, conn: sqlite3.Connection, keys: Iterable[PairKey]):
        for key in keys:
            conn.execute(
                """
                DELETE FROM tracker_pairs
                WHERE symbol = ? AND buy_ex = ? AND buy_mt = ? AND sell_ex = ? AND sell_mt = ?
                """,
                key,
            )

    def record_spread(
        self,
        symbol: str,
        buy_ex: str,
        buy_mt: str,
        sell_ex: str,
        sell_mt: str,
        entry_spread: float,
        exit_spread: float,
        *,
        now_ts: float | None = None,
    ):
        ts = float(now_ts) if now_ts is not None else time.time()
        key = self._pair_key(symbol, buy_ex, buy_mt, sell_ex, sell_mt)
        invalid_fields = self._invalid_numeric_fields(ts, entry_spread, exit_spread)
        with self._lock:
            if invalid_fields:
                self._record_hot_path_attempt_locked(key)
                self._record_hot_path_rejection_locked(key, invalid_fields=invalid_fields)
        if invalid_fields:
            return
        entry_v = float(entry_spread)
        exit_v = float(exit_spread)

        new_state = self._state(entry_v, exit_v)
        record_cutoff = self._memory_cutoff(ts)
        state_cutoff = self._storage_cutoff(ts)
        with self._lock:
            pair_dirty = False
            if key not in self._pairs:
                if self.max_pairs > 0 and len(self._pairs) >= self.max_pairs:
                    return
                if self.track_enable_entry_spread_pct > 0 and entry_v < self.track_enable_entry_spread_pct:
                    return
                pair_dirty = True

            ps = self._pairs[key]
            ps.last_seen_ts = ts
            if ps._prune_events(state_cutoff):
                pair_dirty = True
            ps._prune_block_meta(state_cutoff)
            ps._prune_episodes(state_cutoff)
            event_session_id = self._active_session_id or ps.current_session_id or self._ephemeral_session_id
            event_block_id = ps.current_block_id if ps.history_enabled and ps.current_block_id else None
            if new_state in (-1, 1):
                if ps.last_state in (-1, 1) and new_state != ps.last_state:
                    ps.inverted_events.append(TrackerEvent(ts, "inverted", event_session_id, event_block_id))
                    ps.last_crossover_ts = ts
                    pair_dirty = True
                if new_state == 1 and ps.last_state != 1:
                    ps.entry_events.append(TrackerEvent(ts, "entry", event_session_id, event_block_id))
                    pair_dirty = True
                if new_state == -1 and ps.last_state != -1:
                    ps.exit_events.append(TrackerEvent(ts, "exit", event_session_id, event_block_id))
                    pair_dirty = True
                if ps.last_state != new_state:
                    ps.last_state = new_state
                    pair_dirty = True

            if (not ps.history_enabled) and (entry_v >= self.history_enable_entry_spread_pct):
                ps.history_enabled = True
                pair_dirty = True

            if ps.history_enabled and self.record_interval_sec >= 0:
                preserve_after_ts = self._last_flush_at if self._last_flush_at > 0.0 else float("-inf")
                if ps._prune_records(record_cutoff, preserve_after_ts=preserve_after_ts):
                    pair_dirty = True
                timestamp_status = self._timestamp_status_locked(ps, ts)
                if timestamp_status == "exact_duplicate":
                    self._record_hot_path_attempt_locked(key)
                    self._record_hot_path_rejection_locked(key, invalid_fields=["duplicate_ts"])
                    ps.exact_duplicate_ts_rejections += 1
                    self._emit_tracker_record_rejected(
                        key=key,
                        ts=ts,
                        entry_spread=entry_v,
                        exit_spread=exit_v,
                        invalid_fields=["duplicate_ts"],
                        reason="exact_duplicate_ts",
                        last_record_ts=ps._last_record_ts,
                    )
                    return
                if timestamp_status == "retrograde":
                    self._record_hot_path_attempt_locked(key)
                    self._record_hot_path_rejection_locked(key, invalid_fields=["retrograde_ts"])
                    ps.retrograde_ts_rejections += 1
                    self._emit_tracker_record_rejected(
                        key=key,
                        ts=ts,
                        entry_spread=entry_v,
                        exit_spread=exit_v,
                        invalid_fields=["retrograde_ts"],
                        reason="retrograde_ts_rejected",
                        last_record_ts=ps._last_record_ts,
                    )
                    return
                clock_jitter_ts = timestamp_status == "clock_jitter"
                if clock_jitter_ts:
                    ps.clock_jitter_events += 1
                effective_ts = self._adjust_record_ts(ps._last_record_ts, ts)
                record_delta_sec = (effective_ts - ps._last_record_ts) if ps._last_record_ts > 0.0 else 0.0
                gap_detected = bool(ps._last_record_ts > 0.0 and record_delta_sec > self.gap_threshold_sec)
                monotonic = bool(ps._last_record_ts <= 0.0 or effective_ts > ps._last_record_ts)
                should_record = (
                    not ps.records
                    or (effective_ts - ps._last_record_ts) >= max(self.record_interval_sec, 0.0)
                )
                if should_record:
                    self._record_hot_path_attempt_locked(key)
                    fingerprint = self._record_fingerprint(ts, entry_v, exit_v, self.record_interval_sec)
                    if self._is_light_duplicate_locked(ps, fingerprint):
                        self._record_hot_path_rejection_locked(key, invalid_fields=["duplicate_payload"])
                        self._emit_tracker_record_rejected(
                            key=key,
                            ts=ts,
                            entry_spread=entry_v,
                            exit_spread=exit_v,
                            invalid_fields=["duplicate_payload"],
                            reason="light_duplicate_payload",
                            last_record_ts=ps._last_record_ts,
                        )
                        return
                    block_id = self._ensure_block_locked(key, ps, effective_ts)
                    if ps._last_record_ts > 0.0 and ps.current_block_record_count > 0:
                        ps.current_block_max_gap_sec = max(ps.current_block_max_gap_sec, effective_ts - ps._last_record_ts)
                    spread_outlier_warning = self._is_pair_spread_outlier_locked(ps, entry_v)
                    if spread_outlier_warning:
                        ps.spread_outlier_warnings += 1
                    ps.records.append(
                        SpreadRecord(
                            timestamp=effective_ts,
                            entry_spread_pct=entry_v,
                            exit_spread_pct=exit_v,
                            session_id=self._active_session_id or ps.current_session_id,
                            block_id=block_id,
                        )
                    )
                    self._remember_record_locked(ps, fingerprint=fingerprint, entry_spread=entry_v)
                    ps.current_block_end_ts = effective_ts
                    ps.current_block_record_count += 1
                    self._update_open_block_meta_locked(ps)
                    ps._last_record_ts = effective_ts
                    ps._stats_dirty = True
                    ps._episodes_cache_valid = False
                    pair_dirty = True
                    if self.audit_collector is not None:
                        self.audit_collector.record_tracker_record(
                            pair_id=self._pair_id(key),
                            ts=ts,
                            entry=entry_v,
                            exit=exit_v,
                            session_id=self._active_session_id or ps.current_session_id,
                            block_id=block_id,
                            record_delta_sec=record_delta_sec,
                            gap_detected=gap_detected,
                            monotonic=monotonic,
                            invalid_fields=list(invalid_fields),
                            exact_duplicate_ts=False,
                            retrograde_ts_rejected=False,
                            clock_jitter_ts=clock_jitter_ts,
                            spread_outlier_warning=spread_outlier_warning,
                        )
                    self._emit_event(
                        {
                            "kind": "tracker_record",
                            "pair_key": self._pair_id(key),
                            "record_ts": effective_ts,
                            "entry": entry_v,
                            "exit": exit_v,
                            "session_id": self._active_session_id or ps.current_session_id,
                            "block_id": block_id,
                            "delta_ts": record_delta_sec,
                            "gap_detected": gap_detected,
                            "gap_threshold_sec": self.gap_threshold_sec,
                            "numeric_valid": not invalid_fields,
                            "invalid_fields": list(invalid_fields),
                            "timestamp_monotonic": monotonic,
                            "exact_duplicate_ts": False,
                            "retrograde_ts_rejected": False,
                            "clock_jitter_ts": clock_jitter_ts,
                            "spread_outlier_warning": spread_outlier_warning,
                        }
                    )
                    if self.max_records_per_pair > 0:
                        while len(ps.records) > self.max_records_per_pair:
                            ps.records.popleft()
                            ps._stats_dirty = True
                            pair_dirty = True

            if pair_dirty:
                self._mark_dirty(key)

    def get_pair_stats_obj(self, symbol: str, buy_ex: str, buy_mt: str, sell_ex: str, sell_mt: str) -> Optional[PairStats]:
        key = self._pair_key(symbol, buy_ex, buy_mt, sell_ex, sell_mt)
        with self._lock:
            return self._pairs.get(key)

    def get_pair_stats(self, symbol: str, buy_ex: str, buy_mt: str, sell_ex: str, sell_mt: str) -> Optional[Dict]:
        key = self._pair_key(symbol, buy_ex, buy_mt, sell_ex, sell_mt)
        with self._lock:
            ps = self._pairs.get(key)
            if not ps:
                return None
            ps.ensure_stats()
            return {
                "inverted_count": ps.inverted_count,
                "total_entries": ps.total_entries,
                "total_exits": ps.total_exits,
                "last_crossover_ts": ps.last_crossover_ts,
                "min_spread": ps.min_spread,
                "max_spread": ps.max_spread,
                "avg_spread": round(ps.avg_spread, 4),
                "record_count": len(ps.records),
                "last_seen_ts": ps.last_seen_ts,
                "exact_duplicate_ts_rejections": int(ps.exact_duplicate_ts_rejections),
                "retrograde_ts_rejections": int(ps.retrograde_ts_rejections),
                "clock_jitter_events": int(ps.clock_jitter_events),
                "spread_outlier_warnings": int(ps.spread_outlier_warnings),
                "abandoned_episode_count": int(ps.abandoned_episode_count),
            }

    def get_history(self, symbol: str, buy_ex: str, buy_mt: str, sell_ex: str, sell_mt: str, limit: int = 200) -> List[Dict]:
        key = self._pair_key(symbol, buy_ex, buy_mt, sell_ex, sell_mt)
        with self._lock:
            ps = self._pairs.get(key)
            if not ps or not ps.records:
                return []
            records = list(ps.records)[-max(int(limit), 0) :]
        return [
            {
                "timestamp": record.timestamp,
                "entry_spread": round(record.entry_spread_pct, 4),
                "exit_spread": round(record.exit_spread_pct, 4),
            }
            for record in records
        ]

    def get_feature_history(
        self,
        symbol: str,
        buy_ex: str,
        buy_mt: str,
        sell_ex: str,
        sell_mt: str,
        *,
        limit: int = 15,
        feature_names: list[str] | None = None,
    ) -> list[list[float]]:
        key = self._pair_key(symbol, buy_ex, buy_mt, sell_ex, sell_mt)
        normalized_names = tuple(str(name) for name in (feature_names or []))
        cache_key = (key, normalized_names, int(limit))
        with self._lock:
            ps = self._pairs.get(key)
            if ps is None or not ps.records:
                return []
            last_record_ts = float(ps._last_record_ts or 0.0)
            cached = self._feature_history_cache.get(cache_key)
            if cached and float(cached.get("last_record_ts", 0.0) or 0.0) == last_record_ts:
                return [list(row) for row in list(cached.get("rows") or [])]
            self._ensure_episode_cache_locked(ps)
            records = list(ps.records)
            episodes = list(ps.episodes)
        if not records:
            return []
        lookback_sec = max(self._WINDOWS.get("8h", 8 * 60 * 60), int(limit) * max(int(self.record_interval_sec), 1))
        cutoff = float(records[-1].timestamp) - float(lookback_sec)
        recent_records = [
            {
                "timestamp": float(record.timestamp),
                "entry_spread": float(record.entry_spread_pct),
                "exit_spread": float(record.exit_spread_pct),
                "session_id": int(record.session_id or 0),
                "block_id": int(record.block_id or 0),
            }
            for record in records
            if float(record.timestamp) >= cutoff
        ]
        rows = build_feature_rows(
            recent_records,
            feature_names=list(feature_names or []),
            episodes=episodes,
            cost_estimate_pct=float(getattr(getattr(self, "config", None), "default_cost_estimate_pct", 0.30)),
        )
        rows = rows[-max(int(limit), 0) :]
        with self._lock:
            self._feature_history_cache[cache_key] = {
                "last_record_ts": float(last_record_ts),
                "rows": [list(row) for row in rows],
            }
        return [list(row) for row in rows]

    def save_hourly_health_digest(self, digest: dict[str, Any]) -> dict[str, Any]:
        if self.db_path is None:
            return dict(digest)
        payload = dict(digest)
        hour_start_ts = float(payload.get("hour_start_ts", 0.0) or 0.0)
        if hour_start_ts <= 0.0:
            raise ValueError("hour_start_ts is required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tracker_hourly_health(
                    hour_start_ts, hour_end_ts, records_total, records_rejected, rejection_rate_pct,
                    exchanges_active, exchanges_circuit_open_json, pairs_with_records, pairs_with_gaps,
                    cross_exchange_flags, avg_book_age_json, episode_count, quality_verdict, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hour_start_ts) DO UPDATE SET
                    hour_end_ts = excluded.hour_end_ts,
                    records_total = excluded.records_total,
                    records_rejected = excluded.records_rejected,
                    rejection_rate_pct = excluded.rejection_rate_pct,
                    exchanges_active = excluded.exchanges_active,
                    exchanges_circuit_open_json = excluded.exchanges_circuit_open_json,
                    pairs_with_records = excluded.pairs_with_records,
                    pairs_with_gaps = excluded.pairs_with_gaps,
                    cross_exchange_flags = excluded.cross_exchange_flags,
                    avg_book_age_json = excluded.avg_book_age_json,
                    episode_count = excluded.episode_count,
                    quality_verdict = excluded.quality_verdict,
                    created_at = excluded.created_at
                """,
                (
                    hour_start_ts,
                    float(payload.get("hour_end_ts", 0.0) or 0.0),
                    int(payload.get("records_total", 0) or 0),
                    int(payload.get("records_rejected", 0) or 0),
                    float(payload.get("rejection_rate_pct", 0.0) or 0.0),
                    int(payload.get("exchanges_active", 0) or 0),
                    json.dumps(list(payload.get("exchanges_circuit_open") or []), sort_keys=True),
                    int(payload.get("pairs_with_records", 0) or 0),
                    int(payload.get("pairs_with_gaps", 0) or 0),
                    int(payload.get("cross_exchange_flags", 0) or 0),
                    json.dumps(dict(payload.get("avg_book_age_sec") or {}), sort_keys=True),
                    int(payload.get("episode_count", 0) or 0),
                    str(payload.get("quality_verdict") or "healthy"),
                    float(payload.get("created_at", time.time()) or time.time()),
                ),
            )
        return payload

    def list_hourly_health(self, *, limit: int = 24) -> list[dict[str, Any]]:
        if self.db_path is None or not self.db_path.is_file():
            return []
        with self._connect() as conn:
            rows = list(
                conn.execute(
                    """
                    SELECT hour_start_ts, hour_end_ts, records_total, records_rejected, rejection_rate_pct,
                           exchanges_active, exchanges_circuit_open_json, pairs_with_records, pairs_with_gaps,
                           cross_exchange_flags, avg_book_age_json, episode_count, quality_verdict, created_at
                    FROM tracker_hourly_health
                    ORDER BY hour_start_ts DESC
                    LIMIT ?
                    """,
                    (max(int(limit), 1),),
                )
            )
        payload: list[dict[str, Any]] = []
        for row in rows:
            payload.append(
                {
                    "hour_start_ts": float(row["hour_start_ts"]),
                    "hour_end_ts": float(row["hour_end_ts"]),
                    "records_total": int(row["records_total"]),
                    "records_rejected": int(row["records_rejected"]),
                    "rejection_rate_pct": float(row["rejection_rate_pct"]),
                    "exchanges_active": int(row["exchanges_active"]),
                    "exchanges_circuit_open": list(json.loads(str(row["exchanges_circuit_open_json"] or "[]"))),
                    "pairs_with_records": int(row["pairs_with_records"]),
                    "pairs_with_gaps": int(row["pairs_with_gaps"]),
                    "cross_exchange_flags": int(row["cross_exchange_flags"]),
                    "avg_book_age_sec": dict(json.loads(str(row["avg_book_age_json"] or "{}"))),
                    "episode_count": int(row["episode_count"]),
                    "quality_verdict": str(row["quality_verdict"] or "healthy"),
                    "created_at": float(row["created_at"]),
                }
            )
        return payload

    def get_latest_hourly_health(self) -> dict[str, Any] | None:
        rows = self.list_hourly_health(limit=1)
        return rows[0] if rows else None

    def disable_open_blocks(self, *, reason: str) -> int:
        if self.db_path is None:
            return 0
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE tracker_pair_blocks
                SET disabled_reason = ?, updated_at = ?
                WHERE is_open = 1 AND COALESCE(disabled_reason, '') = ''
                """,
                (str(reason), time.time()),
            )
        return int(cursor.rowcount or 0)

    def get_pair_episodes(
        self,
        symbol: str,
        buy_ex: str,
        buy_mt: str,
        sell_ex: str,
        sell_mt: str,
        *,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        key = self._pair_key(symbol, buy_ex, buy_mt, sell_ex, sell_mt)
        with self._lock:
            ps = self._pairs.get(key)
            if ps is not None:
                self._ensure_episode_cache_locked(ps)
                episodes = list(ps.episodes)[-max(int(limit), 0) :]
                return [self._episode_payload(episode) for episode in episodes]

        if self.db_path is None or not self.db_path.is_file():
            return []
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id
                FROM tracker_pairs
                WHERE symbol = ? AND buy_ex = ? AND buy_mt = ? AND sell_ex = ? AND sell_mt = ?
                """,
                key,
            ).fetchone()
            if row is None:
                return []
            episode_rows = list(
                conn.execute(
                    """
                    SELECT start_ts, peak_ts, end_ts, duration_sec, peak_entry_spread, exit_spread_at_close,
                           baseline_median, baseline_mad, activation_threshold, release_threshold,
                           session_id, block_id, source_version, is_closed
                    FROM tracker_pair_episodes
                    WHERE pair_id = ?
                    ORDER BY end_ts ASC, start_ts ASC
                    LIMIT ?
                    """,
                    (int(row["id"]), max(int(limit), 0)),
                )
            )
        return [self._episode_payload(episode) for episode in self._episodes_from_rows(episode_rows)]

    def get_pair_recurring_context(
        self,
        symbol: str,
        buy_ex: str,
        buy_mt: str,
        sell_ex: str,
        sell_mt: str,
        *,
        current_entry: float,
        now_ts: float | None = None,
    ) -> dict[str, object]:
        key = self._pair_key(symbol, buy_ex, buy_mt, sell_ex, sell_mt)
        with self._lock:
            ps = self._pairs.get(key)
            if ps is not None:
                self._ensure_episode_cache_locked(ps)
                context = build_recurring_context_from_episodes(
                    list(ps.episodes),
                    current_entry=float(current_entry),
                    now_ts=float(now_ts) if now_ts is not None else float(ps._last_record_ts or time.time()),
                    min_total_spread_pct=self.min_total_spread_pct,
                )
                return context

        episodes = self.get_pair_episodes(symbol, buy_ex, buy_mt, sell_ex, sell_mt, limit=500)
        typed_episodes = [
            TrackerEpisode(
                start_ts=float(item["start_ts"]),
                peak_ts=float(item["peak_ts"]),
                end_ts=float(item["end_ts"]),
                duration_sec=float(item["duration_sec"]),
                peak_entry_spread=float(item["peak_entry_spread"]),
                exit_spread_at_close=float(item["exit_spread_at_close"]),
                baseline_median=float(item["baseline_median"]),
                baseline_mad=float(item["baseline_mad"]),
                activation_threshold=float(item["activation_threshold"]),
                release_threshold=float(item["release_threshold"]),
                session_id=int(item["session_id"]),
                block_id=int(item["block_id"]),
                source_version=str(item["source_version"]),
                is_closed=bool(item["is_closed"]),
            )
            for item in episodes
        ]
        return build_recurring_context_from_episodes(
            typed_episodes,
            current_entry=float(current_entry),
            now_ts=float(now_ts) if now_ts is not None else time.time(),
            min_total_spread_pct=self.min_total_spread_pct,
        )

    def prune(self, *, now_ts: float | None = None):
        now = float(now_ts) if now_ts is not None else time.time()
        storage_cutoff = self._storage_cutoff(now)
        storage_stale: list[PairKey] = []
        with self._lock:
            before = len(self._pairs)
            stale = [key for key, stats in self._pairs.items() if (stats.last_seen_ts or 0.0) < storage_cutoff]
            for key in stale:
                self._drop_pair_locked(key)
                storage_stale.append(key)
            if self.max_pairs > 0 and len(self._pairs) > self.max_pairs:
                by_lru = sorted(self._pairs.items(), key=lambda item: item[1].last_seen_ts)
                excess = len(self._pairs) - self.max_pairs
                for key, _ in by_lru[:excess]:
                    self._drop_pair_locked(key, delete_storage=False)
            after = len(self._pairs)
            if before != after:
                logger.info("[Tracker] Pruned %s pairs (%s -> %s)", before - after, before, after)

        if self.db_path is not None and storage_stale:
            with self._connect() as conn:
                self._delete_pairs_from_storage(conn, storage_stale)

    def flush_to_storage(self, *, now_ts: float | None = None, force: bool = False) -> bool:
        if self.db_path is None:
            return False
        ts = float(now_ts) if now_ts is not None else time.time()
        memory_cutoff = self._memory_cutoff(ts)
        storage_cutoff = self._storage_cutoff(ts)
        record_rewrite_cutoff = min(memory_cutoff, float(self._last_flush_at)) if self._last_flush_at > 0.0 else None
        snapshot_record_cutoff = float(record_rewrite_cutoff) if record_rewrite_cutoff is not None else float("-inf")
        with self._lock:
            rejection_stats_payload = self._hot_path_rejection_snapshot_locked()
            rejection_stats_dirty = bool(self._hot_path_rejection_stats_dirty)
            if not force and not self._dirty_pairs and not self._deleted_pairs and not rejection_stats_dirty:
                return True
            total_dirty_pairs = len(self._dirty_pairs)
            if force:
                dirty_keys = list(self._pairs.keys())
            else:
                dirty_keys = sorted(
                    self._dirty_pairs,
                    key=lambda item: float(self._pairs.get(item).last_seen_ts if self._pairs.get(item) is not None else 0.0),
                    reverse=True,
                )
                if len(dirty_keys) > _FLUSH_PAIR_BATCH_MAX:
                    dirty_keys = dirty_keys[:_FLUSH_PAIR_BATCH_MAX]
            deleted_keys = list(self._deleted_pairs)
            snapshots = []
            for key in dirty_keys:
                ps = self._pairs.get(key)
                if ps is None:
                    continue
                ps._prune_events(storage_cutoff)
                preserve_after_ts = self._last_flush_at if self._last_flush_at > 0.0 else float("-inf")
                ps._prune_records(memory_cutoff, preserve_after_ts=preserve_after_ts)
                ps._prune_block_meta(storage_cutoff)
                ps._prune_episodes(storage_cutoff)
                self._ensure_episode_cache_locked(ps)
                snapshots.append(
                    self._pair_snapshot(
                        key,
                        ps,
                        state_cutoff=storage_cutoff,
                        record_cutoff=snapshot_record_cutoff,
                    )
                )
        if force or total_dirty_pairs > _FLUSH_PAIR_BATCH_MAX:
            logger.info(
                "[Tracker] Flush start: force=%s dirty_pairs=%s batch_pairs=%s deleted_pairs=%s",
                force,
                total_dirty_pairs,
                len(dirty_keys),
                len(deleted_keys),
            )

        started_at = time.perf_counter()
        snapshot_rebindings: list[tuple[PairKey, int, dict[int, int]]] = []
        records_before = 0
        records_after = 0
        try:
            with self._connect() as conn:
                records_before_row = conn.execute("SELECT COUNT(*) FROM tracker_records").fetchone()
                records_before = int(records_before_row[0] if records_before_row is not None else 0)
                conn.execute("BEGIN")
                if deleted_keys:
                    self._delete_pairs_from_storage(conn, deleted_keys)
                for snapshot in snapshots:
                    key = snapshot["key"]
                    if not isinstance(key, tuple):
                        continue
                    records = snapshot["records"]
                    inv = snapshot["inverted_events"]
                    ent = snapshot["entry_events"]
                    ext = snapshot["exit_events"]
                    has_payload = bool(records or inv or ent or ext)
                    storage_pair_id = int(snapshot["storage_pair_id"] or 0)
                    if not has_payload and storage_pair_id <= 0:
                        continue
                    if not storage_pair_id:
                        conn.execute(
                            """
                            INSERT INTO tracker_pairs(
                                symbol, buy_ex, buy_mt, sell_ex, sell_mt,
                                last_state, last_seen_ts, last_crossover_ts, history_enabled
                            )
                            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(symbol, buy_ex, buy_mt, sell_ex, sell_mt) DO UPDATE SET
                                last_state = excluded.last_state,
                                last_seen_ts = excluded.last_seen_ts,
                                last_crossover_ts = excluded.last_crossover_ts,
                                history_enabled = excluded.history_enabled
                            """,
                            (
                                key[0], key[1], key[2], key[3], key[4],
                                int(snapshot["last_state"]),
                                float(snapshot["last_seen_ts"]),
                                float(snapshot["last_crossover_ts"]),
                                1 if snapshot["history_enabled"] else 0,
                            ),
                        )
                        row = conn.execute(
                            """
                            SELECT id FROM tracker_pairs
                            WHERE symbol = ? AND buy_ex = ? AND buy_mt = ? AND sell_ex = ? AND sell_mt = ?
                            """,
                            key,
                        ).fetchone()
                        if row is None:
                            raise RuntimeError(f"Tracker pair upsert failed for {self._pair_id(key)}")
                        storage_pair_id = int(row["id"])
                    else:
                        conn.execute(
                            """
                            UPDATE tracker_pairs
                            SET last_state = ?, last_seen_ts = ?, last_crossover_ts = ?, history_enabled = ?
                            WHERE id = ?
                            """,
                            (
                                int(snapshot["last_state"]),
                                float(snapshot["last_seen_ts"]),
                                float(snapshot["last_crossover_ts"]),
                                1 if snapshot["history_enabled"] else 0,
                                storage_pair_id,
                            ),
                            )
                    pair_record_rewrite_cutoff = record_rewrite_cutoff
                    if pair_record_rewrite_cutoff is None:
                        pair_record_rewrite_cutoff = min((float(record[0]) for record in records), default=float("inf"))
                    self._delete_pair_storage_range(
                        conn,
                        pair_id=storage_pair_id,
                        storage_cutoff=storage_cutoff,
                        record_rewrite_cutoff=pair_record_rewrite_cutoff,
                    )
                    if not has_payload:
                        continue
                    block_metadata = {
                        int(item["runtime_block_id"]): {
                            "session_id": int(item["session_id"]),
                            "start_ts": float(item["start_ts"]),
                            "end_ts": float(item["end_ts"]),
                            "record_count": int(item["record_count"]),
                            "max_gap_sec": float(item["max_gap_sec"]),
                            "boundary_reason": self._normalize_boundary_reason(str(item["boundary_reason"])),
                            "is_open": bool(item["is_open"]),
                        }
                        for item in snapshot.get("blocks", [])
                    }
                    grouped = self._group_records_by_block(records)
                    current_block_ids = sorted({int(block_id) for block_id in grouped.keys()} | {int(block_id) for block_id in block_metadata.keys() if int(block_id) != 0})
                    runtime_to_storage_block_id: dict[int, int] = {}
                    for runtime_block_id in current_block_ids:
                        block_group = grouped.get(int(runtime_block_id), {})
                        block_meta = block_metadata.get(int(runtime_block_id), {})
                        session_id = int(block_meta.get("session_id") or block_group.get("session_id") or self._active_session_id or 0)
                        start_ts_value = float(block_group.get("start_ts", block_meta.get("start_ts", snapshot["last_seen_ts"])))
                        end_ts_value = float(block_group.get("end_ts", block_meta.get("end_ts", start_ts_value)))
                        record_count_value = int(block_group.get("record_count", block_meta.get("record_count", 0)))
                        max_gap_value = float(block_group.get("max_gap_sec", block_meta.get("max_gap_sec", 0.0)))
                        boundary_reason = self._normalize_boundary_reason(str(block_meta.get("boundary_reason", "initial")))
                        is_open = 1 if bool(block_meta.get("is_open", False)) else 0
                        if runtime_block_id > 0:
                            conn.execute(
                                """
                                UPDATE tracker_pair_blocks
                                SET session_id = ?, start_ts = ?, end_ts = ?, record_count = ?, max_gap_sec = ?,
                                    boundary_reason = ?, is_open = ?, updated_at = ?
                                WHERE id = ?
                                """,
                                (
                                    session_id,
                                    start_ts_value,
                                    end_ts_value,
                                    record_count_value,
                                    max_gap_value,
                                    boundary_reason,
                                    is_open,
                                    ts,
                                    int(runtime_block_id),
                                ),
                            )
                            runtime_to_storage_block_id[int(runtime_block_id)] = int(runtime_block_id)
                        else:
                            cursor = conn.execute(
                                """
                                INSERT INTO tracker_pair_blocks(
                                    pair_id, session_id, start_ts, end_ts, record_count, max_gap_sec, boundary_reason,
                                    selected_for_training, disabled_reason, manual_override, notes, is_open, created_at, updated_at
                                )
                                VALUES(?, ?, ?, ?, ?, ?, ?, 1, '', 0, '', ?, ?, ?)
                                """,
                                (
                                    storage_pair_id,
                                    session_id,
                                    start_ts_value,
                                    end_ts_value,
                                    record_count_value,
                                    max_gap_value,
                                    boundary_reason,
                                    is_open,
                                    ts,
                                    ts,
                                ),
                            )
                            runtime_to_storage_block_id[int(runtime_block_id)] = int(cursor.lastrowid)
                    mapped_inv = [
                        (
                            storage_pair_id,
                            float(event_ts),
                            int(session_id),
                            int(runtime_to_storage_block_id.get(int(block_id), int(block_id))) if block_id is not None else None,
                        )
                        for event_ts, session_id, block_id in inv
                    ]
                    mapped_inv = list(dict.fromkeys(mapped_inv))
                    mapped_ent = [
                        (
                            storage_pair_id,
                            float(event_ts),
                            int(session_id),
                            int(runtime_to_storage_block_id.get(int(block_id), int(block_id))) if block_id is not None else None,
                        )
                        for event_ts, session_id, block_id in ent
                    ]
                    mapped_ent = list(dict.fromkeys(mapped_ent))
                    mapped_ext = [
                        (
                            storage_pair_id,
                            float(event_ts),
                            int(session_id),
                            int(runtime_to_storage_block_id.get(int(block_id), int(block_id))) if block_id is not None else None,
                        )
                        for event_ts, session_id, block_id in ext
                    ]
                    mapped_ext = list(dict.fromkeys(mapped_ext))
                    mapped_records = [
                        (
                            storage_pair_id,
                            float(record[0]),
                            float(record[1]),
                            float(record[2]),
                            int(record[3]),
                            int(runtime_to_storage_block_id.get(int(record[4]), int(record[4]))),
                        )
                        for record in records
                    ]
                    mapped_records = list(dict.fromkeys(mapped_records))
                    if inv:
                        conn.executemany(
                            """
                            INSERT INTO tracker_events(pair_id, event_type, ts, session_id, block_id)
                            VALUES(?, 'inverted', ?, ?, ?)
                            """,
                            mapped_inv,
                        )
                    if ent:
                        conn.executemany(
                            """
                            INSERT INTO tracker_events(pair_id, event_type, ts, session_id, block_id)
                            VALUES(?, 'entry', ?, ?, ?)
                            """,
                            mapped_ent,
                        )
                    if ext:
                        conn.executemany(
                            """
                            INSERT INTO tracker_events(pair_id, event_type, ts, session_id, block_id)
                            VALUES(?, 'exit', ?, ?, ?)
                            """,
                            mapped_ext,
                        )
                    if records:
                        conn.executemany(
                            """
                            INSERT INTO tracker_records(pair_id, ts, entry_spread_pct, exit_spread_pct, session_id, block_id)
                            VALUES(?, ?, ?, ?, ?, ?)
                            """,
                            mapped_records,
                        )
                    episodes = snapshot.get("episodes", [])
                    conn.execute(
                        """
                        DELETE FROM tracker_pair_episodes
                        WHERE pair_id = ? AND end_ts >= ?
                        """,
                        (storage_pair_id, float(storage_cutoff)),
                    )
                    if episodes:
                        conn.executemany(
                            """
                            INSERT INTO tracker_pair_episodes(
                                pair_id, session_id, block_id, start_ts, peak_ts, end_ts, duration_sec,
                                peak_entry_spread, exit_spread_at_close, baseline_median, baseline_mad,
                                activation_threshold, release_threshold, source_version, is_closed
                            )
                            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            [
                                (
                                    storage_pair_id,
                                    int(item["session_id"]),
                                    int(runtime_to_storage_block_id.get(int(item["block_id"]), int(item["block_id"]))),
                                    float(item["start_ts"]),
                                    float(item["peak_ts"]),
                                    float(item["end_ts"]),
                                    float(item["duration_sec"]),
                                    float(item["peak_entry_spread"]),
                                    float(item["exit_spread_at_close"]),
                                    float(item["baseline_median"]),
                                    float(item["baseline_mad"]),
                                    float(item["activation_threshold"]),
                                    float(item["release_threshold"]),
                                    str(item["source_version"]),
                                    1 if bool(item["is_closed"]) else 0,
                                )
                                for item in episodes
                            ],
                        )
                    conn.execute(
                        """
                        DELETE FROM tracker_pair_blocks
                        WHERE pair_id = ?
                          AND id NOT IN (
                            SELECT DISTINCT block_id
                            FROM tracker_records
                            WHERE pair_id = ? AND block_id IS NOT NULL
                          )
                        """,
                        (storage_pair_id, storage_pair_id),
                    )
                    # Recompute persisted block metadata from the full retained record set.
                    # Without this, blocks that straddle the rewrite cutoff keep older records
                    # in SQLite but get start/count rewritten from only the recent snapshot rows.
                    for storage_block_id in sorted(
                        {
                            int(block_id)
                            for block_id in runtime_to_storage_block_id.values()
                            if int(block_id) > 0
                        }
                    ):
                        self._recompute_block_from_db(conn, int(storage_block_id))
                    snapshot_rebindings.append((key, storage_pair_id, runtime_to_storage_block_id))
                self._write_meta(
                    conn,
                    {
                        "schema_version": _SCHEMA_VERSION,
                        "record_interval_sec": self.record_interval_sec,
                        "tracking_window_sec": self.storage_window_sec,
                        "tracker_memory_window_sec": self.memory_window_sec,
                        "gap_threshold_sec": self.gap_threshold_sec,
                        "min_total_spread_pct": self.min_total_spread_pct,
                        "quality_fix_activated_at": self.quality_fix_activated_at,
                        "last_flush_at": float(ts),
                        _HOT_PATH_REJECTION_META_KEY: json.dumps(rejection_stats_payload, sort_keys=True),
                    },
                )
                conn.commit()
                self._last_flush_at = float(ts)
                records_after_row = conn.execute("SELECT COUNT(*) FROM tracker_records").fetchone()
                records_after = int(records_after_row[0] if records_after_row is not None else 0)
            self._run_wal_checkpoint()
        except Exception as exc:
            logger.error("[Tracker] Failed to flush SQLite storage: %s", exc)
            self._emit_event({"kind": "tracker_error", "error": str(exc)})
            return False

        with self._lock:
            for key, storage_pair_id, block_id_map in snapshot_rebindings:
                ps = self._pairs.get(key)
                if ps is None:
                    continue
                ps.storage_pair_id = int(storage_pair_id)
                if block_id_map:
                    for record in ps.records:
                        mapped = block_id_map.get(int(record.block_id), int(record.block_id))
                        record.block_id = int(mapped)
                    for events in (ps.inverted_events, ps.entry_events, ps.exit_events):
                        for event in events:
                            if event.block_id is None:
                                continue
                            event.block_id = int(block_id_map.get(int(event.block_id), int(event.block_id)))
                    for episode in ps.episodes:
                        episode.block_id = int(block_id_map.get(int(episode.block_id), int(episode.block_id)))
                    remapped_block_meta: Dict[int, TrackerBlockMeta] = {}
                    for runtime_block_id, meta in ps.block_meta.items():
                        storage_block_id = int(block_id_map.get(int(runtime_block_id), int(runtime_block_id)))
                        existing = remapped_block_meta.get(storage_block_id)
                        if existing is None:
                            remapped_block_meta[storage_block_id] = meta
                        else:
                            existing.start_ts = min(existing.start_ts, meta.start_ts)
                            existing.end_ts = max(existing.end_ts, meta.end_ts)
                            existing.record_count = max(existing.record_count, meta.record_count)
                            existing.max_gap_sec = max(existing.max_gap_sec, meta.max_gap_sec)
                            existing.is_open = bool(existing.is_open or meta.is_open)
                    ps.block_meta = remapped_block_meta
                    ps.current_block_id = int(block_id_map.get(int(ps.current_block_id), int(ps.current_block_id))) if ps.current_block_id else 0
            for key in dirty_keys:
                ps = self._pairs.get(key)
                if ps is None:
                    continue
                ps._prune_records(memory_cutoff)
            if force:
                self._dirty_pairs.clear()
            else:
                for key in dirty_keys:
                    self._dirty_pairs.discard(key)
            for key in deleted_keys:
                self._deleted_pairs.discard(key)
            self._hot_path_rejection_stats_dirty = False
        self._emit_event(
            {
                "kind": "tracker_flush",
                "duration_ms": (time.perf_counter() - started_at) * 1000.0,
                "pairs_flushed": len(dirty_keys),
                "force": bool(force),
                "records_before": int(records_before),
                "records_after": int(records_after),
                "commit_success": True,
                "storage_stats": self.get_storage_stats(),
            }
        )
        if rejection_stats_dirty or force:
            self._emit_event(
                {
                    "kind": "tracker_rejection_stats",
                    **rejection_stats_payload,
                }
            )
        return True

    def load_from_storage(self, *, now_ts: float | None = None) -> int:
        if self.db_path is None or not self.db_path.is_file():
            return 0
        now = float(now_ts) if now_ts is not None else time.time()
        state_cutoff = self._storage_cutoff(now)
        record_cutoff = self._memory_cutoff(now)
        loaded_pairs: dict[int, tuple[PairKey, PairStats]] = {}
        try:
            with self._connect() as conn:
                meta = {
                    row["key"]: row["value"]
                    for row in conn.execute("SELECT key, value FROM tracker_meta")
                    if row["key"] in _TRACKER_META_KEYS
                }
                self._last_flush_at = float(meta.get("last_flush_at", 0.0) or 0.0)
                if self._last_flush_at <= 0.0:
                    max_record_row = conn.execute("SELECT MAX(ts) FROM tracker_records").fetchone()
                    self._last_flush_at = self._coerce_float(max_record_row[0] if max_record_row is not None else 0.0, 0.0)
                self.quality_fix_activated_at = self._coerce_float(meta.get("quality_fix_activated_at"), self.quality_fix_activated_at)
                meta_with_rejection_stats = {
                    row["key"]: row["value"]
                    for row in conn.execute("SELECT key, value FROM tracker_meta")
                }
                self._load_hot_path_rejection_stats(meta_with_rejection_stats)
                for row in conn.execute(
                    """
                    SELECT id, symbol, buy_ex, buy_mt, sell_ex, sell_mt,
                           last_state, last_seen_ts, last_crossover_ts, history_enabled
                    FROM tracker_pairs
                    """
                ):
                    key = self._pair_key(row["symbol"], row["buy_ex"], row["buy_mt"], row["sell_ex"], row["sell_mt"])
                    ps = PairStats(
                        storage_pair_id=int(row["id"]),
                        last_state=int(row["last_state"]),
                        last_seen_ts=float(row["last_seen_ts"]),
                        last_crossover_ts=float(row["last_crossover_ts"]),
                        history_enabled=bool(row["history_enabled"]),
                    )
                    loaded_pairs[int(row["id"])] = (key, ps)
                for row in conn.execute(
                    """
                    SELECT pair_id, event_type, ts, session_id, block_id
                    FROM tracker_events
                    WHERE ts >= ?
                    ORDER BY ts ASC
                    """,
                    (state_cutoff,),
                ):
                    pair = loaded_pairs.get(int(row["pair_id"]))
                    if pair is None:
                        continue
                    _, ps = pair
                    event = TrackerEvent(
                        timestamp=float(row["ts"]),
                        event_type=str(row["event_type"]),
                        session_id=int(row["session_id"] or 0),
                        block_id=int(row["block_id"]) if row["block_id"] is not None else None,
                    )
                    if event.event_type == "inverted":
                        ps.inverted_events.append(event)
                    elif event.event_type == "entry":
                        ps.entry_events.append(event)
                    else:
                        ps.exit_events.append(event)
                for row in conn.execute(
                    """
                    SELECT pair_id, ts, entry_spread_pct, exit_spread_pct, session_id, block_id
                    FROM tracker_records
                    WHERE ts >= ?
                    ORDER BY ts ASC
                    """,
                    (record_cutoff,),
                ):
                    pair = loaded_pairs.get(int(row["pair_id"]))
                    if pair is None:
                        continue
                    _, ps = pair
                    record = SpreadRecord(
                        timestamp=float(row["ts"]),
                        entry_spread_pct=float(row["entry_spread_pct"]),
                        exit_spread_pct=float(row["exit_spread_pct"]),
                        session_id=int(row["session_id"] or 0),
                        block_id=int(row["block_id"] or 0),
                    )
                    ps.records.append(record)
                    ps._last_record_ts = max(ps._last_record_ts, record.timestamp)
                    ps._stats_dirty = True
                for row in conn.execute(
                    """
                    SELECT pair_id, start_ts, peak_ts, end_ts, duration_sec, peak_entry_spread,
                           exit_spread_at_close, baseline_median, baseline_mad, activation_threshold,
                           release_threshold, session_id, block_id, source_version, is_closed
                    FROM tracker_pair_episodes
                    WHERE end_ts >= ?
                    ORDER BY end_ts ASC, start_ts ASC
                    """,
                    (state_cutoff,),
                ):
                    pair = loaded_pairs.get(int(row["pair_id"]))
                    if pair is None:
                        continue
                    _, ps = pair
                    ps.episodes.append(
                        TrackerEpisode(
                            start_ts=float(row["start_ts"]),
                            peak_ts=float(row["peak_ts"]),
                            end_ts=float(row["end_ts"]),
                            duration_sec=float(row["duration_sec"]),
                            peak_entry_spread=float(row["peak_entry_spread"]),
                            exit_spread_at_close=float(row["exit_spread_at_close"]),
                            baseline_median=float(row["baseline_median"]),
                            baseline_mad=float(row["baseline_mad"]),
                            activation_threshold=float(row["activation_threshold"]),
                            release_threshold=float(row["release_threshold"]),
                            session_id=int(row["session_id"] or 0),
                            block_id=int(row["block_id"] or 0),
                            source_version=str(row["source_version"] or _EPISODE_SOURCE_VERSION),
                            is_closed=bool(row["is_closed"]),
                        )
                    )
                    ps._episodes_cache_valid = True
        except Exception as exc:
            logger.warning("[Tracker] Could not read SQLite state: %s", exc)
            return 0

        with self._lock:
            self._pairs = defaultdict(PairStats)
            count = 0
            for _, (key, ps) in loaded_pairs.items():
                if (ps.last_seen_ts or 0.0) < state_cutoff and not ps.records and not ps.inverted_events and not ps.entry_events and not ps.exit_events:
                    continue
                self._pairs[key] = ps
                count += 1
            self._dirty_pairs.clear()
            self._deleted_pairs.clear()
        logger.info("[Tracker] Restored %s pairs from %s", count, self.db_path.name if self.db_path else "storage")
        return count

    def get_training_config(self) -> Dict[str, object]:
        return {
            "gap_threshold_sec": self.gap_threshold_sec,
            "min_total_spread_pct": self.min_total_spread_pct,
            "record_interval_sec": self.record_interval_sec,
            "window_sec": self.storage_window_sec,
            "storage_window_sec": self.storage_window_sec,
            "tracker_memory_window_sec": self.memory_window_sec,
            "active_session_id": self._active_session_id,
            "quality_fix_activated_at": self.quality_fix_activated_at,
            "hot_path_rejection_stats": self._hot_path_rejection_snapshot_locked(),
        }

    @staticmethod
    def _float_close(left: float | None, right: float | None, *, tolerance: float = 1e-6) -> bool:
        if left is None or right is None:
            return left is None and right is None
        return abs(float(left) - float(right)) <= float(tolerance)

    def _collect_training_quality_state(
        self,
        conn: sqlite3.Connection,
        *,
        sequence_length: int = _DEFAULT_CLEAN_SEQUENCE_LENGTH,
        prediction_horizon_sec: int = _DEFAULT_CLEAN_PREDICTION_HORIZON_SEC,
    ) -> dict[str, Any]:
        session_rows = list(
            conn.execute(
                """
                SELECT id, started_at, ended_at, status, record_interval_sec, tracking_window_sec, gap_threshold_sec,
                       approved_for_training, excluded_reason, manual_override, notes
                FROM tracker_capture_sessions
                ORDER BY id ASC
                """
            )
        )
        session_meta = {int(row["id"]): row for row in session_rows}
        block_rows = list(
            conn.execute(
                """
                SELECT id, pair_id, session_id, start_ts, end_ts, record_count, max_gap_sec, boundary_reason,
                       selected_for_training, disabled_reason, manual_override, notes, is_open
                FROM tracker_pair_blocks
                ORDER BY session_id ASC, id ASC
                """
            )
        )
        actual_by_block: dict[int, dict[str, float]] = {}
        for row in conn.execute(
            """
            SELECT
                block_id,
                COUNT(*) AS actual_count,
                MIN(ts) AS start_ts,
                MAX(ts) AS end_ts
            FROM tracker_records
            WHERE block_id IS NOT NULL
            GROUP BY block_id
            """
        ):
            block_id = int(row["block_id"] or 0)
            if block_id <= 0:
                continue
            actual_by_block[block_id] = {
                "count": float(row["actual_count"]),
                "start_ts": float(row["start_ts"]),
                "end_ts": float(row["end_ts"]),
                "max_gap_sec": 0.0,
            }
        # Compute actual max_gap per block inside SQLite instead of
        # materializing all tracker_records rows in Python.
        for row in conn.execute(
            """
            SELECT block_id, MAX(gap) AS max_gap
            FROM (
                SELECT
                    block_id,
                    ts - LAG(ts) OVER (PARTITION BY block_id ORDER BY ts) AS gap
                FROM tracker_records
                WHERE block_id IS NOT NULL
            )
            WHERE gap IS NOT NULL
            GROUP BY block_id
            """
        ):
            block_id = int(row["block_id"] or 0)
            if block_id in actual_by_block:
                actual_by_block[block_id]["max_gap_sec"] = float(row["max_gap"] or 0.0)

        blocks_by_session: dict[int, list[dict[str, Any]]] = defaultdict(list)
        block_lookup: dict[int, dict[str, Any]] = {}
        block_ranges_by_session_pair: dict[tuple[int, int], list[tuple[float, float, int]]] = defaultdict(list)
        for row in block_rows:
            session_id = int(row["session_id"] or 0)
            block_id = int(row["id"])
            actual = actual_by_block.get(block_id, {})
            actual_count = int(actual.get("count", 0.0))
            actual_start_ts = float(actual.get("start_ts", row["start_ts"])) if actual_count else None
            actual_end_ts = float(actual.get("end_ts", row["end_ts"])) if actual_count else None
            actual_max_gap_sec = float(actual.get("max_gap_sec", row["max_gap_sec"])) if actual_count else 0.0
            zero_record_block = actual_count <= 0 or int(row["record_count"]) <= 0
            count_mismatch = actual_count != int(row["record_count"])
            range_mismatch = actual_count > 0 and (
                not self._float_close(actual_start_ts, float(row["start_ts"]))
                or not self._float_close(actual_end_ts, float(row["end_ts"]))
            )
            block_payload = {
                "id": block_id,
                "pair_id": int(row["pair_id"]),
                "session_id": session_id,
                "start_ts": float(row["start_ts"]),
                "end_ts": float(row["end_ts"]),
                "record_count": int(row["record_count"]),
                "max_gap_sec": float(row["max_gap_sec"]),
                "boundary_reason": str(row["boundary_reason"]),
                "selected_for_training": bool(row["selected_for_training"]),
                "disabled_reason": str(row["disabled_reason"] or ""),
                "manual_override": bool(row["manual_override"]),
                "notes": str(row["notes"] or ""),
                "is_open": bool(row["is_open"]),
                "actual_record_count": actual_count,
                "actual_start_ts": actual_start_ts,
                "actual_end_ts": actual_end_ts,
                "actual_max_gap_sec": actual_max_gap_sec,
                "zero_record_block": zero_record_block,
                "record_count_mismatch": count_mismatch,
                "range_mismatch": range_mismatch,
            }
            block_payload["critical_anomaly_count"] = int(
                zero_record_block
                + count_mismatch
                + range_mismatch
            )
            blocks_by_session[session_id].append(block_payload)
            block_lookup[block_id] = block_payload
            range_start = actual_start_ts if actual_start_ts is not None else float(row["start_ts"])
            range_end = actual_end_ts if actual_end_ts is not None else float(row["end_ts"])
            block_ranges_by_session_pair[(session_id, int(row["pair_id"]))].append((range_start, range_end, block_id))

        for ranges in block_ranges_by_session_pair.values():
            ranges.sort(key=lambda item: (float(item[0]), float(item[1]), int(item[2])))

        missing_event_blocks_by_session: Counter[int] = Counter()
        event_rows = list(
            conn.execute(
                """
                SELECT pair_id, session_id, block_id, ts
                FROM tracker_events
                ORDER BY session_id ASC, ts ASC
                """
            )
        )
        for row in event_rows:
            session_id = int(row["session_id"] or 0)
            pair_id = int(row["pair_id"] or 0)
            block_id = int(row["block_id"] or 0) if row["block_id"] is not None else 0
            ts_value = float(row["ts"])
            block_row = block_lookup.get(block_id)
            block_matches = (
                block_row is not None
                and int(block_row["pair_id"]) == pair_id
                and int(block_row["session_id"]) == session_id
                and float(block_row.get("actual_start_ts") or block_row["start_ts"]) <= ts_value <= float(block_row.get("actual_end_ts") or block_row["end_ts"])
            )
            if block_matches:
                continue
            matched = False
            for start_ts, end_ts, candidate_block_id in block_ranges_by_session_pair.get((session_id, pair_id), []):
                if float(start_ts) <= ts_value <= float(end_ts):
                    matched = True
                    break
            if not matched:
                missing_event_blocks_by_session[session_id] += 1

        sessions: dict[int, dict[str, Any]] = {}
        summary = {
            "critical_sessions": 0,
            "training_ready_sessions": 0,
            "burn_in_passed_sessions": 0,
            "checkpoint_ready_sessions": 0,
            "zero_record_blocks": 0,
            "record_count_mismatches": 0,
            "range_mismatches": 0,
            "missing_event_blocks": 0,
        }
        for session_id, row in session_meta.items():
            session_blocks = blocks_by_session.get(session_id, [])
            data_start_ts = min(
                (
                    float(block.get("actual_start_ts"))
                    for block in session_blocks
                    if block.get("actual_start_ts") is not None
                ),
                default=0.0,
            )
            data_end_ts = max(
                (
                    float(block.get("actual_end_ts"))
                    for block in session_blocks
                    if block.get("actual_end_ts") is not None
                ),
                default=0.0,
            )
            session_span_sec = max(0.0, float(data_end_ts) - float(data_start_ts)) if data_end_ts > 0.0 else 0.0
            zero_record_blocks = sum(1 for block in session_blocks if bool(block["zero_record_block"]))
            count_mismatches = sum(1 for block in session_blocks if bool(block["record_count_mismatch"]))
            range_mismatches = sum(1 for block in session_blocks if bool(block["range_mismatch"]))
            open_blocks_after_close = sum(1 for block in session_blocks if bool(block["is_open"]) and str(row["status"]) != "open")
            critical_anomaly_count = (
                int(zero_record_blocks)
                + int(count_mismatches)
                + int(range_mismatches)
                + int(open_blocks_after_close)
                + int(missing_event_blocks_by_session.get(session_id, 0))
            )
            checkpoint_ready = critical_anomaly_count == 0 and session_span_sec >= float(_BURN_IN_CHECKPOINT_SEC) and str(row["status"]) != "open"
            burn_in_passed = critical_anomaly_count == 0 and session_span_sec >= float(_BURN_IN_MIN_SEC) and str(row["status"]) != "open"
            training_ready = burn_in_passed and session_span_sec >= self._required_session_span_sec(
                sequence_length=int(sequence_length),
                prediction_horizon_sec=int(prediction_horizon_sec),
                record_interval_sec=float(row["record_interval_sec"]),
            )
            trainable_block_count = sum(
                1
                for block in session_blocks
                if self._block_is_trainable(
                    block,
                    record_interval_sec=float(row["record_interval_sec"]),
                    sequence_length=2,
                )
            )
            selected_block_count = sum(1 for block in session_blocks if bool(block["selected_for_training"]))
            exception_block_count = sum(1 for block in session_blocks if self._block_is_exception(block))
            quality_status = self._session_quality_status(
                status=str(row["status"]),
                critical_anomaly_count=critical_anomaly_count,
                session_span_sec=session_span_sec,
                record_interval_sec=float(row["record_interval_sec"]),
                sequence_length=int(sequence_length),
                prediction_horizon_sec=int(prediction_horizon_sec),
            )
            session_payload = {
                "session_id": session_id,
                "data_start_ts": float(data_start_ts),
                "data_end_ts": float(data_end_ts),
                "session_span_sec": float(session_span_sec),
                "zero_record_blocks": int(zero_record_blocks),
                "record_count_mismatches": int(count_mismatches),
                "range_mismatches": int(range_mismatches),
                "missing_event_blocks": int(missing_event_blocks_by_session.get(session_id, 0)),
                "open_blocks_after_close": int(open_blocks_after_close),
                "critical_anomaly_count": int(critical_anomaly_count),
                "checkpoint_ready": bool(checkpoint_ready),
                "burn_in_passed": bool(burn_in_passed),
                "training_ready": bool(training_ready),
                "quality_status": quality_status,
                "selected_block_count_strict": int(selected_block_count),
                "trainable_block_count_strict": int(trainable_block_count),
                "exception_block_count_strict": int(exception_block_count),
            }
            sessions[session_id] = session_payload
            summary["critical_sessions"] += int(critical_anomaly_count > 0)
            summary["training_ready_sessions"] += int(training_ready)
            summary["burn_in_passed_sessions"] += int(burn_in_passed)
            summary["checkpoint_ready_sessions"] += int(checkpoint_ready)
            summary["zero_record_blocks"] += int(zero_record_blocks)
            summary["record_count_mismatches"] += int(count_mismatches)
            summary["range_mismatches"] += int(range_mismatches)
            summary["missing_event_blocks"] += int(missing_event_blocks_by_session.get(session_id, 0))
        return {
            "sessions": sessions,
            "blocks": block_lookup,
            "summary": summary,
        }

    def update_gap_threshold_sec(self, gap_threshold_sec: float) -> float:
        self.gap_threshold_sec = self._resolve_gap_threshold_sec(gap_threshold_sec, self.record_interval_sec)
        if self.db_path is not None:
            with self._connect() as conn:
                self._write_meta(conn, {"gap_threshold_sec": self.gap_threshold_sec})
                if self._active_session_id:
                    conn.execute(
                        """
                        UPDATE tracker_capture_sessions
                        SET gap_threshold_sec = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (self.gap_threshold_sec, time.time(), self._active_session_id),
                    )
        return self.gap_threshold_sec

    def update_min_total_spread_pct(self, min_total_spread_pct: float) -> float:
        self.min_total_spread_pct = self._resolve_min_total_spread_pct(min_total_spread_pct)
        if self.db_path is not None:
            with self._connect() as conn:
                self._write_meta(conn, {"min_total_spread_pct": self.min_total_spread_pct})
        return self.min_total_spread_pct

    @staticmethod
    def _training_exception_sql(alias: str = "b") -> str:
        return (
            f"({alias}.is_open = 1 OR {alias}.selected_for_training = 0 OR {alias}.manual_override = 1 "
            f"OR COALESCE({alias}.disabled_reason, '') != '' OR {alias}.record_count <= 1 "
            f"OR {alias}.boundary_reason IN ('auto_gap', 'manual_split', 'manual_merge'))"
        )

    @staticmethod
    def _block_is_exception(block: dict[str, object]) -> bool:
        return bool(
            block.get("is_open")
            or not block.get("selected_for_training", True)
            or block.get("manual_override")
            or str(block.get("disabled_reason") or "").strip()
            or int(block.get("record_count") or 0) <= 1
            or str(block.get("boundary_reason") or "") in {"auto_gap", "manual_split", "manual_merge"}
        )

    @staticmethod
    def _block_exception_reasons(block: dict[str, object]) -> list[str]:
        reasons: list[str] = []
        if bool(block.get("is_open")):
            reasons.append("open_block")
        if (not bool(block.get("selected_for_training", True))) or str(block.get("disabled_reason") or "").strip():
            reasons.append("block_excluded")
        if bool(block.get("manual_override")):
            reasons.append("manual_override")
        boundary_reason = str(block.get("boundary_reason") or "")
        if boundary_reason in {"auto_gap", "manual_split", "manual_merge"}:
            reasons.append(boundary_reason)
        if int(block.get("record_count") or 0) <= 1:
            reasons.append("too_short")
        # Preserve deterministic order while deduplicating.
        return list(dict.fromkeys(reasons))

    @staticmethod
    def _block_is_trainable(
        block: dict[str, object],
        *,
        record_interval_sec: float,
        sequence_length: int = 2,
    ) -> bool:
        if SpreadTracker._block_is_exception(block):
            return False
        if int(block.get("record_count") or 0) < max(int(sequence_length), 2):
            return False
        max_gap_sec = float(block.get("actual_max_gap_sec", block.get("max_gap_sec", 0.0)) or 0.0)
        return max_gap_sec <= (2.0 * max(float(record_interval_sec), 0.0))

    @staticmethod
    def _required_session_span_sec(*, sequence_length: int, prediction_horizon_sec: int, record_interval_sec: float) -> float:
        return max(
            float(prediction_horizon_sec),
            float(prediction_horizon_sec) + max(int(sequence_length) - 1, 0) * max(float(record_interval_sec), 0.0),
        )

    @classmethod
    def _session_quality_status(
        cls,
        *,
        status: str,
        critical_anomaly_count: int,
        session_span_sec: float,
        record_interval_sec: float,
        sequence_length: int,
        prediction_horizon_sec: int,
    ) -> str:
        if status == "open":
            return "open"
        if critical_anomaly_count > 0:
            return "quarantined"
        if session_span_sec < float(_BURN_IN_CHECKPOINT_SEC):
            return "collecting"
        if session_span_sec < float(_BURN_IN_MIN_SEC):
            return "checkpoint_ready"
        if session_span_sec >= cls._required_session_span_sec(
            sequence_length=sequence_length,
            prediction_horizon_sec=prediction_horizon_sec,
            record_interval_sec=record_interval_sec,
        ):
            return "training_ready"
        return "burn_in_passed"

    @classmethod
    def _session_meets_run_gates(
        cls,
        *,
        status: str,
        critical_anomaly_count: int,
        session_span_sec: float,
        record_interval_sec: float,
        sequence_length: int,
        prediction_horizon_sec: int,
    ) -> bool:
        if str(status) == "open":
            return False
        if int(critical_anomaly_count) > 0:
            return False
        return float(session_span_sec) >= cls._required_session_span_sec(
            sequence_length=int(sequence_length),
            prediction_horizon_sec=int(prediction_horizon_sec),
            record_interval_sec=float(record_interval_sec),
        )

    @staticmethod
    def _session_review_state(*, status: str, approved: bool, exception_count: int, manual_override: bool) -> str:
        if status == "open":
            return "open"
        if not approved:
            return "excluded"
        if exception_count > 0 and not manual_override:
            return "needs_review"
        if exception_count > 0 and manual_override:
            return "reviewed_ok"
        return "auto_approved"

    def list_training_sessions(
        self,
        *,
        include_open: bool = True,
        include_blocks_preview: bool = True,
        summary_only: bool = False,
        _quality_state: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        if self.db_path is None or not self.db_path.is_file():
            return {
                "config": self.get_training_config(),
                "sessions": [],
                "summary": {
                    "total_sessions": 0,
                    "approved_sessions": 0,
                    "trainable_sessions": 0,
                    "exception_sessions": 0,
                },
            }
        with self._connect() as conn:
            quality_state = _quality_state if _quality_state is not None else self._collect_training_quality_state(conn)
            if summary_only:
                block_session_ids = {int(block.get("session_id", 0) or 0) for block in quality_state["blocks"].values()}
                minimal_rows = list(
                    conn.execute(
                        """
                        SELECT id, status, approved_for_training, excluded_reason, manual_override, record_interval_sec
                        FROM tracker_capture_sessions
                        ORDER BY id DESC
                        """
                    )
                )
                approved_sessions = 0
                trainable_sessions = 0
                exception_sessions = 0
                auto_approved_sessions = 0
                review_state_counts: Counter[str] = Counter()
                total_sessions = 0
                for row in minimal_rows:
                    session_id = int(row["id"])
                    if session_id not in block_session_ids:
                        continue
                    status = str(row["status"])
                    if (not include_open) and status == "open":
                        continue
                    session_quality = quality_state["sessions"].get(session_id, {})
                    approved = bool(row["approved_for_training"])
                    exception_count = int(session_quality.get("critical_anomaly_count", 0))
                    manual_override = bool(row["manual_override"])
                    review_state = self._session_review_state(
                        status=status,
                        approved=approved,
                        exception_count=exception_count,
                        manual_override=manual_override,
                    )
                    can_train = (
                        review_state in {"auto_approved", "reviewed_ok"}
                        and bool(session_quality.get("training_ready", False))
                        and int(session_quality.get("trainable_block_count_strict", 0)) > 0
                    )
                    total_sessions += 1
                    if approved:
                        approved_sessions += 1
                    if can_train:
                        trainable_sessions += 1
                    if exception_count > 0:
                        exception_sessions += 1
                    if review_state == "auto_approved":
                        auto_approved_sessions += 1
                    review_state_counts[review_state] += 1
                return {
                    "config": self.get_training_config(),
                    "sessions": [],
                    "summary": {
                        "total_sessions": total_sessions,
                        "approved_sessions": approved_sessions,
                        "trainable_sessions": trainable_sessions,
                        "exception_sessions": exception_sessions,
                        "auto_approved_sessions": auto_approved_sessions,
                        "training_ready_sessions": int(quality_state["summary"].get("training_ready_sessions", 0)),
                        "burn_in_passed_sessions": int(quality_state["summary"].get("burn_in_passed_sessions", 0)),
                        "checkpoint_ready_sessions": int(quality_state["summary"].get("checkpoint_ready_sessions", 0)),
                        "critical_sessions": int(quality_state["summary"].get("critical_sessions", 0)),
                        "zero_record_blocks": int(quality_state["summary"].get("zero_record_blocks", 0)),
                        "record_count_mismatches": int(quality_state["summary"].get("record_count_mismatches", 0)),
                        "range_mismatches": int(quality_state["summary"].get("range_mismatches", 0)),
                        "missing_event_blocks": int(quality_state["summary"].get("missing_event_blocks", 0)),
                        "review_state_counts": dict(sorted(review_state_counts.items())),
                    },
                }
            session_filter = "" if include_open else "WHERE s.status != 'open'"
            session_rows = list(
                conn.execute(
                    f"""
                    SELECT
                        s.id,
                        s.started_at,
                        s.ended_at,
                        s.status,
                        s.record_interval_sec,
                        s.tracking_window_sec,
                        s.gap_threshold_sec,
                        s.approved_for_training,
                        s.excluded_reason,
                        s.manual_override,
                        s.notes,
                        COALESCE(COUNT(b.id), 0) AS block_count,
                        COALESCE(COUNT(DISTINCT b.pair_id), 0) AS pair_count,
                        COALESCE(COUNT(DISTINCT p.symbol), 0) AS symbol_count,
                        COALESCE(SUM(CASE WHEN b.selected_for_training = 1 THEN 1 ELSE 0 END), 0) AS selected_block_count,
                        COALESCE(SUM(CASE WHEN {self._training_exception_sql('b')} THEN 1 ELSE 0 END), 0) AS exception_block_count
                    FROM tracker_capture_sessions s
                    LEFT JOIN tracker_pair_blocks b ON b.session_id = s.id
                    LEFT JOIN tracker_pairs p ON p.id = b.pair_id
                    {session_filter}
                    GROUP BY s.id
                    HAVING COUNT(b.id) > 0
                    ORDER BY COALESCE(MAX(b.end_ts), s.ended_at, s.started_at) DESC, s.id DESC
                    """
                )
            )
            preview_rows = []
            if include_blocks_preview:
                preview_rows = list(
                    conn.execute(
                        """
                        SELECT
                            b.id,
                            b.session_id,
                            b.start_ts,
                            b.end_ts,
                            b.record_count,
                            b.max_gap_sec,
                            b.boundary_reason,
                            b.selected_for_training,
                            b.disabled_reason,
                            b.manual_override,
                            b.notes,
                            b.is_open,
                            p.symbol,
                            p.buy_ex,
                            p.buy_mt,
                            p.sell_ex,
                            p.sell_mt
                        FROM tracker_pair_blocks b
                        JOIN tracker_pairs p ON p.id = b.pair_id
                        ORDER BY b.session_id DESC, b.start_ts ASC, b.id ASC
                        """
                    )
                )

        preview_by_session: dict[int, list[dict[str, object]]] = defaultdict(list)
        exception_reason_counts_by_session: dict[int, Counter[str]] = defaultdict(Counter)
        for row in preview_rows:
            session_id = int(row["session_id"])
            block_quality = quality_state["blocks"].get(int(row["id"]), {})
            block_payload = {
                "id": int(row["id"]),
                "symbol": str(row["symbol"]),
                "pair_key": f"{row['symbol']}|{row['buy_ex']}|{row['buy_mt']}|{row['sell_ex']}|{row['sell_mt']}",
                "start_ts": float(row["start_ts"]),
                "end_ts": float(row["end_ts"]),
                "record_count": int(row["record_count"]),
                "max_gap_sec": float(row["max_gap_sec"]),
                "boundary_reason": str(row["boundary_reason"]),
                "selected_for_training": bool(row["selected_for_training"]),
                "disabled_reason": str(row["disabled_reason"] or ""),
                "manual_override": bool(row["manual_override"]),
                "notes": str(row["notes"] or ""),
                "is_open": bool(row["is_open"]),
                "actual_record_count": int(block_quality.get("actual_record_count", 0)),
                "actual_max_gap_sec": float(block_quality.get("actual_max_gap_sec", row["max_gap_sec"]) or 0.0),
                "zero_record_block": bool(block_quality.get("zero_record_block", False)),
                "record_count_mismatch": bool(block_quality.get("record_count_mismatch", False)),
                "range_mismatch": bool(block_quality.get("range_mismatch", False)),
            }
            for reason in self._block_exception_reasons(block_payload):
                exception_reason_counts_by_session[session_id][reason] += 1
            if len(preview_by_session[session_id]) >= 5:
                continue
            preview_by_session[session_id].append(block_payload)

        sessions: list[dict[str, object]] = []
        approved_sessions = 0
        trainable_sessions = 0
        exception_sessions = 0
        auto_approved_sessions = 0
        review_state_counts: Counter[str] = Counter()
        for row in session_rows:
            session_quality = quality_state["sessions"].get(int(row["id"]), {})
            approved = bool(row["approved_for_training"])
            exception_count = int(row["exception_block_count"]) + int(session_quality.get("critical_anomaly_count", 0))
            manual_override = bool(row["manual_override"])
            review_state = self._session_review_state(
                status=str(row["status"]),
                approved=approved,
                exception_count=exception_count,
                manual_override=manual_override,
            )
            can_train = (
                review_state in {"auto_approved", "reviewed_ok"}
                and bool(session_quality.get("training_ready", False))
                and int(session_quality.get("trainable_block_count_strict", 0)) > 0
            )
            reason_counter = Counter(exception_reason_counts_by_session.get(int(row["id"]), Counter()))
            if int(session_quality.get("zero_record_blocks", 0)) > 0:
                reason_counter["zero_record_block"] += int(session_quality.get("zero_record_blocks", 0))
            if int(session_quality.get("record_count_mismatches", 0)) > 0:
                reason_counter["record_count_mismatch"] += int(session_quality.get("record_count_mismatches", 0))
            if int(session_quality.get("range_mismatches", 0)) > 0:
                reason_counter["range_mismatch"] += int(session_quality.get("range_mismatches", 0))
            if int(session_quality.get("missing_event_blocks", 0)) > 0:
                reason_counter["missing_event_block"] += int(session_quality.get("missing_event_blocks", 0))
            reason_counts = dict(sorted(reason_counter.items()))
            dominant_reason = ""
            if reason_counts:
                dominant_reason = max(reason_counts.items(), key=lambda item: (item[1], item[0]))[0]
            if approved:
                approved_sessions += 1
            if can_train:
                trainable_sessions += 1
            if exception_count > 0:
                exception_sessions += 1
            if review_state == "auto_approved":
                auto_approved_sessions += 1
            review_state_counts[review_state] += 1
            sessions.append(
                {
                    "id": int(row["id"]),
                    "started_at": float(row["started_at"]),
                    "ended_at": self._coerce_float(row["ended_at"], 0.0),
                    "status": str(row["status"]),
                    "record_interval_sec": float(row["record_interval_sec"]),
                    "tracking_window_sec": float(row["tracking_window_sec"]),
                    "gap_threshold_sec": float(row["gap_threshold_sec"]),
                    "approved_for_training": approved,
                    "excluded_reason": str(row["excluded_reason"] or ""),
                    "manual_override": manual_override,
                    "notes": str(row["notes"] or ""),
                    "data_start_ts": float(session_quality.get("data_start_ts", 0.0) or 0.0),
                    "data_end_ts": float(session_quality.get("data_end_ts", 0.0) or 0.0),
                    "capture_duration_sec": max(
                        0.0,
                        self._coerce_float(row["ended_at"], 0.0) - float(row["started_at"]),
                    ),
                    "data_duration_sec": float(session_quality.get("session_span_sec", 0.0) or 0.0),
                    "block_count": int(row["block_count"]),
                    "pair_count": int(row["pair_count"]),
                    "symbol_count": int(row["symbol_count"]),
                    "selected_block_count": int(row["selected_block_count"]),
                    "trainable_block_count": int(session_quality.get("trainable_block_count_strict", 0)),
                    "exception_block_count": exception_count,
                    "can_train": can_train,
                    "review_state": review_state,
                    "dominant_exception_reason": dominant_reason,
                    "exception_reason_counts": reason_counts,
                    "quality_status": str(session_quality.get("quality_status", "collecting")),
                    "checkpoint_ready": bool(session_quality.get("checkpoint_ready", False)),
                    "burn_in_passed": bool(session_quality.get("burn_in_passed", False)),
                    "training_ready": bool(session_quality.get("training_ready", False)),
                    "session_span_sec": float(session_quality.get("session_span_sec", 0.0) or 0.0),
                    "critical_anomaly_count": int(session_quality.get("critical_anomaly_count", 0)),
                    "zero_record_blocks": int(session_quality.get("zero_record_blocks", 0)),
                    "record_count_mismatches": int(session_quality.get("record_count_mismatches", 0)),
                    "range_mismatches": int(session_quality.get("range_mismatches", 0)),
                    "missing_event_blocks": int(session_quality.get("missing_event_blocks", 0)),
                    "open_blocks_after_close": int(session_quality.get("open_blocks_after_close", 0)),
                    "blocks_preview": preview_by_session.get(int(row["id"]), []),
                }
            )

        return {
            "config": self.get_training_config(),
            "sessions": sessions,
            "summary": {
                "total_sessions": len(sessions),
                "approved_sessions": approved_sessions,
                "trainable_sessions": trainable_sessions,
                "exception_sessions": exception_sessions,
                "auto_approved_sessions": auto_approved_sessions,
                "training_ready_sessions": int(quality_state["summary"].get("training_ready_sessions", 0)),
                "burn_in_passed_sessions": int(quality_state["summary"].get("burn_in_passed_sessions", 0)),
                "checkpoint_ready_sessions": int(quality_state["summary"].get("checkpoint_ready_sessions", 0)),
                "critical_sessions": int(quality_state["summary"].get("critical_sessions", 0)),
                "zero_record_blocks": int(quality_state["summary"].get("zero_record_blocks", 0)),
                "record_count_mismatches": int(quality_state["summary"].get("record_count_mismatches", 0)),
                "range_mismatches": int(quality_state["summary"].get("range_mismatches", 0)),
                "missing_event_blocks": int(quality_state["summary"].get("missing_event_blocks", 0)),
                "review_state_counts": dict(sorted(review_state_counts.items())),
            },
        }

    def get_training_session(self, session_id: int) -> dict[str, object]:
        listing = self.list_training_sessions(include_open=True)
        for session in listing["sessions"]:
            if int(session["id"]) == int(session_id):
                return session
        raise KeyError(f"Session {session_id} not found")

    def update_training_session(
        self,
        session_id: int,
        *,
        approved_for_training: Optional[bool] = None,
        excluded_reason: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict[str, object]:
        if self.db_path is None:
            raise RuntimeError("Tracker storage is not configured")
        updates = []
        params: list[Any] = []
        if approved_for_training is not None:
            updates.append("approved_for_training = ?")
            params.append(1 if approved_for_training else 0)
            if excluded_reason is None:
                excluded_reason = "" if approved_for_training else "manual_exclusion"
        if excluded_reason is not None:
            updates.append("excluded_reason = ?")
            params.append(str(excluded_reason))
        if notes is not None:
            updates.append("notes = ?")
            params.append(str(notes))
        if approved_for_training is not None or excluded_reason is not None or notes is not None:
            updates.append("manual_override = 1")
        if not updates:
            return self.get_training_session(session_id)
        updates.append("updated_at = ?")
        params.append(time.time())
        params.append(int(session_id))
        with self._connect() as conn:
            cursor = conn.execute(f"UPDATE tracker_capture_sessions SET {', '.join(updates)} WHERE id = ?", params)
            if cursor.rowcount <= 0:
                raise KeyError(f"Session {session_id} not found")
        return self.get_training_session(session_id)

    def get_session_exception_blocks(self, session_id: int) -> dict[str, object]:
        session = self.get_training_session(session_id)
        listing = self.list_training_blocks(session_id=int(session_id), include_open=True)
        blocks = listing["sessions"][0]["blocks"] if listing["sessions"] else []
        exception_blocks = [block for block in blocks if self._block_is_exception(block)]
        return {
            "session_id": int(session_id),
            "session": session,
            "blocks": exception_blocks,
            "summary": {
                "exception_block_count": len(exception_blocks),
                "trainable_block_count": int(session["trainable_block_count"]),
            },
        }

    def get_approved_training_session_ids(
        self,
        *,
        include_open: bool = False,
        sequence_length: int = _DEFAULT_CLEAN_SEQUENCE_LENGTH,
        prediction_horizon_sec: int = _DEFAULT_CLEAN_PREDICTION_HORIZON_SEC,
    ) -> list[int]:
        if self.db_path is None or not self.db_path.is_file():
            return []
        allowed_status = {"auto_approved", "reviewed_ok"}
        listing = self.list_training_sessions(include_open=include_open)
        return [
            int(session["id"])
            for session in sorted(
                listing["sessions"],
                key=lambda session: (float(session.get("data_end_ts") or 0.0), int(session["id"])),
            )
            if session["approved_for_training"]
            and session["review_state"] in allowed_status
            and self._session_meets_run_gates(
                status=str(session["status"]),
                critical_anomaly_count=int(session.get("critical_anomaly_count", 0)),
                session_span_sec=float(session.get("session_span_sec", 0.0) or 0.0),
                record_interval_sec=float(session.get("record_interval_sec", self.record_interval_sec) or self.record_interval_sec),
                sequence_length=int(sequence_length),
                prediction_horizon_sec=int(prediction_horizon_sec),
            )
            and int(session["trainable_block_count"]) > 0
        ]

    def preview_training_cohorts(
        self,
        *,
        session_ids: Optional[list[int]] = None,
        sequence_length: int = _DEFAULT_CLEAN_SEQUENCE_LENGTH,
        prediction_horizon_sec: int = _DEFAULT_CLEAN_PREDICTION_HORIZON_SEC,
        summary_only: bool = False,
    ) -> dict[str, object]:
        listing = self.list_training_sessions(include_open=False, include_blocks_preview=not summary_only)
        allowed = {int(session_id) for session_id in session_ids} if session_ids is not None else None
        sessions = [
            session
            for session in listing["sessions"]
            if session["review_state"] in {"auto_approved", "reviewed_ok"}
            and session["approved_for_training"]
            and session["status"] != "open"
            and session["trainable_block_count"] > 0
            and self._session_meets_run_gates(
                status=str(session["status"]),
                critical_anomaly_count=int(session.get("critical_anomaly_count", 0)),
                session_span_sec=float(session.get("session_span_sec", 0.0) or 0.0),
                record_interval_sec=float(session.get("record_interval_sec", self.record_interval_sec) or self.record_interval_sec),
                sequence_length=int(sequence_length),
                prediction_horizon_sec=int(prediction_horizon_sec),
            )
            and (allowed is None or int(session["id"]) in allowed)
        ]
        sessions.sort(key=lambda session: (float(session["data_end_ts"] or 0.0), int(session["id"])))
        folds: list[dict[str, object]] = []
        if len(sessions) >= 3:
            for test_index in range(2, len(sessions)):
                train_sessions = sessions[: test_index - 1]
                validation_session = sessions[test_index - 1]
                test_session = sessions[test_index]
                if not train_sessions:
                    continue
                folds.append(
                    {
                        "fold": len(folds) + 1,
                        "train_session_ids": [int(session["id"]) for session in train_sessions],
                        "validation_session_ids": [int(validation_session["id"])],
                        "test_session_ids": [int(test_session["id"])],
                        "train_data_end_ts": max(float(session["data_end_ts"] or 0.0) for session in train_sessions),
                        "validation_data_end_ts": float(validation_session["data_end_ts"] or 0.0),
                        "test_data_end_ts": float(test_session["data_end_ts"] or 0.0),
                    }
                )
        return {
            "mode": "expanding_walk_forward",
            "sessions": [] if summary_only else sessions,
            "folds": [] if summary_only else folds,
            "summary": {
                "eligible_sessions": len(sessions),
                "eligible_blocks": sum(int(session["trainable_block_count"]) for session in sessions),
                "fold_count": len(folds),
                "coverage_start_ts": min((float(session["data_start_ts"] or 0.0) for session in sessions), default=0.0),
                "coverage_end_ts": max((float(session["data_end_ts"] or 0.0) for session in sessions), default=0.0),
            },
        }

    def list_training_blocks(
        self,
        *,
        session_id: int | None = None,
        include_open: bool = True,
        include_block_details: bool = True,
    ) -> dict[str, object]:
        if self.db_path is None or not self.db_path.is_file():
            return {"config": self.get_training_config(), "sessions": [], "summary": {"total_sessions": 0, "total_blocks": 0}}
        with self._connect() as conn:
            quality_state = self._collect_training_quality_state(conn)
            params: list[Any] = []
            session_filter = ""
            if session_id is not None:
                session_filter = "WHERE s.id = ?"
                params.append(int(session_id))
            session_rows = list(
                conn.execute(
                    f"""
                    SELECT
                        s.id,
                        s.started_at,
                        s.ended_at,
                        s.status,
                        s.record_interval_sec,
                        s.tracking_window_sec,
                        s.gap_threshold_sec,
                        COALESCE(COUNT(b.id), 0) AS block_count,
                        COALESCE(SUM(CASE WHEN b.selected_for_training = 1 THEN 1 ELSE 0 END), 0) AS selected_block_count,
                        COALESCE(SUM(CASE WHEN b.selected_for_training = 1 AND b.is_open = 0 THEN 1 ELSE 0 END), 0) AS trainable_block_count
                    FROM tracker_capture_sessions s
                    LEFT JOIN tracker_pair_blocks b ON b.session_id = s.id
                    {session_filter}
                    GROUP BY s.id
                    HAVING COUNT(b.id) > 0
                    ORDER BY s.started_at DESC, s.id DESC
                    """,
                    params,
                )
            )
            block_rows = []
            if include_block_details:
                block_params: list[Any] = []
                block_filter = []
                if session_id is not None:
                    block_filter.append("b.session_id = ?")
                    block_params.append(int(session_id))
                if not include_open:
                    block_filter.append("b.is_open = 0")
                where_clause = f"WHERE {' AND '.join(block_filter)}" if block_filter else ""
                block_rows = list(
                    conn.execute(
                        f"""
                        SELECT
                            b.id,
                            b.session_id,
                            b.start_ts,
                            b.end_ts,
                            b.record_count,
                            b.max_gap_sec,
                            b.boundary_reason,
                            b.selected_for_training,
                            b.disabled_reason,
                            b.manual_override,
                            b.notes,
                            b.is_open,
                            p.symbol,
                            p.buy_ex,
                            p.buy_mt,
                            p.sell_ex,
                            p.sell_mt
                        FROM tracker_pair_blocks b
                        JOIN tracker_pairs p ON p.id = b.pair_id
                        {where_clause}
                        ORDER BY b.session_id DESC, p.symbol ASC, b.start_ts ASC, b.id ASC
                        """,
                        block_params,
                    )
                )

        blocks_by_session: dict[int, list[dict[str, object]]] = defaultdict(list)
        total_trainable = 0
        if include_block_details:
            for row in block_rows:
                is_open = bool(row["is_open"])
                selected = bool(row["selected_for_training"])
                block_quality = quality_state["blocks"].get(int(row["id"]), {})
                trainable = self._block_is_trainable(
                    {
                        "is_open": is_open,
                        "selected_for_training": selected,
                        "manual_override": bool(row["manual_override"]),
                        "disabled_reason": str(row["disabled_reason"] or ""),
                        "record_count": int(row["record_count"]),
                        "actual_max_gap_sec": float(block_quality.get("actual_max_gap_sec", row["max_gap_sec"]) or 0.0),
                        "boundary_reason": str(row["boundary_reason"]),
                        "zero_record_block": bool(block_quality.get("zero_record_block", False)),
                        "record_count_mismatch": bool(block_quality.get("record_count_mismatch", False)),
                        "range_mismatch": bool(block_quality.get("range_mismatch", False)),
                    },
                    record_interval_sec=self.record_interval_sec,
                )
                if trainable:
                    total_trainable += 1
                block = {
                    "id": int(row["id"]),
                    "session_id": int(row["session_id"]),
                    "pair_key": f"{row['symbol']}|{row['buy_ex']}|{row['buy_mt']}|{row['sell_ex']}|{row['sell_mt']}",
                    "symbol": str(row["symbol"]),
                    "buy_ex": str(row["buy_ex"]),
                    "buy_mt": str(row["buy_mt"]),
                    "sell_ex": str(row["sell_ex"]),
                    "sell_mt": str(row["sell_mt"]),
                    "start_ts": float(row["start_ts"]),
                    "end_ts": float(row["end_ts"]),
                    "duration_sec": max(0.0, float(row["end_ts"]) - float(row["start_ts"])),
                    "record_count": int(row["record_count"]),
                    "max_gap_sec": float(row["max_gap_sec"]),
                    "boundary_reason": str(row["boundary_reason"]),
                    "selected_for_training": selected,
                    "disabled_reason": str(row["disabled_reason"] or ""),
                    "manual_override": bool(row["manual_override"]),
                    "notes": str(row["notes"] or ""),
                    "is_open": is_open,
                    "trainable": trainable,
                    "actual_record_count": int(block_quality.get("actual_record_count", 0)),
                    "actual_start_ts": block_quality.get("actual_start_ts"),
                    "actual_end_ts": block_quality.get("actual_end_ts"),
                    "actual_max_gap_sec": float(block_quality.get("actual_max_gap_sec", row["max_gap_sec"]) or 0.0),
                    "zero_record_block": bool(block_quality.get("zero_record_block", False)),
                    "record_count_mismatch": bool(block_quality.get("record_count_mismatch", False)),
                    "range_mismatch": bool(block_quality.get("range_mismatch", False)),
                    "critical_anomaly_count": int(block_quality.get("critical_anomaly_count", 0)),
                }
                blocks_by_session[int(row["session_id"])].append(block)
        else:
            total_trainable = int(sum(int(session.get("trainable_block_count_strict", 0) or 0) for session in quality_state["sessions"].values()))

        sessions = []
        for row in session_rows:
            session_blocks = blocks_by_session.get(int(row["id"]), [])
            session_quality = quality_state["sessions"].get(int(row["id"]), {})
            sessions.append(
                {
                    "id": int(row["id"]),
                    "started_at": float(row["started_at"]),
                    "ended_at": self._coerce_float(row["ended_at"], 0.0),
                    "status": str(row["status"]),
                    "record_interval_sec": float(row["record_interval_sec"]),
                    "tracking_window_sec": float(row["tracking_window_sec"]),
                    "gap_threshold_sec": float(row["gap_threshold_sec"]),
                    "block_count": int(row["block_count"]),
                    "selected_block_count": int(row["selected_block_count"]),
                    "trainable_block_count": (
                        int(sum(1 for block in session_blocks if bool(block["trainable"])))
                        if include_block_details
                        else int(session_quality.get("trainable_block_count_strict", 0))
                    ),
                    "quality_status": str(session_quality.get("quality_status", "collecting")),
                    "critical_anomaly_count": int(session_quality.get("critical_anomaly_count", 0)),
                    "session_span_sec": float(session_quality.get("session_span_sec", 0.0) or 0.0),
                    "blocks": session_blocks if include_block_details else [],
                }
            )

        return {
            "config": self.get_training_config(),
            "sessions": sessions,
            "summary": {
                "total_sessions": len(sessions),
                "total_blocks": int(sum(int(session["block_count"]) for session in sessions)),
                "trainable_blocks": total_trainable,
            },
        }

    def get_training_block(self, block_id: int) -> dict[str, object]:
        listing = self.list_training_blocks(include_open=True)
        for session in listing["sessions"]:
            for block in session["blocks"]:
                if int(block["id"]) == int(block_id):
                    return block
        raise KeyError(f"Block {block_id} not found")

    def update_training_block(
        self,
        block_id: int,
        *,
        selected_for_training: Optional[bool] = None,
        notes: Optional[str] = None,
        disabled_reason: Optional[str] = None,
    ) -> dict[str, object]:
        if self.db_path is None:
            raise RuntimeError("Tracker storage is not configured")
        updates = []
        params: list[Any] = []
        if selected_for_training is not None:
            updates.append("selected_for_training = ?")
            params.append(1 if selected_for_training else 0)
            if disabled_reason is None:
                disabled_reason = "" if selected_for_training else "manual_unselect"
        if notes is not None:
            updates.append("notes = ?")
            params.append(str(notes))
        if disabled_reason is not None:
            updates.append("disabled_reason = ?")
            params.append(str(disabled_reason))
        if not updates:
            return self.get_training_block(block_id)
        updates.append("updated_at = ?")
        params.append(time.time())
        params.append(int(block_id))
        with self._connect() as conn:
            conn.execute(f"UPDATE tracker_pair_blocks SET {', '.join(updates)} WHERE id = ?", params)
        return self.get_training_block(block_id)

    def _recompute_block_from_db(self, conn: sqlite3.Connection, block_id: int):
        rows = list(conn.execute("SELECT ts FROM tracker_records WHERE block_id = ? ORDER BY ts ASC", (int(block_id),)))
        if not rows:
            conn.execute("DELETE FROM tracker_pair_blocks WHERE id = ?", (int(block_id),))
            return
        timestamps = [float(row["ts"]) for row in rows]
        max_gap = 0.0
        for prev, curr in zip(timestamps, timestamps[1:]):
            max_gap = max(max_gap, curr - prev)
        conn.execute(
            """
            UPDATE tracker_pair_blocks
            SET start_ts = ?, end_ts = ?, record_count = ?, max_gap_sec = ?, updated_at = ?
            WHERE id = ?
            """,
            (timestamps[0], timestamps[-1], len(timestamps), max_gap, time.time(), int(block_id)),
        )

    def _run_wal_checkpoint(self, *, mode: str = "TRUNCATE") -> None:
        if self.db_path is None or not self.db_path.exists():
            return
        try:
            with self._connect() as conn:
                conn.execute(f"PRAGMA wal_checkpoint({str(mode).upper()})")
        except Exception as exc:
            logger.debug("[Tracker] WAL checkpoint skipped: %s", exc)

    def _reconcile_storage_conn(
        self,
        conn: sqlite3.Connection,
        *,
        pair_ids: Optional[Iterable[int]] = None,
        session_ids: Optional[Iterable[int]] = None,
    ) -> dict[str, int]:
        pair_filter = [int(pair_id) for pair_id in pair_ids or [] if int(pair_id) > 0]
        session_filter = [int(session_id) for session_id in session_ids or [] if int(session_id) > 0]
        conditions: list[str] = []
        params: list[Any] = []
        if pair_filter:
            placeholders = ",".join("?" for _ in pair_filter)
            conditions.append(f"pair_id IN ({placeholders})")
            params.extend(pair_filter)
        if session_filter:
            placeholders = ",".join("?" for _ in session_filter)
            conditions.append(f"session_id IN ({placeholders})")
            params.extend(session_filter)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        block_rows = list(
            conn.execute(
                f"""
                SELECT id, pair_id, session_id
                FROM tracker_pair_blocks
                {where_clause}
                ORDER BY session_id ASC, id ASC
                """,
                params,
            )
        )
        deleted_zero_record_blocks = 0
        updated_block_ranges = 0
        reassigned_event_blocks = 0
        affected_pair_ids: set[int] = set()
        block_ranges_by_session_pair: dict[tuple[int, int], list[tuple[float, float, int]]] = defaultdict(list)

        for block_row in block_rows:
            block_id = int(block_row["id"])
            pair_id = int(block_row["pair_id"])
            affected_pair_ids.add(pair_id)
            record_rows = list(
                conn.execute(
                    """
                    SELECT ts
                    FROM tracker_records
                    WHERE block_id = ?
                    ORDER BY ts ASC
                    """,
                    (block_id,),
                )
            )
            if not record_rows:
                conn.execute("DELETE FROM tracker_pair_blocks WHERE id = ?", (block_id,))
                deleted_zero_record_blocks += 1
                continue
            timestamps = [float(row["ts"]) for row in record_rows]
            max_gap = 0.0
            for previous, current in zip(timestamps, timestamps[1:]):
                max_gap = max(max_gap, float(current) - float(previous))
            conn.execute(
                """
                UPDATE tracker_pair_blocks
                SET start_ts = ?, end_ts = ?, record_count = ?, max_gap_sec = ?, updated_at = ?
                WHERE id = ?
                """,
                (timestamps[0], timestamps[-1], len(timestamps), max_gap, time.time(), block_id),
            )
            updated_block_ranges += 1
            block_ranges_by_session_pair[(int(block_row["session_id"]), pair_id)].append((timestamps[0], timestamps[-1], block_id))

        event_conditions: list[str] = []
        event_params: list[Any] = []
        if pair_filter:
            placeholders = ",".join("?" for _ in pair_filter)
            event_conditions.append(f"pair_id IN ({placeholders})")
            event_params.extend(pair_filter)
        if session_filter:
            placeholders = ",".join("?" for _ in session_filter)
            event_conditions.append(f"session_id IN ({placeholders})")
            event_params.extend(session_filter)
        event_where = f"WHERE {' AND '.join(event_conditions)}" if event_conditions else ""
        event_rows = list(
            conn.execute(
                f"""
                SELECT rowid, pair_id, session_id, block_id, ts
                FROM tracker_events
                {event_where}
                ORDER BY session_id ASC, ts ASC
                """,
                event_params,
            )
        )
        for event_row in event_rows:
            session_id = int(event_row["session_id"] or 0)
            pair_id = int(event_row["pair_id"] or 0)
            ts_value = float(event_row["ts"])
            matched_block_id = None
            for start_ts, end_ts, block_id in block_ranges_by_session_pair.get((session_id, pair_id), []):
                if float(start_ts) <= ts_value <= float(end_ts):
                    matched_block_id = int(block_id)
                    break
            current_block_id = int(event_row["block_id"] or 0) if event_row["block_id"] is not None else 0
            if current_block_id != int(matched_block_id or 0):
                conn.execute(
                    "UPDATE tracker_events SET block_id = ? WHERE rowid = ?",
                    (matched_block_id, int(event_row["rowid"])),
                )
                reassigned_event_blocks += 1
            if pair_id > 0:
                affected_pair_ids.add(pair_id)

        for pair_id in sorted(affected_pair_ids):
            self._rebuild_episodes_for_pair_conn(conn, int(pair_id))
        return {
            "deleted_zero_record_blocks": int(deleted_zero_record_blocks),
            "updated_block_ranges": int(updated_block_ranges),
            "reassigned_event_blocks": int(reassigned_event_blocks),
        }

    def split_block(self, block_id: int, split_ts: float) -> dict[str, object]:
        if self.db_path is None:
            raise RuntimeError("Tracker storage is not configured")
        split_ts = float(split_ts)
        pair_id = 0
        with self._connect() as conn:
            block_row = conn.execute("SELECT * FROM tracker_pair_blocks WHERE id = ?", (int(block_id),)).fetchone()
            if block_row is None:
                raise KeyError(f"Block {block_id} not found")
            if bool(block_row["is_open"]):
                raise ValueError("Open blocks cannot be split")
            pair_id = int(block_row["pair_id"])
            records = list(conn.execute("SELECT ts FROM tracker_records WHERE block_id = ? ORDER BY ts ASC", (int(block_id),)))
            keep_ts = [float(row["ts"]) for row in records if float(row["ts"]) <= split_ts]
            move_ts = [float(row["ts"]) for row in records if float(row["ts"]) > split_ts]
            if not keep_ts or not move_ts:
                raise ValueError("Split timestamp must divide the block into two non-empty sides")
            cursor = conn.execute(
                """
                INSERT INTO tracker_pair_blocks(
                    pair_id, session_id, start_ts, end_ts, record_count, max_gap_sec, boundary_reason,
                    selected_for_training, disabled_reason, manual_override, notes, is_open, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, 0, 0.0, 'manual_split', ?, ?, 1, ?, 0, ?, ?)
                """,
                (
                    int(block_row["pair_id"]),
                    int(block_row["session_id"]),
                    move_ts[0],
                    move_ts[-1],
                    int(block_row["selected_for_training"]),
                    str(block_row["disabled_reason"] or ""),
                    str(block_row["notes"] or ""),
                    time.time(),
                    time.time(),
                ),
            )
            new_block_id = int(cursor.lastrowid)
            conn.execute("UPDATE tracker_records SET block_id = ? WHERE block_id = ? AND ts > ?", (new_block_id, int(block_id), split_ts))
            conn.execute("UPDATE tracker_events SET block_id = ? WHERE block_id = ? AND ts > ?", (new_block_id, int(block_id), split_ts))
            conn.execute(
                """
                UPDATE tracker_pair_blocks
                SET manual_override = 1, boundary_reason = 'manual_split', updated_at = ?
                WHERE id IN (?, ?)
                """,
                (time.time(), int(block_id), new_block_id),
            )
            self._recompute_block_from_db(conn, int(block_id))
            self._recompute_block_from_db(conn, new_block_id)
            self._reconcile_storage_conn(conn, session_ids=[int(block_row["session_id"])])
        if pair_id:
            self._refresh_pair_records_from_storage(pair_id)
        return {"block_id": int(block_id), "new_block_id": new_block_id}

    def merge_next_block(self, block_id: int) -> dict[str, object]:
        if self.db_path is None:
            raise RuntimeError("Tracker storage is not configured")
        pair_id = 0
        with self._connect() as conn:
            block_row = conn.execute("SELECT * FROM tracker_pair_blocks WHERE id = ?", (int(block_id),)).fetchone()
            if block_row is None:
                raise KeyError(f"Block {block_id} not found")
            if bool(block_row["is_open"]):
                raise ValueError("Open blocks cannot be merged")
            pair_id = int(block_row["pair_id"])
            next_row = conn.execute(
                """
                SELECT * FROM tracker_pair_blocks
                WHERE pair_id = ? AND session_id = ? AND start_ts > ?
                ORDER BY start_ts ASC, id ASC
                LIMIT 1
                """,
                (int(block_row["pair_id"]), int(block_row["session_id"]), float(block_row["end_ts"])),
            ).fetchone()
            if next_row is None:
                raise ValueError("No adjacent next block available for merge")
            if bool(next_row["is_open"]):
                raise ValueError("Cannot merge with an open next block")
            next_block_id = int(next_row["id"])
            conn.execute("UPDATE tracker_records SET block_id = ? WHERE block_id = ?", (int(block_id), next_block_id))
            conn.execute("UPDATE tracker_events SET block_id = ? WHERE block_id = ?", (int(block_id), next_block_id))
            conn.execute(
                """
                UPDATE tracker_pair_blocks
                SET manual_override = 1, boundary_reason = 'manual_merge', updated_at = ?
                WHERE id = ?
                """,
                (time.time(), int(block_id)),
            )
            conn.execute("DELETE FROM tracker_pair_blocks WHERE id = ?", (next_block_id,))
            self._recompute_block_from_db(conn, int(block_id))
            self._reconcile_storage_conn(conn, session_ids=[int(block_row["session_id"])])
        if pair_id:
            self._refresh_pair_records_from_storage(pair_id)
        return {"block_id": int(block_id), "merged_block_id": next_block_id}

    def resegment_sessions(
        self,
        *,
        session_ids: Optional[list[int]] = None,
        threshold_sec: Optional[float] = None,
    ) -> dict[str, object]:
        if self.db_path is None:
            raise RuntimeError("Tracker storage is not configured")
        effective_threshold = self._resolve_gap_threshold_sec(
            threshold_sec if threshold_sec is not None else self.gap_threshold_sec,
            self.record_interval_sec,
        )
        processed_sessions: list[int] = []
        recreated_blocks = 0
        with self._connect() as conn:
            candidate_rows = list(conn.execute("SELECT id FROM tracker_capture_sessions WHERE status != 'open' ORDER BY id ASC"))
            target_session_ids = [int(row["id"]) for row in candidate_rows]
            if session_ids is not None:
                requested = {int(value) for value in session_ids}
                target_session_ids = [value for value in target_session_ids if value in requested]
            rebuilt_pair_ids: set[int] = set()
            for current_session_id in target_session_ids:
                processed_sessions.append(current_session_id)
                pair_rows = list(conn.execute("SELECT DISTINCT pair_id FROM tracker_records WHERE session_id = ? ORDER BY pair_id ASC", (current_session_id,)))
                conn.execute("UPDATE tracker_events SET block_id = NULL WHERE session_id = ?", (current_session_id,))
                conn.execute("DELETE FROM tracker_pair_blocks WHERE session_id = ?", (current_session_id,))
                for pair_row in pair_rows:
                    pair_id = int(pair_row["pair_id"])
                    rebuilt_pair_ids.add(pair_id)
                    records = list(
                        conn.execute(
                            """
                            SELECT ts FROM tracker_records
                            WHERE session_id = ? AND pair_id = ?
                            ORDER BY ts ASC
                            """,
                            (current_session_id, pair_id),
                        )
                    )
                    if not records:
                        continue
                    block_start = 0.0
                    block_end = 0.0
                    last_ts = 0.0
                    current_block_id = 0
                    block_count = 0
                    block_max_gap = 0.0
                    for index, record_row in enumerate(records):
                        ts_value = float(record_row["ts"])
                        needs_new = index == 0 or (last_ts > 0.0 and (ts_value - last_ts) > effective_threshold)
                        if needs_new:
                            if current_block_id:
                                conn.execute(
                                    """
                                    UPDATE tracker_pair_blocks
                                    SET start_ts = ?, end_ts = ?, record_count = ?, max_gap_sec = ?, is_open = 0, updated_at = ?
                                    WHERE id = ?
                                    """,
                                    (block_start, block_end, block_count, block_max_gap, time.time(), current_block_id),
                                )
                                recreated_blocks += 1
                            boundary_reason = "legacy_import" if index == 0 else "auto_gap"
                            cursor = conn.execute(
                                """
                                INSERT INTO tracker_pair_blocks(
                                    pair_id, session_id, start_ts, end_ts, record_count, max_gap_sec, boundary_reason,
                                    selected_for_training, disabled_reason, manual_override, notes, is_open, created_at, updated_at
                                )
                                VALUES(?, ?, ?, ?, 0, 0.0, ?, 1, '', 0, '', 0, ?, ?)
                                """,
                                (pair_id, current_session_id, ts_value, ts_value, boundary_reason, time.time(), time.time()),
                            )
                            current_block_id = int(cursor.lastrowid)
                            block_start = ts_value
                            block_end = ts_value
                            block_count = 0
                            block_max_gap = 0.0
                        elif last_ts > 0.0:
                            block_max_gap = max(block_max_gap, ts_value - last_ts)
                            block_end = ts_value
                        block_count += 1
                        conn.execute(
                            """
                            UPDATE tracker_records
                            SET block_id = ?
                            WHERE pair_id = ? AND session_id = ? AND ts = ?
                            """,
                            (current_block_id, pair_id, current_session_id, ts_value),
                        )
                        last_ts = ts_value
                    if current_block_id:
                        conn.execute(
                            """
                            UPDATE tracker_pair_blocks
                            SET start_ts = ?, end_ts = ?, record_count = ?, max_gap_sec = ?, is_open = 0, updated_at = ?
                            WHERE id = ?
                            """,
                            (block_start, block_end, block_count, block_max_gap, time.time(), current_block_id),
                        )
                        recreated_blocks += 1
                    block_rows = list(
                        conn.execute(
                            """
                            SELECT id, start_ts, end_ts
                            FROM tracker_pair_blocks
                            WHERE session_id = ? AND pair_id = ?
                            ORDER BY start_ts ASC
                            """,
                            (current_session_id, pair_id),
                        )
                    )
                    for event_row in conn.execute(
                        """
                        SELECT event_type, ts
                        FROM tracker_events
                        WHERE session_id = ? AND pair_id = ?
                        ORDER BY ts ASC
                        """,
                        (current_session_id, pair_id),
                    ):
                        event_ts = float(event_row["ts"])
                        matching_block_id: Optional[int] = None
                        for block_row in block_rows:
                            if float(block_row["start_ts"]) <= event_ts <= float(block_row["end_ts"]):
                                matching_block_id = int(block_row["id"])
                                break
                        conn.execute(
                            """
                            UPDATE tracker_events
                            SET block_id = ?
                            WHERE pair_id = ? AND session_id = ? AND event_type = ? AND ts = ?
                            """,
                            (matching_block_id, pair_id, current_session_id, str(event_row["event_type"]), event_ts),
                        )
            for pair_id in rebuilt_pair_ids:
                self._rebuild_episodes_for_pair_conn(conn, pair_id)
            if processed_sessions:
                self._reconcile_storage_conn(conn, session_ids=processed_sessions)
        self.update_gap_threshold_sec(effective_threshold)
        for pair_id in rebuilt_pair_ids:
            self._refresh_pair_records_from_storage(pair_id)
        return {"session_ids": processed_sessions, "recreated_blocks": recreated_blocks, "gap_threshold_sec": effective_threshold}

    def get_selected_training_block_ids(
        self,
        *,
        trainable_only: bool = True,
        session_ids: Optional[list[int]] = None,
        sequence_length: int = 2,
    ) -> list[int]:
        if self.db_path is None or not self.db_path.is_file():
            return []
        listing = self.list_training_blocks(session_id=None, include_open=False)
        normalized_ids = {int(session_id) for session_id in session_ids or [] if int(session_id) > 0}
        selected_ids: list[int] = []
        for session in listing["sessions"]:
            if normalized_ids and int(session["id"]) not in normalized_ids:
                continue
            for block in session["blocks"]:
                if not bool(block.get("selected_for_training", True)):
                    continue
                if trainable_only and not self._block_is_trainable(
                    block,
                    record_interval_sec=float(session.get("record_interval_sec", self.record_interval_sec) or self.record_interval_sec),
                    sequence_length=max(int(sequence_length), 2),
                ):
                    continue
                selected_ids.append(int(block["id"]))
        return selected_ids

    def create_training_run(
        self,
        *,
        block_ids: Optional[list[int]] = None,
        session_ids: Optional[list[int]] = None,
        sequence_length: int,
        prediction_horizon_sec: int,
        artifact_dir: str = "",
    ) -> dict[str, object]:
        if self.db_path is None:
            raise RuntimeError("Tracker storage is not configured")
        target_sequence_length = max(int(sequence_length), 1)
        target_horizon_sec = max(int(prediction_horizon_sec), 1)
        unique_session_ids = [int(session_id) for session_id in dict.fromkeys(session_ids or []) if int(session_id) > 0]
        unique_block_ids = [int(block_id) for block_id in dict.fromkeys(block_ids or []) if int(block_id) > 0]
        now = time.time()
        session_listing = self.list_training_sessions(include_open=False)
        sessions_by_id = {int(session["id"]): session for session in session_listing["sessions"]}
        if not unique_session_ids:
            if unique_block_ids:
                block_listing = self.list_training_blocks(include_open=False)
                unique_session_ids = [
                    int(session["id"])
                    for session in block_listing["sessions"]
                    if any(int(block["id"]) in set(unique_block_ids) for block in session["blocks"])
                ]
            else:
                unique_session_ids = self.get_approved_training_session_ids(
                    include_open=False,
                    sequence_length=target_sequence_length,
                    prediction_horizon_sec=target_horizon_sec,
                )
        if not unique_session_ids:
            raise ValueError("No approved sessions available for training")
        session_order = {int(session_id): index for index, session_id in enumerate(unique_session_ids)}
        selected_sessions = [sessions_by_id.get(int(session_id)) for session_id in unique_session_ids]
        session_payloads = [session for session in selected_sessions if session is not None]
        if len(session_payloads) != len(unique_session_ids):
            raise ValueError("One or more requested sessions were not found.")
        for session in session_payloads:
            if not self._session_meets_run_gates(
                status=str(session["status"]),
                critical_anomaly_count=int(session.get("critical_anomaly_count", 0)),
                session_span_sec=float(session.get("session_span_sec", 0.0) or 0.0),
                record_interval_sec=float(session.get("record_interval_sec", self.record_interval_sec) or self.record_interval_sec),
                sequence_length=target_sequence_length,
                prediction_horizon_sec=target_horizon_sec,
            ):
                raise ValueError(
                    f"Session {int(session['id'])} does not meet clean-cycle quality gates for sequence_length={target_sequence_length} "
                    f"and prediction_horizon_sec={target_horizon_sec}."
                )
            if not bool(session.get("approved_for_training", False)):
                raise ValueError(f"Session {int(session['id'])} is not approved for training.")
        if not unique_block_ids:
            unique_block_ids = self.get_selected_training_block_ids(
                trainable_only=True,
                session_ids=[int(session["id"]) for session in session_payloads],
                sequence_length=target_sequence_length,
            )
        if not unique_block_ids:
            raise ValueError("No selected blocks available for training")
        block_listing = self.list_training_blocks(include_open=False)
        blocks_by_id = {
            int(block["id"]): (session, block)
            for session in block_listing["sessions"]
            for block in session["blocks"]
        }
        for block_id in unique_block_ids:
            session_block = blocks_by_id.get(int(block_id))
            if session_block is None:
                raise ValueError(f"Block {int(block_id)} was not found.")
            session_payload, block_payload = session_block
            if not self._block_is_trainable(
                block_payload,
                record_interval_sec=float(session_payload.get("record_interval_sec", self.record_interval_sec) or self.record_interval_sec),
                sequence_length=target_sequence_length,
            ):
                raise ValueError(f"Block {int(block_id)} does not satisfy clean-cycle block gates.")
        with self._connect() as conn:
            placeholders = ",".join("?" for _ in unique_block_ids)
            block_rows = list(
                conn.execute(
                    f"""
                    SELECT
                        b.id,
                        b.session_id,
                        b.start_ts,
                        b.end_ts,
                        b.record_count,
                        p.symbol,
                        p.buy_ex,
                        p.buy_mt,
                        p.sell_ex,
                        p.sell_mt
                    FROM tracker_pair_blocks b
                    JOIN tracker_pairs p ON p.id = b.pair_id
                    WHERE b.id IN ({placeholders}) AND b.selected_for_training = 1 AND b.is_open = 0
                    ORDER BY b.session_id ASC, b.start_ts ASC, b.id ASC
                    """,
                    unique_block_ids,
                )
            )
            if not block_rows:
                raise ValueError("No trainable closed blocks selected")
            estimated_window_count = sum(max(0, int(row["record_count"]) - target_sequence_length) for row in block_rows)
            if estimated_window_count <= 0:
                raise ValueError(
                    f"Selected sessions do not contain enough contiguous records for sequence_length={target_sequence_length}."
                )
            cursor = conn.execute(
                """
                INSERT INTO ml_training_runs(
                    created_at, started_at, finished_at, status, state_path, artifact_dir, audit_path, error,
                    threshold_sec, sequence_length, prediction_horizon_sec, selected_block_count, result_json
                )
                VALUES(?, NULL, NULL, 'queued', ?, ?, '', '', ?, ?, ?, ?, '{}')
                """,
                (
                    now,
                    str(self.db_path),
                    str(artifact_dir or ""),
                    self.gap_threshold_sec,
                    target_sequence_length,
                    target_horizon_sec,
                    len(block_rows),
                ),
            )
            run_id = int(cursor.lastrowid)
            conn.executemany(
                """
                INSERT INTO ml_training_run_sessions(
                    run_id, session_id, position, started_at, ended_at, data_start_ts, data_end_ts, status,
                    approved_for_training, excluded_reason, block_count, trainable_block_count, exception_block_count
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        int(row["id"]),
                        session_order.get(int(row["id"]), index),
                        float(row["started_at"]),
                        self._coerce_float(row["ended_at"], 0.0),
                        self._coerce_float(row["data_start_ts"], 0.0),
                        self._coerce_float(row["data_end_ts"], 0.0),
                        str(row["status"]),
                        1 if bool(row["approved_for_training"]) else 0,
                        str(row["excluded_reason"] or ""),
                        int(row["block_count"]),
                        int(row["trainable_block_count"]),
                        int(row["exception_block_count"]),
                    )
                    for index, row in enumerate(session_payloads)
                ],
            )
            conn.executemany(
                """
                INSERT INTO ml_training_run_blocks(
                    run_id, block_id, session_id, pair_key, start_ts, end_ts, record_count
                )
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        int(row["id"]),
                        int(row["session_id"]),
                        f"{row['symbol']}|{row['buy_ex']}|{row['buy_mt']}|{row['sell_ex']}|{row['sell_mt']}",
                        float(row["start_ts"]),
                        float(row["end_ts"]),
                        int(row["record_count"]),
                    )
                    for row in block_rows
                ],
            )
        return self.get_training_run(run_id)

    def update_training_run(
        self,
        run_id: int,
        *,
        status: Optional[str] = None,
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
        error: Optional[str] = None,
        result: Optional[dict[str, Any]] = None,
        artifact_dir: Optional[str] = None,
        audit_path: Optional[str] = None,
    ) -> dict[str, object]:
        if self.db_path is None:
            raise RuntimeError("Tracker storage is not configured")
        updates = []
        params: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            params.append(str(status))
        if started_at is not None:
            updates.append("started_at = ?")
            params.append(float(started_at))
        if finished_at is not None:
            updates.append("finished_at = ?")
            params.append(float(finished_at))
        if error is not None:
            updates.append("error = ?")
            params.append(str(error))
        if result is not None:
            updates.append("result_json = ?")
            params.append(json.dumps(result, sort_keys=True))
        if artifact_dir is not None:
            updates.append("artifact_dir = ?")
            params.append(str(artifact_dir))
        if audit_path is not None:
            updates.append("audit_path = ?")
            params.append(str(audit_path))
        if updates:
            params.append(int(run_id))
            with self._connect() as conn:
                conn.execute(f"UPDATE ml_training_runs SET {', '.join(updates)} WHERE id = ?", params)
        return self.get_training_run(run_id)

    def get_training_run(self, run_id: int) -> dict[str, object]:
        if self.db_path is None:
            raise RuntimeError("Tracker storage is not configured")
        with self._connect() as conn:
            run_row = conn.execute("SELECT * FROM ml_training_runs WHERE id = ?", (int(run_id),)).fetchone()
            if run_row is None:
                raise KeyError(f"Training run {run_id} not found")
            session_rows = list(
                conn.execute(
                    """
                    SELECT
                        session_id,
                        position,
                        started_at,
                        ended_at,
                        data_start_ts,
                        data_end_ts,
                        status,
                        approved_for_training,
                        excluded_reason,
                        block_count,
                        trainable_block_count,
                        exception_block_count
                    FROM ml_training_run_sessions
                    WHERE run_id = ?
                    ORDER BY position ASC, session_id ASC
                    """,
                    (int(run_id),),
                )
            )
            block_rows = list(
                conn.execute(
                    """
                    SELECT block_id, session_id, pair_key, start_ts, end_ts, record_count
                    FROM ml_training_run_blocks
                    WHERE run_id = ?
                    ORDER BY session_id ASC, start_ts ASC, block_id ASC
                    """,
                    (int(run_id),),
                )
            )
        raw_result = str(run_row["result_json"] or "{}")
        try:
            result_payload = json.loads(raw_result)
        except Exception:
            result_payload = {}
        sessions_payload = [
            {
                "session_id": int(row["session_id"]),
                "started_at": float(row["started_at"]),
                "ended_at": float(row["ended_at"]),
                "data_start_ts": float(row["data_start_ts"]),
                "data_end_ts": float(row["data_end_ts"]),
                "status": str(row["status"]),
                "approved_for_training": bool(row["approved_for_training"]),
                "excluded_reason": str(row["excluded_reason"] or ""),
                "block_count": int(row["block_count"]),
                "trainable_block_count": int(row["trainable_block_count"]),
                "exception_block_count": int(row["exception_block_count"]),
            }
            for row in session_rows
        ]
        if not sessions_payload:
            derived_session_ids = sorted({int(row["session_id"]) for row in block_rows})
            sessions_payload = [
                {
                    "session_id": session_id,
                    "started_at": 0.0,
                    "ended_at": 0.0,
                    "data_start_ts": min(float(row["start_ts"]) for row in block_rows if int(row["session_id"]) == session_id),
                    "data_end_ts": max(float(row["end_ts"]) for row in block_rows if int(row["session_id"]) == session_id),
                    "status": "unknown",
                    "approved_for_training": True,
                    "excluded_reason": "",
                    "block_count": sum(1 for row in block_rows if int(row["session_id"]) == session_id),
                    "trainable_block_count": sum(1 for row in block_rows if int(row["session_id"]) == session_id),
                    "exception_block_count": 0,
                }
                for session_id in derived_session_ids
            ]
        return {
            "id": int(run_row["id"]),
            "created_at": float(run_row["created_at"]),
            "started_at": self._coerce_float(run_row["started_at"], 0.0),
            "finished_at": self._coerce_float(run_row["finished_at"], 0.0),
            "status": str(run_row["status"]),
            "state_path": str(run_row["state_path"]),
            "artifact_dir": str(run_row["artifact_dir"] or ""),
            "audit_path": str(run_row["audit_path"] or ""),
            "error": str(run_row["error"] or ""),
            "threshold_sec": float(run_row["threshold_sec"]),
            "sequence_length": int(run_row["sequence_length"]),
            "prediction_horizon_sec": int(run_row["prediction_horizon_sec"]),
            "selected_block_count": int(run_row["selected_block_count"]),
            "result": result_payload,
            "sessions": sessions_payload,
            "blocks": [
                {
                    "block_id": int(row["block_id"]),
                    "session_id": int(row["session_id"]),
                    "pair_key": str(row["pair_key"]),
                    "start_ts": float(row["start_ts"]),
                    "end_ts": float(row["end_ts"]),
                    "record_count": int(row["record_count"]),
                }
                for row in block_rows
            ],
        }

    def get_latest_training_run(self) -> dict[str, object]:
        if self.db_path is None:
            raise RuntimeError("Tracker storage is not configured")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id
                FROM ml_training_runs
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            raise KeyError("No training runs found")
        return self.get_training_run(int(row["id"]))

    def get_training_quality_report(
        self,
        *,
        sequence_length: int = _DEFAULT_CLEAN_SEQUENCE_LENGTH,
        prediction_horizon_sec: int = _DEFAULT_CLEAN_PREDICTION_HORIZON_SEC,
        summary_only: bool = False,
    ) -> dict[str, Any]:
        if self.db_path is None or not self.db_path.is_file():
            return {
                "config": self.get_training_config(),
                "summary": {
                    "total_sessions": 0,
                    "critical_sessions": 0,
                    "training_ready_sessions": 0,
                    "burn_in_passed_sessions": 0,
                    "checkpoint_ready_sessions": 0,
                },
                "sessions": [],
            }
        with self._connect() as conn:
            quality_state = self._collect_training_quality_state(conn)
        listing = self.list_training_sessions(
            include_open=True,
            include_blocks_preview=False,
            summary_only=summary_only,
            _quality_state=quality_state,
        )
        if summary_only:
            return {
                "config": self.get_training_config(),
                "summary": dict(listing["summary"]),
                "sequence_length": int(sequence_length),
                "prediction_horizon_sec": int(prediction_horizon_sec),
                "sessions": [],
            }
        sessions = []
        for session in listing["sessions"]:
            dynamic_status = self._session_quality_status(
                status=str(session["status"]),
                critical_anomaly_count=int(session.get("critical_anomaly_count", 0)),
                session_span_sec=float(session.get("session_span_sec", 0.0) or 0.0),
                record_interval_sec=float(session.get("record_interval_sec", self.record_interval_sec) or self.record_interval_sec),
                sequence_length=int(sequence_length),
                prediction_horizon_sec=int(prediction_horizon_sec),
            )
            session_payload = dict(session)
            session_payload["dynamic_quality_status"] = dynamic_status
            session_payload["dynamic_training_ready"] = bool(session.get("can_train", False))
            sessions.append(session_payload)
        return {
            "config": self.get_training_config(),
            "summary": dict(listing["summary"]),
            "sequence_length": int(sequence_length),
            "prediction_horizon_sec": int(prediction_horizon_sec),
            "sessions": sessions,
        }

    def reset_training_cycle(self, *, archive_root: str | Path | None = None) -> dict[str, Any]:
        if self.db_path is None:
            raise RuntimeError("Tracker storage is not configured")
        now = time.time()
        self.flush_to_storage(now_ts=now, force=True)
        if self._active_session_id:
            self.close_active_session(status="interrupted", ended_at=now)
        quality_report = self.get_training_quality_report()
        storage_stats = self.get_storage_stats()
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
        archive_base = Path(archive_root) if archive_root is not None else self.db_path.parent.parent / "archive" / "tracker_cycles"
        archive_dir = archive_base / timestamp
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived_files: list[str] = []
        for suffix in ("", "-wal", "-shm"):
            source_path = Path(f"{self.db_path}{suffix}")
            if not source_path.exists():
                continue
            target_path = archive_dir / source_path.name
            shutil.copy2(str(source_path), str(target_path))
            archived_files.append(str(target_path))
        manifest = {
            "archived_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "source_db_path": str(self.db_path),
            "archived_files": archived_files,
            "storage_stats": storage_stats,
            "quality_report_summary": quality_report.get("summary", {}),
        }
        (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        with self._lock:
            self._pairs = defaultdict(PairStats)
            self._dirty_pairs.clear()
            self._deleted_pairs.clear()
            self._last_flush_at = 0.0
            self._active_session_id = 0
            self._ephemeral_session_id = 1
            self._ephemeral_block_id = 1
        with self._connect() as conn:
            conn.executescript(
                """
                DROP TABLE IF EXISTS ml_training_run_blocks;
                DROP TABLE IF EXISTS ml_training_run_sessions;
                DROP TABLE IF EXISTS ml_training_runs;
                DROP TABLE IF EXISTS tracker_pair_episodes;
                DROP TABLE IF EXISTS tracker_pair_blocks;
                DROP TABLE IF EXISTS tracker_capture_sessions;
                DROP TABLE IF EXISTS tracker_events;
                DROP TABLE IF EXISTS tracker_records;
                DROP TABLE IF EXISTS tracker_pairs;
                DROP TABLE IF EXISTS tracker_meta;
                """
            )
        self._initialize_storage()
        self._open_runtime_session()
        return {
            "archive_dir": str(archive_dir),
            "manifest_path": str(archive_dir / "manifest.json"),
            "archived_files": archived_files,
            "quality_report_summary": quality_report.get("summary", {}),
            "new_db_path": str(self.db_path),
            "new_active_session_id": int(self._active_session_id),
        }

    def get_storage_stats(self) -> Dict[str, object]:
        stats: Dict[str, object] = {
            "db_path": str(self.db_path) if self.db_path is not None else "",
            "record_interval_sec": self.record_interval_sec,
            "window_sec": self.storage_window_sec,
            "storage_window_sec": self.storage_window_sec,
            "tracker_memory_window_sec": self.memory_window_sec,
            "max_records_per_pair": self.max_records_per_pair,
            "gap_threshold_sec": self.gap_threshold_sec,
            "min_total_spread_pct": self.min_total_spread_pct,
            "active_session_id": self._active_session_id,
            "quality_fix_activated_at": self.quality_fix_activated_at,
            "last_flush_at": self._last_flush_at,
            "pairs_in_memory": len(self._pairs),
            "dirty_pairs_pending": len(self._dirty_pairs),
            "deleted_pairs_pending": len(self._deleted_pairs),
            "db_size_bytes": self.db_path.stat().st_size if self.db_path is not None and self.db_path.exists() else 0,
            "pairs_persisted": 0,
            "records_total": 0,
            "events_total": 0,
            "episodes_total": 0,
            "sessions_total": 0,
            "blocks_total": 0,
            "selected_training_blocks": 0,
            "open_blocks": 0,
        }
        stats.update(self._hot_path_rejection_snapshot_locked())
        if self.db_path is None or not self.db_path.is_file():
            return stats
        try:
            with self._connect() as conn:
                stats["pairs_persisted"] = int(conn.execute("SELECT COUNT(*) FROM tracker_pairs").fetchone()[0])
                stats["records_total"] = int(conn.execute("SELECT COUNT(*) FROM tracker_records").fetchone()[0])
                stats["events_total"] = int(conn.execute("SELECT COUNT(*) FROM tracker_events").fetchone()[0])
                stats["episodes_total"] = int(conn.execute("SELECT COUNT(*) FROM tracker_pair_episodes").fetchone()[0])
                stats["sessions_total"] = int(conn.execute("SELECT COUNT(*) FROM tracker_capture_sessions").fetchone()[0])
                stats["blocks_total"] = int(conn.execute("SELECT COUNT(*) FROM tracker_pair_blocks").fetchone()[0])
                stats["selected_training_blocks"] = int(conn.execute("SELECT COUNT(*) FROM tracker_pair_blocks WHERE selected_for_training = 1").fetchone()[0])
                stats["open_blocks"] = int(conn.execute("SELECT COUNT(*) FROM tracker_pair_blocks WHERE is_open = 1").fetchone()[0])
        except Exception as exc:
            logger.warning("[Tracker] Failed to query storage stats: %s", exc)
        return stats

    def save_state(self, path: str | Path) -> bool:
        path = Path(path)
        now = time.time()
        cutoff = self._memory_cutoff(now)
        data: Dict[str, object] = {
            "saved_at": now,
            "window_sec": self.storage_window_sec,
            "tracker_memory_window_sec": self.memory_window_sec,
            "pairs": {},
        }
        with self._lock:
            for key, ps in self._pairs.items():
                pair_id = self._pair_id(key)
                snapshot = self._pair_snapshot(key, ps, state_cutoff=cutoff, record_cutoff=cutoff)
                inv = snapshot["inverted_events"]
                ent = snapshot["entry_events"]
                ext = snapshot["exit_events"]
                recs = snapshot["records"]
                if not inv and not ent and not ext and not recs:
                    continue
                data["pairs"][pair_id] = {
                    "last_state": snapshot["last_state"],
                    "last_crossover_ts": snapshot["last_crossover_ts"],
                    "last_seen_ts": snapshot["last_seen_ts"],
                    "history_enabled": snapshot["history_enabled"],
                    "inverted_events": [event[0] for event in inv],
                    "entry_events": [event[0] for event in ent],
                    "exit_events": [event[0] for event in ext],
                    "records": [{"ts": record[0], "entry": round(record[1], 6), "exit": round(record[2], 6)} for record in recs],
                }
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            tmp.replace(path)
            logger.info("[Tracker] Exported legacy JSON state (%s pairs)", len(data["pairs"]))
            return True
        except Exception as exc:
            logger.error("[Tracker] Failed to export legacy JSON state: %s", exc)
            return False

    def load_state(self, path: str | Path) -> int:
        path = Path(path)
        if not path.is_file():
            return 0
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[Tracker] Could not read legacy JSON state file: %s", exc)
            return 0
        if not isinstance(raw, dict) or "pairs" not in raw:
            return 0
        cutoff = self._memory_cutoff(time.time())
        count = 0
        with self._lock:
            for pair_id, payload in raw["pairs"].items():
                if not isinstance(payload, dict):
                    continue
                parts = pair_id.split("|")
                if len(parts) != 5:
                    continue
                key: PairKey = tuple(parts)  # type: ignore[assignment]
                ps = self._pairs[key]
                ps.last_state = int(payload.get("last_state", 0))
                ps.last_crossover_ts = float(payload.get("last_crossover_ts", 0.0))
                ps.last_seen_ts = float(payload.get("last_seen_ts", 0.0))
                ps.history_enabled = bool(payload.get("history_enabled", False))
                for ts_value in payload.get("inverted_events", []):
                    if float(ts_value) >= cutoff:
                        ps.inverted_events.append(TrackerEvent(float(ts_value), "inverted"))
                for ts_value in payload.get("entry_events", []):
                    if float(ts_value) >= cutoff:
                        ps.entry_events.append(TrackerEvent(float(ts_value), "entry"))
                for ts_value in payload.get("exit_events", []):
                    if float(ts_value) >= cutoff:
                        ps.exit_events.append(TrackerEvent(float(ts_value), "exit"))
                for record in payload.get("records", []):
                    if not isinstance(record, dict):
                        continue
                    ts_value = float(record.get("ts", 0.0))
                    if ts_value < cutoff:
                        continue
                    ps.records.append(SpreadRecord(timestamp=ts_value, entry_spread_pct=float(record.get("entry", 0.0)), exit_spread_pct=float(record.get("exit", 0.0))))
                    ps._last_record_ts = max(ps._last_record_ts, ts_value)
                    ps._stats_dirty = True
                self._mark_dirty(key)
                count += 1
        logger.info("[Tracker] Imported %s pairs from legacy JSON %s", count, path.name)
        return count

    def get_inverted_counts(
        self,
        symbol: str,
        buy_ex: str,
        buy_mt: str,
        sell_ex: str,
        sell_mt: str,
        *,
        now_ts: Optional[float] = None,
    ) -> Dict[str, int]:
        key = self._pair_key(symbol, buy_ex, buy_mt, sell_ex, sell_mt)
        ts = float(now_ts) if now_ts is not None else time.time()
        with self._lock:
            ps = self._pairs.get(key)
            if not ps or not ps.inverted_events:
                return {name: 0 for name in self._WINDOWS}
            events = [event.timestamp for event in ps.inverted_events]
        out: Dict[str, int] = {}
        for name, seconds in self._WINDOWS.items():
            cutoff = ts - seconds
            index = bisect_left(events, cutoff)
            out[name] = len(events) - index
        return out

    def batch_enrich(self, keys: List[PairKey], now_ts: float) -> Dict[PairKey, Optional[Dict]]:
        cutoffs = {name: now_ts - seconds for name, seconds in self._WINDOWS.items()}
        results: Dict[PairKey, Optional[Dict]] = {}
        with self._lock:
            for key in keys:
                if key in results:
                    continue
                ps = self._pairs.get(key)
                if not ps:
                    results[key] = None
                    continue
                if ps.inverted_events:
                    inv_counts = {}
                    events = [event.timestamp for event in ps.inverted_events]
                    for name, cutoff in cutoffs.items():
                        idx = bisect_left(events, cutoff)
                        inv_counts[name] = len(events) - idx
                else:
                    inv_counts = dict(self._ZERO_COUNTS)
                results[key] = {
                    "inverted_counts": inv_counts,
                    "inverted_count": inv_counts.get("4h", 0),
                    "total_entries": len(ps.entry_events),
                    "total_exits": len(ps.exit_events),
                    "last_crossover_ts": ps.last_crossover_ts,
                    "history_points": len(ps.records),
                    "last_record_ts": float(ps._last_record_ts),
                    "current_block_id": int(ps.current_block_id or 0),
                    "history_enabled": bool(ps.history_enabled),
                }
        return results

    def batch_record(self, records: list, *, now_ts: float | None = None):
        if not records:
            return
        ts = float(now_ts) if now_ts is not None else time.time()
        record_cutoff = self._memory_cutoff(ts)
        state_cutoff = self._storage_cutoff(ts)
        track_gate = self.track_enable_entry_spread_pct
        hist_gate = self.history_enable_entry_spread_pct
        rec_interval = max(self.record_interval_sec, 0.0)
        max_recs = self.max_records_per_pair
        with self._lock:
            pruned: set[PairKey] = set()
            pairs_at_cap = self.max_pairs > 0 and len(self._pairs) >= self.max_pairs
            for rec in records:
                symbol, buy_ex, buy_mt, sell_ex, sell_mt, entry_v, exit_v = rec
                key = self._pair_key(symbol, buy_ex, buy_mt, sell_ex, sell_mt)
                invalid_fields = self._invalid_numeric_fields(ts, entry_v, exit_v)
                if invalid_fields:
                    self._record_hot_path_attempt_locked(key)
                    self._record_hot_path_rejection_locked(key, invalid_fields=invalid_fields)
                    continue
                entry_v = float(entry_v)
                exit_v = float(exit_v)
                new_state = self._state(entry_v, exit_v)
                pair_dirty = False
                if key not in self._pairs:
                    if pairs_at_cap:
                        continue
                    if track_gate > 0 and entry_v < track_gate:
                        continue
                pair_dirty = True
                ps = self._pairs[key]
                ps.last_seen_ts = ts
                if key not in pruned:
                    if ps._prune_events(state_cutoff):
                        pair_dirty = True
                    preserve_after_ts = self._last_flush_at if self._last_flush_at > 0.0 else float("-inf")
                    if ps._prune_records(record_cutoff, preserve_after_ts=preserve_after_ts):
                        pair_dirty = True
                    ps._prune_block_meta(state_cutoff)
                    ps._prune_episodes(state_cutoff)
                    pruned.add(key)
                event_session_id = self._active_session_id or ps.current_session_id or self._ephemeral_session_id
                event_block_id = ps.current_block_id if ps.history_enabled and ps.current_block_id else None
                if new_state in (-1, 1):
                    if ps.last_state in (-1, 1) and new_state != ps.last_state:
                        ps.inverted_events.append(TrackerEvent(ts, "inverted", event_session_id, event_block_id))
                        ps.last_crossover_ts = ts
                        pair_dirty = True
                    if new_state == 1 and ps.last_state != 1:
                        ps.entry_events.append(TrackerEvent(ts, "entry", event_session_id, event_block_id))
                        pair_dirty = True
                    if new_state == -1 and ps.last_state != -1:
                        ps.exit_events.append(TrackerEvent(ts, "exit", event_session_id, event_block_id))
                        pair_dirty = True
                    if ps.last_state != new_state:
                        ps.last_state = new_state
                        pair_dirty = True
                if (not ps.history_enabled) and (entry_v >= hist_gate):
                    ps.history_enabled = True
                    pair_dirty = True
                if ps.history_enabled and rec_interval >= 0:
                    timestamp_status = self._timestamp_status_locked(ps, ts)
                    if timestamp_status == "exact_duplicate":
                        self._record_hot_path_attempt_locked(key)
                        self._record_hot_path_rejection_locked(key, invalid_fields=["duplicate_ts"])
                        ps.exact_duplicate_ts_rejections += 1
                        self._emit_tracker_record_rejected(
                            key=key,
                            ts=ts,
                            entry_spread=entry_v,
                            exit_spread=exit_v,
                            invalid_fields=["duplicate_ts"],
                            reason="exact_duplicate_ts",
                            last_record_ts=ps._last_record_ts,
                        )
                        continue
                    if timestamp_status == "retrograde":
                        self._record_hot_path_attempt_locked(key)
                        self._record_hot_path_rejection_locked(key, invalid_fields=["retrograde_ts"])
                        ps.retrograde_ts_rejections += 1
                        self._emit_tracker_record_rejected(
                            key=key,
                            ts=ts,
                            entry_spread=entry_v,
                            exit_spread=exit_v,
                            invalid_fields=["retrograde_ts"],
                            reason="retrograde_ts_rejected",
                            last_record_ts=ps._last_record_ts,
                        )
                        continue
                    clock_jitter_ts = timestamp_status == "clock_jitter"
                    if clock_jitter_ts:
                        ps.clock_jitter_events += 1
                    effective_ts = self._adjust_record_ts(ps._last_record_ts, ts)
                    record_delta_sec = (effective_ts - ps._last_record_ts) if ps._last_record_ts > 0.0 else 0.0
                    gap_detected = bool(ps._last_record_ts > 0.0 and record_delta_sec > self.gap_threshold_sec)
                    monotonic = bool(ps._last_record_ts <= 0.0 or effective_ts > ps._last_record_ts)
                    should_record = (not ps.records or (effective_ts - ps._last_record_ts) >= rec_interval)
                    if should_record:
                        self._record_hot_path_attempt_locked(key)
                        fingerprint = self._record_fingerprint(ts, entry_v, exit_v, self.record_interval_sec)
                        if self._is_light_duplicate_locked(ps, fingerprint):
                            self._record_hot_path_rejection_locked(key, invalid_fields=["duplicate_payload"])
                            self._emit_tracker_record_rejected(
                                key=key,
                                ts=ts,
                                entry_spread=entry_v,
                                exit_spread=exit_v,
                                invalid_fields=["duplicate_payload"],
                                reason="light_duplicate_payload",
                                last_record_ts=ps._last_record_ts,
                            )
                            continue
                        block_id = self._ensure_block_locked(key, ps, effective_ts)
                        if ps._last_record_ts > 0.0 and ps.current_block_record_count > 0:
                            ps.current_block_max_gap_sec = max(ps.current_block_max_gap_sec, effective_ts - ps._last_record_ts)
                        spread_outlier_warning = self._is_pair_spread_outlier_locked(ps, entry_v)
                        if spread_outlier_warning:
                            ps.spread_outlier_warnings += 1
                        ps.records.append(
                            SpreadRecord(
                                timestamp=effective_ts,
                                entry_spread_pct=entry_v,
                                exit_spread_pct=exit_v,
                                session_id=self._active_session_id or ps.current_session_id,
                                block_id=block_id,
                            )
                        )
                        self._remember_record_locked(ps, fingerprint=fingerprint, entry_spread=entry_v)
                        ps.current_block_end_ts = effective_ts
                        ps.current_block_record_count += 1
                        self._update_open_block_meta_locked(ps)
                        ps._last_record_ts = effective_ts
                        ps._stats_dirty = True
                        ps._episodes_cache_valid = False
                        pair_dirty = True
                        if self.audit_collector is not None:
                            self.audit_collector.record_tracker_record(
                                pair_id=self._pair_id(key),
                                ts=ts,
                                entry=entry_v,
                                exit=exit_v,
                                session_id=self._active_session_id or ps.current_session_id,
                                block_id=block_id,
                                record_delta_sec=record_delta_sec,
                                gap_detected=gap_detected,
                                monotonic=monotonic,
                                invalid_fields=list(invalid_fields),
                                exact_duplicate_ts=False,
                                retrograde_ts_rejected=False,
                                clock_jitter_ts=clock_jitter_ts,
                                spread_outlier_warning=spread_outlier_warning,
                            )
                        self._emit_event(
                            {
                                "kind": "tracker_record",
                                "pair_key": self._pair_id(key),
                                "record_ts": effective_ts,
                                "entry": entry_v,
                                "exit": exit_v,
                                "session_id": self._active_session_id or ps.current_session_id,
                                "block_id": block_id,
                                "delta_ts": record_delta_sec,
                                "gap_detected": gap_detected,
                                "gap_threshold_sec": self.gap_threshold_sec,
                                "numeric_valid": not invalid_fields,
                                "invalid_fields": list(invalid_fields),
                                "timestamp_monotonic": monotonic,
                                "exact_duplicate_ts": False,
                                "retrograde_ts_rejected": False,
                                "clock_jitter_ts": clock_jitter_ts,
                                "spread_outlier_warning": spread_outlier_warning,
                            }
                        )
                        if max_recs > 0:
                            while len(ps.records) > max_recs:
                                ps.records.popleft()
                                ps._stats_dirty = True
                                pair_dirty = True
                if pair_dirty:
                    self._mark_dirty(key)
