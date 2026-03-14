from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, NamedTuple

try:
    import numpy as np
    import pandas as pd
    _VECTORIZE_AVAILABLE = True
except ImportError:
    _VECTORIZE_AVAILABLE = False

_VECTORIZE_MIN_RECORDS = 5000


class _WindowStats(NamedTuple):
    mean: float
    std: float
    max: float
    position_in_range: float
    zscore_vs_mean: float
    trend_slope_per_hour: float

V1_MICRO_FEATURE_NAMES = [
    "entry_spread",
    "exit_spread",
    "delta_entry",
    "delta_exit",
    "delta2_entry",
    "delta2_exit",
    "rolling_std_entry",
    "rolling_std_exit",
    "zscore_entry",
    "zscore_exit",
]

V2_MULTISCALE_FEATURE_NAMES = [
    *V1_MICRO_FEATURE_NAMES,
    "mean_entry_30m",
    "std_entry_30m",
    "max_entry_30m",
    "episode_count_30m",
    "position_in_range_30m",
    "mean_entry_2h",
    "std_entry_2h",
    "max_entry_2h",
    "episode_count_2h",
    "trend_slope_2h",
    "mean_entry_8h",
    "std_entry_8h",
    "max_entry_8h",
    "episode_count_8h",
    "zscore_vs_8h",
]

V3_EXIT_AWARE_FEATURE_NAMES = [
    *V2_MULTISCALE_FEATURE_NAMES,
    # Exit multi-scale (6)
    "mean_exit_2h",
    "exit_p10_2h",
    "std_exit_2h",
    "mean_exit_8h",
    "exit_p10_8h",
    "zscore_exit_vs_8h",
    # Cycle / profitability (6)
    "mean_total_spread_2h",
    "median_episode_duration_2h",
    "episode_close_rate_2h",
    "mean_total_spread_8h",
    "median_episode_duration_8h",
    "mean_reversion_speed_8h",
    # Entry-exit relation (3)
    "current_spread_total",
    "entry_exit_ratio_2h",
    "viability_score",
]

DEFAULT_FEATURE_CONTRACT_VERSION = "v3_exit_aware_40"

FEATURE_CONTRACTS: dict[str, list[str]] = {
    "v1_micro_10": list(V1_MICRO_FEATURE_NAMES),
    "v2_multiscale_25": list(V2_MULTISCALE_FEATURE_NAMES),
    "v3_exit_aware_40": list(V3_EXIT_AWARE_FEATURE_NAMES),
}

FEATURE_NAMES = list(FEATURE_CONTRACTS[DEFAULT_FEATURE_CONTRACT_VERSION])

_MICRO_ROLLING_WINDOW = 5
_MULTISCALE_WINDOWS = {
    "30m": 30 * 60,
    "2h": 2 * 60 * 60,
    "8h": 8 * 60 * 60,
}


@dataclass(slots=True)
class NormalizedRecord:
    timestamp: float
    entry_spread: float
    exit_spread: float
    session_id: int = 0
    block_id: int = 0


@dataclass(slots=True)
class NormalizedEpisode:
    end_ts: float
    start_ts: float = 0.0
    exit_spread: float = 0.0
    duration_sec: float = 0.0
    peak_entry_spread: float = 0.0
    total_spread: float = 0.0
    is_closed: bool = True


class _RollingEntryWindow:
    def __init__(self, window_sec: float):
        self.window_sec = float(window_sec)
        self.points: deque[tuple[float, float, float]] = deque()
        self.max_points: deque[tuple[float, float]] = deque()
        self.min_points: deque[tuple[float, float]] = deque()
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_t = 0.0
        self.sum_t2 = 0.0
        self.sum_ty = 0.0

    def push(self, ts: float, entry: float) -> None:
        t_hr = float(ts) / 3600.0
        point = (float(ts), float(entry), t_hr)
        self.points.append(point)
        self.sum_y += float(entry)
        self.sum_y2 += float(entry) * float(entry)
        self.sum_t += t_hr
        self.sum_t2 += t_hr * t_hr
        self.sum_ty += t_hr * float(entry)
        while self.max_points and self.max_points[-1][1] <= float(entry):
            self.max_points.pop()
        self.max_points.append((float(ts), float(entry)))
        while self.min_points and self.min_points[-1][1] >= float(entry):
            self.min_points.pop()
        self.min_points.append((float(ts), float(entry)))
        self._expire(float(ts) - self.window_sec)

    def _expire(self, cutoff: float) -> None:
        while self.points and self.points[0][0] < cutoff:
            ts, entry, t_hr = self.points.popleft()
            self.sum_y -= entry
            self.sum_y2 -= entry * entry
            self.sum_t -= t_hr
            self.sum_t2 -= t_hr * t_hr
            self.sum_ty -= t_hr * entry
            if self.max_points and self.max_points[0] == (ts, entry):
                self.max_points.popleft()
            if self.min_points and self.min_points[0] == (ts, entry):
                self.min_points.popleft()

    def stats(self, current_entry: float) -> _WindowStats:
        count = len(self.points)
        if count <= 0:
            return _WindowStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        mean = self.sum_y / count
        variance = max((self.sum_y2 / count) - (mean * mean), 0.0)
        std = math.sqrt(variance)
        max_entry = float(self.max_points[0][1]) if self.max_points else 0.0
        min_entry = float(self.min_points[0][1]) if self.min_points else 0.0
        entry_range = max_entry - min_entry
        position = ((float(current_entry) - min_entry) / entry_range) if entry_range > 1e-9 else 0.0
        zscore = ((float(current_entry) - mean) / std) if std > 1e-9 else 0.0
        denominator = (count * self.sum_t2) - (self.sum_t * self.sum_t)
        slope = (
            ((count * self.sum_ty) - (self.sum_t * self.sum_y)) / denominator
            if abs(denominator) > 1e-9
            else 0.0
        )
        return _WindowStats(
            mean=float(mean),
            std=float(std),
            max=float(max_entry),
            position_in_range=float(position),
            zscore_vs_mean=float(zscore),
            trend_slope_per_hour=float(slope),
        )


class _RollingEpisodeCounter:
    def __init__(self, window_sec: float):
        self.window_sec = float(window_sec)
        self.active: deque[float] = deque()
        self.cursor = 0

    def advance(self, episodes: list[NormalizedEpisode], current_ts: float) -> int:
        current_ts = float(current_ts)
        while self.cursor < len(episodes) and float(episodes[self.cursor].end_ts) <= current_ts:
            self.active.append(float(episodes[self.cursor].end_ts))
            self.cursor += 1
        cutoff = current_ts - self.window_sec
        while self.active and self.active[0] < cutoff:
            self.active.popleft()
        return len(self.active)


_EPS = 0.01  # Floor for all denominators to avoid inf/nan


class _ExitWindowStats(NamedTuple):
    mean: float
    std: float
    p10: float


class _RollingExitWindow:
    """Rolling window for exit spread stats. Exit values are NEGATIVE when inverted."""

    def __init__(self, window_sec: float):
        self.window_sec = float(window_sec)
        self.points: deque[tuple[float, float]] = deque()  # (ts, exit_spread)
        self.sum_y = 0.0
        self.sum_y2 = 0.0

    def push(self, ts: float, exit_spread: float) -> None:
        self.points.append((float(ts), float(exit_spread)))
        self.sum_y += float(exit_spread)
        self.sum_y2 += float(exit_spread) * float(exit_spread)
        self._expire(float(ts) - self.window_sec)

    def _expire(self, cutoff: float) -> None:
        while self.points and self.points[0][0] < cutoff:
            _, val = self.points.popleft()
            self.sum_y -= val
            self.sum_y2 -= val * val

    def stats(self) -> _ExitWindowStats:
        count = len(self.points)
        if count <= 0:
            return _ExitWindowStats(0.0, 0.0, 0.0)
        mean = self.sum_y / count
        variance = max((self.sum_y2 / count) - (mean * mean), 0.0)
        std = math.sqrt(variance)
        # P10: 10th percentile — sort current buffer (bounded size, max ~1920 for 8h@15s)
        if count <= 2:
            p10 = mean
        else:
            sorted_vals = sorted(v for _, v in self.points)
            idx = max(0, int(count * 0.10) - 1)
            p10 = sorted_vals[idx]
        return _ExitWindowStats(mean=float(mean), std=float(std), p10=float(p10))


class _EpisodeRollingStats(NamedTuple):
    mean_total_spread: float
    median_duration_sec: float
    close_rate: float
    mean_reversion_speed: float


class _RollingEpisodeStats:
    """Rolling window stats over closed episodes (ALL episodes, no min_total_spread filter).

    Coorte matura for close_rate: only episodes with start_ts <= current_ts - maturation_sec
    count in the denominator, avoiding censoring bias from recently-started episodes.
    """

    _DEFAULT_MATURATION_SEC = 2 * 60 * 60  # 2h

    def __init__(self, window_sec: float, maturation_sec: float = 0.0):
        self.window_sec = float(window_sec)
        self.maturation_sec = float(maturation_sec) if maturation_sec > 0 else self._DEFAULT_MATURATION_SEC
        self.episodes: deque[NormalizedEpisode] = deque()
        self.cursor = 0

    def advance(self, all_episodes: list[NormalizedEpisode], current_ts: float) -> None:
        """Ingest episodes whose end_ts <= current_ts, expire old ones."""
        current_ts = float(current_ts)
        while self.cursor < len(all_episodes) and float(all_episodes[self.cursor].end_ts) <= current_ts:
            self.episodes.append(all_episodes[self.cursor])
            self.cursor += 1
        cutoff = current_ts - self.window_sec
        while self.episodes and float(self.episodes[0].end_ts) < cutoff:
            self.episodes.popleft()

    def stats(self, current_ts: float) -> _EpisodeRollingStats:
        if not self.episodes:
            return _EpisodeRollingStats(0.0, 0.0, 0.0, 0.0)

        total_spreads: list[float] = []
        durations: list[float] = []
        reversion_speeds: list[float] = []

        for ep in self.episodes:
            if not ep.is_closed:
                continue
            total_spreads.append(float(ep.total_spread))
            dur = float(ep.duration_sec)
            durations.append(dur)
            if dur > 0.0:
                speed_per_hour = float(ep.total_spread) / (dur / 3600.0)
                reversion_speeds.append(speed_per_hour)

        mean_ts = sum(total_spreads) / len(total_spreads) if total_spreads else 0.0
        if durations:
            sorted_dur = sorted(durations)
            mid = len(sorted_dur) // 2
            median_dur = sorted_dur[mid] if len(sorted_dur) % 2 else (sorted_dur[mid - 1] + sorted_dur[mid]) / 2.0
        else:
            median_dur = 0.0
        mean_rev = sum(reversion_speeds) / len(reversion_speeds) if reversion_speeds else 0.0

        # Coorte matura close_rate: closed episodes whose start_ts is old enough
        maturation_cutoff = float(current_ts) - self.maturation_sec
        mature_closed = sum(1 for ep in self.episodes if ep.is_closed and float(ep.start_ts) <= maturation_cutoff)
        # All episodes in window started before maturation cutoff (denominator)
        mature_total = sum(1 for ep in self.episodes if float(ep.start_ts) <= maturation_cutoff)
        close_rate = float(mature_closed) / max(mature_total, 1) if mature_total > 0 else 0.0

        return _EpisodeRollingStats(
            mean_total_spread=float(mean_ts),
            median_duration_sec=float(median_dur),
            close_rate=float(close_rate),
            mean_reversion_speed=float(mean_rev),
        )


def feature_schema_hash(feature_names: list[str]) -> str:
    return hashlib.sha1(",".join(str(name) for name in feature_names).encode("utf-8")).hexdigest()


def feature_contract_version_for_names(feature_names: list[str]) -> str | None:
    normalized = list(feature_names or [])
    for version, expected in FEATURE_CONTRACTS.items():
        if normalized == list(expected):
            return version
    return None


def supported_feature_contract_versions() -> list[str]:
    return list(FEATURE_CONTRACTS.keys())


def resolve_feature_contract(feature_names: list[str] | None = None, *, version: str | None = None) -> tuple[str, list[str]]:
    if version:
        expected = FEATURE_CONTRACTS.get(str(version))
        if expected is None:
            raise ValueError(f"Unsupported feature contract version: {version}")
        if feature_names is not None and list(feature_names) != list(expected):
            raise ValueError(f"Feature names do not match contract {version}")
        return str(version), list(expected)
    if feature_names is None:
        return DEFAULT_FEATURE_CONTRACT_VERSION, list(FEATURE_NAMES)
    resolved = feature_contract_version_for_names(list(feature_names))
    if resolved is None:
        raise ValueError("Unsupported feature schema")
    return resolved, list(FEATURE_CONTRACTS[resolved])


def normalize_record(record: Any, fallback_ts: float = 0.0) -> NormalizedRecord:
    if isinstance(record, NormalizedRecord):
        return record
    if isinstance(record, dict):
        # Fast path for canonical keys (loaded from optimized SQLite)
        ts_val = record.get("timestamp")
        if ts_val is not None:
            return NormalizedRecord(
                timestamp=float(ts_val),
                entry_spread=float(record.get("entry_spread") or 0.0),
                exit_spread=float(record.get("exit_spread") or 0.0),
                session_id=int(record.get("session_id") or 0),
                block_id=int(record.get("block_id") or 0),
            )
        # Legacy keys fallback
        return NormalizedRecord(
            timestamp=_coerce_float(record.get("ts", fallback_ts), fallback_ts),
            entry_spread=_coerce_float(record.get("entry", record.get("entry_spread_pct", 0.0)), 0.0),
            exit_spread=_coerce_float(record.get("exit", record.get("exit_spread_pct", 0.0)), 0.0),
            session_id=int(record.get("session_id") or 0),
            block_id=int(record.get("block_id") or 0),
        )
    timestamp = _coerce_float(getattr(record, "timestamp", getattr(record, "ts", fallback_ts)), fallback_ts)
    entry_spread = _coerce_float(
        getattr(record, "entry_spread", getattr(record, "entry_spread_pct", getattr(record, "entry", 0.0))),
        0.0,
    )
    exit_spread = _coerce_float(
        getattr(record, "exit_spread", getattr(record, "exit_spread_pct", getattr(record, "exit", 0.0))),
        0.0,
    )
    return NormalizedRecord(
        timestamp=float(timestamp),
        entry_spread=float(entry_spread),
        exit_spread=float(exit_spread),
        session_id=int(getattr(record, "session_id", 0) or 0),
        block_id=int(getattr(record, "block_id", 0) or 0),
    )


def normalize_episode(episode: Any) -> NormalizedEpisode:
    if isinstance(episode, NormalizedEpisode):
        return episode
    if isinstance(episode, dict):
        return NormalizedEpisode(
            end_ts=_coerce_float(episode.get("end_ts", 0.0), 0.0),
            start_ts=_coerce_float(episode.get("start_ts", 0.0), 0.0),
            exit_spread=_coerce_float(episode.get("exit_spread_at_close", episode.get("exit_spread", 0.0)), 0.0),
            duration_sec=_coerce_float(episode.get("duration_sec", 0.0), 0.0),
            peak_entry_spread=_coerce_float(episode.get("peak_entry_spread", 0.0), 0.0),
            total_spread=_coerce_float(episode.get("total_spread", 0.0), 0.0),
            is_closed=bool(episode.get("is_closed", True)),
        )
    end_ts = _coerce_float(getattr(episode, "end_ts", 0.0), 0.0)
    start_ts = _coerce_float(getattr(episode, "start_ts", 0.0), 0.0)
    exit_spread = _coerce_float(getattr(episode, "exit_spread_at_close", getattr(episode, "exit_spread", 0.0)), 0.0)
    duration_sec = _coerce_float(getattr(episode, "duration_sec", 0.0), 0.0)
    peak_entry = _coerce_float(getattr(episode, "peak_entry_spread", 0.0), 0.0)
    # total_spread: use property/method if available, else compute
    raw_total = getattr(episode, "total_spread", None)
    if callable(raw_total):
        total = _coerce_float(raw_total(), 0.0)
    elif raw_total is not None:
        total = _coerce_float(raw_total, 0.0)
    else:
        total = peak_entry + exit_spread
    is_closed = bool(getattr(episode, "is_closed", True))
    return NormalizedEpisode(
        end_ts=end_ts, start_ts=start_ts, exit_spread=exit_spread,
        duration_sec=duration_sec, peak_entry_spread=peak_entry,
        total_spread=total, is_closed=is_closed,
    )


def _build_feature_rows_vectorized(
    records: list[dict],
    episodes: list[NormalizedEpisode],
    *,
    contract_version: str = "v2_multiscale_25",
    cost_estimate_pct: float = 0.30,
) -> list[list[float]]:
    """Vectorized feature computation using numpy/pandas. ~50x faster than Python loops."""
    is_v3 = contract_version == "v3_exit_aware_40"
    n = len(records)
    timestamps = np.empty(n, dtype=np.float64)
    entries = np.empty(n, dtype=np.float64)
    exits = np.empty(n, dtype=np.float64)
    for i, r in enumerate(records):
        timestamps[i] = float(r["timestamp"])
        entries[i] = float(r["entry_spread"])
        exits[i] = float(r.get("exit_spread") or 0.0)

    # --- Micro features (10) ---
    delta_e = np.empty(n, dtype=np.float64)
    delta_e[0] = 0.0
    delta_e[1:] = np.diff(entries)
    delta_x = np.empty(n, dtype=np.float64)
    delta_x[0] = 0.0
    delta_x[1:] = np.diff(exits)
    delta2_e = np.empty(n, dtype=np.float64)
    delta2_e[0] = 0.0
    delta2_e[1:] = np.diff(delta_e)
    delta2_x = np.empty(n, dtype=np.float64)
    delta2_x[0] = 0.0
    delta2_x[1:] = np.diff(delta_x)

    # Rolling std/zscore (window=5, min_periods=2, population std ddof=0)
    se = pd.Series(entries)
    sx = pd.Series(exits)
    with np.errstate(invalid="ignore"):
        std_e = se.rolling(5, min_periods=2).std(ddof=0).fillna(0.0).values
        mean_e = se.rolling(5, min_periods=2).mean().values
        mean_e[0] = entries[0]
        z_e = np.where(std_e > 1e-9, (entries - mean_e) / std_e, 0.0)
        std_x = sx.rolling(5, min_periods=2).std(ddof=0).fillna(0.0).values
        mean_x = sx.rolling(5, min_periods=2).mean().values
        mean_x[0] = exits[0]
        z_x = np.where(std_x > 1e-9, (exits - mean_x) / std_x, 0.0)

    # --- Multiscale features (15) ---
    dt_index = pd.to_datetime(timestamps, unit="s")
    # Use absolute t_hr to match the Python _RollingEntryWindow implementation
    t_hr = timestamps / 3600.0
    df = pd.DataFrame(
        {"entry": entries, "t_hr": t_hr, "t_hr2": t_hr * t_hr, "ty": t_hr * entries},
        index=dt_index,
    )

    window_secs = {"30m": 1800, "2h": 7200, "8h": 28800}
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    maxs: dict[str, np.ndarray] = {}
    mins: dict[str, np.ndarray] = {}
    slopes: dict[str, np.ndarray] = {}
    ep_counts: dict[str, np.ndarray] = {}

    with np.errstate(invalid="ignore", divide="ignore"):
        for name, sec in window_secs.items():
            win = f"{sec}s"
            r = df["entry"].rolling(win, min_periods=1)
            means[name] = r.mean().values
            stds[name] = r.std(ddof=0).fillna(0.0).values
            maxs[name] = r.max().values
            mins[name] = r.min().values
            # Trend slope via rolling linear regression sums
            r_all = df.rolling(win, min_periods=1)
            count = r_all["entry"].count().values
            sum_y = r_all["entry"].sum().values
            sum_t = r_all["t_hr"].sum().values
            sum_t2 = r_all["t_hr2"].sum().values
            sum_ty = r_all["ty"].sum().values
            denom = count * sum_t2 - sum_t * sum_t
            slopes[name] = np.where(np.abs(denom) > 1e-9, (count * sum_ty - sum_t * sum_y) / denom, 0.0)

        # Position in range (30m)
        range_30m = maxs["30m"] - mins["30m"]
        position_30m = np.where(range_30m > 1e-9, (entries - mins["30m"]) / range_30m, 0.0)

        # Zscore vs 8h mean
        zscore_8h = np.where(stds["8h"] > 1e-9, (entries - means["8h"]) / stds["8h"], 0.0)

    # Episode counts via searchsorted
    ep_end_ts = np.array(sorted(float(e.end_ts) for e in episodes), dtype=np.float64) if episodes else np.array([], dtype=np.float64)
    for name, sec in window_secs.items():
        if ep_end_ts.size > 0:
            right = np.searchsorted(ep_end_ts, timestamps, side="right")
            left = np.searchsorted(ep_end_ts, timestamps - sec, side="left")
            ep_counts[name] = (right - left).astype(np.float64)
        else:
            ep_counts[name] = np.zeros(n, dtype=np.float64)

    # Assemble v2 columns (25): must match V2_MULTISCALE_FEATURE_NAMES order exactly
    columns = [
        entries, exits, delta_e, delta_x, delta2_e, delta2_x,
        std_e, std_x, z_e, z_x,
        means["30m"], stds["30m"], maxs["30m"], ep_counts["30m"], position_30m,
        means["2h"], stds["2h"], maxs["2h"], ep_counts["2h"], slopes["2h"],
        means["8h"], stds["8h"], maxs["8h"], ep_counts["8h"], zscore_8h,
    ]

    # --- V3 exit-aware features (15) ---
    if is_v3:
        df_exit = pd.DataFrame({"exit": exits}, index=dt_index)
        exit_means: dict[str, np.ndarray] = {}
        exit_stds: dict[str, np.ndarray] = {}
        exit_p10s: dict[str, np.ndarray] = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for name, sec in [("2h", 7200), ("8h", 28800)]:
                win = f"{sec}s"
                r_exit = df_exit["exit"].rolling(win, min_periods=1)
                exit_means[name] = r_exit.mean().values
                exit_stds[name] = r_exit.std(ddof=0).fillna(0.0).values
                exit_p10s[name] = r_exit.quantile(0.10).fillna(0.0).values
            zscore_exit_8h_vec = np.where(
                exit_stds["8h"] > _EPS,
                (exits - exit_means["8h"]) / np.maximum(exit_stds["8h"], _EPS),
                0.0,
            )

        # --- Cycle features (6) via episode arrays ---
        # Pre-compute episode data as sorted numpy arrays
        closed_episodes = [ep for ep in episodes if ep.is_closed]
        if closed_episodes:
            ep_end_arr = np.array([float(e.end_ts) for e in closed_episodes], dtype=np.float64)
            ep_start_arr = np.array([float(e.start_ts) for e in closed_episodes], dtype=np.float64)
            ep_total_arr = np.array([float(e.total_spread) for e in closed_episodes], dtype=np.float64)
            ep_dur_arr = np.array([float(e.duration_sec) for e in closed_episodes], dtype=np.float64)
            # Sort by end_ts for searchsorted
            sort_idx = np.argsort(ep_end_arr)
            ep_end_sorted = ep_end_arr[sort_idx]
            ep_start_sorted = ep_start_arr[sort_idx]
            ep_total_sorted = ep_total_arr[sort_idx]
            ep_dur_sorted = ep_dur_arr[sort_idx]
        else:
            ep_end_sorted = np.array([], dtype=np.float64)
            ep_start_sorted = np.array([], dtype=np.float64)
            ep_total_sorted = np.array([], dtype=np.float64)
            ep_dur_sorted = np.array([], dtype=np.float64)

        mean_total_2h = np.zeros(n, dtype=np.float64)
        median_dur_2h = np.zeros(n, dtype=np.float64)
        close_rate_2h = np.zeros(n, dtype=np.float64)
        mean_total_8h = np.zeros(n, dtype=np.float64)
        median_dur_8h = np.zeros(n, dtype=np.float64)
        mean_rev_speed_8h = np.zeros(n, dtype=np.float64)

        maturation_sec = 2 * 60 * 60  # 2h maturation for close_rate

        if ep_end_sorted.size > 0:
            for i_rec in range(n):
                ts_now = timestamps[i_rec]
                for name, sec, out_mean, out_dur, out_rate_or_speed, is_rate in [
                    ("2h", 7200, mean_total_2h, median_dur_2h, close_rate_2h, True),
                    ("8h", 28800, mean_total_8h, median_dur_8h, mean_rev_speed_8h, False),
                ]:
                    right = int(np.searchsorted(ep_end_sorted, ts_now, side="right"))
                    left = int(np.searchsorted(ep_end_sorted, ts_now - sec, side="left"))
                    if right > left:
                        window_totals = ep_total_sorted[left:right]
                        window_durs = ep_dur_sorted[left:right]
                        window_starts = ep_start_sorted[left:right]
                        out_mean[i_rec] = float(np.mean(window_totals))
                        out_dur[i_rec] = float(np.median(window_durs))
                        if is_rate:
                            # Coorte matura close_rate
                            mature_mask = window_starts <= (ts_now - maturation_sec)
                            mature_count = int(np.sum(mature_mask))
                            out_rate_or_speed[i_rec] = 1.0 if mature_count > 0 else 0.0
                            # All episodes in window are closed (from closed_episodes),
                            # so close_rate for mature coorte = mature_closed / mature_total = 1.0
                            # unless we also track unclosed episodes. Since we only have closed,
                            # close_rate is the ratio of mature closed episodes to all in window.
                            if mature_count > 0:
                                out_rate_or_speed[i_rec] = float(mature_count) / float(right - left)
                        else:
                            # Mean reversion speed (total_spread / duration_hours)
                            valid = window_durs > 0.0
                            if np.any(valid):
                                speeds = window_totals[valid] / (window_durs[valid] / 3600.0)
                                out_rate_or_speed[i_rec] = float(np.mean(speeds))

        # --- Relation features (3) ---
        current_spread_total_vec = entries + exits  # NO abs
        entry_exit_ratio_vec = np.where(
            np.abs(exit_means["2h"]) > _EPS,
            means["2h"] / np.maximum(np.abs(exit_means["2h"]), _EPS),
            0.0,
        )
        viability_vec = np.where(
            stds["2h"] > _EPS,
            (entries + exit_means["2h"] - cost_estimate_pct) / np.maximum(stds["2h"], _EPS),
            0.0,
        )

        columns.extend([
            exit_means["2h"], exit_p10s["2h"], exit_stds["2h"],
            exit_means["8h"], exit_p10s["8h"], zscore_exit_8h_vec,
            mean_total_2h, median_dur_2h, close_rate_2h,
            mean_total_8h, median_dur_8h, mean_rev_speed_8h,
            current_spread_total_vec, entry_exit_ratio_vec, viability_vec,
        ])

    result = np.column_stack(columns)
    return result.tolist()


def build_feature_rows(
    records: list[Any],
    *,
    feature_names: list[str] | None = None,
    feature_contract_version: str | None = None,
    episodes: list[Any] | None = None,
    cost_estimate_pct: float = 0.30,
) -> list[list[float]]:
    contract_version, expected_names = resolve_feature_contract(feature_names, version=feature_contract_version)
    if not records:
        return []
    first = records[0]
    # Vectorized path: canonical dict records with enough volume for numpy/pandas to pay off
    if (
        _VECTORIZE_AVAILABLE
        and isinstance(first, dict)
        and "timestamp" in first
        and len(records) >= _VECTORIZE_MIN_RECORDS
        and contract_version in ("v2_multiscale_25", "v3_exit_aware_40")
    ):
        normalized_episodes = sorted(
            [normalize_episode(episode) for episode in list(episodes or [])],
            key=lambda item: item.end_ts,
        )
        return _build_feature_rows_vectorized(
            records, normalized_episodes,
            contract_version=contract_version, cost_estimate_pct=cost_estimate_pct,
        )
    # Fast path: canonical dict records (Python loop, avoids NormalizedRecord allocations)
    if isinstance(first, dict) and "timestamp" in first:
        micro_features = _build_micro_feature_lists_dicts(records)
        if contract_version == "v1_micro_10":
            return micro_features
        normalized_episodes = sorted(
            [normalize_episode(episode) for episode in list(episodes or [])],
            key=lambda item: item.end_ts,
        )
        multiscale_features = _build_multiscale_feature_lists_dicts(
            records, normalized_episodes,
            contract_version=contract_version, cost_estimate_pct=cost_estimate_pct,
        )
        return [micro + multi for micro, multi in zip(micro_features, multiscale_features)]
    # Slow path: needs NormalizedRecord conversion + sort
    normalized_records = [
        normalize_record(record, fallback_ts=float(index))
        for index, record in enumerate(records)
    ]
    normalized_records.sort(key=lambda item: item.timestamp)
    micro_features = _build_micro_feature_lists(normalized_records)
    if contract_version == "v1_micro_10":
        return micro_features
    normalized_episodes = sorted(
        [normalize_episode(episode) for episode in list(episodes or [])],
        key=lambda item: item.end_ts,
    )
    multiscale_features = _build_multiscale_feature_lists(
        normalized_records, normalized_episodes,
        contract_version=contract_version, cost_estimate_pct=cost_estimate_pct,
    )
    return [micro + multi for micro, multi in zip(micro_features, multiscale_features)]


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _rolling_std_zscore(buffer: deque[float], current: float) -> tuple[float, float]:
    if len(buffer) < 2:
        return 0.0, 0.0
    mean = sum(buffer) / len(buffer)
    variance = sum((value - mean) ** 2 for value in buffer) / len(buffer)
    std = math.sqrt(max(variance, 0.0))
    if std < 1e-9:
        return 0.0, 0.0
    return float(std), float((current - mean) / std)


def _build_micro_feature_lists(records: list[NormalizedRecord]) -> list[list[float]]:
    """Return micro features as lists (order matches V1_MICRO_FEATURE_NAMES)."""
    rows: list[list[float]] = []
    previous_entry: float | None = None
    previous_exit: float | None = None
    previous_delta_entry: float | None = None
    previous_delta_exit: float | None = None
    entry_buffer: deque[float] = deque(maxlen=_MICRO_ROLLING_WINDOW)
    exit_buffer: deque[float] = deque(maxlen=_MICRO_ROLLING_WINDOW)
    for record in records:
        entry_spread = float(record.entry_spread)
        exit_spread = float(record.exit_spread)
        delta_entry = 0.0 if previous_entry is None else entry_spread - previous_entry
        delta_exit = 0.0 if previous_exit is None else exit_spread - previous_exit
        delta2_entry = 0.0 if previous_delta_entry is None else delta_entry - previous_delta_entry
        delta2_exit = 0.0 if previous_delta_exit is None else delta_exit - previous_delta_exit
        entry_buffer.append(entry_spread)
        exit_buffer.append(exit_spread)
        rolling_std_entry, zscore_entry = _rolling_std_zscore(entry_buffer, entry_spread)
        rolling_std_exit, zscore_exit = _rolling_std_zscore(exit_buffer, exit_spread)
        rows.append([
            entry_spread,
            exit_spread,
            delta_entry,
            delta_exit,
            delta2_entry,
            delta2_exit,
            rolling_std_entry,
            rolling_std_exit,
            zscore_entry,
            zscore_exit,
        ])
        previous_entry = entry_spread
        previous_exit = exit_spread
        previous_delta_entry = delta_entry
        previous_delta_exit = delta_exit
    return rows


def _build_micro_feature_lists_dicts(records: list[dict]) -> list[list[float]]:
    """Micro features from dict records — avoids NormalizedRecord allocation."""
    rows: list[list[float]] = []
    previous_entry: float | None = None
    previous_exit: float | None = None
    previous_delta_entry: float | None = None
    previous_delta_exit: float | None = None
    entry_buffer: deque[float] = deque(maxlen=_MICRO_ROLLING_WINDOW)
    exit_buffer: deque[float] = deque(maxlen=_MICRO_ROLLING_WINDOW)
    for record in records:
        entry_spread = float(record["entry_spread"])
        exit_spread = float(record.get("exit_spread") or 0.0)
        delta_entry = 0.0 if previous_entry is None else entry_spread - previous_entry
        delta_exit = 0.0 if previous_exit is None else exit_spread - previous_exit
        delta2_entry = 0.0 if previous_delta_entry is None else delta_entry - previous_delta_entry
        delta2_exit = 0.0 if previous_delta_exit is None else delta_exit - previous_delta_exit
        entry_buffer.append(entry_spread)
        exit_buffer.append(exit_spread)
        rolling_std_entry, zscore_entry = _rolling_std_zscore(entry_buffer, entry_spread)
        rolling_std_exit, zscore_exit = _rolling_std_zscore(exit_buffer, exit_spread)
        rows.append([
            entry_spread, exit_spread,
            delta_entry, delta_exit,
            delta2_entry, delta2_exit,
            rolling_std_entry, rolling_std_exit,
            zscore_entry, zscore_exit,
        ])
        previous_entry = entry_spread
        previous_exit = exit_spread
        previous_delta_entry = delta_entry
        previous_delta_exit = delta_exit
    return rows


def _build_multiscale_feature_lists_dicts(
    records: list[dict],
    episodes: list[NormalizedEpisode],
    *,
    contract_version: str = "v2_multiscale_25",
    cost_estimate_pct: float = 0.30,
) -> list[list[float]]:
    """Multiscale features from dict records — avoids NormalizedRecord allocation.

    When contract_version is v3_exit_aware_40, appends 15 exit/cycle/relation features.
    """
    is_v3 = contract_version == "v3_exit_aware_40"
    entry_windows = {
        name: _RollingEntryWindow(seconds)
        for name, seconds in _MULTISCALE_WINDOWS.items()
    }
    episode_windows = {
        name: _RollingEpisodeCounter(seconds)
        for name, seconds in _MULTISCALE_WINDOWS.items()
    }
    # V3 rolling windows for exit spreads and episode stats
    exit_windows: dict[str, _RollingExitWindow] = {}
    episode_stats: dict[str, _RollingEpisodeStats] = {}
    if is_v3:
        exit_windows = {
            "2h": _RollingExitWindow(2 * 60 * 60),
            "8h": _RollingExitWindow(8 * 60 * 60),
        }
        episode_stats = {
            "2h": _RollingEpisodeStats(2 * 60 * 60),
            "8h": _RollingEpisodeStats(8 * 60 * 60),
        }
    rows: list[list[float]] = []
    for record in records:
        current_ts = float(record["timestamp"])
        current_entry = float(record["entry_spread"])
        current_exit = float(record.get("exit_spread") or 0.0)
        for window in entry_windows.values():
            window.push(current_ts, current_entry)
        counts = {
            name: episode_windows[name].advance(episodes, current_ts)
            for name in episode_windows
        }
        s30 = entry_windows["30m"].stats(current_entry)
        s2h = entry_windows["2h"].stats(current_entry)
        s8h = entry_windows["8h"].stats(current_entry)
        row = [
            s30.mean, s30.std, s30.max, float(counts["30m"]), s30.position_in_range,
            s2h.mean, s2h.std, s2h.max, float(counts["2h"]), s2h.trend_slope_per_hour,
            s8h.mean, s8h.std, s8h.max, float(counts["8h"]), s8h.zscore_vs_mean,
        ]
        if is_v3:
            # Push exit windows
            for ew in exit_windows.values():
                ew.push(current_ts, current_exit)
            for es in episode_stats.values():
                es.advance(episodes, current_ts)
            ex2h = exit_windows["2h"].stats()
            ex8h = exit_windows["8h"].stats()
            ep2h = episode_stats["2h"].stats(current_ts)
            ep8h = episode_stats["8h"].stats(current_ts)
            # Exit multi-scale (6)
            zscore_exit_8h = (current_exit - ex8h.mean) / max(ex8h.std, _EPS)
            # Cycle / profitability (6) — already computed by _RollingEpisodeStats
            # Entry-exit relation (3)
            current_spread_total = current_entry + current_exit  # NO abs
            entry_exit_ratio = s2h.mean / max(abs(ex2h.mean), _EPS)
            viability = (current_entry + ex2h.mean - cost_estimate_pct) / max(s2h.std, _EPS)
            row.extend([
                ex2h.mean, ex2h.p10, ex2h.std,
                ex8h.mean, ex8h.p10, zscore_exit_8h,
                ep2h.mean_total_spread, ep2h.median_duration_sec, ep2h.close_rate,
                ep8h.mean_total_spread, ep8h.median_duration_sec, ep8h.mean_reversion_speed,
                current_spread_total, entry_exit_ratio, viability,
            ])
        rows.append(row)
    return rows


def _build_multiscale_feature_lists(
    records: list[NormalizedRecord],
    episodes: list[NormalizedEpisode],
    *,
    contract_version: str = "v2_multiscale_25",
    cost_estimate_pct: float = 0.30,
) -> list[list[float]]:
    """Return multiscale features as lists (order matches V2 extra names).

    When contract_version is v3_exit_aware_40, appends 15 exit/cycle/relation features.
    """
    is_v3 = contract_version == "v3_exit_aware_40"
    entry_windows = {
        name: _RollingEntryWindow(seconds)
        for name, seconds in _MULTISCALE_WINDOWS.items()
    }
    episode_windows = {
        name: _RollingEpisodeCounter(seconds)
        for name, seconds in _MULTISCALE_WINDOWS.items()
    }
    exit_windows: dict[str, _RollingExitWindow] = {}
    episode_stats_windows: dict[str, _RollingEpisodeStats] = {}
    if is_v3:
        exit_windows = {
            "2h": _RollingExitWindow(2 * 60 * 60),
            "8h": _RollingExitWindow(8 * 60 * 60),
        }
        episode_stats_windows = {
            "2h": _RollingEpisodeStats(2 * 60 * 60),
            "8h": _RollingEpisodeStats(8 * 60 * 60),
        }
    rows: list[list[float]] = []
    for record in records:
        current_ts = float(record.timestamp)
        current_entry = float(record.entry_spread)
        current_exit = float(record.exit_spread)
        for window in entry_windows.values():
            window.push(current_ts, current_entry)
        counts = {
            name: episode_windows[name].advance(episodes, current_ts)
            for name in episode_windows
        }
        s30 = entry_windows["30m"].stats(current_entry)
        s2h = entry_windows["2h"].stats(current_entry)
        s8h = entry_windows["8h"].stats(current_entry)
        row = [
            s30.mean, s30.std, s30.max, float(counts["30m"]), s30.position_in_range,
            s2h.mean, s2h.std, s2h.max, float(counts["2h"]), s2h.trend_slope_per_hour,
            s8h.mean, s8h.std, s8h.max, float(counts["8h"]), s8h.zscore_vs_mean,
        ]
        if is_v3:
            for ew in exit_windows.values():
                ew.push(current_ts, current_exit)
            for es in episode_stats_windows.values():
                es.advance(episodes, current_ts)
            ex2h = exit_windows["2h"].stats()
            ex8h = exit_windows["8h"].stats()
            ep2h = episode_stats_windows["2h"].stats(current_ts)
            ep8h = episode_stats_windows["8h"].stats(current_ts)
            zscore_exit_8h = (current_exit - ex8h.mean) / max(ex8h.std, _EPS)
            current_spread_total = current_entry + current_exit
            entry_exit_ratio = s2h.mean / max(abs(ex2h.mean), _EPS)
            viability = (current_entry + ex2h.mean - cost_estimate_pct) / max(s2h.std, _EPS)
            row.extend([
                ex2h.mean, ex2h.p10, ex2h.std,
                ex8h.mean, ex8h.p10, zscore_exit_8h,
                ep2h.mean_total_spread, ep2h.median_duration_sec, ep2h.close_rate,
                ep8h.mean_total_spread, ep8h.median_duration_sec, ep8h.mean_reversion_speed,
                current_spread_total, entry_exit_ratio, viability,
            ])
        rows.append(row)
    return rows
