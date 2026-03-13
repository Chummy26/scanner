from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, NamedTuple


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

DEFAULT_FEATURE_CONTRACT_VERSION = "v2_multiscale_25"

FEATURE_CONTRACTS: dict[str, list[str]] = {
    "v1_micro_10": list(V1_MICRO_FEATURE_NAMES),
    "v2_multiscale_25": list(V2_MULTISCALE_FEATURE_NAMES),
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
        return NormalizedEpisode(end_ts=_coerce_float(episode.get("end_ts", 0.0), 0.0))
    return NormalizedEpisode(end_ts=_coerce_float(getattr(episode, "end_ts", 0.0), 0.0))


def build_feature_rows(
    records: list[Any],
    *,
    feature_names: list[str] | None = None,
    feature_contract_version: str | None = None,
    episodes: list[Any] | None = None,
) -> list[list[float]]:
    contract_version, expected_names = resolve_feature_contract(feature_names, version=feature_contract_version)
    normalized_records = [
        normalize_record(record, fallback_ts=float(index))
        for index, record in enumerate(records)
    ]
    normalized_records.sort(key=lambda item: item.timestamp)
    if not normalized_records:
        return []
    micro_features = _build_micro_feature_lists(normalized_records)
    if contract_version == "v1_micro_10":
        return micro_features
    normalized_episodes = sorted(
        [normalize_episode(episode) for episode in list(episodes or [])],
        key=lambda item: item.end_ts,
    )
    multiscale_features = _build_multiscale_feature_lists(normalized_records, normalized_episodes)
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


def _build_multiscale_feature_lists(
    records: list[NormalizedRecord],
    episodes: list[NormalizedEpisode],
) -> list[list[float]]:
    """Return multiscale features as lists (order matches V2 extra names)."""
    entry_windows = {
        name: _RollingEntryWindow(seconds)
        for name, seconds in _MULTISCALE_WINDOWS.items()
    }
    episode_windows = {
        name: _RollingEpisodeCounter(seconds)
        for name, seconds in _MULTISCALE_WINDOWS.items()
    }
    rows: list[list[float]] = []
    for record in records:
        current_ts = float(record.timestamp)
        current_entry = float(record.entry_spread)
        for window in entry_windows.values():
            window.push(current_ts, current_entry)
        counts = {
            name: episode_windows[name].advance(episodes, current_ts)
            for name in episode_windows
        }
        s30 = entry_windows["30m"].stats(current_entry)
        s2h = entry_windows["2h"].stats(current_entry)
        s8h = entry_windows["8h"].stats(current_entry)
        rows.append([
            s30.mean,
            s30.std,
            s30.max,
            float(counts["30m"]),
            s30.position_in_range,
            s2h.mean,
            s2h.std,
            s2h.max,
            float(counts["2h"]),
            s2h.trend_slope_per_hour,
            s8h.mean,
            s8h.std,
            s8h.max,
            float(counts["8h"]),
            s8h.zscore_vs_mean,
        ])
    return rows
