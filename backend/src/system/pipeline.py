import time
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Tuple
from .models import Instrument, FundingData, Opportunity, set_fee_config, calc_net_estimated_pct
from .universe import UniverseManager
from .history import FundingHistoryManager

logger = logging.getLogger(__name__)

# Aligned to 1 minute to ensure we hit 11:55, 12:00, 12:05 etc.
NORMAL_INTERVAL_SEC = 60 

LOCAL_SUGGESTION_MIN_SPREAD_PCT = 0.05
LOCAL_SUGGESTION_STREAK_NEGATIVE_TOLERANCE_PCT = -0.08  # Keep streak if spread dips slightly negative.
LOCAL_SUGGESTION_TOP_N = 150
LOCAL_SUGGESTION_TOP_N_BOOST = 200
LOCAL_SUGGESTION_TOP_N_BOOST_MINUTES = {59, 0}
LOCAL_SUGGESTION_TTL_SEC = 6 * 3600
LOCAL_SUGGESTION_WINDOW_START_MIN = 50
LOCAL_SUGGESTION_WINDOW_END_MIN = 1
CRITICAL_SCAN_START_MIN = 57
LOCAL_SUGGESTION_SLOT_MINUTES = [50, 55, 57, 58, 59, 0, 1]
LOCAL_SUGGESTION_FOCUS_SLOTS = {59, 0}
LOCAL_SUGGESTION_SLOT_WEIGHTS = {
    50: 1.0,
    55: 1.0,
    57: 1.0,
    58: 1.5,
    59: 2.0,
    0: 2.0,
    1: 1.5
}
LOCAL_SUGGESTION_MIN_HITS_FLOOR = 3
LOCAL_SUGGESTION_SLOT_COUNT = len(LOCAL_SUGGESTION_SLOT_MINUTES)
LOCAL_SUGGESTION_SMART_WINDOW_COUNT = 3
LOCAL_SUGGESTION_SMART_REQUIRED_OK = 2
LOCAL_SUGGESTION_SMART_MIN_AVG_SPREAD_PCT = LOCAL_SUGGESTION_MIN_SPREAD_PCT
LOCAL_SUGGESTION_STICKY_MAX_EXTRA = 50
LOCAL_SUGGESTION_MIN_KEEP_HOURS = 36
LOCAL_SUGGESTION_MEMORY_TTL_SEC = 7 * 24 * 3600
LOCAL_SUGGESTION_INVERSION_TTL_SEC = 3 * 24 * 3600
LOCAL_SUGGESTION_LIFETIME_DECAY = 0.96
LOCAL_SUGGESTION_RETURN_MIN_CONSISTENCY = 0.5
LOCAL_SUGGESTION_RETURN_MIN_TOTAL = 1.0
LOCAL_SUGGESTION_RETURN_MIN_SPREAD_PCT = 0.09
LOCAL_SUGGESTION_STREAK_GRACE_HOURS = 6
LOCAL_SUGGESTION_STREAK_CACHE_MIN = 2
ELITE_SUGGESTION_MIN_OK_WINDOWS = 3
ELITE_SUGGESTION_TTL_SEC = 3 * 24 * 3600
CRITICAL_WINDOW_MINUTES = {58, 59, 0, 1}
CRITICAL_ANALYSIS_TTL_SEC = 7 * 24 * 3600
SPOT_SHORT_MIN_RATE_PCT = LOCAL_SUGGESTION_MIN_SPREAD_PCT
SPOT_SHORT_MAX_ITEMS = 200
LOCAL_SUGGESTION_TOLERANCE_LEVELS = {
    "strict": LOCAL_SUGGESTION_SLOT_COUNT,
    "high": max(LOCAL_SUGGESTION_SLOT_COUNT - 1, 1),
    "medium": max(LOCAL_SUGGESTION_SLOT_COUNT - 2, 1),
    "low": max(LOCAL_SUGGESTION_SLOT_COUNT - 3, 1),
    "minimal": max(LOCAL_SUGGESTION_SLOT_COUNT - 4, 1)
}

class Pipeline:
    def __init__(self, universe_mgr: UniverseManager, out_dir: str = "out/prod"):
        self.universe_mgr = universe_mgr
        self.out_dir = out_dir
        self.running = False
        
        self.history_mgr = FundingHistoryManager(os.path.join(out_dir, "history", "funding"))
        self.memory_path = os.path.join(out_dir, "..", "config", "intelligence_memory.json")
        self.suggestion_config_path = os.path.join(out_dir, "..", "config", "suggestions_config.json")
        self.elite_path = os.path.join(out_dir, "history", "elite_suggestions.json")
        self.critical_path = os.path.join(out_dir, "history", "critical_analysis.json")
        self.funding_config_path = os.path.join(out_dir, "..", "..", "config", "funding_config.json")
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.elite_path), exist_ok=True)
        self.intelligence_memory = self._load_json(self.memory_path, {})
        self.critical_state = self._load_json(self.critical_path, {})
        if not isinstance(self.critical_state, dict):
            self.critical_state = {}
        if not isinstance(self.critical_state.get("items"), dict):
            self.critical_state["items"] = {}
        self.suggestion_tolerance_level, self.suggestion_required_hits = self._load_suggestion_tolerance()
        self.suggestion_streak_negative_tolerance_pct = self._load_suggestion_negative_tolerance()
        self.suggestion_streak_grace_hours = self._load_suggestion_streak_grace_hours()
        self._hydrate_intelligence_scores()

        fee_config = self._load_json(self.funding_config_path, {})
        set_fee_config(fee_config)
        
        os.makedirs(os.path.join(out_dir, "latest"), exist_ok=True)
        self.latest_funding_map: Dict[str, Dict[str, FundingData]] = {}
        self.current_snapshot: Dict[str, Any] = {}

    def _load_json(self, path: str, default: Any) -> Any:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return default

    def _atomic_write_json(self, path: str, data: Any):
        temp_path = f"{path}.tmp"
        max_retries = 3
        for i in range(max_retries):
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                # Check if destination exists and try to remove it if replace fails
                if os.path.exists(path):
                    try:
                        os.replace(temp_path, path)
                    except OSError:
                        time.sleep(0.5)
                        os.remove(path)
                        os.replace(temp_path, path)
                else:
                    os.replace(temp_path, path)
                return # Success
            except Exception as e:
                logger.error(f"Atomic write attempt {i+1} failed for {path}: {e}")
                time.sleep(1)
        
        if os.path.exists(temp_path): 
            try: os.remove(temp_path)
            except: pass

    def _load_suggestion_tolerance(self) -> tuple[str, int]:
        level = os.getenv("TEAM_OP_SUGGESTION_TOLERANCE")
        if level is None or not level.strip():
            level = os.getenv("GEMINI_SUGGESTION_TOLERANCE", "medium")
        level = level.strip().lower()
        min_hits = None
        config = self._load_json(self.suggestion_config_path, {})
        if isinstance(config, dict):
            if config.get("tolerance_level"):
                level = str(config.get("tolerance_level")).strip().lower()
            min_hits = config.get("min_hits")

        env_min_hits = os.getenv("TEAM_OP_SUGGESTION_MIN_HITS")
        if env_min_hits is None or not env_min_hits.strip():
            env_min_hits = os.getenv("GEMINI_SUGGESTION_MIN_HITS", "")
        env_min_hits = env_min_hits.strip()
        if env_min_hits.isdigit():
            min_hits = int(env_min_hits)

        if isinstance(min_hits, int):
            min_hits = max(1, min(min_hits, LOCAL_SUGGESTION_SLOT_COUNT))
            return "custom", min_hits

        if level not in LOCAL_SUGGESTION_TOLERANCE_LEVELS:
            level = "medium"
        return level, LOCAL_SUGGESTION_TOLERANCE_LEVELS[level]

    def _load_suggestion_negative_tolerance(self) -> float:
        default_value = LOCAL_SUGGESTION_STREAK_NEGATIVE_TOLERANCE_PCT
        config = self._load_json(self.suggestion_config_path, {})
        value = None
        if isinstance(config, dict):
            value = config.get("streak_negative_tolerance_pct")

        env_value = os.getenv("TEAM_OP_SUGGESTION_STREAK_NEGATIVE_TOLERANCE_PCT")
        if env_value is None or not env_value.strip():
            env_value = os.getenv("GEMINI_SUGGESTION_STREAK_NEGATIVE_TOLERANCE_PCT", "")
        if env_value and env_value.strip():
            value = env_value.strip()

        if value is None:
            return default_value
        try:
            return float(value)
        except (TypeError, ValueError):
            return default_value

    def _load_suggestion_streak_grace_hours(self) -> int:
        default_value = LOCAL_SUGGESTION_STREAK_GRACE_HOURS
        config = self._load_json(self.suggestion_config_path, {})
        value = None
        if isinstance(config, dict):
            value = config.get("streak_grace_hours")

        env_value = os.getenv("TEAM_OP_SUGGESTION_STREAK_GRACE_HOURS")
        if env_value is None or not env_value.strip():
            env_value = os.getenv("GEMINI_SUGGESTION_STREAK_GRACE_HOURS", "")
        if env_value and env_value.strip():
            value = env_value.strip()

        if value is None:
            return int(default_value)
        try:
            return max(1, int(float(value)))
        except (TypeError, ValueError):
            return int(default_value)

    def _hydrate_intelligence_scores(self):
        if not isinstance(self.intelligence_memory, dict):
            return
        changed = False
        for entry in self.intelligence_memory.values():
            if not isinstance(entry, dict):
                continue
            windows = entry.get("windows")
            if not isinstance(windows, dict) or not windows:
                continue
            required_hits = entry.get("tolerance_required_hits")
            if (not isinstance(required_hits, int)
                or entry.get("tolerance_level") != self.suggestion_tolerance_level
                or required_hits != self.suggestion_required_hits):
                required_hits = self.suggestion_required_hits
                entry["tolerance_level"] = self.suggestion_tolerance_level
                entry["tolerance_required_hits"] = required_hits
                changed = True
            min_hits_floor = max(LOCAL_SUGGESTION_MIN_HITS_FLOOR, required_hits - 1)
            items = []
            latest_dt = None
            latest_id = None
            for win_id, win in windows.items():
                if not isinstance(win, dict):
                    continue
                dt = self._parse_window_id(win_id)
                if not dt:
                    continue
                slots = set()
                for s in win.get("slots", []):
                    try:
                        slots.add(int(s))
                    except (TypeError, ValueError):
                        continue
                weighted_hits = self._get_weighted_hits(slots)
                win["weighted_hits"] = weighted_hits
                win["focus_hit"] = bool(slots & LOCAL_SUGGESTION_FOCUS_SLOTS)
                win["min_hits_floor"] = min_hits_floor
                spread_min_pct = self._get_window_min_spread_pct(win.get("slot_spreads"))
                spread_ok = True
                if spread_min_pct is not None:
                    spread_ok = spread_min_pct >= self.suggestion_streak_negative_tolerance_pct
                win["spread_min_pct"] = spread_min_pct
                win["spread_ok"] = spread_ok
                win["ok"] = len(slots) >= min_hits_floor and weighted_hits >= required_hits and spread_ok
                items.append((dt, win["ok"]))
                if latest_dt is None or dt > latest_dt:
                    latest_dt = dt
                    latest_id = win_id
            if not items:
                continue
            items.sort(key=lambda x: x[0])
            lifetime_ok = 0.0
            lifetime_total = 0.0
            for _, ok in items:
                lifetime_ok = lifetime_ok * LOCAL_SUGGESTION_LIFETIME_DECAY + (1.0 if ok else 0.0)
                lifetime_total = lifetime_total * LOCAL_SUGGESTION_LIFETIME_DECAY + 1.0
            entry["lifetime_ok"] = lifetime_ok
            entry["lifetime_total"] = lifetime_total
            entry["consistency_score"] = lifetime_ok / lifetime_total if lifetime_total else 0.0
        if changed:
            self._atomic_write_json(self.memory_path, self.intelligence_memory)
            if latest_id:
                entry["last_finalized_window_id"] = latest_id

    def _local_tz(self):
        return datetime.now().astimezone().tzinfo

    def _parse_window_id(self, window_id: str) -> Optional[datetime]:
        try:
            return datetime.strptime(window_id, "%Y%m%d%H").replace(tzinfo=self._local_tz())
        except Exception:
            return None

    def _get_suggestion_top_n(self, now_dt: Optional[datetime]) -> int:
        if now_dt and now_dt.minute in LOCAL_SUGGESTION_TOP_N_BOOST_MINUTES:
            return max(LOCAL_SUGGESTION_TOP_N, LOCAL_SUGGESTION_TOP_N_BOOST)
        return LOCAL_SUGGESTION_TOP_N

    def _get_min_hits_floor(self) -> int:
        return max(LOCAL_SUGGESTION_MIN_HITS_FLOOR, self.suggestion_required_hits - 1)

    def _get_weighted_hits(self, slots: set[int]) -> float:
        return sum(LOCAL_SUGGESTION_SLOT_WEIGHTS.get(slot, 1.0) for slot in slots)

    def _get_window_min_spread_pct(self, slot_spreads: Any) -> Optional[float]:
        values = []
        if isinstance(slot_spreads, dict):
            iterable = slot_spreads.values()
        elif isinstance(slot_spreads, (list, tuple)):
            iterable = slot_spreads
        else:
            return None
        for val in iterable:
            try:
                values.append(float(val))
            except (TypeError, ValueError):
                continue
        if not values:
            return None
        return min(values)

    def _update_lifetime_score(self, entry: Dict[str, Any], ok: bool):
        decay = LOCAL_SUGGESTION_LIFETIME_DECAY
        lifetime_ok = float(entry.get("lifetime_ok") or 0.0)
        lifetime_total = float(entry.get("lifetime_total") or 0.0)
        lifetime_ok = lifetime_ok * decay + (1.0 if ok else 0.0)
        lifetime_total = lifetime_total * decay + 1.0
        entry["lifetime_ok"] = lifetime_ok
        entry["lifetime_total"] = lifetime_total
        entry["consistency_score"] = lifetime_ok / lifetime_total if lifetime_total else 0.0

    def _finalize_previous_window(self, entry: Dict[str, Any], window_id: Optional[str]):
        if not window_id:
            return
        if entry.get("last_finalized_window_id") == window_id:
            return
        windows = entry.get("windows")
        if not isinstance(windows, dict):
            return
        win = windows.get(window_id)
        if not isinstance(win, dict):
            return
        self._update_lifetime_score(entry, bool(win.get("ok")))
        entry["last_finalized_window_id"] = window_id

    def _update_streak_cache(self, entry: Dict[str, Any], now_dt: datetime) -> int:
        current = self._get_consecutive_ok_windows(entry, now_dt=now_dt, allow_resume=False)
        prev = entry.get("streak_current")
        if isinstance(prev, int) and current < prev and prev >= LOCAL_SUGGESTION_STREAK_CACHE_MIN:
            entry["streak_cached"] = prev
            entry["streak_cached_at"] = time.time()
        entry["streak_current"] = current
        entry["streak_grace_hours"] = self.suggestion_streak_grace_hours
        return current

    def _get_recent_ok_ratio(self, entry: Dict[str, Any], max_windows: int) -> float:
        windows = entry.get("windows", {})
        if not isinstance(windows, dict) or max_windows <= 0:
            return 0.0
        pairs = []
        for win_id, win in windows.items():
            dt = self._parse_window_id(win_id)
            if not dt or not isinstance(win, dict):
                continue
            pairs.append((dt, bool(win.get("ok"))))
        if not pairs:
            return 0.0
        pairs.sort(key=lambda x: x[0], reverse=True)
        recent = pairs[:max_windows]
        if not recent:
            return 0.0
        ok_count = sum(1 for _, ok in recent if ok)
        return ok_count / len(recent)

    def _normalize_interval_hours(self, value: Optional[float]) -> float:
        try:
            hours = float(value)
        except (TypeError, ValueError):
            return 1.0
        return hours if hours > 0 else 1.0

    def _calc_next_funding_time(self, next_ts: Optional[int], interval_hours: Optional[float]) -> int:
        now_ms = int(time.time() * 1000)
        try:
            ts = int(float(next_ts)) if next_ts else None
        except (TypeError, ValueError):
            ts = None
        if ts and ts > now_ms:
            return ts
        hours = self._normalize_interval_hours(interval_hours or 8.0)
        sec = hours * 3600
        return int(((time.time() // sec) + 1) * sec * 1000)

    def _get_opportunity_interval_hours(self, opp: Opportunity) -> float:
        s_hours = self._normalize_interval_hours(opp.short_interval)
        l_hours = self._normalize_interval_hours(opp.long_interval)
        if abs(s_hours - l_hours) < 1e-6:
            return s_hours
        return max(s_hours, l_hours)

    def _get_interval_step_hours(self, interval_hours: float) -> int:
        if interval_hours <= 1:
            return 1
        step = int(round(interval_hours))
        return max(step, 1)

    def _get_entry_ttl_sec(self, interval_hours: Optional[float]) -> int:
        hours = self._normalize_interval_hours(interval_hours)
        base = LOCAL_SUGGESTION_TTL_SEC
        keep_hours = self._get_window_keep_hours(hours)
        ttl_hours = max(keep_hours, hours + 1)
        return max(base, int(ttl_hours * 3600))

    def _get_window_keep_hours(self, interval_hours: Optional[float]) -> int:
        hours = self._normalize_interval_hours(interval_hours)
        step_hours = self._get_interval_step_hours(hours)
        required_windows = max(ELITE_SUGGESTION_MIN_OK_WINDOWS, LOCAL_SUGGESTION_SMART_WINDOW_COUNT, 2)
        keep_span = (required_windows - 1) * step_hours + 1
        return max(LOCAL_SUGGESTION_MIN_KEEP_HOURS, 3, int(keep_span))

    def _is_window_aligned(self, opp: Opportunity, window_dt: Optional[datetime], interval_hours: float) -> bool:
        if interval_hours <= 1:
            return True
        if not window_dt:
            return True
        next_times = []
        for ts in (opp.short_funding.next_funding_time, opp.long_funding.next_funding_time):
            if isinstance(ts, (int, float)) and ts:
                next_times.append(int(ts))
        if not next_times:
            return True
        next_dt = datetime.fromtimestamp(min(next_times) / 1000, tz=window_dt.tzinfo)
        return next_dt.replace(minute=0, second=0, microsecond=0) == window_dt

    def _get_consecutive_ok_windows(self, entry: Dict[str, Any], now_dt: Optional[datetime] = None, allow_resume: bool = False) -> int:
        if now_dt is None:
            now_dt = datetime.now().astimezone()
        windows_raw = entry.get("windows")
        if not isinstance(windows_raw, dict) or not windows_raw:
            return 0
        step_hours = entry.get("interval_step_hours")
        if not isinstance(step_hours, (int, float)) or step_hours <= 0:
            step_hours = self._get_interval_step_hours(entry.get("interval_hours") or 1.0)
        items = []
        for win_id, win in windows_raw.items():
            if not isinstance(win, dict):
                continue
            dt = self._parse_window_id(win_id)
            if not dt:
                continue
            items.append((dt, bool(win.get("ok"))))
        if not items:
            return 0
        items.sort(key=lambda x: x[0])
        last_dt, last_ok = items[-1]
        if not last_ok:
            return 0
        grace_hours = entry.get("streak_grace_hours") or self.suggestion_streak_grace_hours
        try:
            grace_hours = max(step_hours, int(float(grace_hours)))
        except (TypeError, ValueError):
            grace_hours = max(step_hours, int(self.suggestion_streak_grace_hours))
        gap = now_dt - last_dt if last_dt else timedelta(hours=step_hours + 1)
        if last_dt and gap > timedelta(hours=step_hours):
            if not allow_resume:
                return 0
            cached = entry.get("streak_cached")
            cached_at = entry.get("streak_cached_at")
            if cached and cached_at and gap <= timedelta(hours=grace_hours):
                try:
                    if time.time() - float(cached_at) <= grace_hours * 3600:
                        return max(1, int(cached))
                except (TypeError, ValueError):
                    pass
            return 0
        count = 1
        prev_dt = last_dt
        for dt, ok in reversed(items[:-1]):
            if not ok:
                break
            if prev_dt - dt != timedelta(hours=step_hours):
                break
            count += 1
            prev_dt = dt
        if allow_resume:
            cached = entry.get("streak_cached")
            cached_at = entry.get("streak_cached_at")
            if cached and cached_at:
                try:
                    if time.time() - float(cached_at) <= grace_hours * 3600:
                        return max(count, int(cached))
                except (TypeError, ValueError):
                    pass
        return count

    def _get_previous_window_id(self, window_id: str) -> Optional[str]:
        dt = self._parse_window_id(window_id)
        if not dt:
            return None
        prev = dt - timedelta(hours=1)
        return prev.strftime("%Y%m%d%H")

    def _get_window_id(self, now_dt: datetime) -> Optional[str]:
        if now_dt.minute >= LOCAL_SUGGESTION_WINDOW_START_MIN:
            window_dt = (now_dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        elif now_dt.minute <= LOCAL_SUGGESTION_WINDOW_END_MIN:
            window_dt = now_dt.replace(minute=0, second=0, microsecond=0)
        else:
            return None
        return window_dt.strftime("%Y%m%d%H")

    def _filter_top_window_opps(self, opportunities: List[Opportunity], now_dt: Optional[datetime] = None) -> List[Opportunity]:
        min_spread_raw = LOCAL_SUGGESTION_MIN_SPREAD_PCT / 100.0
        top_n = self._get_suggestion_top_n(now_dt)
        unique = []
        seen = set()
        for o in opportunities:
            if not getattr(o, "is_valid", True):
                continue
            if o.spread_raw < min_spread_raw:
                continue
            key = (o.asset, o.short_inst.exchange, o.long_inst.exchange, o.short_inst.symbol, o.long_inst.symbol)
            if key in seen:
                continue
            seen.add(key)
            unique.append(o)
            if len(unique) >= top_n:
                break
        return unique

    def _get_recent_ok_windows(self, entry: Dict[str, Any], max_windows: int) -> List[str]:
        windows = entry.get("windows", {})
        if not isinstance(windows, dict) or max_windows <= 0:
            return []
        pairs = []
        for win_id, win in windows.items():
            dt = self._parse_window_id(win_id)
            if not dt:
                continue
            ok = bool(win.get("ok")) if isinstance(win, dict) else False
            pairs.append((dt, win_id, ok))
        pairs.sort(key=lambda x: x[0], reverse=True)
        recent = pairs[:max_windows]
        return [win_id for _, win_id, ok in recent if ok]

    def _get_intelligence_candidates(self, opportunities: List[Opportunity], base_candidates: List[Opportunity]) -> List[Opportunity]:
        if not self.intelligence_memory:
            return base_candidates
        now = time.time()
        sticky_keys = set()
        for key, entry in self.intelligence_memory.items():
            if not isinstance(entry, dict):
                continue
            data = entry.get("data") or {}
            if not data.get("is_valid", True):
                continue
            last_seen = entry.get("last_seen")
            ttl_sec = self._get_entry_ttl_sec(entry.get("interval_hours"))
            if not last_seen or (now - last_seen > ttl_sec):
                continue
            avg_spread = entry.get("spread_avg_pct")
            if avg_spread is None:
                avg_spread = entry.get("spread_last_pct")
            if avg_spread is None and not entry.get("qualified"):
                continue
            if avg_spread is not None and avg_spread < LOCAL_SUGGESTION_SMART_MIN_AVG_SPREAD_PCT and not entry.get("qualified"):
                continue
            sticky_keys.add(key)

        if not sticky_keys:
            return base_candidates

        min_spread_raw = LOCAL_SUGGESTION_MIN_SPREAD_PCT / 100.0
        if self.suggestion_streak_negative_tolerance_pct < 0:
            min_spread_raw = self.suggestion_streak_negative_tolerance_pct / 100.0
        seen_full = set(
            (o.asset, o.short_inst.exchange, o.long_inst.exchange, o.short_inst.symbol, o.long_inst.symbol)
            for o in base_candidates
        )
        existing_keys = set(
            f"{o.asset}_{o.short_inst.exchange}_{o.long_inst.exchange}"
            for o in base_candidates
        )
        extras = []
        for o in opportunities:
            if o.spread_raw < min_spread_raw:
                continue
            simple_key = f"{o.asset}_{o.short_inst.exchange}_{o.long_inst.exchange}"
            if simple_key not in sticky_keys or simple_key in existing_keys:
                continue
            full_key = (o.asset, o.short_inst.exchange, o.long_inst.exchange, o.short_inst.symbol, o.long_inst.symbol)
            if full_key in seen_full:
                continue
            extras.append(o)
            seen_full.add(full_key)
            existing_keys.add(simple_key)
            if len(extras) >= LOCAL_SUGGESTION_STICKY_MAX_EXTRA:
                break

        return base_candidates + extras

    def _prune_entry_windows(self, entry: Dict[str, Any], now_dt: datetime):
        windows = entry.get("windows")
        if not isinstance(windows, dict):
            return
        keep_hours = self._get_window_keep_hours(entry.get("interval_hours"))
        cutoff = now_dt - timedelta(hours=keep_hours)
        removed = False
        for win_id in list(windows.keys()):
            win_dt = self._parse_window_id(win_id)
            if not win_dt or win_dt < cutoff:
                windows.pop(win_id, None)
                removed = True
        if removed:
            self._update_spread_stats(entry)

    def _prune_inversion_events(self, entry: Dict[str, Any], now: float):
        events = entry.get("inversion_events")
        if not isinstance(events, list):
            return
        cutoff_ms = int((now - LOCAL_SUGGESTION_INVERSION_TTL_SEC) * 1000)
        filtered = []
        for item in events:
            try:
                ts = int(item)
            except (TypeError, ValueError):
                continue
            if ts >= cutoff_ms:
                filtered.append(ts)
        entry["inversion_events"] = filtered

    def _get_latest_rate(self, exchange: str, symbol: str) -> Optional[float]:
        ex = str(exchange or "")
        sym = str(symbol or "")
        if not ex or not sym:
            return None
        if ex.lower() == "spot":
            return 0.0
        data = self.latest_funding_map.get(ex, {}).get(sym)
        if not data:
            return None
        try:
            return float(data.rate)
        except (TypeError, ValueError):
            return None

    def _refresh_inversion_signals(self, now: float):
        if not self.latest_funding_map:
            return
        for entry in self.intelligence_memory.values():
            if not isinstance(entry, dict):
                continue
            data = entry.get("data")
            if not isinstance(data, dict):
                continue
            short_ex = data.get("short_exchange")
            long_ex = data.get("long_exchange")
            short_symbol = data.get("short_symbol")
            long_symbol = data.get("long_symbol")
            if not short_ex or not long_ex or not short_symbol or not long_symbol:
                continue
            short_rate = self._get_latest_rate(short_ex, short_symbol)
            long_rate = self._get_latest_rate(long_ex, long_symbol)
            if short_rate is None or long_rate is None:
                continue
            current_sign = 1 if short_rate > long_rate else -1 if short_rate < long_rate else 0
            prev_sign = entry.get("last_spread_sign")
            if current_sign in (-1, 1) and prev_sign in (-1, 1) and current_sign != prev_sign:
                events = entry.get("inversion_events")
                if not isinstance(events, list):
                    events = []
                events.append(int(now * 1000))
                entry["inversion_events"] = events
            if current_sign in (-1, 1):
                entry["last_spread_sign"] = current_sign

    def _prune_intelligence(self, now: float, now_dt: Optional[datetime] = None):
        if now_dt is None:
            now_dt = datetime.now().astimezone()
        to_delete = []
        for k, v in self.intelligence_memory.items():
            last_seen = v.get("last_seen")
            if last_seen and now - last_seen > LOCAL_SUGGESTION_MEMORY_TTL_SEC:
                to_delete.append(k)
                continue
            cached_at = v.get("streak_cached_at")
            grace_hours = v.get("streak_grace_hours") or self.suggestion_streak_grace_hours
            if cached_at:
                try:
                    if now - float(cached_at) > float(grace_hours) * 3600:
                        v.pop("streak_cached", None)
                        v.pop("streak_cached_at", None)
                except (TypeError, ValueError):
                    v.pop("streak_cached", None)
                    v.pop("streak_cached_at", None)
            self._prune_entry_windows(v, now_dt)
            self._prune_inversion_events(v, now)
        for k in to_delete:
            del self.intelligence_memory[k]

    def _update_intelligence(self, window_candidates: List[Opportunity], window_id: Optional[str], now_dt: datetime):
        now = time.time()
        if not window_id:
            self._prune_intelligence(now, now_dt=now_dt)
            self._atomic_write_json(self.memory_path, self.intelligence_memory)
            return

        slot_minute = now_dt.minute
        window_dt = self._parse_window_id(window_id)
        for o in window_candidates:
            spread_pct = o.spread_raw * 100.0
            key = f"{o.asset}_{o.short_inst.exchange}_{o.long_inst.exchange}"
            existing = self.intelligence_memory.get(key)
            entry = existing if isinstance(existing, dict) else {}
            entry.setdefault("first_seen", now)
            entry["data"] = o.to_dict()
            interval_hours = self._get_opportunity_interval_hours(o)
            prev_interval_raw = entry.get("interval_hours")
            if prev_interval_raw is not None:
                prev_interval = self._normalize_interval_hours(prev_interval_raw)
                if abs(prev_interval - interval_hours) > 0.01:
                    entry.pop("windows", None)
                    entry.pop("qualified", None)
                    entry.pop("qualified_at", None)
                    entry.pop("qualified_windows", None)
                    entry.pop("spread_count", None)
                    entry.pop("spread_avg_pct", None)
                    entry.pop("spread_min_pct", None)
                    entry.pop("spread_max_pct", None)
                    entry.pop("window_id", None)
                    entry.pop("last_slot_minute", None)
                    entry.pop("lifetime_ok", None)
                    entry.pop("lifetime_total", None)
                    entry.pop("consistency_score", None)
                    entry.pop("last_finalized_window_id", None)
                    entry["first_seen"] = now
            entry["interval_hours"] = interval_hours
            entry["interval_step_hours"] = self._get_interval_step_hours(interval_hours)
            entry["ttl_sec"] = self._get_entry_ttl_sec(interval_hours)
            entry["spread_last_pct"] = spread_pct
            entry["tolerance_level"] = self.suggestion_tolerance_level
            entry["tolerance_required_hits"] = self.suggestion_required_hits
            entry["min_spread_pct"] = LOCAL_SUGGESTION_MIN_SPREAD_PCT
            entry["top_n"] = LOCAL_SUGGESTION_TOP_N
            entry["slot_minutes"] = LOCAL_SUGGESTION_SLOT_MINUTES
            entry["smart_window_count"] = LOCAL_SUGGESTION_SMART_WINDOW_COUNT
            entry["smart_required_ok"] = LOCAL_SUGGESTION_SMART_REQUIRED_OK
            entry["smart_min_avg_spread_pct"] = LOCAL_SUGGESTION_SMART_MIN_AVG_SPREAD_PCT
            if not isinstance(entry.get("inversion_events"), list):
                entry["inversion_events"] = []

            if not self._is_window_aligned(o, window_dt, interval_hours):
                if existing:
                    self.intelligence_memory[key] = entry
                continue

            prev_window_id = entry.get("window_id")
            if prev_window_id and prev_window_id != window_id:
                self._finalize_previous_window(entry, prev_window_id)

            entry["last_seen"] = now
            entry["window_id"] = window_id
            entry["last_slot_minute"] = slot_minute

            windows = entry.get("windows", {})
            window = windows.get(window_id, {})
            slots = set(window.get("slots", []))
            slots.add(slot_minute)
            window["slots"] = sorted(slots)
            window["hits"] = len(slots)
            window["required_hits"] = self.suggestion_required_hits
            min_hits_floor = self._get_min_hits_floor()
            weighted_hits = self._get_weighted_hits(slots)
            window["min_hits_floor"] = min_hits_floor
            window["weighted_hits"] = weighted_hits
            window["focus_hit"] = bool(slots & LOCAL_SUGGESTION_FOCUS_SLOTS)
            window["last_seen"] = now
            slot_spreads = window.get("slot_spreads")
            if not isinstance(slot_spreads, dict):
                slot_spreads = {}
            slot_spreads[str(slot_minute)] = spread_pct
            window["slot_spreads"] = slot_spreads
            spread_min_pct = self._get_window_min_spread_pct(slot_spreads)
            spread_ok = True
            if spread_min_pct is not None:
                spread_ok = spread_min_pct >= self.suggestion_streak_negative_tolerance_pct
            window["spread_min_pct"] = spread_min_pct
            window["spread_ok"] = spread_ok
            window["ok"] = (
                len(slots) >= min_hits_floor
                and weighted_hits >= self.suggestion_required_hits
                and spread_ok
            )
            windows[window_id] = window
            entry["windows"] = windows
            self._update_spread_stats(entry)

            step_hours = entry.get("interval_step_hours", 1)
            prev_id = self._get_previous_window_id(window_id)
            if step_hours != 1:
                prev_dt = self._parse_window_id(window_id)
                if prev_dt:
                    prev_id = (prev_dt - timedelta(hours=step_hours)).strftime("%Y%m%d%H")
            if prev_id and windows.get(prev_id, {}).get("ok") and window.get("ok"):
                entry["qualified"] = True
                entry.setdefault("qualified_at", now)
                entry["qualified_windows"] = [prev_id, window_id]
                entry["qualified_reason"] = "consecutive_ok"

            if not entry.get("qualified"):
                recent_ok = self._get_recent_ok_windows(entry, LOCAL_SUGGESTION_SMART_WINDOW_COUNT)
                entry["recent_ok_count"] = len(recent_ok)
                if len(recent_ok) >= LOCAL_SUGGESTION_SMART_REQUIRED_OK:
                    avg_spread = entry.get("spread_avg_pct")
                    if avg_spread is None:
                        avg_spread = entry.get("spread_last_pct")
                    if avg_spread is not None and avg_spread >= LOCAL_SUGGESTION_SMART_MIN_AVG_SPREAD_PCT:
                        entry["qualified"] = True
                        entry.setdefault("qualified_at", now)
                        entry["qualified_windows"] = recent_ok[:LOCAL_SUGGESTION_SMART_REQUIRED_OK]
                        entry["qualified_reason"] = "recent_ok"
            else:
                entry["recent_ok_count"] = len(self._get_recent_ok_windows(entry, LOCAL_SUGGESTION_SMART_WINDOW_COUNT))

            if not entry.get("qualified"):
                consistency = entry.get("consistency_score")
                if not isinstance(consistency, (int, float)):
                    consistency = self._get_recent_ok_ratio(entry, LOCAL_SUGGESTION_SMART_WINDOW_COUNT)
                    entry["consistency_score"] = consistency
                spread_ok = entry.get("spread_last_pct")
                if spread_ok is None:
                    spread_ok = spread_pct
                should_return = (
                    window.get("ok")
                    and window.get("focus_hit")
                    and float(spread_ok) >= LOCAL_SUGGESTION_RETURN_MIN_SPREAD_PCT
                    and isinstance(entry.get("lifetime_total"), (int, float))
                    and float(entry.get("lifetime_total")) >= LOCAL_SUGGESTION_RETURN_MIN_TOTAL
                    and consistency >= LOCAL_SUGGESTION_RETURN_MIN_CONSISTENCY
                )
                if should_return:
                    entry["qualified"] = True
                    entry.setdefault("qualified_at", now)
                    entry["qualified_windows"] = [window_id]
                    entry["qualified_reason"] = "returning_consistency"
            self._update_streak_cache(entry, now_dt)
            self.intelligence_memory[key] = entry

        self._refresh_inversion_signals(now)
        self._prune_intelligence(now, now_dt=now_dt)
        self._atomic_write_json(self.memory_path, self.intelligence_memory)

    def _update_spread_stats(self, entry: Dict[str, Any]):
        windows = entry.get("windows", {})
        if not isinstance(windows, dict):
            return
        spreads: List[float] = []
        for win in windows.values():
            if not isinstance(win, dict):
                continue
            slot_spreads = win.get("slot_spreads", {})
            if isinstance(slot_spreads, dict):
                for val in slot_spreads.values():
                    try:
                        spreads.append(float(val))
                    except (TypeError, ValueError):
                        continue
            elif isinstance(slot_spreads, list):
                for val in slot_spreads:
                    try:
                        spreads.append(float(val))
                    except (TypeError, ValueError):
                        continue
        if not spreads:
            return
        entry["spread_count"] = len(spreads)
        entry["spread_avg_pct"] = sum(spreads) / len(spreads)
        entry["spread_min_pct"] = min(spreads)
        entry["spread_max_pct"] = max(spreads)

    def _load_elite_suggestions(self) -> Dict[str, Any]:
        payload = self._load_json(self.elite_path, {})
        return payload if isinstance(payload, dict) else {}

    def _save_elite_suggestions(self, payload: Dict[str, Any]):
        self._atomic_write_json(self.elite_path, payload)

    def _update_elite_suggestions(self, now_ts: float):
        now_dt = datetime.now().astimezone()
        payload = self._load_elite_suggestions()
        items = payload.get("items")
        if not isinstance(items, dict):
            items = {}
        ttl_sec = payload.get("ttl_sec") or ELITE_SUGGESTION_TTL_SEC

        for key in list(items.keys()):
            item = items.get(key, {})
            last_seen = item.get("last_seen") or item.get("updated_at_ts")
            if last_seen and now_ts - float(last_seen) > ttl_sec:
                items.pop(key, None)

        for key, entry in self.intelligence_memory.items():
            if not isinstance(entry, dict):
                continue
            consecutive = self._get_consecutive_ok_windows(entry, now_dt=now_dt, allow_resume=True)
            if consecutive < ELITE_SUGGESTION_MIN_OK_WINDOWS:
                continue
            data = entry.get("data") or {}
            items[key] = {
                "key": key,
                "coin": data.get("coin"),
                "short_exchange": data.get("short_exchange"),
                "long_exchange": data.get("long_exchange"),
                "short_symbol": data.get("short_symbol"),
                "long_symbol": data.get("long_symbol"),
                "consecutive_ok_windows": consecutive,
                "qualified_reason": entry.get("qualified_reason"),
                "qualified_at": entry.get("qualified_at"),
                "spread_avg_pct": entry.get("spread_avg_pct"),
                "spread_min_pct": entry.get("spread_min_pct"),
                "spread_max_pct": entry.get("spread_max_pct"),
                "spread_last_pct": entry.get("spread_last_pct"),
                "last_seen": entry.get("last_seen"),
                "interval_hours": entry.get("interval_hours"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "updated_at_ts": now_ts
            }

        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "ttl_sec": ttl_sec,
            "min_ok_windows": ELITE_SUGGESTION_MIN_OK_WINDOWS,
            "count": len(items),
            "items": items
        }
        self._save_elite_suggestions(payload)

    def _load_critical_analysis(self) -> Dict[str, Any]:
        payload = self._load_json(self.critical_path, {})
        if not isinstance(payload, dict):
            payload = {}
        items = payload.get("items")
        if not isinstance(items, dict):
            items = {}
        payload["items"] = items
        return payload

    def _save_critical_analysis(self, payload: Dict[str, Any]):
        self._atomic_write_json(self.critical_path, payload)

    def _update_critical_analysis(self, now_dt: datetime, window_candidates: Optional[List[Opportunity]] = None,
                                  opportunities: Optional[List[Opportunity]] = None):
        if now_dt.minute not in CRITICAL_WINDOW_MINUTES:
            return
        if not self.latest_funding_map:
            return
        allowed_assets = set()
        if window_candidates:
            allowed_assets = {o.asset for o in window_candidates if o and o.asset}
        elif opportunities:
            top_candidates = self._filter_top_window_opps(opportunities, now_dt)
            allowed_assets = {o.asset for o in top_candidates if o and o.asset}
        if not allowed_assets:
            return
        payload = self.critical_state if isinstance(self.critical_state, dict) else {}
        items = payload.get("items")
        if not isinstance(items, dict):
            items = {}
        now_ts = time.time()
        minute_key = str(now_dt.minute)

        for asset in allowed_assets:
            insts = self.universe_mgr.get_instruments_for_asset(asset)
            if len(insts) < 2:
                continue
            for i in range(len(insts)):
                for j in range(len(insts)):
                    if i == j:
                        continue
                    s_inst = insts[i]
                    l_inst = insts[j]
                    s_data = self.latest_funding_map.get(s_inst.exchange, {}).get(s_inst.symbol)
                    l_data = self.latest_funding_map.get(l_inst.exchange, {}).get(l_inst.symbol)
                    if not s_data or not l_data:
                        continue
                    opp = Opportunity(asset, s_inst, l_inst, s_data, l_data)
                    spread_pct = opp.spread_raw * 100.0
                    key = self._make_suggestion_key(
                        asset,
                        s_inst.exchange,
                        l_inst.exchange,
                        s_inst.symbol,
                        l_inst.symbol
                    )
                    if not key:
                        continue

                    entry = items.get(key) if isinstance(items.get(key), dict) else {}
                    count = int(entry.get("count", 0)) + 1
                    sum_spread = float(entry.get("sum_spread_pct", 0.0)) + spread_pct
                    sum_abs = float(entry.get("sum_abs_spread_pct", 0.0)) + abs(spread_pct)
                    min_spread = entry.get("min_spread_pct")
                    max_spread = entry.get("max_spread_pct")
                    min_spread = spread_pct if min_spread is None else min(float(min_spread), spread_pct)
                    max_spread = spread_pct if max_spread is None else max(float(max_spread), spread_pct)
                    pos_count = int(entry.get("positive_count", 0)) + (1 if spread_pct >= 0 else 0)
                    neg_count = int(entry.get("negative_count", 0)) + (1 if spread_pct < 0 else 0)
                    minute_counts = entry.get("minute_counts")
                    if not isinstance(minute_counts, dict):
                        minute_counts = {}
                    minute_counts[minute_key] = int(minute_counts.get(minute_key, 0)) + 1
                    minute_last = entry.get("minute_last_spread_pct")
                    if not isinstance(minute_last, dict):
                        minute_last = {}
                    minute_last[minute_key] = spread_pct

                    entry.update({
                        "key": key,
                        "coin": asset,
                        "short_exchange": s_inst.exchange,
                        "long_exchange": l_inst.exchange,
                        "short_symbol": s_inst.symbol,
                        "long_symbol": l_inst.symbol,
                        "short_interval_hours": opp.short_interval,
                        "long_interval_hours": opp.long_interval,
                        "interval_hours": self._get_opportunity_interval_hours(opp),
                        "is_valid": opp.is_valid,
                        "count": count,
                        "sum_spread_pct": sum_spread,
                        "sum_abs_spread_pct": sum_abs,
                        "avg_spread_pct": sum_spread / count,
                        "avg_abs_spread_pct": sum_abs / count,
                        "min_spread_pct": min_spread,
                        "max_spread_pct": max_spread,
                        "positive_count": pos_count,
                        "negative_count": neg_count,
                        "last_spread_pct": spread_pct,
                        "last_short_rate_pct": opp.short_funding.rate * 100.0,
                        "last_long_rate_pct": opp.long_funding.rate * 100.0,
                        "last_seen_ts": now_ts,
                        "last_seen_local": now_dt.isoformat(),
                        "minute_counts": minute_counts,
                        "minute_last_spread_pct": minute_last
                    })
                    if not entry.get("first_seen_ts"):
                        entry["first_seen_ts"] = now_ts
                        entry["first_seen_local"] = now_dt.isoformat()
                    items[key] = entry

        cutoff = now_ts - CRITICAL_ANALYSIS_TTL_SEC
        for key in list(items.keys()):
            entry = items.get(key)
            if not isinstance(entry, dict):
                items.pop(key, None)
                continue
            last_seen = entry.get("last_seen_ts")
            if not isinstance(last_seen, (int, float)) or last_seen < cutoff:
                items.pop(key, None)

        payload.update({
            "ok": True,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "updated_at_ts": now_ts,
            "minutes": sorted(CRITICAL_WINDOW_MINUTES, key=lambda x: (x - 58) % 60),
            "ttl_sec": CRITICAL_ANALYSIS_TTL_SEC,
            "scope": "top_100_assets",
            "scope_count": len(allowed_assets),
            "count": len(items),
            "items": items
        })
        self.critical_state = payload
        self._save_critical_analysis(payload)

    def _save_top_window_snapshot(self, window_candidates: List[Opportunity], window_id: Optional[str], now_dt: datetime):
        if not window_id:
            return

        payload = [o.to_dict() for o in window_candidates]
        snapshot = {
            "ts": int(time.time() * 1000),
            "timestamp_utc": now_dt.astimezone(timezone.utc).isoformat(),
            "timestamp_local": now_dt.isoformat(),
            "window_id": window_id,
            "min_spread_pct": LOCAL_SUGGESTION_MIN_SPREAD_PCT,
            "tolerance_level": self.suggestion_tolerance_level,
            "required_hits": self.suggestion_required_hits,
            "slot_minutes": LOCAL_SUGGESTION_SLOT_MINUTES,
            "count": len(payload),
            "data": payload
        }

        date_str = now_dt.strftime("%Y-%m-%d")
        path = os.path.join(self.out_dir, "history", "top_100", f"{date_str}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot) + "\n")
            logger.info(f"Saved Top 100 snapshot for {now_dt.hour:02d}:{now_dt.minute:02d}")
        except Exception as e:
            logger.error(f"Failed to save Top 100 snapshot: {e}")

    def _make_suggestion_key(self, coin: str, short_ex: str, long_ex: str,
                             short_symbol: str = "", long_symbol: str = "") -> str:
        if not coin or not short_ex or not long_ex:
            return ""
        return "|".join([
            str(coin).upper().strip(),
            str(short_ex).lower().strip(),
            str(long_ex).lower().strip(),
            str(short_symbol or "").upper().strip(),
            str(long_symbol or "").upper().strip()
        ])

    def _iter_top_100_history_paths(self, days: int = 2):
        days = max(1, min(days, 7))
        now_dt = datetime.now().astimezone()
        for i in range(days):
            date_str = (now_dt - timedelta(days=i)).strftime("%Y-%m-%d")
            path = os.path.join(self.out_dir, "history", "top_100", f"{date_str}.jsonl")
            if os.path.exists(path):
                yield path

    def _get_history_spread_stats(self, keys: Dict[str, Any], days: int = 2) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {k: {"sum": 0.0, "count": 0} for k in keys}
        for path in self._iter_top_100_history_paths(days=days):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            snap = json.loads(line)
                        except Exception:
                            continue
                        seen = set()
                        for row in snap.get("data", []) or []:
                            coin = row.get("coin")
                            short_ex = row.get("short_exchange")
                            long_ex = row.get("long_exchange")
                            short_symbol = row.get("short_symbol")
                            long_symbol = row.get("long_symbol")
                            full_key = self._make_suggestion_key(coin, short_ex, long_ex, short_symbol, long_symbol)
                            simple_key = self._make_suggestion_key(coin, short_ex, long_ex)
                            target_key = full_key if full_key in stats else simple_key if simple_key in stats else ""
                            if not target_key or target_key in seen:
                                continue
                            spread_pct = row.get("spread_pct")
                            if spread_pct is None:
                                continue
                            try:
                                stats[target_key]["sum"] += float(spread_pct)
                                stats[target_key]["count"] += 1
                                seen.add(target_key)
                            except (TypeError, ValueError):
                                continue
            except Exception as e:
                logger.error(f"Failed to read history file {path}: {e}")
        return stats

    def _build_spot_short_list(self) -> List[Dict[str, Any]]:
        inst_map = {(i.exchange, i.symbol): i for i in self.universe_mgr.get_all_instruments_flat()}
        candidates: List[Dict[str, Any]] = []
        min_rate_pct = SPOT_SHORT_MIN_RATE_PCT
        for ex_name, rates in self.latest_funding_map.items():
            for symbol, funding in rates.items():
                if not funding:
                    continue
                rate_pct = funding.rate * 100.0
                if rate_pct < min_rate_pct:
                    continue
                inst = inst_map.get((ex_name, symbol))
                if not inst:
                    continue
                interval_hours = self._normalize_interval_hours(funding.interval_hours)
                spot_symbol = symbol.split(':')[0] if symbol else ""
                candidates.append({
                    "coin": inst.canonical_base,
                    "short_exchange": ex_name,
                    "short_symbol": inst.symbol,
                    "spot_symbol": spot_symbol,
                    "short_rate_pct": rate_pct,
                    "short_funding_interval": interval_hours,
                    "short_next_funding_time": self._calc_next_funding_time(funding.next_funding_time, interval_hours),
                    "short_mark_price": funding.mark_price
                })

        if not candidates:
            return []

        candidates.sort(key=lambda x: x.get("short_rate_pct", 0), reverse=True)
        unique: List[Dict[str, Any]] = []
        seen = set()
        for row in candidates:
            key = (row.get("coin"), row.get("short_exchange"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
            if len(unique) >= SPOT_SHORT_MAX_ITEMS:
                break

        spot_requests: Dict[str, set] = {}
        for row in unique:
            spot_symbol = row.get("spot_symbol")
            if not spot_symbol:
                continue
            spot_requests.setdefault(row.get("short_exchange"), set()).add(spot_symbol)

        spot_prices: Dict[str, Dict[str, float]] = {}
        for ex_name, symbols in spot_requests.items():
            adapter = self.universe_mgr.adapters.get(ex_name)
            if not adapter:
                continue
            try:
                prices = adapter.get_spot_prices(list(symbols))
            except Exception as e:
                logger.error(f"[{ex_name}] Spot prices failed: {e}")
                prices = {}
            if isinstance(prices, dict) and prices:
                spot_prices[ex_name] = prices

        now_iso = datetime.now(timezone.utc).isoformat()
        for row in unique:
            ex_name = row.get("short_exchange")
            spot_symbol = row.get("spot_symbol")
            spot_price = spot_prices.get(ex_name, {}).get(spot_symbol)
            mark_price = row.get("short_mark_price")
            if spot_price and mark_price and spot_price > 0:
                basis = ((mark_price / spot_price) - 1.0) * 100.0
                row["basis_pct"] = basis
                row["entry_pct"] = basis
            else:
                row["basis_pct"] = None
                row["entry_pct"] = None
            row["spot_price"] = spot_price
            row["spot_exchange"] = ex_name
            row["long_exchange"] = "spot"
            row["long_symbol"] = spot_symbol
            row["long_rate_pct"] = 0.0
            row["long_funding_interval"] = None
            row["long_next_funding_time"] = None
            row["spread_pct"] = row.get("short_rate_pct", 0.0)
            row["funding_net_estimated_pct"] = calc_net_estimated_pct(
                row.get("short_rate_pct", 0.0),
                row.get("short_exchange"),
                row.get("long_exchange")
            )
            row["is_spot_short"] = True
            row["ts"] = now_iso

        return unique

    def _build_spot_short_opportunities(
        self,
        spot_shorts: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Opportunity]]:
        if spot_shorts is None:
            spot_shorts = self._build_spot_short_list()
        if not spot_shorts:
            return spot_shorts, []

        inst_map = {(i.exchange, i.symbol): i for i in self.universe_mgr.get_all_instruments_flat()}
        now_ms = int(time.time() * 1000)
        opportunities: List[Opportunity] = []

        for row in spot_shorts:
            short_ex = row.get("short_exchange")
            short_symbol = row.get("short_symbol")
            spot_symbol = row.get("spot_symbol") or row.get("long_symbol")
            if not short_ex or not short_symbol or not spot_symbol:
                continue
            short_inst = inst_map.get((short_ex, short_symbol))
            short_funding = self.latest_funding_map.get(short_ex, {}).get(short_symbol)
            if not short_inst or not short_funding:
                continue
            base = str(row.get("coin") or short_inst.canonical_base or "").upper()
            quote = "USDT"
            if isinstance(spot_symbol, str) and "/" in spot_symbol:
                base_part, quote_part = spot_symbol.split("/", 1)
                if base_part:
                    base = base_part.strip().upper()
                if quote_part:
                    quote = quote_part.strip().upper()

            long_inst = Instrument(
                "spot",
                spot_symbol,
                base,
                quote,
                short_inst.volume_24h,
                True
            )
            long_funding = FundingData(
                "spot",
                spot_symbol,
                0.0,
                now_ms,
                short_funding.next_funding_time,
                row.get("spot_price"),
                short_funding.interval_hours
            )
            asset = row.get("coin") or short_inst.canonical_base
            opportunities.append(Opportunity(asset, short_inst, long_inst, short_funding, long_funding))

        return spot_shorts, opportunities

    def _update_ui_status(self, message: str, state: str = "running"):
        self.current_snapshot["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        if "status" not in self.current_snapshot:
            self.current_snapshot["status"] = {}
        self.current_snapshot["status"].update({"state": state, "message": message})
        
        # We only want to overwrite the full file with status if it's the first time
        # Otherwise, we just keep the previous data and update the status
        self._atomic_write_json(os.path.join(self.out_dir, "ui_snapshot.json"), self.current_snapshot)

    def run_forever(self):
        self.running = True
        logger.info("Pipeline started (Dynamic Interval Mode: 5m normal / 1m critical).")
        
        if not self.current_snapshot:
            primary = self._load_json(os.path.join(self.out_dir, "ui_snapshot.json"), {})
            if not primary or "opportunities_total" not in primary:
                primary = self._load_json(os.path.join(self.out_dir, "latest", "ui_snapshot.json"), {})
            if not primary:
                primary = {"status": {"state": "starting"}}
            if "status" not in primary:
                primary["status"] = {"state": "starting"}
            self.current_snapshot = primary

        while self.running:
            try:
                self._update_ui_status("Refreshing Universe...")
                self.universe_mgr.refresh_universe()
                
                self._update_ui_status("Capturing Funding...")
                self.latest_funding_map = self._snapshot_funding()
                
                self._update_ui_status("Ranking...")
                min_spread_raw = min(self.suggestion_streak_negative_tolerance_pct, 0.0) / 100.0
                all_opportunities = self._rank_opportunities(self.latest_funding_map, min_spread_raw=min_spread_raw)
                opportunities = [o for o in all_opportunities if o.spread_raw > 0]
                spot_shorts, spot_opportunities = self._build_spot_short_opportunities()
                suggestion_opportunities = opportunities + spot_opportunities
                suggestion_opportunities.sort(key=lambda x: x.spread_raw, reverse=True)

                now_dt = datetime.now().astimezone()
                window_id = self._get_window_id(now_dt)
                window_candidates = self._filter_top_window_opps(suggestion_opportunities, now_dt) if window_id else []
                critical_candidates = self._filter_top_window_opps(opportunities, now_dt) if window_id else []
                intelligence_candidates = self._get_intelligence_candidates(
                    all_opportunities + spot_opportunities,
                    window_candidates
                ) if window_id else []

                # Intelligence and Top 100 Snapshots (windowed)
                self._update_intelligence(intelligence_candidates, window_id, now_dt)
                self._save_top_window_snapshot(window_candidates, window_id, now_dt)
                self._update_elite_suggestions(now_dt.timestamp())
                self._update_critical_analysis(now_dt, window_candidates=critical_candidates, opportunities=opportunities)
                
                # Dynamic Scheduling Logic
                now_ts = now_dt.timestamp()
                
                if now_dt.minute >= CRITICAL_SCAN_START_MIN or now_dt.minute == 0:
                    # 1-minute intervals near funding turn
                    next_scan_ts = (int(now_ts // 60) + 1) * 60
                    interval_desc = "1m (Critical)"
                elif now_dt.minute == 1:
                    # Exit critical window after HH:01
                    next_scan_ts = (int(now_ts // 300) + 1) * 300
                    interval_desc = "5m (Post-hour)"
                else:
                    # 5-minute intervals aligned to clock (5, 10, 15...)
                    next_five_ts = (int(now_ts // 300) + 1) * 300
                    critical_dt = now_dt.replace(minute=CRITICAL_SCAN_START_MIN, second=0, microsecond=0)
                    critical_ts = critical_dt.timestamp()
                    if now_dt.minute < CRITICAL_SCAN_START_MIN and critical_ts < next_five_ts:
                        next_scan_ts = critical_ts
                        interval_desc = "5m (Pre-critical)"
                    else:
                        next_scan_ts = next_five_ts
                        interval_desc = "5m (Normal)"
                
                self._save_results(opportunities)
                self._save_ui_snapshot(opportunities, next_scan=next_scan_ts, spot_shorts=spot_shorts)
                self.history_mgr.save_snapshots(self.latest_funding_map)

                local_next = datetime.fromtimestamp(next_scan_ts).strftime('%H:%M:%S')
                logger.info("="*30)
                logger.info(f" SCAN COMPLETE: {len(opportunities)} pairs found")
                logger.info(f" Mode: {interval_desc}")
                logger.info(f" Next Execution: {local_next} (Local)")
                logger.info("="*30)
                
                self._update_ui_status("Idle")
                
                # Wait logic
                sleep_time = next_scan_ts - time.time()
                if sleep_time > 0:
                    logger.info(f"Sleeping for {int(sleep_time)}s...")
                    refresh_flag = os.path.join(self.out_dir, "refresh_now.flag")
                    while sleep_time > 0:
                        if os.path.exists(refresh_flag):
                            try:
                                os.remove(refresh_flag)
                            except OSError:
                                pass
                            logger.info("Refresh flag detected. Running scan early.")
                            break
                        time.sleep(min(1, sleep_time))
                        sleep_time = next_scan_ts - time.time()
                else:
                    logger.warning(f"Scan took too long, skipping sleep (diff: {int(sleep_time)}s)")
            except Exception as e:
                logger.error(f"Pipeline loop error: {e}", exc_info=True)
                time.sleep(10)

    def _snapshot_funding(self) -> Dict[str, Dict[str, FundingData]]:
        funding_map = {}
        by_ex = {}
        for inst in self.universe_mgr.get_all_instruments_flat():
            by_ex.setdefault(inst.exchange, []).append(inst)

        for ex_name, insts in by_ex.items():
            adapter = self.universe_mgr.adapters.get(ex_name)
            if adapter:
                try:
                    rates = adapter.get_funding_rates(insts)
                    funding_map[ex_name] = {r.symbol: r for r in rates}
                    logger.info(f"[{ex_name}] Captured {len(rates)} rates")
                except Exception as e:
                    logger.error(f"[{ex_name}] Snapshot failed: {e}")
        return funding_map

    def _rank_opportunities(self, funding_map: Dict[str, Dict[str, FundingData]],
                            min_spread_raw: Optional[float] = None) -> List[Opportunity]:
        if min_spread_raw is None:
            min_spread_raw = 0.0
        opps = []
        for asset in self.universe_mgr.get_canonical_assets():
            insts = self.universe_mgr.get_instruments_for_asset(asset)
            if len(insts) < 2: continue
            for i in range(len(insts)):
                for j in range(len(insts)):
                    if i == j: continue
                    s_data = funding_map.get(insts[i].exchange, {}).get(insts[i].symbol)
                    l_data = funding_map.get(insts[j].exchange, {}).get(insts[j].symbol)
                    if s_data and l_data:
                        opp = Opportunity(asset, insts[i], insts[j], s_data, l_data)
                        if opp.spread_raw > min_spread_raw:
                            opps.append(opp)
        opps.sort(key=lambda x: x.spread_raw, reverse=True)
        return opps

    def _save_results(self, opportunities: List[Opportunity]):
        data = [o.to_dict() for o in opportunities]
        self._atomic_write_json(os.path.join(self.out_dir, "latest", "ranking.json"), data)

    def _get_local_suggestions(self, now: float) -> List[Dict[str, Any]]:
        suggestions = []
        tracked = []
        now_dt = datetime.now().astimezone()
        cutoff_ms = int((now - LOCAL_SUGGESTION_INVERSION_TTL_SEC) * 1000)
        for v in self.intelligence_memory.values():
            if not v.get("qualified"):
                continue
            last_seen = v.get("last_seen")
            display_ttl_sec = LOCAL_SUGGESTION_TTL_SEC
            if not last_seen or now - last_seen > display_ttl_sec:
                continue
            d = v.get("data")
            if not d:
                continue
            if not d.get("is_valid", True):
                continue
            payload = dict(d)
            consecutive = self._get_consecutive_ok_windows(v, now_dt=now_dt, allow_resume=True)
            payload["consecutive_ok_windows"] = consecutive
            payload["spread_last_pct"] = v.get("spread_last_pct", d.get("spread_pct"))
            payload["qualified_reason"] = v.get("qualified_reason")
            payload["recent_ok_count"] = v.get("recent_ok_count")
            inversion_events = v.get("inversion_events")
            if isinstance(inversion_events, list):
                payload["inversion_count_3d"] = sum(
                    1 for ts in inversion_events
                    if isinstance(ts, (int, float)) and int(ts) >= cutoff_ms
                )
            else:
                payload["inversion_count_3d"] = 0
            if consecutive > 0:
                step_hours = v.get("interval_step_hours")
                if not isinstance(step_hours, (int, float)) or step_hours <= 0:
                    step_hours = self._get_interval_step_hours(v.get("interval_hours") or 1.0)
                payload["streak_hours"] = float(consecutive) * float(step_hours)
                windows_raw = v.get("windows") if isinstance(v.get("windows"), dict) else {}
                latest_dt = None
                for win_id in windows_raw.keys():
                    dt = self._parse_window_id(win_id)
                    if not dt:
                        continue
                    if latest_dt is None or dt > latest_dt:
                        latest_dt = dt
                if latest_dt:
                    start_dt = latest_dt - timedelta(hours=(consecutive - 1) * step_hours)
                    payload["streak_start_local"] = start_dt.isoformat()
                    payload["streak_start_window_id"] = start_dt.strftime("%Y%m%d%H")
            full_key = self._make_suggestion_key(
                payload.get("coin"),
                payload.get("short_exchange"),
                payload.get("long_exchange"),
                payload.get("short_symbol"),
                payload.get("long_symbol")
            )
            simple_key = self._make_suggestion_key(
                payload.get("coin"),
                payload.get("short_exchange"),
                payload.get("long_exchange")
            )
            tracked.append({
                "payload": payload,
                "entry": v,
                "full_key": full_key,
                "simple_key": simple_key
            })
            suggestions.append(payload)

        if tracked:
            keys = {item["full_key"]: True for item in tracked if item["full_key"]}
            keys.update({item["simple_key"]: True for item in tracked if item["simple_key"]})
            stats = self._get_history_spread_stats(keys)
            for item in tracked:
                payload = item["payload"]
                entry = item["entry"]
                full = item["full_key"]
                simple = item["simple_key"]
                entry_avg = entry.get("spread_avg_pct") if isinstance(entry, dict) else None
                entry_count = entry.get("spread_count") if isinstance(entry, dict) else None
                if entry_avg is not None:
                    payload["spread_avg_pct"] = entry_avg
                    if entry_count is not None:
                        payload["spread_sample_count"] = entry_count
                    continue
                if full and stats.get(full, {}).get("count"):
                    avg = stats[full]["sum"] / stats[full]["count"]
                    payload["spread_avg_pct"] = avg
                    payload["spread_sample_count"] = stats[full]["count"]
                    if isinstance(entry, dict):
                        entry["spread_avg_pct"] = avg
                        entry["spread_count"] = stats[full]["count"]
                elif simple and stats.get(simple, {}).get("count"):
                    avg = stats[simple]["sum"] / stats[simple]["count"]
                    payload["spread_avg_pct"] = avg
                    payload["spread_sample_count"] = stats[simple]["count"]
                    if isinstance(entry, dict):
                        entry["spread_avg_pct"] = avg
                        entry["spread_count"] = stats[simple]["count"]

        suggestions.sort(key=lambda x: x.get("spread_avg_pct", x.get("spread_pct", 0)), reverse=True)
        return suggestions

    def _save_ui_snapshot(
        self,
        opportunities: List[Opportunity],
        next_scan: float,
        spot_shorts: Optional[List[Dict[str, Any]]] = None
    ):
        valid_opps = [o for o in opportunities if o.is_valid]
        invalid_opps = [o for o in opportunities if not o.is_valid]
        now_ts = time.time()
        
        assets_payload = []
        for asset in self.universe_mgr.get_canonical_assets():
            insts = self.universe_mgr.get_instruments_for_asset(asset)
            if insts:
                assets_payload.append({
                    "asset": asset,
                    "exchanges": list(set(i.exchange for i in insts)),
                    "total_volume_24h": sum(i.volume_24h for i in insts),
                    "has_funding": any(self.latest_funding_map.get(i.exchange, {}).get(i.symbol) for i in insts)
                })

        local_suggestions = self._get_local_suggestions(time.time())
        suggestions_1h = [d for d in local_suggestions if d.get('short_funding_interval') == 1.0]
        suggestions_stability = [d for d in local_suggestions if d.get('short_funding_interval') != 1.0]
        if spot_shorts is None:
            spot_shorts = self._build_spot_short_list()
        memory_total = len(self.intelligence_memory)
        memory_qualified = 0
        memory_active = 0
        for v in self.intelligence_memory.values():
            if v.get("qualified"):
                memory_qualified += 1
            last_seen = v.get("last_seen")
            ttl_sec = self._get_entry_ttl_sec(v.get("interval_hours"))
            if last_seen and now_ts - last_seen <= ttl_sec:
                memory_active += 1

        self.current_snapshot.update({
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "next_scan_at": next_scan * 1000,
            "run_id": f"scan_{int(time.time())}",
            "summary": {
                "coins_count": self.universe_mgr.get_eligible_canonical_count(),
                "symbols_with_pairs": len(self.universe_mgr.get_all_instruments_flat()),
                "pairs_valid_total": len(valid_opps),
                "opportunities_valid_count": len(valid_opps)
            },
            "assets": sorted(assets_payload, key=lambda x: x["total_volume_24h"], reverse=True),
            "opportunities_valid": [o.to_dict() for o in valid_opps[:200]],
            "opportunities_invalid": [o.to_dict() for o in invalid_opps[:100]],
            "opportunities_total": len(opportunities),
            "opportunities_valid_count": len(valid_opps),
            "opportunities_invalid_count": len(invalid_opps),
            "suggestions_1h": suggestions_1h,
            "suggestions_stability": suggestions_stability,
            "spot_shorts": spot_shorts,
            "diagnostics": {
                "system": {
                    "state": self.current_snapshot.get("status", {}).get("state"),
                    "message": self.current_snapshot.get("status", {}).get("message"),
                    "last_universe_refresh": self.universe_mgr.last_refresh,
                    "memory_total": memory_total,
                    "memory_active": memory_active,
                    "memory_qualified": memory_qualified
                },
                "universe": self.universe_mgr.get_stats(),
                "causality": {
                    "universe_total_assets": self.universe_mgr.get_eligible_canonical_count(),
                    "snapshot_total_rates": sum(len(v) for v in self.latest_funding_map.values()),
                    "ranking_total_candidates": len(opportunities),
                    "deep_check_valid": len(valid_opps)
                }
            }
        })
        self._atomic_write_json(os.path.join(self.out_dir, "ui_snapshot.json"), self.current_snapshot)
        self._atomic_write_json(os.path.join(self.out_dir, "latest", "ui_snapshot.json"), self.current_snapshot)
        self._save_funding_snapshot()

    def _save_funding_snapshot(self) -> None:
        payload = {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "updated_at_ts": int(time.time() * 1000),
            "exchanges": {}
        }
        for ex_name, rates in self.latest_funding_map.items():
            if not rates:
                continue
            ex_payload = {}
            for symbol, funding in rates.items():
                if not funding:
                    continue
                ex_payload[symbol] = {
                    "rate": funding.rate,
                    "next_funding_time": funding.next_funding_time,
                    "interval_hours": funding.interval_hours,
                    "mark_price": funding.mark_price,
                    "timestamp": funding.timestamp
                }
            if ex_payload:
                payload["exchanges"][ex_name] = ex_payload
        self._atomic_write_json(os.path.join(self.out_dir, "latest", "funding_snapshot.json"), payload)
