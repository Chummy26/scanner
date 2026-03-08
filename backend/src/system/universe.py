import json
import os
import time
import logging
from typing import Dict, List, Set, Optional
from .adapters import AdapterFactory, BaseAdapter
from .models import Instrument

logger = logging.getLogger(__name__)

ALLOWED_EXCHANGES = {"gate", "mexc", "kucoin", "bingx", "xt", "bitget"}

class UniverseManager:
    def __init__(self, state_file: str = "universe_state.json", exchanges: List[str] = None):
        self.state_file = state_file
        self.exchanges = [ex for ex in (exchanges or list(ALLOWED_EXCHANGES)) if ex in ALLOWED_EXCHANGES]
        self.adapters = {ex: AdapterFactory.create_adapter(ex) for ex in self.exchanges}
        self.instruments: Dict[str, List[Instrument]] = {} # asset -> [Instrument]
        self.stats: Dict[str, Dict] = {} # exchange -> stats
        self.last_refresh = 0
        self.refresh_interval = 3600 # 1 hour

    def load_state(self):
        if not os.path.exists(self.state_file): return
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.last_refresh = data.get('last_refresh', 0)
                self.stats = data.get('stats', {})
                self.instruments = {}
                for asset, inst_list in data.get('instruments_data', {}).items():
                    self.instruments[asset] = [Instrument(**i) for i in inst_list]
                logger.info(f"Loaded {len(self.instruments)} assets from state.")
        except Exception as e: logger.error(f"Load state failed: {e}")

    def save_state(self):
        try:
            data = {
                'last_refresh': self.last_refresh,
                'stats': self.stats,
                'instruments_data': {a: [i.to_dict() for i in l] for a, l in self.instruments.items()}
            }
            with open(self.state_file + ".tmp", 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            os.replace(self.state_file + ".tmp", self.state_file)
        except Exception as e: logger.error(f"Save state failed: {e}")

    def refresh_universe(self):
        logger.info("--- UNIVERSE REFRESH START ---")
        temp_all_instruments = []
        for ex_name, adapter in self.adapters.items():
            try:
                instruments, stats = adapter.get_universe()
                self.stats[ex_name] = stats
                temp_all_instruments.extend(instruments)
                logger.info(f"[{ex_name}] Found {len(instruments)} instruments")
            except Exception as e:
                logger.error(f"[{ex_name}] Refresh failed: {e}")

        self._build_indexes(temp_all_instruments)
        self.last_refresh = time.time()
        self.save_state()
        logger.info(f"--- UNIVERSE REFRESH COMPLETE: {len(self.instruments)} canonical assets ---")

    def _build_indexes(self, all_instruments: List[Instrument]):
        def has_non_ascii(value: str) -> bool:
            return any(ord(ch) > 127 for ch in value)

        def normalize_alias(value: str) -> str:
            return value.replace(' ', '').replace('-', '')

        alias_map: Dict[str, str] = {}
        for inst in all_instruments:
            alias = inst.alias_base if isinstance(inst.alias_base, str) else ""
            if alias and has_non_ascii(alias):
                canonical = inst.canonical_base
                if canonical and not has_non_ascii(canonical):
                    alias_key = normalize_alias(alias)
                    if alias_key:
                        existing = alias_map.get(alias_key)
                        if existing and existing != canonical:
                            logger.warning(
                                "[universe] Alias conflict for %s: %s vs %s",
                                alias_key.encode('unicode_escape').decode('ascii'),
                                existing,
                                canonical
                            )
                        else:
                            alias_map[alias_key] = canonical

        if alias_map:
            for inst in all_instruments:
                base_key = normalize_alias(inst.base)
                if has_non_ascii(base_key) and base_key in alias_map:
                    inst.canonical_base = alias_map[base_key]

        def escape_unicode(value: str) -> str:
            try:
                return value.encode('unicode_escape').decode('ascii')
            except Exception:
                return value

        unresolved = sorted({
            normalize_alias(inst.base)
            for inst in all_instruments
            if has_non_ascii(inst.base) and normalize_alias(inst.base) not in alias_map
        })
        if unresolved:
            escaped = [escape_unicode(v) for v in unresolved]
            logger.warning(f"[universe] Non-ASCII bases without alias: {escaped}")

        self.instruments = {}
        for inst in all_instruments:
            # CORRECT: Instrument has canonical_base, not asset
            self.instruments.setdefault(inst.canonical_base, []).append(inst)

    def get_canonical_assets(self) -> List[str]: return sorted(list(self.instruments.keys()))
    def get_instruments_for_asset(self, asset: str) -> List[Instrument]: return self.instruments.get(asset, [])
    def get_all_instruments_flat(self) -> List[Instrument]: return [i for l in self.instruments.values() for i in l]
    def get_stats(self) -> Dict[str, Dict]: return self.stats
    def get_eligible_canonical_count(self) -> int: return len(self.instruments)
