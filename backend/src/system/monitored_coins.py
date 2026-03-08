
import json
import os
import logging
import time
from typing import List, Dict

logger = logging.getLogger(__name__)

class MonitoredCoinsManager:
    def __init__(self, file_path: str = "out/prod/config/monitored_history.json"):
        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.monitored_coins = self._load()

    def _load(self) -> List[Dict]:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for entry in data:
                            if isinstance(entry, dict):
                                self._ensure_defaults(entry)
                        return data
                    return []
            except Exception as e:
                logger.error(f"Failed to load monitored coins: {e}")
        return []

    def reload(self):
        """Reloads the monitored coins from the file."""
        self.monitored_coins = self._load()

    def _save(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.monitored_coins, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save monitored coins: {e}")

    def _ensure_defaults(self, entry: Dict):
        defaults = {
            "short_amount_tokens": None,
            "long_amount_tokens": None,
            "entry_short_rate_pct": None,
            "entry_long_rate_pct": None,
            "entry_spread_pct": None,
            "entry_ts": None,
            "entry_ts_locked": False,
            "alert_spread_below_pct": None,
            "alert_interval_change": False
        }
        for key, value in defaults.items():
            if key not in entry:
                entry[key] = value
        if entry.get("short_amount_tokens") is None and entry.get("short_amount_usdt") is not None:
            entry["short_amount_tokens"] = entry.get("short_amount_usdt")
        if entry.get("long_amount_tokens") is None and entry.get("long_amount_usdt") is not None:
            entry["long_amount_tokens"] = entry.get("long_amount_usdt")

    def add_coin(self, symbol: str, exchanges: List[str], short_exchange: str = None, long_exchange: str = None):
        symbol = symbol.upper()
        # Check for duplicates
        for entry in self.monitored_coins:
            if entry['symbol'] == symbol:
                # Update exchanges if already exists
                entry['exchanges'] = list(set(entry['exchanges'] + exchanges))
                entry['short_exchange'] = short_exchange or entry.get('short_exchange')
                entry['long_exchange'] = long_exchange or entry.get('long_exchange')
                self._ensure_defaults(entry)
                self._save()
                return
        
        entry = {
            "symbol": symbol,
            "exchanges": exchanges,
            "short_exchange": short_exchange,
            "long_exchange": long_exchange,
            "added_at": int(time.time()),
            "status": "PENDING",
            "last_validation": None,
            "error_message": None,
            "active": True
        }
        self._ensure_defaults(entry)
        self.monitored_coins.append(entry)
        self._save()

    def update_status(self, symbol: str, status: str, error_message: str = None):
        symbol = symbol.upper()
        for entry in self.monitored_coins:
            if entry['symbol'] == symbol:
                entry['status'] = status
                entry['error_message'] = error_message
                entry['last_validation'] = int(time.time() * 1000)
                break
        self._save()

    def toggle_active(self, symbol: str, active: bool):
        symbol = symbol.upper()
        for entry in self.monitored_coins:
            if entry['symbol'] == symbol:
                entry['active'] = active
                break
        self._save()

    def remove_coin(self, symbol: str):
        symbol = symbol.upper()
        self.monitored_coins = [c for c in self.monitored_coins if c['symbol'] != symbol]
        self._save()

    def update_coin(self, symbol: str, fields: Dict) -> bool:
        symbol = symbol.upper()
        allowed = {
            "short_amount_tokens",
            "long_amount_tokens",
            "entry_short_rate_pct",
            "entry_long_rate_pct",
            "entry_spread_pct",
            "entry_ts",
            "entry_ts_locked",
            "alert_spread_below_pct",
            "alert_interval_change"
        }
        allowed.update({"short_amount_usdt", "long_amount_usdt"})
        for entry in self.monitored_coins:
            if entry['symbol'] == symbol:
                self._ensure_defaults(entry)
                for key, value in fields.items():
                    if key in allowed:
                        entry[key] = value
                self._save()
                return True
        return False

    def get_coins(self) -> List[Dict]:
        return self.monitored_coins
