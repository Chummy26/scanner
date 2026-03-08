import time
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import re

FEE_CONFIG: Dict[str, Any] = {
    "default": {"taker_fee_bps_open": 0.0, "taker_fee_bps_close": 0.0},
    "by_exchange": {}
}

def set_fee_config(config: Optional[Dict[str, Any]]) -> None:
    if not isinstance(config, dict):
        return
    costs = config.get("costs")
    if not isinstance(costs, dict):
        return
    default = costs.get("default")
    by_exchange = costs.get("by_exchange")
    if isinstance(default, dict):
        FEE_CONFIG["default"] = default
    if isinstance(by_exchange, dict):
        FEE_CONFIG["by_exchange"] = by_exchange

def _get_costs_for_exchange(exchange: str) -> Dict[str, Any]:
    ex = str(exchange or "").strip().lower()
    by_ex = FEE_CONFIG.get("by_exchange", {})
    if isinstance(by_ex, dict) and ex in by_ex and isinstance(by_ex[ex], dict):
        return by_ex[ex]
    default = FEE_CONFIG.get("default")
    return default if isinstance(default, dict) else {}

def _get_taker_bps(exchange: str) -> float:
    costs = _get_costs_for_exchange(exchange)
    default = FEE_CONFIG.get("default") if isinstance(FEE_CONFIG.get("default"), dict) else {}
    open_bps = costs.get("taker_fee_bps_open", default.get("taker_fee_bps_open", 0.0))
    close_bps = costs.get("taker_fee_bps_close", default.get("taker_fee_bps_close", 0.0))
    try:
        return max(float(open_bps), 0.0) + max(float(close_bps), 0.0)
    except (TypeError, ValueError):
        return 0.0

def calc_net_estimated_pct(spread_pct: float, short_exchange: str, long_exchange: str) -> float:
    taker_bps_total = _get_taker_bps(short_exchange) + _get_taker_bps(long_exchange)
    fee_pct = taker_bps_total / 100.0
    try:
        return float(spread_pct) - fee_pct
    except (TypeError, ValueError):
        return spread_pct

MIN_VOLUME_USD = 1000.0
NORMALIZE_BASE = {
    "XBT": "BTC",
    "PEPE1000": "PEPE", "1000PEPE": "PEPE", "1000000PEPE": "PEPE",
    "SHIB1000": "SHIB", "1000SHIB": "SHIB",
    "LUNC1000": "LUNC", "1000LUNC": "LUNC",
    "BONK1000": "BONK", "1000BONK": "BONK",
    "FLOKI1000": "FLOKI", "1000FLOKI": "FLOKI",
    "XEC1000": "XEC", "1000XEC": "XEC",
    "SATS1000": "SATS", "1000SATS": "SATS",
    "RATS1000": "RATS", "1000RATS": "RATS",
    "CAT1000": "CAT", "1000CAT": "CAT",
    "MOG1000000": "MOG", "1000000MOG": "MOG",
}

class Instrument:
    def __init__(self, exchange: str, symbol: str, base: str, quote: str, volume_24h: float, active: bool, **kwargs):
        self.exchange = exchange
        self.symbol = symbol 
        self.base = base.upper()
        self.quote = quote.upper()
        self.volume_24h = float(volume_24h)
        self.active = active
        self.alias_base = kwargs.get("alias_base")
        self.canonical_base = self._normalize(self.base)

    def _normalize(self, b: str) -> str:
        b = b.replace(' ', '').replace('-', '')
        if re.match(r'^(1000|1000000|100000|100|10|K|1M|1K)', b):
            stripped = re.sub(r'^(1000|1000000|100000|100|10|K|1M|1K)', '', b)
            if len(stripped) > 1: b = stripped
        return NORMALIZE_BASE.get(b, b)

    def to_dict(self):
        payload = {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "base": self.base,
            "quote": self.quote,
            "volume_24h": self.volume_24h,
            "active": self.active
        }
        if self.alias_base:
            payload["alias_base"] = self.alias_base
        return payload

    def __repr__(self):
        return f"<{self.exchange}:{self.symbol} ({self.canonical_base})>"

class FundingData:
    def __init__(self, exchange: str, symbol: str, rate: float, timestamp: int, 
                 next_funding_time: Optional[int], mark_price: Optional[float] = None, 
                 interval_hours: float = 8.0):
        self.exchange = exchange
        self.symbol = symbol
        self.rate = float(rate)
        self.timestamp = int(timestamp)
        self.next_funding_time = int(next_funding_time) if next_funding_time else None
        self.mark_price = float(mark_price) if mark_price else None
        self.interval_hours = float(interval_hours or 8.0)

class Opportunity:
    def __init__(self, asset: str, short_inst: Instrument, long_inst: Instrument, 
                 short_funding: FundingData, long_funding: FundingData):
        self.asset = asset
        self.short_inst = short_inst
        self.long_inst = long_inst
        self.short_funding = short_funding
        self.long_funding = long_funding
        
        self.short_interval = short_funding.interval_hours
        self.long_interval = long_funding.interval_hours
        self.spread_raw = short_funding.rate - long_funding.rate
        self.is_valid = (self.short_interval == self.long_interval)
        self.invalid_reason = None if self.is_valid else f"INTERVAL_MISMATCH ({self.short_interval}h vs {self.long_interval}h)"

    def to_dict(self) -> Dict[str, Any]:
        s_price = self.short_funding.mark_price
        l_price = self.long_funding.mark_price
        entry_pct = ((s_price / l_price) - 1.0) * 100.0 if s_price and l_price and l_price > 0 else 0.0
            
        now_ms = int(time.time() * 1000)
        def get_next(f_time, interval):
            if f_time and f_time > now_ms: return f_time
            sec = interval * 3600
            return int(((time.time() // sec) + 1) * sec * 1000)

        return {
            "coin": self.asset,
            "short_exchange": self.short_inst.exchange,
            "short_symbol": self.short_inst.symbol,
            "short_mark_price": self.short_funding.mark_price,
            "long_exchange": self.long_inst.exchange,
            "long_symbol": self.long_inst.symbol,
            "long_mark_price": self.long_funding.mark_price,
            "spread_pct": self.spread_raw * 100.0,
            "short_rate_pct": self.short_funding.rate * 100.0,
            "long_rate_pct": self.long_funding.rate * 100.0,
            "short_funding_interval": self.short_interval,
            "long_funding_interval": self.long_interval,
            "short_next_funding_time": get_next(self.short_funding.next_funding_time, self.short_interval),
            "long_next_funding_time": get_next(self.long_funding.next_funding_time, self.long_interval),
            "is_valid": self.is_valid,
            "invalid_reason": self.invalid_reason,
            "entry_pct": entry_pct,
            "basis_pct": entry_pct,
            "funding_net_estimated_pct": calc_net_estimated_pct(
                self.spread_raw * 100.0,
                self.short_inst.exchange,
                self.long_inst.exchange
            ),
            "ts": datetime.now(timezone.utc).isoformat()
        }
