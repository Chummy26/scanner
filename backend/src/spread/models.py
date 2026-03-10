"""Data models for the spread arbitrage engine."""

import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class OrderBookSnapshot:
    """Snapshot of an order book from a single exchange/market."""
    exchange: str
    symbol: str
    market_type: str  # "spot" | "futures"
    bids: List[Tuple[float, float]] = field(default_factory=list)  # [(price, qty), ...]
    asks: List[Tuple[float, float]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    connected: bool = True  # WS connection alive at time of snapshot

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def best_bid_qty(self) -> Optional[float]:
        return self.bids[0][1] if self.bids else None

    @property
    def best_ask_qty(self) -> Optional[float]:
        return self.asks[0][1] if self.asks else None

    def is_valid(self) -> bool:
        return bool(self.bids and self.asks and self.best_bid and self.best_ask)


@dataclass
class SpreadOpportunity:
    """A calculated spread arbitrage opportunity between two exchanges."""
    asset: str  # e.g. "BTC"
    arb_type: str  # "spot_futures" | "futures_futures"
    buy_exchange: str
    sell_exchange: str
    buy_market_type: str  # "spot" | "futures"
    sell_market_type: str  # "spot" | "futures"
    buy_price: float  # best ask (to buy)
    sell_price: float  # best bid (to sell)
    entry_spread_pct: float  # (sell/buy - 1) * 100
    exit_spread_pct: float  # spread if reversed (buy_price/sell_price - 1)*100
    buy_volume_24h: float = 0.0
    sell_volume_24h: float = 0.0
    inverted_count: int = 0  # inversions in tracking window
    inverted_counts: Dict[str, int] = field(default_factory=dict)  # rolling window counts (e.g., 30m/1h/4h/8h/24h)
    funding_rate_buy: Optional[float] = None
    funding_rate_sell: Optional[float] = None

    # Funding metadata (enriched by ws_manager from futures contract info)
    funding_interval_buy: Optional[float] = None   # hours (1, 4, 8)
    funding_interval_sell: Optional[float] = None
    next_settlement_buy: Optional[int] = None       # unix ms timestamp
    next_settlement_sell: Optional[int] = None

    # Open interest (normalized to USDT value)
    open_interest_buy: Optional[float] = None
    open_interest_sell: Optional[float] = None

    # Position limits (max notional USDT)
    position_limit_buy: Optional[float] = None
    position_limit_sell: Optional[float] = None

    timestamp: float = field(default_factory=time.time)
    buy_symbol: str = ""
    sell_symbol: str = ""

    # Book age (seconds since last orderbook update)
    buy_book_age: float = -1.0
    sell_book_age: float = -1.0

    # Withdrawal/deposit status (enriched by ws_manager)
    buy_withdraw_status: Optional[bool] = None  # True = withdrawals active on at least one chain
    sell_withdraw_status: Optional[bool] = None
    buy_deposit_status: Optional[bool] = None  # True = deposits active on at least one chain
    sell_deposit_status: Optional[bool] = None

    # Extra fields enriched by ws_manager
    total_entries: int = 0
    total_exits: int = 0
    last_crossover_ts: float = 0.0
    # ML context
    ml_score: int = 0
    ml_context: Optional[Dict[str, Any]] = None

    def pair_key(self) -> str:
        """Stable key for caching, delta updates, and detail lookups."""
        return "|".join(
            [
                str(self.asset or "").strip().upper(),
                str(self.buy_exchange or "").strip().lower(),
                str(self.buy_market_type or "").strip().lower(),
                str(self.sell_exchange or "").strip().lower(),
                str(self.sell_market_type or "").strip().lower(),
            ]
        )

    def _scanner_base_dict(self) -> Dict[str, Any]:
        buy_type_upper = self.buy_market_type.upper()
        sell_type_upper = self.sell_market_type.upper()
        code = f"{buy_type_upper}_{self.buy_exchange}_{self.asset}-USDT_{self.sell_exchange}_{sell_type_upper}"
        return {
            "pairKey": self.pair_key(),
            "symbol": self.asset,
            "current": "USDT",
            "code": code,
            "type": "opportunity",
            "buyFrom": self.buy_exchange,
            "sellTo": self.sell_exchange,
            "buyType": buy_type_upper,
            "sellType": sell_type_upper,
            "buyPrice": round(self.buy_price, 8),
            "sellPrice": round(self.sell_price, 8),
            "buyVol24": round(self.buy_volume_24h, 2),
            "sellVol24": round(self.sell_volume_24h, 2),
            "entrySpread": round(self.entry_spread_pct, 4),
            "exitSpread": round(self.exit_spread_pct, 4),
            "histCruzamento": {
                "inverted_count": self.inverted_count,
                "inverted_counts": self.inverted_counts or None,
                "totalEntries": self.total_entries,
                "totalExits": self.total_exits,
            },
        }

    def to_scanner_lite_dict(self) -> Dict[str, Any]:
        """Compact payload for the main scanner board and incremental WS updates."""
        from datetime import datetime, timezone
        d = self._scanner_base_dict()
        d["timestamp"] = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        d["mlScore"] = self.ml_score
        if d["histCruzamento"].get("inverted_counts") is None:
            d["histCruzamento"].pop("inverted_counts", None)
        if self.funding_rate_buy is not None:
            d["buyFundingRate"] = self.funding_rate_buy
        if self.funding_rate_sell is not None:
            d["sellFundingRate"] = self.funding_rate_sell
        if self.buy_book_age >= 0:
            d["buyBookAge"] = self.buy_book_age
        if self.sell_book_age >= 0:
            d["sellBookAge"] = self.sell_book_age
        if self.buy_withdraw_status is not None:
            d["buyWithdrawOk"] = self.buy_withdraw_status
        if self.sell_withdraw_status is not None:
            d["sellWithdrawOk"] = self.sell_withdraw_status
        if self.buy_deposit_status is not None:
            d["buyDepositOk"] = self.buy_deposit_status
        if self.sell_deposit_status is not None:
            d["sellDepositOk"] = self.sell_deposit_status
        if self.funding_interval_buy is not None:
            d["buyFundingInterval"] = self.funding_interval_buy
        if self.funding_interval_sell is not None:
            d["sellFundingInterval"] = self.funding_interval_sell
        if self.ml_context:
            d["signalAction"] = str(self.ml_context.get("signal_action") or "WAIT")
            d["signalReasonCode"] = str(self.ml_context.get("signal_reason_code") or "")
            d["modelStatus"] = str(self.ml_context.get("model_status") or "missing_ml_context")
            d["modelVersion"] = str(self.ml_context.get("model_version") or "unavailable")
            d["successProbability"] = self.ml_context.get("success_probability")
            d["driftStatus"] = str(self.ml_context.get("drift_status") or "unknown")
            d["inferenceLatencyMs"] = self.ml_context.get("inference_latency_ms")
            d["contextStrength"] = str(self.ml_context.get("context_strength") or "unknown")
            d["rangeStatus"] = str(self.ml_context.get("range_status") or "unknown")
            d["entryPositionLabel"] = str(self.ml_context.get("entry_position_label") or "unknown")
            d["etaAlignmentStatus"] = str(self.ml_context.get("eta_alignment_status") or "unknown")
            d["displayEtaSeconds"] = self.ml_context.get("display_eta_seconds")
            d["recommendedEntryRange"] = self.ml_context.get("recommended_entry_range") or "--"
            d["recommendedExitRange"] = self.ml_context.get("recommended_exit_range") or "--"
        return d

    def to_scanner_dict(self) -> Dict[str, Any]:
        """Convert to the JSON format expected by the frontend scanner WebSocket.

        Matches legacy scanner format: UPPERCASE market types, code field, funding rates.
        """
        from datetime import datetime, timezone
        ts = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        d: Dict[str, Any] = self._scanner_base_dict()
        d["timestamp"] = ts
        d["mlScore"] = self.ml_score
        d["mlContext"] = self.ml_context
        # Don't send null (cleaner payload for clients that might not expect it).
        if d["histCruzamento"].get("inverted_counts") is None:
            d["histCruzamento"].pop("inverted_counts", None)
        if self.funding_rate_buy is not None:
            d["buyFundingRate"] = self.funding_rate_buy
        if self.funding_rate_sell is not None:
            d["sellFundingRate"] = self.funding_rate_sell
        if self.last_crossover_ts > 0:
            d["histCruzamento"]["lastCrossoverTimestamp"] = int(self.last_crossover_ts * 1000)
        if self.buy_book_age >= 0:
            d["buyBookAge"] = self.buy_book_age
        if self.sell_book_age >= 0:
            d["sellBookAge"] = self.sell_book_age
        if self.buy_withdraw_status is not None:
            d["buyWithdrawOk"] = self.buy_withdraw_status
        if self.sell_withdraw_status is not None:
            d["sellWithdrawOk"] = self.sell_withdraw_status
        if self.buy_deposit_status is not None:
            d["buyDepositOk"] = self.buy_deposit_status
        if self.sell_deposit_status is not None:
            d["sellDepositOk"] = self.sell_deposit_status
        # Funding metadata
        if self.funding_interval_buy is not None:
            d["buyFundingInterval"] = self.funding_interval_buy
        if self.funding_interval_sell is not None:
            d["sellFundingInterval"] = self.funding_interval_sell
        if self.next_settlement_buy is not None:
            d["buyNextSettlement"] = self.next_settlement_buy
        if self.next_settlement_sell is not None:
            d["sellNextSettlement"] = self.next_settlement_sell
        # Open interest (USDT)
        if self.open_interest_buy is not None:
            d["buyOpenInterest"] = round(self.open_interest_buy, 2)
        if self.open_interest_sell is not None:
            d["sellOpenInterest"] = round(self.open_interest_sell, 2)
        # Position limits (USDT)
        if self.position_limit_buy is not None:
            d["buyPositionLimit"] = round(self.position_limit_buy, 2)
        if self.position_limit_sell is not None:
            d["sellPositionLimit"] = round(self.position_limit_sell, 2)
        
        # Add ML Context
        if self.ml_context is not None:
            d["mlContext"] = self.ml_context
            
        return d


@dataclass
class ExchangeConfig:
    """Configuration for a single exchange connection."""
    name: str
    enabled: bool = True
    spot_enabled: bool = True
    futures_enabled: bool = True
    spot_feed_mode: str = "depth_ws"
    futures_feed_mode: str = "depth_ws"
    symbols: List[str] = field(default_factory=list)  # e.g. ["BTC", "ETH", "SOL"]


@dataclass
class SpreadConfig:
    """Global spread engine configuration."""
    min_spread_pct: float = 0.2
    tracker_min_spread_pct: float = 0.05
    max_spread_pct: float = 50.0  # Cap to filter symbol name collisions
    min_volume_usd: float = 0.0  # 0 = top-of-book pricing (matches ArbMaster); L2 depth still available via /_export/v1/orderbook
    min_inverted: int = 0
    tracking_window_sec: int = 604800  # 7d
    tracker_memory_window_sec: int = 43_200  # 12h
    broadcast_interval_sec: float = 0.15  # Real-time updates (~6.6x per second)
    depth_limit: int = 5  # Minimum depth supported by MEXC/Gate/XT (they accept 5,10,20)
    # Use all discovered symbols (no cap). The per-exchange filtering ensures
    # each exchange only subscribes to symbols it actually lists.
    max_symbols: int = 0  # 0 = unlimited
    symbol_discovery_enabled: bool = True
    symbol_discovery_max_symbols: int = 0  # 0 = unlimited
    min_discovery_volume_usd: float = 100_000  # Filter coins below this 24h volume during discovery (0 = disabled)
    min_opportunity_volume_usd: float = 500  # Filter opportunities where BOTH sides < this volume (0 = disabled)
    startup_warmup_sec: float = 20.0

    # Tracker settings (history + invertidas counting)
    tracker_record_interval_sec: float = 15.0
    tracker_epsilon_pct: float = 0.02
    tracker_max_records_per_pair: int = 0  # 0 = derive from tracker_memory_window_sec / tracker_record_interval_sec
    tracker_gap_threshold_sec: float = 0.0  # 0 = derive from record interval (max(60, 4 * interval))
    tracker_capture_mode: str = "continuous_all_pairs"
    min_total_spread_pct: float = 1.0
    label_cost_floor_pct: float = 1.0
    label_percentile: int = 70
    label_episode_window_days: int = 5
    tracker_db_path: str = ""
    symbols: List[str] = field(default_factory=lambda: [
        "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT",
        "LINK", "MATIC", "UNI", "ATOM", "FIL", "APT", "ARB", "OP",
        "SUI", "SEI", "TIA", "INJ", "PEPE", "WIF", "BONK", "FLOKI",
    ])
    exchanges: List[ExchangeConfig] = field(default_factory=lambda: [
        ExchangeConfig("mexc", spot_feed_mode="ticker_first", futures_feed_mode="ticker_first"),
        ExchangeConfig("bingx", spot_feed_mode="ticker_first", futures_feed_mode="ticker_first"),
        ExchangeConfig("gate", spot_feed_mode="ticker_first", futures_feed_mode="ticker_first"),
        ExchangeConfig("kucoin", spot_feed_mode="ticker_first", futures_feed_mode="ticker_first"),
        ExchangeConfig("xt", spot_feed_mode="ticker_first", futures_feed_mode="ticker_first"),
        ExchangeConfig("bitget", spot_feed_mode="ticker_first", futures_feed_mode="ticker_first"),
    ])
