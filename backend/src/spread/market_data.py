"""Fetches 24h volume and funding rate data from exchange REST APIs.

Runs periodically (every 60s) and caches results. Data is used to enrich
spread opportunities with volume and funding rate information.

Uses asyncio.to_thread + urllib (same as ticker fallbacks) to avoid
event-loop starvation that caused aiohttp requests to silently timeout.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple

from .exchanges.base import _fetch_json_sync

logger = logging.getLogger(__name__)

# Keep base normalization consistent with symbol discovery / WS subscriptions.
from .symbol_discovery import normalize_base

# Cache: (exchange, symbol, market_type) -> {"volume_24h": float, "funding_rate": float, "ts": float}
_cache: Dict[str, dict] = {}
_CACHE_TTL = 120  # seconds


def _cache_key(exchange: str, symbol: str, market_type: str) -> str:
    return f"{exchange}:{symbol}:{market_type}"


VOLUME_UNKNOWN = -1.0  # Sentinel: volume not yet fetched / not available

# Tracks whether the first full market data fetch has completed.
# After this, VOLUME_UNKNOWN means "symbol doesn't exist on that exchange"
# rather than "hasn't been fetched yet".
_market_data_ready = False


def get_volume(exchange: str, symbol: str, market_type: str) -> float:
    """Get cached 24h volume in USDT.

    Returns VOLUME_UNKNOWN (-1.0) when no cache entry exists (not yet fetched).
    Returns 0.0 when the exchange genuinely reports zero volume.
    """
    key = _cache_key(exchange, symbol, market_type)
    entry = _cache.get(key)
    if entry and (time.time() - entry.get("ts", 0)) < _CACHE_TTL * 3:
        return entry.get("volume_24h", 0.0)
    return VOLUME_UNKNOWN


def get_funding_rate(exchange: str, symbol: str) -> Optional[float]:
    """Get cached funding rate for futures."""
    key = _cache_key(exchange, symbol, "futures")
    entry = _cache.get(key)
    if entry and (time.time() - entry.get("ts", 0)) < _CACHE_TTL * 3:
        return entry.get("funding_rate")
    return None


async def fetch_all_market_data(exchanges: list, symbols: list):
    """Fetch volume and funding data from all exchanges. Called periodically (60s)."""
    global _market_data_ready
    tasks = []
    for ex in exchanges:
        tasks.append(_fetch_exchange_data(ex, symbols))
    await asyncio.gather(*tasks, return_exceptions=True)
    _market_data_ready = True


async def _fetch_exchange_data(exchange: str, symbols: list):
    """Fetch data for a single exchange."""
    try:
        fetcher = _EXCHANGE_FETCHERS.get(exchange)
        if fetcher:
            await fetcher(symbols)
    except Exception as e:
        logger.debug(f"[MarketData] {exchange} fetch error: {e}")


# ---------------------------------------------------------------------------
# Exchange-specific fetchers
# ---------------------------------------------------------------------------

async def _fetch_mexc(symbols: list):
    """MEXC: spot tickers + futures tickers."""
    # Spot 24h tickers
    try:
        data = await asyncio.to_thread(_fetch_json_sync, "https://api.mexc.com/api/v3/ticker/24hr")
        if isinstance(data, list):
            for item in data:
                sym = str(item.get("symbol", ""))
                if sym.endswith("USDT"):
                    base = normalize_base(sym[:-4])
                    vol = float(item.get("quoteVolume", 0) or 0)
                    key = _cache_key("mexc", base, "spot")
                    _cache[key] = {"volume_24h": vol, "ts": time.time()}
    except Exception:
        pass

    # Futures tickers (includes funding)
    try:
        data = await asyncio.to_thread(_fetch_json_sync, "https://contract.mexc.com/api/v1/contract/ticker")
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("symbol", ""))
                if "_USDT" in sym:
                    base = normalize_base(sym.split("_")[0])
                    # MEXC futures: `amount24` = USDT turnover, `volume24` = contract count
                    vol = float(item.get("amount24", 0) or item.get("volume24", 0) or 0)
                    fr = item.get("fundingRate")
                    funding = float(fr) if fr is not None else None
                    key = _cache_key("mexc", base, "futures")
                    _cache[key] = {"volume_24h": vol, "funding_rate": funding, "ts": time.time()}
    except Exception:
        pass


async def _fetch_bingx(symbols: list):
    """BingX: spot + swap tickers."""
    # Spot
    try:
        ts_ms = int(time.time() * 1000)
        url = f"https://open-api.bingx.com/openApi/spot/v1/ticker/24hr?timestamp={ts_ms}"
        data = await asyncio.to_thread(_fetch_json_sync, url)
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("symbol", ""))
                if "-USDT" in sym:
                    base = normalize_base(sym.split("-")[0])
                    vol = float(item.get("quoteVolume", 0) or 0)
                    key = _cache_key("bingx", base, "spot")
                    _cache[key] = {"volume_24h": vol, "ts": time.time()}
    except Exception:
        pass

    # Swap (futures)
    try:
        data = await asyncio.to_thread(_fetch_json_sync, "https://open-api.bingx.com/openApi/swap/v2/quote/ticker")
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("symbol", ""))
                if "-USDT" in sym:
                    base = normalize_base(sym.split("-")[0])
                    vol = float(item.get("quoteVolume", 0) or 0)
                    key = _cache_key("bingx", base, "futures")
                    _cache[key] = {"volume_24h": vol, "ts": time.time()}
    except Exception:
        pass

    # Funding rates
    try:
        data = await asyncio.to_thread(_fetch_json_sync, "https://open-api.bingx.com/openApi/swap/v2/quote/premiumIndex")
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("symbol", ""))
                if "-USDT" in sym:
                    base = normalize_base(sym.split("-")[0])
                    fr = item.get("lastFundingRate")
                    if fr is not None:
                        key = _cache_key("bingx", base, "futures")
                        entry = _cache.get(key, {})
                        entry["funding_rate"] = float(fr)
                        entry["ts"] = time.time()
                        _cache[key] = entry
    except Exception:
        pass


async def _fetch_gate(symbols: list):
    """Gate.io: spot + futures tickers."""
    # Spot
    try:
        data = await asyncio.to_thread(_fetch_json_sync, "https://api.gateio.ws/api/v4/spot/tickers")
        if isinstance(data, list):
            for item in data:
                pair = str(item.get("currency_pair", ""))
                if pair.endswith("_USDT"):
                    base = normalize_base(pair[:-5])
                    vol = float(item.get("quote_volume", 0) or 0)
                    key = _cache_key("gate", base, "spot")
                    _cache[key] = {"volume_24h": vol, "ts": time.time()}
    except Exception:
        pass

    # Futures
    try:
        data = await asyncio.to_thread(_fetch_json_sync, "https://api.gateio.ws/api/v4/futures/usdt/tickers")
        if isinstance(data, list):
            for item in data:
                contract = str(item.get("contract", ""))
                if contract.endswith("_USDT"):
                    base = normalize_base(contract[:-5])
                    vol = float(item.get("quote_volume_24h", 0) or item.get("volume_24h_quote", 0) or 0)
                    fr = item.get("funding_rate")
                    funding = float(fr) if fr is not None else None
                    key = _cache_key("gate", base, "futures")
                    _cache[key] = {"volume_24h": vol, "funding_rate": funding, "ts": time.time()}
    except Exception:
        pass


async def _fetch_bitget(symbols: list):
    """Bitget: spot + futures tickers."""
    # Spot
    try:
        data = await asyncio.to_thread(_fetch_json_sync, "https://api.bitget.com/api/v2/spot/market/tickers")
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("symbol", ""))
                if sym.endswith("USDT"):
                    base = normalize_base(sym[:-4])
                    vol = float(item.get("usdtVolume", 0) or item.get("quoteVolume", 0) or 0)
                    key = _cache_key("bitget", base, "spot")
                    _cache[key] = {"volume_24h": vol, "ts": time.time()}
    except Exception:
        pass

    # Futures
    try:
        data = await asyncio.to_thread(_fetch_json_sync, "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES")
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("symbol", ""))
                if sym.endswith("USDT"):
                    base = normalize_base(sym[:-4])
                    vol = float(item.get("usdtVolume", 0) or item.get("quoteVolume", 0) or 0)
                    fr = item.get("fundingRate")
                    funding = float(fr) if fr is not None else None
                    key = _cache_key("bitget", base, "futures")
                    _cache[key] = {"volume_24h": vol, "funding_rate": funding, "ts": time.time()}
    except Exception:
        pass


async def _fetch_kucoin(symbols: list):
    """KuCoin: spot + futures tickers."""
    # Spot
    try:
        data = await asyncio.to_thread(_fetch_json_sync, "https://api.kucoin.com/api/v1/market/allTickers")
        tickers = data.get("data", {}).get("ticker", [])
        if isinstance(tickers, list):
            for item in tickers:
                sym = str(item.get("symbol", ""))
                if sym.endswith("-USDT"):
                    base = normalize_base(sym[:-5])
                    vol = float(item.get("volValue", 0) or 0)
                    key = _cache_key("kucoin", base, "spot")
                    _cache[key] = {"volume_24h": vol, "ts": time.time()}
    except Exception:
        pass

    # Futures - KuCoin futures tickers
    try:
        data = await asyncio.to_thread(_fetch_json_sync, "https://api-futures.kucoin.com/api/v1/contracts/active")
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("symbol", ""))
                if sym.endswith("USDTM"):
                    base = normalize_base(sym[:-5])
                    vol = float(item.get("turnoverOf24h", 0) or item.get("volumeOf24h", 0) or 0)
                    fr = item.get("fundingFeeRate")
                    funding = float(fr) if fr is not None else None
                    key = _cache_key("kucoin", base, "futures")
                    _cache[key] = {"volume_24h": vol, "funding_rate": funding, "ts": time.time()}
    except Exception:
        pass


async def _fetch_xt(symbols: list):
    """XT: spot + futures tickers."""
    # Spot
    try:
        data = await asyncio.to_thread(_fetch_json_sync, "https://sapi.xt.com/v4/public/ticker/24h")
        items = data.get("result") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("s", "") or item.get("symbol", ""))
                if sym.endswith("_usdt") or sym.endswith("_USDT"):
                    base = normalize_base(sym.split("_")[0])
                    # XT spot: `v` = USDT turnover (quote volume), `q` = base token quantity
                    vol = float(item.get("v", 0) or item.get("qv", 0) or item.get("quoteVolume", 0) or 0)
                    key = _cache_key("xt", base, "spot")
                    _cache[key] = {"volume_24h": vol, "ts": time.time()}
    except Exception:
        pass

    # Futures
    try:
        data = await asyncio.to_thread(_fetch_json_sync, "https://fapi.xt.com/future/market/v1/public/q/tickers")
        items = data.get("result") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("s", "") or item.get("symbol", ""))
                if "_usdt" in sym.lower():
                    base = normalize_base(sym.split("_")[0])
                    # XT futures uses `v` as quote volume (USDT).
                    vol = float(item.get("v", 0) or item.get("qv", 0) or item.get("amount", 0) or 0)
                    fr = item.get("fr") or item.get("fundingRate")
                    funding = float(fr) if fr is not None else None
                    key = _cache_key("xt", base, "futures")
                    _cache[key] = {"volume_24h": vol, "funding_rate": funding, "ts": time.time()}
    except Exception:
        pass


# Registry
_EXCHANGE_FETCHERS = {
    "mexc": _fetch_mexc,
    "bingx": _fetch_bingx,
    "gate": _fetch_gate,
    "bitget": _fetch_bitget,
    "kucoin": _fetch_kucoin,
    "xt": _fetch_xt,
}


# ---------------------------------------------------------------------------
# Futures contract metadata: OI, funding interval, next settlement, position limits
# ---------------------------------------------------------------------------

_futures_meta_cache: Dict[str, dict] = {}  # key: "exchange:symbol"
_FUTURES_META_TTL = 120  # seconds


def _fmeta_key(exchange: str, symbol: str) -> str:
    return f"{exchange}:{symbol}"


def get_futures_meta(exchange: str, symbol: str) -> Optional[dict]:
    """Get cached futures metadata.

    Returns {"funding_interval": float|None, "next_settlement": int|None,
             "open_interest": float|None, "position_limit": float|None} or None.
    """
    key = _fmeta_key(exchange, symbol)
    entry = _futures_meta_cache.get(key)
    if entry and (time.time() - entry.get("ts", 0)) < _FUTURES_META_TTL * 3:
        return entry
    return None


def _store_futures_meta(exchange: str, symbol: str, *,
                        funding_interval=None, next_settlement=None,
                        open_interest=None, position_limit=None):
    """Store (or merge) futures metadata into cache."""
    key = _fmeta_key(exchange, symbol)
    existing = _futures_meta_cache.get(key, {})
    now = time.time()
    if funding_interval is not None:
        existing["funding_interval"] = funding_interval
    if next_settlement is not None:
        existing["next_settlement"] = next_settlement
    if open_interest is not None:
        existing["open_interest"] = open_interest
    if position_limit is not None:
        existing["position_limit"] = position_limit
    existing["ts"] = now
    _futures_meta_cache[key] = existing


async def fetch_all_futures_meta(exchanges: list, symbols: list):
    """Fetch OI, funding interval, next settlement, position limits for all exchanges."""
    tasks = []
    symbols_set = set(s.upper() for s in symbols)
    for ex in exchanges:
        fetcher = _FUTURES_META_FETCHERS.get(ex)
        if fetcher:
            tasks.append(fetcher(symbols_set))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total = sum(1 for r in results if not isinstance(r, Exception))
    logger.info(f"[MarketData] Futures meta: {total}/{len(tasks)} exchanges OK, "
                f"{len(_futures_meta_cache)} entries cached")


async def _fmeta_mexc(symbols_set: set):
    """MEXC: ticker (OI) + detail (interval, next settle, limits, contractSize)."""
    # 1. Contract detail — interval, next settle time, limits, contractSize
    contract_info = {}  # symbol -> {collectCycle, nextSettleTime, maxVol, contractSize}
    try:
        data = await asyncio.to_thread(
            _fetch_json_sync, "https://contract.mexc.com/api/v1/contract/detail"
        )
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("symbol", ""))
                if "_USDT" in sym:
                    base = normalize_base(sym.split("_")[0])
                    if base in symbols_set:
                        contract_info[base] = {
                            "collectCycle": item.get("collectCycle"),
                            "nextSettleTime": item.get("nextSettleTime"),
                            "maxVol": item.get("maxVol"),
                            "contractSize": float(item.get("contractSize", 0) or 0),
                        }
    except Exception as e:
        logger.debug(f"[MarketData] MEXC detail error: {e}")

    # 2. Ticker — OI (holdVol), markPrice
    try:
        data = await asyncio.to_thread(
            _fetch_json_sync, "https://contract.mexc.com/api/v1/contract/ticker"
        )
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("symbol", ""))
                if "_USDT" in sym:
                    base = normalize_base(sym.split("_")[0])
                    if base not in symbols_set:
                        continue
                    hold_vol = float(item.get("holdVol", 0) or 0)
                    mark_price = float(item.get("lastPrice", 0) or item.get("fairPrice", 0) or 0)
                    info = contract_info.get(base, {})
                    contract_size = info.get("contractSize", 1.0) or 1.0
                    oi_usdt = hold_vol * contract_size * mark_price if mark_price > 0 else None

                    interval = info.get("collectCycle")
                    if interval is not None:
                        try:
                            interval = float(interval)
                        except (TypeError, ValueError):
                            interval = None

                    next_settle = info.get("nextSettleTime")
                    if next_settle is not None:
                        try:
                            next_settle = int(next_settle)
                        except (TypeError, ValueError):
                            next_settle = None

                    max_vol = info.get("maxVol")
                    pos_limit = None
                    if max_vol is not None and mark_price > 0:
                        try:
                            pos_limit = float(max_vol) * contract_size * mark_price
                        except (TypeError, ValueError):
                            pass

                    _store_futures_meta("mexc", base,
                                        funding_interval=interval,
                                        next_settlement=next_settle,
                                        open_interest=oi_usdt,
                                        position_limit=pos_limit)
    except Exception as e:
        logger.debug(f"[MarketData] MEXC ticker meta error: {e}")


async def _fmeta_bingx(symbols_set: set):
    """BingX: premiumIndex (next funding time) + per-symbol OI."""
    # 1. Bulk: premiumIndex — nextFundingTime for all symbols
    next_times = {}
    try:
        data = await asyncio.to_thread(
            _fetch_json_sync, "https://open-api.bingx.com/openApi/swap/v2/quote/premiumIndex"
        )
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("symbol", ""))
                if "-USDT" in sym:
                    base = normalize_base(sym.split("-")[0])
                    if base in symbols_set:
                        nft = item.get("nextFundingTime")
                        if nft is not None:
                            try:
                                next_times[base] = int(nft)
                            except (TypeError, ValueError):
                                pass
    except Exception as e:
        logger.debug(f"[MarketData] BingX premiumIndex meta error: {e}")

    # Store next_settlement + default interval (8h — not exposed in BingX API)
    for base, nft in next_times.items():
        _store_futures_meta("bingx", base,
                            funding_interval=8.0,
                            next_settlement=nft)

    # 2. Per-symbol: OI (already in USDT)
    sem = asyncio.Semaphore(5)

    async def _fetch_oi(base: str):
        async with sem:
            try:
                url = f"https://open-api.bingx.com/openApi/swap/v2/quote/openInterest?symbol={base}-USDT"
                data = await asyncio.to_thread(_fetch_json_sync, url)
                oi_data = data.get("data") if isinstance(data, dict) else data
                if isinstance(oi_data, dict):
                    oi = float(oi_data.get("openInterest", 0) or 0)
                    if oi > 0:
                        _store_futures_meta("bingx", base, open_interest=oi)
            except Exception:
                pass

    # Only fetch OI for symbols that are in the active set (limit API calls)
    targets = [b for b in symbols_set if b in next_times][:100]
    await asyncio.gather(*[_fetch_oi(b) for b in targets], return_exceptions=True)


async def _fmeta_gate(symbols_set: set):
    """Gate.io: single /contracts call returns everything."""
    try:
        data = await asyncio.to_thread(
            _fetch_json_sync, "https://api.gateio.ws/api/v4/futures/usdt/contracts"
        )
        if not isinstance(data, list):
            return
        for item in data:
            contract = str(item.get("name", ""))
            if not contract.endswith("_USDT"):
                continue
            base = normalize_base(contract[:-5])
            if base not in symbols_set:
                continue

            # OI: position_size (contracts) × quanto_multiplier × mark_price
            pos_size = float(item.get("position_size", 0) or 0)
            quanto = float(item.get("quanto_multiplier", 1) or 1)
            mark_price = float(item.get("mark_price", 0) or 0)
            oi_usdt = pos_size * quanto * mark_price if mark_price > 0 else None

            # Funding interval (seconds → hours)
            fi_sec = item.get("funding_interval")
            interval = None
            if fi_sec is not None:
                try:
                    interval = float(fi_sec) / 3600.0
                except (TypeError, ValueError):
                    pass

            # Next settlement (SECONDS → ms)
            next_apply = item.get("funding_next_apply")
            next_settle = None
            if next_apply is not None:
                try:
                    next_settle = int(float(next_apply) * 1000)
                except (TypeError, ValueError):
                    pass

            # Position limit
            order_max = item.get("order_size_max")
            pos_limit = None
            if order_max is not None and mark_price > 0:
                try:
                    pos_limit = float(order_max) * quanto * mark_price
                except (TypeError, ValueError):
                    pass

            _store_futures_meta("gate", base,
                                funding_interval=interval,
                                next_settlement=next_settle,
                                open_interest=oi_usdt,
                                position_limit=pos_limit)
    except Exception as e:
        logger.debug(f"[MarketData] Gate contracts meta error: {e}")


async def _fmeta_bitget(symbols_set: set):
    """Bitget: current-fund-rate (interval, next) + tickers (OI) + contracts (limits)."""
    # 1. Current fund rate — interval + next settlement (bulk)
    try:
        data = await asyncio.to_thread(
            _fetch_json_sync,
            "https://api.bitget.com/api/v2/mix/market/current-fund-rate?productType=USDT-FUTURES"
        )
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("symbol", ""))
                if sym.endswith("USDT"):
                    base = normalize_base(sym[:-4])
                    if base not in symbols_set:
                        continue
                    interval = item.get("fundingRateInterval")
                    try:
                        interval = float(interval) if interval is not None else None
                    except (TypeError, ValueError):
                        interval = None

                    next_update = item.get("nextUpdate")
                    next_settle = None
                    if next_update is not None:
                        try:
                            next_settle = int(next_update)
                        except (TypeError, ValueError):
                            pass

                    _store_futures_meta("bitget", base,
                                        funding_interval=interval,
                                        next_settlement=next_settle)
    except Exception as e:
        logger.debug(f"[MarketData] Bitget fund-rate meta error: {e}")

    # 2. Tickers — OI (openInterestUsd already in USDT in some responses)
    try:
        data = await asyncio.to_thread(
            _fetch_json_sync,
            "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES"
        )
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("symbol", ""))
                if sym.endswith("USDT"):
                    base = normalize_base(sym[:-4])
                    if base not in symbols_set:
                        continue
                    # Try openInterestUsd first, fallback to openInterest × lastPr
                    oi_usdt_raw = item.get("openInterestUsd")
                    if oi_usdt_raw is not None:
                        try:
                            oi = float(oi_usdt_raw)
                            if oi > 0:
                                _store_futures_meta("bitget", base, open_interest=oi)
                                continue
                        except (TypeError, ValueError):
                            pass
                    oi_raw = item.get("openInterest")
                    last_pr = item.get("lastPr") or item.get("markPrice")
                    if oi_raw is not None and last_pr is not None:
                        try:
                            oi = float(oi_raw) * float(last_pr)
                            if oi > 0:
                                _store_futures_meta("bitget", base, open_interest=oi)
                        except (TypeError, ValueError):
                            pass
    except Exception as e:
        logger.debug(f"[MarketData] Bitget tickers OI error: {e}")

    # 3. Contracts — position limits
    try:
        data = await asyncio.to_thread(
            _fetch_json_sync,
            "https://api.bitget.com/api/v2/mix/market/contracts?productType=USDT-FUTURES"
        )
        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("symbol", ""))
                if sym.endswith("USDT"):
                    base = normalize_base(sym[:-4])
                    if base not in symbols_set:
                        continue
                    pos_limit_raw = item.get("posLimit") or item.get("maxPositionNum")
                    if pos_limit_raw is not None:
                        try:
                            _store_futures_meta("bitget", base,
                                                position_limit=float(pos_limit_raw))
                        except (TypeError, ValueError):
                            pass
    except Exception as e:
        logger.debug(f"[MarketData] Bitget contracts limits error: {e}")


async def _fmeta_kucoin(symbols_set: set):
    """KuCoin: single /contracts/active call returns everything."""
    try:
        data = await asyncio.to_thread(
            _fetch_json_sync, "https://api-futures.kucoin.com/api/v1/contracts/active"
        )
        items = data.get("data") if isinstance(data, dict) else data
        if not isinstance(items, list):
            return
        for item in items:
            sym = str(item.get("symbol", ""))
            if not sym.endswith("USDTM"):
                continue
            base = normalize_base(sym[:-5])
            if base not in symbols_set:
                continue

            # OI: openInterest × multiplier × markPrice
            oi_contracts = float(item.get("openInterest", 0) or 0)
            multiplier = float(item.get("multiplier", 1) or 1)
            mark_price = float(item.get("markPrice", 0) or 0)
            oi_usdt = oi_contracts * multiplier * mark_price if mark_price > 0 else None

            # Funding interval: fundingRateGranularity (milliseconds → hours)
            gran_ms = item.get("fundingRateGranularity")
            interval = None
            if gran_ms is not None:
                try:
                    interval = float(gran_ms) / 3_600_000.0
                except (TypeError, ValueError):
                    pass

            # Next settlement: nextFundingRateDateTime (ms) — NOT nextFundingRateTime!
            next_dt = item.get("nextFundingRateDateTime")
            next_settle = None
            if next_dt is not None:
                try:
                    next_settle = int(next_dt)
                except (TypeError, ValueError):
                    pass

            # Position limit: maxRiskLimit (USDT)
            max_risk = item.get("maxRiskLimit")
            pos_limit = None
            if max_risk is not None:
                try:
                    pos_limit = float(max_risk)
                except (TypeError, ValueError):
                    pass

            _store_futures_meta("kucoin", base,
                                funding_interval=interval,
                                next_settlement=next_settle,
                                open_interest=oi_usdt,
                                position_limit=pos_limit)
    except Exception as e:
        logger.debug(f"[MarketData] KuCoin contracts meta error: {e}")


async def _fmeta_xt(symbols_set: set):
    """XT: tickers (OI) + per-symbol funding-rate (interval, next settle)."""
    # 1. Bulk: tickers — OI
    oi_done = set()
    try:
        data = await asyncio.to_thread(
            _fetch_json_sync, "https://fapi.xt.com/future/market/v1/public/q/tickers"
        )
        items = data.get("result") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                sym = str(item.get("s", "") or item.get("symbol", ""))
                if "_usdt" not in sym.lower():
                    continue
                base = normalize_base(sym.split("_")[0])
                if base not in symbols_set:
                    continue
                # Try openInterestUsd first
                oi_usdt_raw = item.get("openInterestUsd") or item.get("oi")
                if oi_usdt_raw is not None:
                    try:
                        oi = float(oi_usdt_raw)
                        if oi > 0:
                            _store_futures_meta("xt", base, open_interest=oi)
                            oi_done.add(base)
                            continue
                    except (TypeError, ValueError):
                        pass
                # Fallback: openInterest (base qty) × lastPrice
                oi_raw = item.get("openInterest")
                last_pr = item.get("p") or item.get("lastPrice")
                if oi_raw is not None and last_pr is not None:
                    try:
                        oi = float(oi_raw) * float(last_pr)
                        if oi > 0:
                            _store_futures_meta("xt", base, open_interest=oi)
                            oi_done.add(base)
                    except (TypeError, ValueError):
                        pass
    except Exception as e:
        logger.debug(f"[MarketData] XT tickers OI error: {e}")

    # 2. Per-symbol: funding-rate detail (interval + next settlement)
    sem = asyncio.Semaphore(5)

    async def _fetch_fr_detail(base: str):
        async with sem:
            try:
                url = f"https://fapi.xt.com/future/market/v1/public/q/funding-rate?symbol={base.lower()}_usdt"
                data = await asyncio.to_thread(_fetch_json_sync, url)
                result = data.get("result") if isinstance(data, dict) else data
                if not isinstance(result, dict):
                    return

                # collectionInternal (typo in API, means "interval") — hours
                ci = result.get("collectionInternal")
                interval = None
                if ci is not None:
                    try:
                        interval = float(ci)
                    except (TypeError, ValueError):
                        pass

                # nextCollectionTime — ms
                nct = result.get("nextCollectionTime")
                next_settle = None
                if nct is not None:
                    try:
                        next_settle = int(nct)
                    except (TypeError, ValueError):
                        pass

                _store_futures_meta("xt", base,
                                    funding_interval=interval,
                                    next_settlement=next_settle)
            except Exception:
                pass

    targets = [b for b in symbols_set if b in oi_done][:80]
    await asyncio.gather(*[_fetch_fr_detail(b) for b in targets], return_exceptions=True)


_FUTURES_META_FETCHERS = {
    "mexc": _fmeta_mexc,
    "bingx": _fmeta_bingx,
    "gate": _fmeta_gate,
    "bitget": _fmeta_bitget,
    "kucoin": _fmeta_kucoin,
    "xt": _fmeta_xt,
}


# ---------------------------------------------------------------------------
# Bulk volume pre-fetch (for symbol discovery filtering)
# ---------------------------------------------------------------------------

async def fetch_detailed_volumes(exchanges: list) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Fetch 24h USDT volume per base, per exchange, per market type.

    Returns:
        {exchange: {"spot": {base: vol}, "futures": {base: vol}}}

    Used to filter which symbols to subscribe on each exchange+market,
    keeping only pairs with sufficient volume.
    """
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    lock = asyncio.Lock()

    async def _accum(exchange: str, market: str, base: str, vol: float):
        async with lock:
            if exchange not in result:
                result[exchange] = {"spot": {}, "futures": {}}
            cur = result[exchange][market].get(base)
            if cur is None or vol > cur:
                result[exchange][market][base] = vol

    async def _fetch_mexc_vol():
        try:
            data = await asyncio.to_thread(_fetch_json_sync, "https://api.mexc.com/api/v3/ticker/24hr")
            if isinstance(data, list):
                for item in data:
                    sym = str(item.get("symbol", ""))
                    if sym.endswith("USDT"):
                        base = normalize_base(sym[:-4])
                        vol = float(item.get("quoteVolume", 0) or 0)
                        await _accum("mexc", "spot", base, vol)
        except Exception:
            pass
        try:
            data = await asyncio.to_thread(_fetch_json_sync, "https://futures.mexc.com/api/v1/contract/ticker")
            items = data.get("data") if isinstance(data, dict) else data
            if isinstance(items, list):
                for item in items:
                    sym = str(item.get("symbol", ""))
                    if sym.endswith("_USDT"):
                        base = normalize_base(sym[:-5])
                        vol = float(item.get("amount24", 0) or 0)
                        await _accum("mexc", "futures", base, vol)
        except Exception:
            pass

    async def _fetch_bingx_vol():
        try:
            ts_ms = int(time.time() * 1000)
            data = await asyncio.to_thread(_fetch_json_sync, f"https://open-api.bingx.com/openApi/spot/v1/ticker/24hr?timestamp={ts_ms}")
            items = data.get("data") if isinstance(data, dict) else data
            if isinstance(items, list):
                for item in items:
                    sym = str(item.get("symbol", ""))
                    if "-USDT" in sym:
                        base = normalize_base(sym.split("-")[0])
                        vol = float(item.get("quoteVolume", 0) or 0)
                        await _accum("bingx", "spot", base, vol)
        except Exception:
            pass
        try:
            data = await asyncio.to_thread(_fetch_json_sync, "https://open-api.bingx.com/openApi/swap/v2/quote/ticker")
            items = data.get("data") if isinstance(data, dict) else data
            if isinstance(items, list):
                for item in items:
                    sym = str(item.get("symbol", ""))
                    if "-USDT" in sym:
                        base = normalize_base(sym.split("-")[0])
                        vol = float(item.get("quoteVolume", 0) or 0)
                        await _accum("bingx", "futures", base, vol)
        except Exception:
            pass

    async def _fetch_gate_vol():
        try:
            data = await asyncio.to_thread(_fetch_json_sync, "https://api.gateio.ws/api/v4/spot/tickers")
            if isinstance(data, list):
                for item in data:
                    pair = str(item.get("currency_pair", ""))
                    if pair.endswith("_USDT"):
                        base = normalize_base(pair[:-5])
                        vol = float(item.get("quote_volume", 0) or 0)
                        await _accum("gate", "spot", base, vol)
        except Exception:
            pass
        try:
            data = await asyncio.to_thread(_fetch_json_sync, "https://api.gateio.ws/api/v4/futures/usdt/tickers")
            if isinstance(data, list):
                for item in data:
                    contract = str(item.get("contract", ""))
                    if contract.endswith("_USDT"):
                        base = normalize_base(contract[:-5])
                        vol = float(item.get("quote_volume_24h", 0) or item.get("volume_24h_quote", 0) or 0)
                        await _accum("gate", "futures", base, vol)
        except Exception:
            pass

    async def _fetch_bitget_vol():
        try:
            data = await asyncio.to_thread(_fetch_json_sync, "https://api.bitget.com/api/v2/spot/market/tickers")
            items = data.get("data") if isinstance(data, dict) else data
            if isinstance(items, list):
                for item in items:
                    sym = str(item.get("symbol", ""))
                    if sym.endswith("USDT"):
                        base = normalize_base(sym[:-4])
                        vol = float(item.get("usdtVolume", 0) or item.get("quoteVolume", 0) or 0)
                        await _accum("bitget", "spot", base, vol)
        except Exception:
            pass
        try:
            data = await asyncio.to_thread(_fetch_json_sync, "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES")
            items = data.get("data") if isinstance(data, dict) else data
            if isinstance(items, list):
                for item in items:
                    sym = str(item.get("symbol", ""))
                    if "USDT" in sym:
                        base = normalize_base(sym.replace("USDT", ""))
                        vol = float(item.get("quoteVolume", 0) or item.get("usdtVolume", 0) or 0)
                        await _accum("bitget", "futures", base, vol)
        except Exception:
            pass

    async def _fetch_kucoin_vol():
        try:
            data = await asyncio.to_thread(_fetch_json_sync, "https://api.kucoin.com/api/v1/market/allTickers")
            tickers = data.get("data", {}).get("ticker", [])
            if isinstance(tickers, list):
                for item in tickers:
                    sym = str(item.get("symbol", ""))
                    if sym.endswith("-USDT"):
                        base = normalize_base(sym[:-5])
                        vol = float(item.get("volValue", 0) or 0)
                        await _accum("kucoin", "spot", base, vol)
        except Exception:
            pass
        try:
            data = await asyncio.to_thread(_fetch_json_sync, "https://api-futures.kucoin.com/api/v1/contracts/active")
            items = data.get("data") if isinstance(data, dict) else data
            if isinstance(items, list):
                for item in items:
                    sym = str(item.get("symbol", ""))
                    if sym.endswith("USDTM"):
                        base = normalize_base(sym[:-5])
                        vol = float(item.get("turnoverOf24h", 0) or item.get("volumeOf24h", 0) or 0)
                        await _accum("kucoin", "futures", base, vol)
        except Exception:
            pass

    async def _fetch_xt_vol():
        try:
            data = await asyncio.to_thread(_fetch_json_sync, "https://sapi.xt.com/v4/public/ticker/24h")
            items = data.get("result") if isinstance(data, dict) else data
            if isinstance(items, list):
                for item in items:
                    sym = str(item.get("s", "") or item.get("symbol", ""))
                    if sym.endswith("_usdt") or sym.endswith("_USDT"):
                        base = normalize_base(sym.split("_")[0])
                        vol = float(item.get("v", 0) or item.get("qv", 0) or 0)
                        await _accum("xt", "spot", base, vol)
        except Exception:
            pass
        try:
            data = await asyncio.to_thread(_fetch_json_sync, "https://fapi.xt.com/future/market/v1/public/q/tickers")
            items = data.get("result") if isinstance(data, dict) else data
            if isinstance(items, list):
                for item in items:
                    sym = str(item.get("s", "") or item.get("symbol", ""))
                    if "_usdt" in sym.lower():
                        base = normalize_base(sym.split("_")[0])
                        vol = float(item.get("v", 0) or item.get("qv", 0) or 0)
                        await _accum("xt", "futures", base, vol)
        except Exception:
            pass

    _vol_fetchers = {
        "mexc": _fetch_mexc_vol,
        "bingx": _fetch_bingx_vol,
        "gate": _fetch_gate_vol,
        "bitget": _fetch_bitget_vol,
        "kucoin": _fetch_kucoin_vol,
        "xt": _fetch_xt_vol,
    }

    tasks = []
    for ex in exchanges:
        fn = _vol_fetchers.get(ex)
        if fn:
            tasks.append(fn())
    await asyncio.gather(*tasks, return_exceptions=True)

    total_entries = sum(
        len(m.get("spot", {})) + len(m.get("futures", {}))
        for m in result.values()
    )
    logger.info(f"[MarketData] Detailed volume: {len(result)} exchanges, {total_entries} entries")
    return result


async def fetch_bulk_volumes(exchanges: list) -> Dict[str, float]:
    """Quick bulk volume fetch — returns max 24h USDT vol per base (across all exchanges).

    Backwards-compatible wrapper around fetch_detailed_volumes.
    """
    detailed = await fetch_detailed_volumes(exchanges)
    volumes: Dict[str, float] = {}
    for _ex, markets in detailed.items():
        for _mkt, base_vols in markets.items():
            for base, vol in base_vols.items():
                cur = volumes.get(base)
                if cur is None or vol > cur:
                    volumes[base] = vol
    logger.info(f"[MarketData] Bulk volume: {len(volumes)} symbols with volume data")
    return volumes


# ---------------------------------------------------------------------------
# Funding history fetch (per-symbol, for details modal)
# ---------------------------------------------------------------------------

_funding_history_cache: Dict[str, dict] = {}
_FUNDING_HISTORY_TTL = 300  # 5 minutes


async def fetch_funding_history(exchange: str, symbol: str) -> list:
    """Fetch recent funding rate history for a symbol from an exchange. Cached 5min."""
    cache_key = f"fh:{exchange}:{symbol}"
    cached = _funding_history_cache.get(cache_key)
    if cached and (time.time() - cached.get("ts", 0)) < _FUNDING_HISTORY_TTL:
        return cached.get("data", [])

    result = []
    try:
        fetcher = _FUNDING_HISTORY_FETCHERS.get(exchange)
        if fetcher:
            result = await fetcher(symbol)
    except Exception as e:
        logger.debug(f"[MarketData] funding_history {exchange}:{symbol} error: {e}")

    _funding_history_cache[cache_key] = {"data": result, "ts": time.time()}
    return result


async def _fh_mexc(symbol: str) -> list:
    url = f"https://contract.mexc.com/api/v1/contract/funding_rate/{symbol}_USDT"
    data = await asyncio.to_thread(_fetch_json_sync, url)
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    return [{"funding_time": int(r.get("settleTime", 0)), "funding_rate": str(r.get("fundingRate", 0))} for r in items[-50:]]


async def _fh_bingx(symbol: str) -> list:
    url = f"https://open-api.bingx.com/openApi/swap/v2/quote/fundingRate?symbol={symbol}-USDT"
    data = await asyncio.to_thread(_fetch_json_sync, url)
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    return [{"funding_time": int(r.get("fundingTime", 0)), "funding_rate": str(r.get("fundingRate", 0))} for r in items[-50:]]


async def _fh_gate(symbol: str) -> list:
    url = f"https://api.gateio.ws/api/v4/futures/usdt/funding_rate?contract={symbol}_USDT&limit=50"
    data = await asyncio.to_thread(_fetch_json_sync, url)
    if not isinstance(data, list):
        return []
    return [{"funding_time": int(float(r.get("t", 0)) * 1000), "funding_rate": str(r.get("r", 0))} for r in data[-50:]]


async def _fh_bitget(symbol: str) -> list:
    url = f"https://api.bitget.com/api/v2/mix/market/history-fund-rate?symbol={symbol}USDT&productType=USDT-FUTURES"
    data = await asyncio.to_thread(_fetch_json_sync, url)
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    return [{"funding_time": int(r.get("fundingTime", 0)), "funding_rate": str(r.get("fundingRate", 0))} for r in items[-50:]]


async def _fh_kucoin(symbol: str) -> list:
    url = f"https://api-futures.kucoin.com/api/v1/contract/funding-rates?symbol={symbol}USDTM"
    data = await asyncio.to_thread(_fetch_json_sync, url)
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    return [{"funding_time": int(r.get("timePoint", 0)), "funding_rate": str(r.get("fundingRate", 0))} for r in items[-50:]]


async def _fh_xt(symbol: str) -> list:
    url = f"https://fapi.xt.com/future/market/v1/public/q/funding-rate?symbol={symbol.lower()}_usdt"
    data = await asyncio.to_thread(_fetch_json_sync, url)
    items = data.get("result") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    return [{"funding_time": int(r.get("createdTime", 0) or r.get("time", 0)), "funding_rate": str(r.get("fundingRate", 0))} for r in items[-50:]]


_FUNDING_HISTORY_FETCHERS = {
    "mexc": _fh_mexc,
    "bingx": _fh_bingx,
    "gate": _fh_gate,
    "bitget": _fh_bitget,
    "kucoin": _fh_kucoin,
    "xt": _fh_xt,
}


# ---------------------------------------------------------------------------
# Bulk funding history (last 24 cycles per exchange:symbol)
# ---------------------------------------------------------------------------

_funding_history_bulk_cache: Dict[str, dict] = {}  # key: "exchange:symbol"
_FUNDING_HISTORY_BULK_TTL = 180  # 3 minutes


def get_funding_history_bulk(exchange: str, symbol: str) -> Optional[list]:
    """Get cached bulk funding rate history (last 24 cycles).

    Returns list of {"funding_time": int_ms, "funding_rate": str} or None.
    """
    key = f"fhb:{exchange}:{symbol}"
    entry = _funding_history_bulk_cache.get(key)
    if entry and (time.time() - entry.get("ts", 0)) < _FUNDING_HISTORY_BULK_TTL * 2:
        return entry.get("data")
    return None


async def fetch_all_funding_history(exchanges: list, symbols: list):
    """Fetch last 24 cycles of funding history for all exchange:symbol pairs.

    Reuses existing _fh_* per-symbol fetchers with semaphore rate limiting.
    Only fetches for symbols in the provided list (active opportunities).
    """
    count = 0
    sem_per_ex: Dict[str, asyncio.Semaphore] = {ex: asyncio.Semaphore(3) for ex in exchanges}

    async def _fetch_one(exchange: str, symbol: str):
        nonlocal count
        cache_key = f"fhb:{exchange}:{symbol}"
        # Skip if recently fetched
        cached = _funding_history_bulk_cache.get(cache_key)
        if cached and (time.time() - cached.get("ts", 0)) < _FUNDING_HISTORY_BULK_TTL:
            return

        sem = sem_per_ex.get(exchange)
        if not sem:
            return
        async with sem:
            fetcher = _FUNDING_HISTORY_FETCHERS.get(exchange)
            if not fetcher:
                return
            try:
                result = await fetcher(symbol)
                if result:
                    _funding_history_bulk_cache[cache_key] = {"data": result, "ts": time.time()}
                    count += 1
            except Exception:
                pass

    tasks = []
    for ex in exchanges:
        if ex not in _FUNDING_HISTORY_FETCHERS:
            continue
        for sym in symbols:
            tasks.append(_fetch_one(ex, sym))

    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(f"[MarketData] Funding history bulk: {count} new entries "
                f"({len(_funding_history_bulk_cache)} total cached)")


# ---------------------------------------------------------------------------
# Network / chain info fetch (per-symbol per-exchange)
# ---------------------------------------------------------------------------

_network_cache: Dict[str, dict] = {}
_NETWORK_CACHE_TTL = 300  # 5 minutes

# Withdraw status cache (populated by network fetches)
_withdraw_cache: Dict[str, dict] = {}


def get_withdraw_status(exchange: str, symbol: str) -> Optional[bool]:
    """Get cached withdrawal status. True = withdrawals active on at least one chain."""
    key = f"{exchange}:{symbol}"
    entry = _withdraw_cache.get(key)
    if entry and (time.time() - entry.get("ts", 0)) < 700:
        return entry.get("withdraw_ok")
    return None


def get_deposit_withdraw_status(exchange: str, symbol: str) -> Optional[dict]:
    """Get cached deposit + withdrawal status.

    Returns {"deposit_ok": bool, "withdraw_ok": bool} or None if not cached.
    """
    key = f"{exchange}:{symbol}"
    entry = _withdraw_cache.get(key)
    if entry and (time.time() - entry.get("ts", 0)) < 700:
        return {"deposit_ok": entry.get("deposit_ok", False), "withdraw_ok": entry.get("withdraw_ok", False)}
    return None


async def fetch_coin_networks(exchange: str, symbol: str) -> list:
    """Fetch network/chain info with deposit/withdrawal status. Cached 5min."""
    cache_key = f"net:{exchange}:{symbol}"
    cached = _network_cache.get(cache_key)
    if cached and (time.time() - cached.get("ts", 0)) < _NETWORK_CACHE_TTL:
        return cached.get("data", [])

    result = []
    try:
        fetcher = _NETWORK_FETCHERS.get(exchange)
        if fetcher:
            result = await fetcher(symbol)
    except Exception as e:
        logger.debug(f"[MarketData] networks {exchange}:{symbol} error: {e}")

    _network_cache[cache_key] = {"data": result, "ts": time.time()}

    # Update withdraw status cache (both deposit + withdraw)
    if result:
        any_withdraw = any(c.get("withdrawEnable") for c in result)
        any_deposit = any(c.get("depositEnable") for c in result)
        _withdraw_cache[f"{exchange}:{symbol}"] = {
            "withdraw_ok": any_withdraw, "deposit_ok": any_deposit, "ts": time.time()
        }

    return result


async def _net_mexc(symbol: str) -> list:
    # NOTE: This endpoint requires API key. Will return [] without one.
    url = "https://api.mexc.com/api/v3/capital/config/getall"
    data = await asyncio.to_thread(_fetch_json_sync, url)
    if not isinstance(data, list):
        return []
    for coin in data:
        if str(coin.get("coin", "")).upper() == symbol.upper():
            nets = coin.get("networkList", [])
            return [{
                "chain": n.get("network", n.get("netWork", "")),
                "withdrawFee": n.get("withdrawFee"),
                "minWithdrawAmount": n.get("withdrawMin"),
                "depositEnable": n.get("depositEnable", False),
                "withdrawEnable": n.get("withdrawEnable", False),
                "contractAddress": n.get("contract", ""),
            } for n in nets]
    return []


async def _net_bingx(symbol: str) -> list:
    # NOTE: This endpoint requires API key. Will return [] without one.
    url = "https://open-api.bingx.com/openApi/wallets/v1/capital/config/getall"
    data = await asyncio.to_thread(_fetch_json_sync, url)
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    for coin in items:
        if str(coin.get("coin", "")).upper() == symbol.upper():
            nets = coin.get("networkList", [])
            return [{
                "chain": n.get("network", ""),
                "withdrawFee": n.get("withdrawFee"),
                "minWithdrawAmount": n.get("withdrawMin"),
                "depositEnable": n.get("depositEnable", False),
                "withdrawEnable": n.get("withdrawEnable", False),
                "contractAddress": n.get("contract", ""),
            } for n in nets]
    return []


async def _net_gate(symbol: str) -> list:
    url = f"https://api.gateio.ws/api/v4/spot/currencies/{symbol}"
    data = await asyncio.to_thread(_fetch_json_sync, url)
    if not isinstance(data, dict):
        return []
    chains = data.get("chains", [])
    return [{
        "chain": c.get("name", ""),
        "withdrawFee": None,
        "minWithdrawAmount": None,
        "withdrawEnable": not c.get("withdraw_disabled", True),
        "depositEnable": not c.get("deposit_disabled", True),
        "contractAddress": c.get("addr", ""),
    } for c in chains]


async def _net_bitget(symbol: str) -> list:
    url = "https://api.bitget.com/api/v2/spot/public/coins"
    data = await asyncio.to_thread(_fetch_json_sync, url)
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    for coin in items:
        if str(coin.get("coin", "")).upper() == symbol.upper():
            chains = coin.get("chains", [])
            return [{
                "chain": c.get("chain", ""),
                "withdrawFee": c.get("withdrawFee"),
                "minWithdrawAmount": c.get("minWithdrawAmount"),
                "withdrawEnable": str(c.get("withdrawable", "")).lower() == "true",
                "depositEnable": str(c.get("rechargeable", "")).lower() == "true",
                "contractAddress": c.get("contractAddress", ""),
            } for c in chains]
    return []


async def _net_kucoin(symbol: str) -> list:
    url = f"https://api.kucoin.com/api/v3/currencies/{symbol}"
    data = await asyncio.to_thread(_fetch_json_sync, url)
    item = data.get("data") if isinstance(data, dict) else data
    if not isinstance(item, dict):
        return []
    chains = item.get("chains", [])
    return [{
        "chain": c.get("chainName", ""),
        "withdrawFee": c.get("withdrawalMinFee"),
        "minWithdrawAmount": c.get("withdrawalMinSize"),
        "depositEnable": c.get("isDepositEnabled", False),
        "withdrawEnable": c.get("isWithdrawEnabled", False),
        "contractAddress": c.get("contractAddress", ""),
    } for c in chains]


async def _net_xt(symbol: str) -> list:
    url = "https://sapi.xt.com/v4/public/wallet/support/currency"
    data = await asyncio.to_thread(_fetch_json_sync, url)
    result = data.get("result") if isinstance(data, dict) else data
    if isinstance(result, list):
        # Filter for the matching currency
        for coin in result:
            if str(coin.get("currency", "")).lower() == symbol.lower():
                chains = coin.get("supportChains", [])
                return [{
                    "chain": c.get("chain", c.get("chainName", "")),
                    "withdrawFee": c.get("withdrawFeeAmount"),
                    "minWithdrawAmount": c.get("withdrawMinAmount"),
                    "depositEnable": c.get("depositEnabled", False),
                    "withdrawEnable": c.get("withdrawEnabled", False),
                    "contractAddress": c.get("contract", ""),
                } for c in chains]
    elif isinstance(result, dict):
        chains = result.get("supportChains", result.get("chains", []))
        return [{
            "chain": c.get("chain", c.get("chainName", "")),
            "withdrawFee": c.get("withdrawFeeAmount"),
            "minWithdrawAmount": c.get("withdrawMinAmount"),
            "depositEnable": c.get("depositEnabled", False),
            "withdrawEnable": c.get("withdrawEnabled", False),
            "contractAddress": c.get("contract", ""),
        } for c in chains]
    return []


_NETWORK_FETCHERS = {
    "mexc": _net_mexc,
    "bingx": _net_bingx,
    "gate": _net_gate,
    "bitget": _net_bitget,
    "kucoin": _net_kucoin,
    "xt": _net_xt,
}


# ---------------------------------------------------------------------------
# Bulk withdrawal status fetch (for scanner table indicators)
# ---------------------------------------------------------------------------

async def fetch_bulk_withdraw_status():
    """Bulk-fetch withdrawal status from exchanges that return all coins in one request.
    Populates _withdraw_cache for use in scanner table indicators."""
    count = 0

    async def _bulk_bitget():
        nonlocal count
        try:
            url = "https://api.bitget.com/api/v2/spot/public/coins"
            data = await asyncio.to_thread(_fetch_json_sync, url)
            items = data.get("data") if isinstance(data, dict) else data
            if not isinstance(items, list):
                return
            now = time.time()
            for coin in items:
                sym = str(coin.get("coin", "")).upper()
                if not sym:
                    continue
                chains = coin.get("chains", [])
                any_wd = any(str(c.get("withdrawable", "")).lower() == "true" for c in chains)
                any_dp = any(str(c.get("rechargeable", "")).lower() == "true" for c in chains)
                _withdraw_cache[f"bitget:{sym}"] = {"withdraw_ok": any_wd, "deposit_ok": any_dp, "ts": now}
                count += 1
        except Exception as e:
            logger.debug(f"[MarketData] bulk withdraw bitget error: {e}")

    async def _bulk_xt():
        nonlocal count
        try:
            url = "https://sapi.xt.com/v4/public/wallet/support/currency"
            data = await asyncio.to_thread(_fetch_json_sync, url)
            result = data.get("result") if isinstance(data, dict) else data
            if not isinstance(result, list):
                return
            now = time.time()
            for coin in result:
                sym = str(coin.get("currency", "")).upper()
                if not sym:
                    continue
                chains = coin.get("supportChains", [])
                any_wd = any(c.get("withdrawEnabled", False) for c in chains)
                any_dp = any(c.get("depositEnabled", False) for c in chains)
                _withdraw_cache[f"xt:{sym}"] = {"withdraw_ok": any_wd, "deposit_ok": any_dp, "ts": now}
                count += 1
        except Exception as e:
            logger.debug(f"[MarketData] bulk withdraw xt error: {e}")

    await asyncio.gather(_bulk_bitget(), _bulk_xt(), return_exceptions=True)
    logger.info(f"[MarketData] Bulk withdraw status: {count} symbols cached")
