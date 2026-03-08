"""Discover all available USDT perpetual/spot symbols from each exchange.

Fetches symbol lists from REST APIs and finds the union of all available
base assets across all exchanges. Returns normalized base symbols (e.g. "BTC").
"""

import asyncio
import logging
import re
from typing import Dict, List, Set

import aiohttp

logger = logging.getLogger(__name__)

# Normalize multiplier prefixes: 1000PEPE -> PEPE, etc.
_MULTIPLIER_RE = re.compile(r'^(1000000|100000|10000|1000|100|10|1M|1K|K)')
_NORMALIZE_MAP = {
    "XBT": "BTC", "XBTUSDTM": "BTC",
    "PEPE1000": "PEPE", "1000PEPE": "PEPE", "1000000PEPE": "PEPE",
    "SHIB1000": "SHIB", "1000SHIB": "SHIB",
    "BONK1000": "BONK", "1000BONK": "BONK",
    "FLOKI1000": "FLOKI", "1000FLOKI": "FLOKI",
    "LUNC1000": "LUNC", "1000LUNC": "LUNC",
    "SATS1000": "SATS", "1000SATS": "SATS",
    "RATS1000": "RATS", "1000RATS": "RATS",
    "CAT1000": "CAT", "1000CAT": "CAT",
    "MOG1000000": "MOG", "1000000MOG": "MOG",
}


def normalize_base(raw: str) -> str:
    raw = raw.strip().upper().replace(" ", "").replace("-", "")
    if raw in _NORMALIZE_MAP:
        return _NORMALIZE_MAP[raw]
    stripped = _MULTIPLIER_RE.sub("", raw)
    if len(stripped) > 1:
        return stripped
    return raw


async def _fetch_json(session: aiohttp.ClientSession, url: str, timeout: int = 15):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                return None
            return await r.json(content_type=None)
    except Exception as e:
        logger.debug(f"[discovery] Failed {url}: {e}")
        return None


# ---- Per-exchange symbol fetchers ----

async def _mexc_symbols(session: aiohttp.ClientSession) -> Dict[str, Set[str]]:
    """MEXC: spot + futures symbols."""
    spot = set()
    futures = set()

    # Spot: /api/v3/defaultSymbols or /api/v3/exchangeInfo
    data = await _fetch_json(session, "https://api.mexc.com/api/v3/exchangeInfo")
    if data and isinstance(data.get("symbols"), list):
        for s in data["symbols"]:
            if s.get("status") == "1" and s.get("quoteAsset") == "USDT":
                base = normalize_base(s.get("baseAsset", ""))
                if base:
                    spot.add(base)

    # Futures: /api/v1/contract/detail (futures.mexc.com works; contract.mexc.com returns 403)
    data = await _fetch_json(session, "https://futures.mexc.com/api/v1/contract/detail")
    if data and isinstance(data.get("data"), list):
        for s in data["data"]:
            if s.get("quoteCoin") == "USDT" and s.get("state") == 0:
                base = normalize_base(s.get("baseCoin", ""))
                if base:
                    futures.add(base)

    logger.info(f"[discovery] MEXC: {len(spot)} spot, {len(futures)} futures")
    return {"spot": spot, "futures": futures}


async def _bingx_symbols(session: aiohttp.ClientSession) -> Dict[str, Set[str]]:
    spot = set()
    futures = set()

    # Spot – BingX changed the response: baseAsset/quoteAsset are gone.
    # Now each entry has "symbol" (e.g. "BTC-USDT") and "status" (1=active).
    data = await _fetch_json(session, "https://open-api.bingx.com/openApi/spot/v1/common/symbols")
    if data and isinstance(data.get("data"), dict):
        for s in data["data"].get("symbols", []):
            # New format: parse from symbol field ("BTC-USDT")
            sym = s.get("symbol", "")
            if sym.endswith("-USDT") and s.get("status") == 1:
                base = normalize_base(sym.replace("-USDT", ""))
                if base:
                    spot.add(base)
            # Legacy fallback: old format with baseAsset/quoteAsset
            elif s.get("quoteAsset") == "USDT":
                base = normalize_base(s.get("baseAsset", ""))
                if base:
                    spot.add(base)

    # Futures / swap
    data = await _fetch_json(session, "https://open-api.bingx.com/openApi/swap/v2/quote/contracts")
    if data and isinstance(data.get("data"), list):
        for s in data["data"]:
            sym = s.get("symbol", "")
            if "-USDT" in sym:
                base = normalize_base(sym.split("-")[0])
                if base:
                    futures.add(base)

    logger.info(f"[discovery] BingX: {len(spot)} spot, {len(futures)} futures")
    return {"spot": spot, "futures": futures}


async def _gate_symbols(session: aiohttp.ClientSession) -> Dict[str, Set[str]]:
    spot = set()
    futures = set()

    # Spot
    data = await _fetch_json(session, "https://api.gateio.ws/api/v4/spot/currency_pairs")
    if isinstance(data, list):
        for s in data:
            pair = s.get("id", "")
            if pair.endswith("_USDT") and s.get("trade_status") == "tradable":
                base = normalize_base(pair.replace("_USDT", ""))
                if base:
                    spot.add(base)

    # Futures
    data = await _fetch_json(session, "https://api.gateio.ws/api/v4/futures/usdt/contracts")
    if isinstance(data, list):
        for s in data:
            name = s.get("name", "")
            if name.endswith("_USDT") and not s.get("in_delisting"):
                base = normalize_base(name.replace("_USDT", ""))
                if base:
                    futures.add(base)

    logger.info(f"[discovery] Gate: {len(spot)} spot, {len(futures)} futures")
    return {"spot": spot, "futures": futures}


async def _kucoin_symbols(session: aiohttp.ClientSession) -> Dict[str, Set[str]]:
    spot = set()
    futures = set()

    # Spot
    data = await _fetch_json(session, "https://api.kucoin.com/api/v2/symbols")
    if data and isinstance(data.get("data"), list):
        for s in data["data"]:
            if s.get("quoteCurrency") == "USDT" and s.get("enableTrading"):
                base = normalize_base(s.get("baseCurrency", ""))
                if base:
                    spot.add(base)

    # Futures
    data = await _fetch_json(session, "https://api-futures.kucoin.com/api/v1/contracts/active")
    if data and isinstance(data.get("data"), list):
        for s in data["data"]:
            sym = s.get("symbol", "")
            if "USDT" in sym:
                base = normalize_base(sym.replace("USDTM", "").replace("USDT", ""))
                if base:
                    futures.add(base)

    logger.info(f"[discovery] KuCoin: {len(spot)} spot, {len(futures)} futures")
    return {"spot": spot, "futures": futures}


async def _xt_symbols(session: aiohttp.ClientSession) -> Dict[str, Set[str]]:
    spot = set()
    futures = set()

    # Spot
    data = await _fetch_json(session, "https://sapi.xt.com/v4/public/symbol")
    if data and isinstance(data.get("result"), dict):
        for s in data["result"].get("symbols", []):
            if s.get("quoteCurrency") == "usdt" and s.get("state") == "ONLINE":
                base = normalize_base(s.get("baseCurrency", ""))
                if base:
                    spot.add(base)

    # Futures
    data = await _fetch_json(session, "https://fapi.xt.com/future/market/v1/public/symbol/list")
    if data and isinstance(data.get("result"), list):
        for s in data["result"]:
            sym = s.get("symbol", "")
            if "_usdt" in sym.lower():
                base = normalize_base(sym.split("_")[0])
                if base:
                    futures.add(base)

    logger.info(f"[discovery] XT: {len(spot)} spot, {len(futures)} futures")
    return {"spot": spot, "futures": futures}


async def _bitget_symbols(session: aiohttp.ClientSession) -> Dict[str, Set[str]]:
    spot = set()
    futures = set()

    # Spot
    data = await _fetch_json(session, "https://api.bitget.com/api/v2/spot/public/symbols")
    if data and isinstance(data.get("data"), list):
        for s in data["data"]:
            if s.get("quoteCoin") == "USDT" and s.get("status") == "online":
                base = normalize_base(s.get("baseCoin", ""))
                if base:
                    spot.add(base)

    # Futures
    data = await _fetch_json(session, "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES")
    if data and isinstance(data.get("data"), list):
        for s in data["data"]:
            sym = s.get("symbol", "")
            if "USDT" in sym:
                base = normalize_base(sym.replace("USDT", ""))
                if base:
                    futures.add(base)

    logger.info(f"[discovery] Bitget: {len(spot)} spot, {len(futures)} futures")
    return {"spot": spot, "futures": futures}


# ---- Main discovery function ----

_EXCHANGE_FETCHERS = {
    "mexc": _mexc_symbols,
    "bingx": _bingx_symbols,
    "gate": _gate_symbols,
    "kucoin": _kucoin_symbols,
    "xt": _xt_symbols,
    "bitget": _bitget_symbols,
}


async def discover_all_symbols(exchanges: List[str] = None) -> Dict[str, Dict[str, Set[str]]]:
    """Discover all available USDT symbols from all exchanges.

    Returns:
        Dict of exchange -> {"spot": set of bases, "futures": set of bases}
    """
    if exchanges is None:
        exchanges = list(_EXCHANGE_FETCHERS.keys())

    result: Dict[str, Dict[str, Set[str]]] = {}

    async with aiohttp.ClientSession(
        headers={"User-Agent": "Mozilla/5.0"}
    ) as session:
        tasks = {}
        for ex in exchanges:
            fetcher = _EXCHANGE_FETCHERS.get(ex)
            if fetcher:
                tasks[ex] = asyncio.create_task(fetcher(session))

        for ex, task in tasks.items():
            try:
                result[ex] = await task
            except Exception as e:
                logger.error(f"[discovery] {ex} failed: {e}")
                result[ex] = {"spot": set(), "futures": set()}

    return result


def get_all_tradeable_bases(exchange_symbols: Dict[str, Dict[str, Set[str]]]) -> List[str]:
    """Get sorted union of all base symbols across all exchanges.

    Only includes symbols that appear in at least 2 exchanges (any market type).
    """
    # Count how many exchanges have each base
    base_exchanges: Dict[str, Set[str]] = {}
    for ex, markets in exchange_symbols.items():
        all_bases = markets.get("spot", set()) | markets.get("futures", set())
        for base in all_bases:
            if base not in base_exchanges:
                base_exchanges[base] = set()
            base_exchanges[base].add(ex)

    # Only keep bases on 2+ exchanges.
    # Sort by availability (more exchanges first), then alphabetically.
    tradeable_with_score = [(len(exs), b) for b, exs in base_exchanges.items() if len(exs) >= 2]
    tradeable_with_score.sort(key=lambda t: (-t[0], t[1]))
    tradeable = [b for _, b in tradeable_with_score]

    logger.info(f"[discovery] Total unique bases: {len(base_exchanges)}, "
                f"tradeable (2+ exchanges): {len(tradeable)}")
    return tradeable
