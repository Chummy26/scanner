import time
import os
import ccxt
import logging
import urllib.request
import json
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from .models import Instrument, FundingData, MIN_VOLUME_USD

logger = logging.getLogger(__name__)

class BaseAdapter(ABC):
    def __init__(self, exchange_id: str, ccxt_id: str = None, spot_ccxt_id: Optional[str] = None):
        self.exchange_id = exchange_id
        real_ccxt_id = ccxt_id or exchange_id
        self.ccxt_id = real_ccxt_id
        self.spot_ccxt_id = real_ccxt_id if spot_ccxt_id is None else spot_ccxt_id

        ccxt_class = getattr(ccxt, real_ccxt_id)
        self.ccxt_exchange = ccxt_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })

        self.ccxt_spot = None
        self.spot_markets_loaded = False
        
        self.markets_loaded = False
        self.failure_count = 0
        self.cb_open_until = 0
        self.CB_THRESHOLD = 3
        self.CB_COOLDOWN = 300 

    def _check_circuit_breaker(self):
        if time.time() < self.cb_open_until:
            raise Exception(f"Circuit Breaker OPEN for {self.exchange_id}. Resuming in {int(self.cb_open_until - time.time())}s")

    def _record_success(self):
        self.failure_count = 0
        self.cb_open_until = 0

    def _record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.CB_THRESHOLD:
            self.cb_open_until = time.time() + self.CB_COOLDOWN
            logger.warning(f"[{self.exchange_id}] Circuit Breaker TRIGGERED.")

    def calculate_next_funding_time(self, interval_hours: float) -> int:
        now = time.time()
        interval_sec = (interval_hours or 8.0) * 3600
        next_ts = (now // interval_sec + 1) * interval_sec
        return int(next_ts * 1000)

    def normalize_next_funding_time(self, value: Any, interval_hours: float) -> int:
        now_ms = int(time.time() * 1000)
        interval_ms = int((interval_hours or 8.0) * 3600 * 1000)
        if value is None:
            return self.calculate_next_funding_time(interval_hours)
        try:
            ts = int(float(value))
        except (TypeError, ValueError):
            return self.calculate_next_funding_time(interval_hours)
        if ts <= 0:
            return self.calculate_next_funding_time(interval_hours)
        if ts < 1_000_000_000_000:
            ts *= 1000
        elif ts > 100_000_000_000_000:
            ts = int(ts / 1000)
        if interval_ms > 0 and ts < now_ms - (5 * 60 * 1000):
            steps = ((now_ms - ts) // interval_ms) + 1
            ts = ts + steps * interval_ms
        if ts > now_ms + (365 * 24 * 3600 * 1000):
            return self.calculate_next_funding_time(interval_hours)
        return ts

    def load_markets_safe(self):
        self._check_circuit_breaker()
        try:
            if not self.markets_loaded:
                self.ccxt_exchange.load_markets()
                self.markets_loaded = True
            self._record_success()
        except Exception as e:
            logger.error(f"[{self.exchange_id}] Failed to load markets: {e}")
            self._record_failure()
            raise

    def _get_spot_client(self):
        if not self.spot_ccxt_id:
            return None
        if self.ccxt_spot is None:
            try:
                spot_class = getattr(ccxt, self.spot_ccxt_id)
            except AttributeError:
                logger.error(f"[{self.exchange_id}] Spot CCXT id not found: {self.spot_ccxt_id}")
                self.spot_ccxt_id = None
                return None
            self.ccxt_spot = spot_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
        return self.ccxt_spot

    @abstractmethod
    def get_universe(self) -> tuple[List[Instrument], Dict]:
        pass

    @abstractmethod
    def get_funding_rates(self, instruments: List[Instrument]) -> List[FundingData]:
        pass

    def get_spot_prices(self, symbols: List[str]) -> Dict[str, float]:
        return {}

    @staticmethod
    def _has_non_ascii(value: str) -> bool:
        return any(ord(ch) > 127 for ch in value)

    @staticmethod
    def _extract_alias_base(info: Dict[str, Any]) -> Optional[str]:
        if not isinstance(info, dict):
            return None
        for key in ("baseCoinName", "base_coin", "baseName", "baseAsset", "base_currency"):
            raw = info.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return None

class GenericCCXTAdapter(BaseAdapter):
    def get_universe(self) -> tuple[List[Instrument], Dict]:
        try:
            self.load_markets_safe()
            tickers = self.ccxt_exchange.fetch_tickers()
            self._record_success()
            
            instruments = []
            stats = {
                "universe_total_count": len(tickers),
                "universe_tradable_count": 0,
                "universe_non_tradable_count": 0,
                "eligible_by_volume_24h_count": 0,
                "excluded_by_volume_24h_count": 0,
                "eligible_ratio_pct": 0.0,
                "volume_audit": {"top_10": []},
                "status": "OK"
            }
            
            candidates = []
            for symbol, ticker in tickers.items():
                if not ticker: continue
                try:
                    market = self.ccxt_exchange.market(symbol)
                except: continue
                
                is_linear = market.get('linear') or (market.get('quote') == 'USDT')
                is_swap = market.get('swap') or (market.get('type') == 'swap')
                
                if not (is_swap and is_linear and market.get('active')):
                    stats["universe_non_tradable_count"] += 1
                    continue
                
                stats["universe_tradable_count"] += 1
                vol = float(ticker.get('quoteVolume') or (float(ticker.get('baseVolume', 0)) * float(ticker.get('last', 0))))
                candidates.append({"symbol": symbol, "volume": vol})
                
                alias_base = None
                info = market.get('info') if isinstance(market, dict) else None
                alias_candidate = self._extract_alias_base(info)
                if isinstance(alias_candidate, str):
                    base_val = market.get('base') or ''
                    if base_val and not self._has_non_ascii(base_val) and self._has_non_ascii(alias_candidate):
                        alias_base = alias_candidate

                if vol >= MIN_VOLUME_USD:
                    instruments.append(Instrument(self.exchange_id, symbol, market['base'], market['quote'], vol, True, alias_base=alias_base))
                    stats["eligible_by_volume_24h_count"] += 1
                else:
                    stats["excluded_by_volume_24h_count"] += 1
            
            candidates.sort(key=lambda x: x["volume"], reverse=True)
            stats["volume_audit"]["top_10"] = candidates[:10]
            if stats["universe_tradable_count"] > 0:
                stats["eligible_ratio_pct"] = (stats["eligible_by_volume_24h_count"] / stats["universe_tradable_count"]) * 100.0
                
            return instruments, stats
        except Exception as e:
            logger.error(f"[{self.exchange_id}] get_universe failed: {e}")
            self._record_failure()
            return [], {"error": str(e), "status": "ERR"}

    def get_funding_rates(self, instruments: List[Instrument]) -> List[FundingData]:
        if not instruments: return []
        try:
            self._check_circuit_breaker()
            symbols = [i.symbol for i in instruments]
            
            if self.ccxt_exchange.has.get('fetchFundingRates'):
                try:
                    rates = self.ccxt_exchange.fetch_funding_rates(symbols)
                    self._record_success()
                    return self._process_ccxt_rates(rates, instruments)
                except Exception as e:
                    logger.warning(f"[{self.exchange_id}] fetchFundingRates failed: {e}")

            tickers = self.ccxt_exchange.fetch_tickers(symbols)
            self._record_success()
            return self._process_ccxt_tickers(tickers, instruments)
            
        except Exception as e:
            logger.error(f"[{self.exchange_id}] get_funding_rates failed: {e}")
            self._record_failure()
            return []

    def get_spot_prices(self, symbols: List[str]) -> Dict[str, float]:
        if not symbols:
            return {}
        client = self._get_spot_client()
        if not client:
            return {}
        try:
            if not self.spot_markets_loaded:
                client.load_markets()
                self.spot_markets_loaded = True
        except Exception as e:
            logger.error(f"[{self.exchange_id}] Spot load markets failed: {e}")
            return {}
        try:
            available = set(client.markets.keys())
            symbols = [s for s in symbols if s in available]
        except Exception:
            pass
        if not symbols:
            return {}
        if len(symbols) > 200:
            symbols = symbols[:200]
        prices: Dict[str, float] = {}
        try:
            if client.has.get('fetchTickers'):
                tickers = client.fetch_tickers(symbols)
                if isinstance(tickers, dict):
                    for sym, ticker in tickers.items():
                        if not ticker:
                            continue
                        price = ticker.get('last')
                        if price is None:
                            price = ticker.get('close')
                        if price is not None:
                            prices[sym] = float(price)
            else:
                for sym in symbols:
                    ticker = client.fetch_ticker(sym)
                    if not ticker:
                        continue
                    price = ticker.get('last')
                    if price is None:
                        price = ticker.get('close')
                    if price is not None:
                        prices[sym] = float(price)
        except Exception as e:
            logger.error(f"[{self.exchange_id}] Spot tickers failed: {e}")
        return prices

    def _process_ccxt_rates(self, rates: Any, instruments: List[Instrument]) -> List[FundingData]:
        results = []
        inst_map = {i.symbol: i for i in instruments}
        iterable = rates.values() if isinstance(rates, dict) else rates
        
        for item in iterable:
            symbol = item.get('symbol')
            if symbol in inst_map:
                rate = item.get('fundingRate')
                if rate is not None:
                    interval = self._extract_interval(item)
                    info = item.get('info') if isinstance(item, dict) else {}
                    if not isinstance(info, dict):
                        info = {}
                    next_raw = (
                        item.get('nextFundingTime') or
                        item.get('nextFundingTimestamp') or
                        item.get('fundingTimestamp') or
                        info.get('nextFundingTime') or
                        info.get('nextFundingTimestamp') or
                        info.get('fundingTime')
                    )
                    next_f = self.normalize_next_funding_time(next_raw, interval)
                    results.append(FundingData(
                        exchange=self.exchange_id, symbol=symbol, rate=float(rate),
                        timestamp=item.get('timestamp') or int(time.time() * 1000),
                        next_funding_time=int(next_f), mark_price=item.get('markPrice'),
                        interval_hours=interval
                    ))
        return results

    def _process_ccxt_tickers(self, tickers: Dict, instruments: List[Instrument]) -> List[FundingData]:
        results = []
        inst_map = {i.symbol: i for i in instruments}
        for symbol, ticker in tickers.items():
            if symbol in inst_map:
                # Mexc often puts fundingRate in ticker.info
                rate = ticker.get('fundingRate')
                if rate is None and 'info' in ticker:
                    rate = ticker['info'].get('fundingRate')
                
                if rate is not None:
                    interval = self._extract_interval(ticker)
                    info = ticker.get('info') if isinstance(ticker, dict) else {}
                    if not isinstance(info, dict):
                        info = {}
                    next_raw = (
                        ticker.get('nextFundingTime') or
                        ticker.get('nextFundingTimestamp') or
                        ticker.get('fundingTimestamp') or
                        info.get('nextFundingTime') or
                        info.get('nextFundingTimestamp') or
                        info.get('fundingTime')
                    )
                    next_f = self.normalize_next_funding_time(next_raw, interval)
                    results.append(FundingData(
                        exchange=self.exchange_id, symbol=symbol, rate=float(rate),
                        timestamp=ticker.get('timestamp') or int(time.time() * 1000),
                        next_funding_time=int(next_f), mark_price=ticker.get('markPrice') or ticker.get('last'),
                        interval_hours=interval
                    ))
        return results

    def _extract_interval(self, item: Dict) -> float:
        info = item.get('info', {})
        for key in ['fundingInterval', 'funding_interval', 'collectCycle', 'fundingRateInterval']:
            if key in info:
                try:
                    val = float(info[key])
                    if val > 100: return val / 3600.0
                    return val
                except: pass
        return 8.0

class MexcAdapter(GenericCCXTAdapter):
    def __init__(self):
        super().__init__('mexc')
        self._funding_cache = {}
        self._ticker_cache = {}
        self._last_cache_refresh = 0

    def get_universe(self) -> tuple[List[Instrument], Dict]:
        try:
            self.load_markets_safe()
            # MEXC fetch_tickers can sometimes return None for some fields
            tickers = self.ccxt_exchange.fetch_tickers()
            self._record_success()
            
            instruments = []
            stats = {
                "universe_total_count": len(tickers),
                "universe_tradable_count": 0,
                "eligible_by_volume_24h_count": 0,
                "excluded_by_volume_24h_count": 0,
                "eligible_ratio_pct": 0.0,
                "status": "OK"
            }
            
            for symbol, ticker in tickers.items():
                if not ticker: continue
                try:
                    market = self.ccxt_exchange.market(symbol)
                except: continue
                
                if not (market.get('swap') and market.get('active')): continue
                
                stats["universe_tradable_count"] += 1
                # Safer volume extraction
                quote_vol = ticker.get('quoteVolume')
                if quote_vol is None:
                    base_vol = float(ticker.get('baseVolume') or 0)
                    last_price = float(ticker.get('last') or 0)
                    quote_vol = base_vol * last_price
                else:
                    quote_vol = float(quote_vol)

                if quote_vol >= MIN_VOLUME_USD:
                    alias_base = None
                    info = market.get('info') if isinstance(market, dict) else None
                    alias_candidate = self._extract_alias_base(info)
                    if isinstance(alias_candidate, str):
                        base_val = market.get('base') or ''
                        if base_val and not self._has_non_ascii(base_val) and self._has_non_ascii(alias_candidate):
                            alias_base = alias_candidate

                    instruments.append(Instrument(self.exchange_id, symbol, market['base'], market['quote'], quote_vol, True, alias_base=alias_base))
                    stats["eligible_by_volume_24h_count"] += 1
            
            if stats["universe_tradable_count"] > 0:
                stats["excluded_by_volume_24h_count"] = max(stats["universe_tradable_count"] - stats["eligible_by_volume_24h_count"], 0)
                stats["eligible_ratio_pct"] = (stats["eligible_by_volume_24h_count"] / stats["universe_tradable_count"]) * 100.0
            return instruments, stats
        except Exception as e:
            logger.error(f"[{self.exchange_id}] get_universe failed: {e}")
            self._record_failure()
            return [], {"error": str(e), "status": "ERR"}

    def _refresh_cache(self):
        if time.time() - self._last_cache_refresh < 20:
            return
        try:
            rate_url = "https://contract.mexc.com/api/v1/contract/funding_rate"
            ticker_url = "https://contract.mexc.com/api/v1/contract/ticker"
            req_rate = urllib.request.Request(rate_url, headers={"User-Agent": "Mozilla/5.0"})
            req_ticker = urllib.request.Request(ticker_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_rate, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            with urllib.request.urlopen(req_ticker, timeout=10) as resp:
                tdata = json.loads(resp.read().decode("utf-8"))
            if data.get('success') and isinstance(data.get('data'), list):
                self._funding_cache = {c.get('symbol'): c for c in data.get('data', []) if c.get('symbol')}
            if tdata.get('success') and isinstance(tdata.get('data'), list):
                self._ticker_cache = {c.get('symbol'): c for c in tdata.get('data', []) if c.get('symbol')}
            self._last_cache_refresh = time.time()
        except Exception as e:
            logger.error(f"[mexc] Cache refresh failed: {e}")

    def _to_api_symbol(self, symbol: str) -> str:
        return symbol.split(':')[0].replace('/', '_')

    def get_funding_rates(self, instruments: List[Instrument]) -> List[FundingData]:
        if not instruments:
            return []
        self._refresh_cache()
        results = []
        if not self._funding_cache:
            return super().get_funding_rates(instruments)
        for inst in instruments:
            api_sym = self._to_api_symbol(inst.symbol)
            funding = self._funding_cache.get(api_sym)
            if not funding:
                continue
            rate = funding.get('fundingRate')
            if rate is None:
                continue
            interval = float(funding.get('collectCycle') or 8.0)
            next_f = self.normalize_next_funding_time(funding.get('nextSettleTime'), interval)
            ticker = self._ticker_cache.get(api_sym, {})
            mark = ticker.get('fairPrice') or ticker.get('indexPrice') or ticker.get('lastPrice')
            results.append(FundingData(
                exchange=self.exchange_id, symbol=inst.symbol, rate=float(rate),
                timestamp=int(funding.get('timestamp') or int(time.time() * 1000)),
                next_funding_time=next_f, mark_price=float(mark) if mark is not None else None,
                interval_hours=interval
            ))
        return results

class KuCoinAdapter(GenericCCXTAdapter):
    def __init__(self): super().__init__('kucoin', ccxt_id='kucoinfutures', spot_ccxt_id='kucoin')
    
    def get_funding_rates(self, instruments: List[Instrument]) -> List[FundingData]:
        try:
            self.load_markets_safe()
            markets = self.ccxt_exchange.fetch_markets()
            market_map = {m['symbol']: m for m in markets}
            results = []
            for inst in instruments:
                m = market_map.get(inst.symbol)
                if m:
                    info = m.get('info', {})
                    rate = info.get('fundingFeeRate')
                    if rate is not None:
                        interval = float(info.get('fundingRateGranularity', 28800000)) / 3600000.0
                        next_raw = info.get('nextFundingRateDateTime') or info.get('nextFundingTime')
                        results.append(FundingData(
                            exchange=self.exchange_id, symbol=inst.symbol, rate=float(rate),
                            timestamp=int(time.time()*1000), next_funding_time=self.normalize_next_funding_time(next_raw, interval),
                            mark_price=float(info.get('markPrice') or 0), interval_hours=interval
                        ))
            if results: return results
        except Exception as e: logger.error(f"[kucoin] Bulk failed: {e}")
        return super().get_funding_rates(instruments)

class BingXAdapter(GenericCCXTAdapter):
    def __init__(self):
        super().__init__('bingx')
        self._interval_cache = {}
        self._next_cache = {}
        self._history_seed_last = {}
        self._history_seed_ttl_sec = 6 * 3600
        self._history_seed_budget = 20
        self._last_cache_save = 0
        self._cache_path = os.path.join("out", "config", "bingx_interval_cache.json")
        self._load_interval_cache()

    def _load_interval_cache(self):
        try:
            if not os.path.exists(self._cache_path):
                return
            with open(self._cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            intervals = payload.get("intervals")
            next_times = payload.get("next_times")
            if isinstance(intervals, dict):
                self._interval_cache = {k: float(v) for k, v in intervals.items() if v is not None}
            if isinstance(next_times, dict):
                self._next_cache = {k: int(float(v)) for k, v in next_times.items() if v is not None}
        except Exception as e:
            logger.error(f"[bingx] Failed to load interval cache: {e}")

    def _save_interval_cache(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_cache_save < 60:
            return
        try:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "intervals": self._interval_cache,
                "next_times": self._next_cache
            }
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self._last_cache_save = now
        except Exception as e:
            logger.error(f"[bingx] Failed to save interval cache: {e}")

    def _infer_interval_hours(self, api_sym: str, next_ms: Optional[int], prev_next: Optional[int]) -> Optional[float]:
        if not next_ms:
            return self._interval_cache.get(api_sym)
        if prev_next and next_ms > prev_next:
            delta_ms = next_ms - prev_next
            hours = delta_ms / 3600000.0
            if 0.25 <= hours <= 24:
                inferred = round(hours, 2)
                self._interval_cache[api_sym] = inferred
                return inferred
        return self._interval_cache.get(api_sym)

    def _seed_interval_from_history(self, api_sym: str) -> Optional[float]:
        now = time.time()
        last_try = self._history_seed_last.get(api_sym)
        if last_try and now - last_try < self._history_seed_ttl_sec:
            return None
        self._history_seed_last[api_sym] = now
        try:
            url = f"https://open-api.bingx.com/openApi/swap/v2/quote/fundingRate?symbol={api_sym}"
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("code") != 0:
                return None
            rows = data.get("data") or []
            if len(rows) < 2:
                return None
            t0 = int(float(rows[0].get("fundingTime")))
            t1 = int(float(rows[1].get("fundingTime")))
            if not t0 or not t1:
                return None
            delta_ms = abs(t0 - t1)
            hours = delta_ms / 3600000.0
            if 0.25 <= hours <= 24:
                inferred = round(hours, 2)
                self._interval_cache[api_sym] = inferred
                return inferred
        except Exception as e:
            logger.error(f"[bingx] History seed failed for {api_sym}: {e}")
        return None
    def get_funding_rates(self, instruments: List[Instrument]) -> List[FundingData]:
        try:
            url = "https://open-api.bingx.com/openApi/swap/v2/quote/premiumIndex"
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get('code') == 0:
                    tickers = {t['symbol']: t for t in data['data']}
                    results = []
                    cache_dirty = False
                    history_seeded = 0
                    for inst in instruments:
                        api_sym = inst.symbol.split(':')[0].replace('/', '-')
                        t = tickers.get(api_sym)
                        if t and t.get('lastFundingRate') is not None:
                            next_raw = t.get('nextFundingTime')
                            next_ms = None
                            try:
                                if next_raw is not None:
                                    next_ms = int(float(next_raw))
                            except Exception:
                                next_ms = None
                            prev_next = self._next_cache.get(api_sym)
                            interval = self._infer_interval_hours(api_sym, next_ms, prev_next)
                            should_seed = interval is None or interval == 8.0
                            if should_seed and history_seeded < self._history_seed_budget:
                                interval = self._seed_interval_from_history(api_sym)
                                if interval is not None:
                                    history_seeded += 1
                                    cache_dirty = True
                            if next_ms is not None and prev_next != next_ms:
                                self._next_cache[api_sym] = next_ms
                                cache_dirty = True
                            interval = interval or 8.0
                            results.append(FundingData(
                                exchange=self.exchange_id, symbol=inst.symbol, rate=float(t['lastFundingRate']),
                                timestamp=int(time.time()*1000), next_funding_time=self.normalize_next_funding_time(next_ms, interval),
                                mark_price=float(t.get('markPrice', 0)), interval_hours=interval
                            ))
                    if cache_dirty:
                        self._save_interval_cache()
                    return results
        except Exception as e: logger.error(f"[bingx] Failed: {e}")
        return super().get_funding_rates(instruments)

class XTAdapter(GenericCCXTAdapter):
    def __init__(self): super().__init__('xt')
    def get_universe(self) -> tuple[List[Instrument], Dict]:
        try:
            url = "https://fapi.xt.com/future/market/v1/public/cg/contracts"
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if not isinstance(data, list):
                return [], {"error": "Invalid XT data", "status": "ERR"}
            instruments = []
            for item in data:
                symbol = item.get('symbol')
                if not symbol: continue
                ccxt_sym = symbol.upper().replace('_', '/') + ":USDT"
                vol = float(item.get('target_volume') or 0)
                if vol >= MIN_VOLUME_USD:
                    instruments.append(Instrument(self.exchange_id, ccxt_sym, item.get('base_asset', symbol.split('_')[0]), 'USDT', vol, True))
            tradable = len(data)
            eligible = len(instruments)
            excluded = max(tradable - eligible, 0)
            ratio = (eligible / tradable) * 100.0 if tradable else 0.0
            return instruments, {
                "universe_total_count": len(data),
                "universe_tradable_count": tradable,
                "eligible_by_volume_24h_count": eligible,
                "excluded_by_volume_24h_count": excluded,
                "eligible_ratio_pct": ratio,
                "status": "OK"
            }
        except Exception as e: return [], {"error": str(e), "status": "ERR"}

    def get_funding_rates(self, instruments: List[Instrument]) -> List[FundingData]:
        try:
            url = "https://fapi.xt.com/future/market/v1/public/cg/contracts"
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list):
                    inst_map = {i.symbol.split(':')[0].replace('/', '_').lower(): i for i in instruments}
                    results = []
                    for item in data:
                        sym_id = item.get('symbol', '').lower()
                        if sym_id in inst_map:
                            inst = inst_map[sym_id]
                            rate = item.get('funding_rate')
                            if rate is not None:
                                interval = float(item.get('collection_internal') or 8.0)
                                results.append(FundingData(
                                    exchange=self.exchange_id, symbol=inst.symbol, rate=float(rate),
                                    timestamp=int(time.time()*1000), next_funding_time=self.normalize_next_funding_time(item.get('next_funding_rate_timestamp'), interval),
                                    mark_price=float(item.get('index_price') or 0), interval_hours=interval
                                ))
                    return results
        except Exception as e: logger.error(f"[xt] Failed: {e}")
        return super().get_funding_rates(instruments)

class BitgetAdapter(GenericCCXTAdapter):
    def __init__(self):
        super().__init__('bitget')
        self._funding_cache = {}
        self._ticker_cache = {}
        self._last_cache_refresh = 0

    def _refresh_cache(self):
        if time.time() - self._last_cache_refresh < 20:
            return
        try:
            rate_url = "https://api.bitget.com/api/v2/mix/market/current-fund-rate?productType=USDT-FUTURES"
            ticker_url = "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES"
            req_rate = urllib.request.Request(rate_url, headers={"User-Agent": "Mozilla/5.0"})
            req_ticker = urllib.request.Request(ticker_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_rate, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            with urllib.request.urlopen(req_ticker, timeout=10) as resp:
                tdata = json.loads(resp.read().decode("utf-8"))
            if data.get('code') == '00000' and isinstance(data.get('data'), list):
                self._funding_cache = {c.get('symbol'): c for c in data.get('data', []) if c.get('symbol')}
            if tdata.get('code') == '00000' and isinstance(tdata.get('data'), list):
                self._ticker_cache = {c.get('symbol'): c for c in tdata.get('data', []) if c.get('symbol')}
            self._last_cache_refresh = time.time()
        except Exception as e:
            logger.error(f"[bitget] Cache refresh failed: {e}")

    def _to_api_symbol(self, symbol: str) -> str:
        return symbol.split(':')[0].replace('/', '')

    def get_funding_rates(self, instruments: List[Instrument]) -> List[FundingData]:
        if not instruments:
            return []
        self._refresh_cache()
        results = []
        if not self._funding_cache:
            return super().get_funding_rates(instruments)
        for inst in instruments:
            api_sym = self._to_api_symbol(inst.symbol)
            funding = self._funding_cache.get(api_sym)
            if not funding:
                continue
            rate = funding.get('fundingRate')
            if rate is None:
                continue
            try:
                interval = float(funding.get('fundingRateInterval') or 8.0)
            except (TypeError, ValueError):
                interval = 8.0
            next_f = self.normalize_next_funding_time(funding.get('nextUpdate'), interval)
            ticker = self._ticker_cache.get(api_sym, {})
            mark = ticker.get('markPrice') or ticker.get('indexPrice') or ticker.get('lastPr')
            ts = ticker.get('ts')
            results.append(FundingData(
                exchange=self.exchange_id, symbol=inst.symbol, rate=float(rate),
                timestamp=int(float(ts)) if ts is not None else int(time.time() * 1000),
                next_funding_time=next_f, mark_price=float(mark) if mark is not None else None,
                interval_hours=interval
            ))
        return results

class AdapterFactory:
    @staticmethod
    def create_adapter(exchange_id: str) -> BaseAdapter:
        adapters = {
            'gate': lambda: GenericCCXTAdapter('gate'), 
            'mexc': MexcAdapter, 
            'kucoin': KuCoinAdapter,
            'bingx': BingXAdapter, 
            'xt': XTAdapter, 
            'bitget': BitgetAdapter
        }
        creator = adapters.get(exchange_id)
        if not creator: raise ValueError(f"Unsupported: {exchange_id}")
        return creator()
