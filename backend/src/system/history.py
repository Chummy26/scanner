
import json
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class FundingHistoryManager:
    def __init__(self, base_dir: str = "out/prod/history/funding"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self._recent_seen = {}

    def _read_tail_lines(self, file_path: str, max_lines: int = 200, max_bytes: int = 512 * 1024) -> List[str]:
        if not os.path.exists(file_path):
            return []
        data = b""
        with open(file_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            while end > 0 and data.count(b"\n") <= max_lines and len(data) < max_bytes:
                read_size = min(8192, end)
                end -= read_size
                f.seek(end)
                data = f.read(read_size) + data
        lines = data.splitlines()[-max_lines:]
        return [line.decode("utf-8", errors="ignore") for line in lines if line.strip()]

    def _get_recent_seen(self, date_str: str, file_path: str) -> Dict:
        seen = self._recent_seen.get(date_str)
        if seen is None:
            seen = {}
            for line in self._read_tail_lines(file_path):
                try:
                    rec = json.loads(line)
                    key = (rec.get("ex"), rec.get("sym"))
                    if key[0] and key[1]:
                        seen[key] = rec.get("ts")
                except Exception:
                    continue
            self._recent_seen[date_str] = seen
        return seen

    def save_record(self, record: Dict):
        """Saves a single record to the daily file with basic duplicate prevention."""
        ts = record.get('ts')
        if not ts: return
        dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
            
        date_str = dt.strftime("%Y-%m-%d")
        file_path = os.path.join(self.base_dir, f"{date_str}.jsonl")

        seen = self._get_recent_seen(date_str, file_path)
        key = (record.get("ex"), record.get("sym"))
        if seen.get(key) == record.get("ts"):
            return
        seen[key] = record.get("ts")

        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to save record: {e}")

    def save_snapshots(self, funding_map: Dict[str, Dict[str, any]]):
        records_by_date = {}
        now_ms = int(time.time() * 1000)
        for ex_name, symbols in funding_map.items():
            for symbol, data in symbols.items():
                # Store by payment time (next funding), but keep settle_ts for reference.
                interval_ms = data.interval_hours * 3600000
                event_ts = data.next_funding_time or now_ms
                settle_ts = event_ts - interval_ms

                # Normalize to prevent drift: 07:59:59 -> 08:00:00
                # Round to nearest 5 minutes to allow for exchange clock drift
                event_ts = (event_ts // 300000) * 300000
                settle_ts = (settle_ts // 300000) * 300000

                record = {
                    "ts": event_ts,
                    "settle_ts": settle_ts,
                    "ex": ex_name,
                    "sym": symbol,
                    "rate": data.rate,
                    "next": data.next_funding_time,
                    "price": data.mark_price,
                    "interval": data.interval_hours
                }
                dt = datetime.fromtimestamp(event_ts / 1000, tz=timezone.utc)
                date_str = dt.strftime("%Y-%m-%d")
                records_by_date.setdefault(date_str, []).append(record)

        for date_str, records in records_by_date.items():
            file_path = os.path.join(self.base_dir, f"{date_str}.jsonl")
            seen = self._get_recent_seen(date_str, file_path)
            lines = []
            for record in records:
                key = (record.get("ex"), record.get("sym"))
                if seen.get(key) == record.get("ts"):
                    continue
                seen[key] = record.get("ts")
                lines.append(json.dumps(record))
            if lines:
                try:
                    with open(file_path, "a", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")
                except Exception as e:
                    logger.error(f"Failed to save batch records: {e}")

    def get_history(self, exchange_a: str, exchange_b: str, symbol: str, days: int = 7) -> Dict:
        """Retrieves and merges history for two exchanges."""
        from collections import defaultdict
        merged = defaultdict(lambda: {"ts": 0, "rates": {}, "intervals": {}, "settle_ts": None})
        
        # Look back X days
        now = datetime.now(timezone.utc)
        for i in range(days + 1):
            date_str = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            file_path = os.path.join(self.base_dir, f"{date_str}.jsonl")
            if not os.path.exists(file_path): continue
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        # More precise matching: remove only '/' and ':USDT' but keep multipliers
                        rec_sym_clean = rec['sym'].split(':')[0].replace('/', '')
                        target_sym_clean = symbol.replace('/', '') # 'symbol' passed to method is already canonical or raw
                        
                        if (rec_sym_clean == target_sym_clean or rec.get('canonical_base') == symbol) and rec['ex'] in (exchange_a, exchange_b):
                            event_ts = rec['ts']
                            norm_ts = event_ts
                            
                            merged[norm_ts]["ts"] = norm_ts
                            merged[norm_ts]["rates"][rec['ex']] = rec['rate']
                            if rec.get('interval') is not None:
                                merged[norm_ts]["intervals"][rec['ex']] = rec.get('interval')
                            if rec.get('settle_ts') is not None:
                                merged[norm_ts]["settle_ts"] = rec.get('settle_ts')
                    except: continue
        
        # Sort by timestamp
        sorted_history = sorted(merged.values(), key=lambda x: x['ts'])
        return {
            "symbol": symbol,
            "ex_a": exchange_a,
            "ex_b": exchange_b,
            "history": sorted_history
        }
