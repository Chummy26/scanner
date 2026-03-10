from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from statistics import mean
from typing import Any


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "min": round(min(data), 6),
        "max": round(max(data), 6),
        "mean": round(mean(data), 6),
        "p50": round(_percentile(data, 50.0), 6),
        "p95": round(_percentile(data, 95.0), 6),
        "p99": round(_percentile(data, 99.0), 6),
    }


class RuntimePerfMonitor:
    def __init__(self, *, max_samples: int = 512, max_events: int = 256):
        self._lock = threading.Lock()
        self._max_samples = max(64, int(max_samples))
        self._max_events = max(32, int(max_events))
        self._route_samples: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self._max_samples))
        self._scanner_cycles: deque[dict[str, Any]] = deque(maxlen=self._max_events)
        self._broadcast_events: deque[dict[str, Any]] = deque(maxlen=self._max_events)
        self._event_loop_lag_ms: deque[float] = deque(maxlen=self._max_samples)
        self._cache_samples: deque[dict[str, Any]] = deque(maxlen=self._max_events)
        self._slow_events: deque[dict[str, Any]] = deque(maxlen=self._max_events)

    def record_route(
        self,
        route_name: str,
        *,
        latency_ms: float,
        status_code: int,
        payload_bytes: int = 0,
        rows: int = 0,
        source: str = "",
        path: str = "",
    ) -> None:
        sample = {
            "ts": round(time.time(), 6),
            "route": str(route_name),
            "path": str(path),
            "latency_ms": round(float(latency_ms), 6),
            "status_code": int(status_code),
            "payload_bytes": int(payload_bytes),
            "rows": int(rows),
            "source": str(source or ""),
        }
        with self._lock:
            self._route_samples[str(route_name)].append(sample)
            if sample["latency_ms"] >= 1000.0:
                self._slow_events.append({"kind": "route", **sample})

    def record_scanner_cycle(self, payload: dict[str, Any]) -> None:
        sample = {"ts": round(time.time(), 6), **payload}
        sample.setdefault("kind", "ws_manager")
        with self._lock:
            self._scanner_cycles.append(sample)
            if float(sample.get("total_ms") or 0.0) >= 1000.0:
                self._slow_events.append({"kind": "scanner_cycle", **sample})

    def record_broadcast(self, channel: str, payload: dict[str, Any]) -> None:
        sample = {"ts": round(time.time(), 6), "channel": str(channel), **payload}
        with self._lock:
            self._broadcast_events.append(sample)
            if float(sample.get("total_ms") or 0.0) >= 1000.0:
                self._slow_events.append({"kind": "broadcast", **sample})

    def record_event_loop_lag(self, lag_ms: float) -> None:
        value = round(max(float(lag_ms), 0.0), 6)
        with self._lock:
            self._event_loop_lag_ms.append(value)
            if value >= 250.0:
                self._slow_events.append({"kind": "event_loop_lag", "ts": round(time.time(), 6), "lag_ms": value})

    def record_cache_state(self, payload: dict[str, Any]) -> None:
        sample = {"ts": round(time.time(), 6), **payload}
        with self._lock:
            self._cache_samples.append(sample)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            route_samples = {name: list(samples) for name, samples in self._route_samples.items()}
            scanner_cycles = list(self._scanner_cycles)
            broadcast_events = list(self._broadcast_events)
            lag_values = list(self._event_loop_lag_ms)
            cache_samples = list(self._cache_samples)
            slow_events = list(self._slow_events)

        route_summary: dict[str, Any] = {}
        for name, samples in route_samples.items():
            latencies = [float(item.get("latency_ms") or 0.0) for item in samples]
            payloads = [float(item.get("payload_bytes") or 0.0) for item in samples]
            rows = [float(item.get("rows") or 0.0) for item in samples]
            sources = defaultdict(int)
            errors = 0
            for item in samples:
                source = str(item.get("source") or "")
                if source:
                    sources[source] += 1
                if int(item.get("status_code") or 0) >= 500:
                    errors += 1
            route_summary[name] = {
                "latency_ms": _numeric_summary(latencies),
                "payload_bytes": _numeric_summary(payloads),
                "rows": _numeric_summary(rows),
                "error_count": errors,
                "source_counts": dict(sources),
            }

        def _cycle_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
            def _metric(metric: str) -> dict[str, float]:
                return _numeric_summary([float(item.get(metric) or 0.0) for item in samples])

            return {
                "count": len(samples),
                "total_ms": _metric("total_ms"),
                "calculate_ms": _metric("calculate_ms"),
                "ingest_filter_ms": _metric("ingest_filter_ms"),
                "batch_record_ms": _metric("batch_record_ms"),
                "market_enrich_ms": _metric("market_enrich_ms"),
                "tracker_enrich_ms": _metric("tracker_enrich_ms"),
                "history_fetch_ms": _metric("history_fetch_ms"),
                "ml_analyze_ms": _metric("ml_analyze_ms"),
                "ml_render_ms": _metric("ml_render_ms"),
                "filter_ms": _metric("filter_ms"),
                "lite_refresh_ms": _metric("lite_refresh_ms"),
                "ml_predictions_drained": _metric("ml_predictions_drained"),
                "ml_refresh_queue_size": _metric("ml_refresh_queue_size"),
                "ml_cache_size": _metric("ml_cache_size"),
                "tracker_records_enqueued": _metric("tracker_records_enqueued"),
                "tracker_rejections_enqueued": _metric("tracker_rejections_enqueued"),
                "tracker_pending_records": _metric("tracker_pending_records"),
                "tracker_pending_rejections": _metric("tracker_pending_rejections"),
                "tracker_records_drained": _metric("tracker_records_drained"),
                "tracker_rejections_drained": _metric("tracker_rejections_drained"),
                "opportunities_before_filter": _metric("opportunities_before_filter"),
                "opportunities_after_filter": _metric("opportunities_after_filter"),
                "snapshot_lock_ms": _metric("snapshot_lock_ms"),
                "symbol_calc_ms": _metric("symbol_calc_ms"),
                "aging_ms": _metric("aging_ms"),
                "callback_ms": _metric("callback_ms"),
                "sort_ms": _metric("sort_ms"),
                "symbols_to_calc_count": _metric("symbols_to_calc_count"),
                "dirty_symbols_pending": _metric("dirty_symbols_pending"),
                "stale_count": _metric("stale_count"),
                "spread_record_count": _metric("spread_record_count"),
                "opportunity_count": _metric("opportunity_count"),
            }

        broadcast_summary: dict[str, Any] = {}
        by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in broadcast_events:
            by_channel[str(item.get("channel") or "unknown")].append(item)
        for channel, samples in by_channel.items():
            broadcast_summary[channel] = {
                "total_ms": _numeric_summary([float(item.get("total_ms") or 0.0) for item in samples]),
                "payload_bytes": _numeric_summary([float(item.get("payload_bytes") or 0.0) for item in samples]),
                "client_count": _numeric_summary([float(item.get("client_count") or 0.0) for item in samples]),
            }

        cycle_by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in scanner_cycles:
            cycle_by_kind[str(item.get("kind") or "unknown")].append(item)

        cycle_summary_by_kind = {kind: _cycle_summary(samples) for kind, samples in cycle_by_kind.items()}

        return {
            "routes": route_summary,
            "scanner_cycle": cycle_summary_by_kind.get("ws_manager", _cycle_summary([])),
            "scanner_cycle_by_kind": cycle_summary_by_kind,
            "broadcasts": broadcast_summary,
            "event_loop_lag_ms": _numeric_summary(lag_values),
            "cache_state": cache_samples[-1] if cache_samples else {},
            "slow_events": slow_events[-25:],
        }
