from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

from .ml_dataset import build_dataset_bundle
from .train_model import run_training_loop


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return path


def _append_ndjson(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":")))
            handle.write("\n")


def _iter_ndjson(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (float(percentile) / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[int(rank)])
    return float(ordered[lower] + ((ordered[upper] - ordered[lower]) * (rank - lower)))


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not cleaned:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": len(cleaned),
        "min": min(cleaned),
        "max": max(cleaned),
        "mean": sum(cleaned) / len(cleaned),
        "p95": _percentile(cleaned, 95.0),
        "p99": _percentile(cleaned, 99.0),
    }


def default_runtime_audit_dir(base_dir: Path | None = None) -> Path:
    root = (
        Path(base_dir)
        if base_dir is not None
        else Path(__file__).resolve().parent.parent.parent / "out" / "runtime_audit"
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / stamp


def create_sqlite_snapshot(source_path: Path, target_path: Path) -> Path:
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() == ".sqlite":
        with sqlite3.connect(source_path, timeout=30.0) as source, sqlite3.connect(target_path, timeout=30.0) as target:
            source.backup(target)
        return target_path
    shutil.copy2(source_path, target_path)
    return target_path


class RuntimeAuditCollector:
    def __init__(
        self,
        output_dir: Path,
        *,
        duration_sec: int = 7_200,
        status_sample_sec: int = 5,
        resource_sample_sec: int = 60,
        record_interval_sec: float = 15.0,
        gap_threshold_sec: float = 60.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.ndjson"
        self.samples_path = self.output_dir / "samples.ndjson"
        self.alerts_path = self.output_dir / "alerts.ndjson"
        self.api_probe_path = self.output_dir / "api_probe.ndjson"
        self.summary_path = self.output_dir / "summary.json"
        self.runtime_report_path = self.output_dir / "runtime_audit.md"
        self.final_report_path = self.output_dir / "final_audit.md"
        self.manual_verification_path = self.output_dir / "manual_data_verification.md"
        self.dataset_summary_path = self.output_dir / "dataset_summary.json"
        self.signal_confirmation_path = self.output_dir / "signal_confirmations.json"
        self.manifest_path = self.output_dir / "manifest.json"
        self.snapshot_path = self.output_dir / "sqlite_snapshot.sqlite"
        self.training_artifact_dir = self.output_dir / "training_artifacts"
        self._lock = threading.Lock()
        self._started_perf = time.perf_counter()
        self._last_probability_by_pair: dict[str, float] = {}
        self._last_record_perf_by_pair: dict[str, float] = {}
        self._last_exchange_state: dict[str, tuple[bool, bool]] = {}
        self._duration_sec = int(duration_sec)
        self._status_sample_sec = max(1, int(status_sample_sec))
        self._resource_sample_sec = max(1, int(resource_sample_sec))
        self._record_interval_sec = float(record_interval_sec)
        self._gap_threshold_sec = float(gap_threshold_sec)
        self._ws_manager: Any = None
        self._manifest = {
            "started_at_utc": _utc_now_iso(),
            "duration_sec": self._duration_sec,
            "status_sample_sec": self._status_sample_sec,
            "resource_sample_sec": self._resource_sample_sec,
            "record_interval_sec": self._record_interval_sec,
            "gap_threshold_sec": self._gap_threshold_sec,
        }
        _write_json(self.manifest_path, self._manifest)

    def attach_ws_manager(self, ws_manager: Any):
        self._ws_manager = ws_manager
        if hasattr(ws_manager, "set_runtime_audit_session"):
            ws_manager.set_runtime_audit_session(self)
        elif hasattr(ws_manager, "attach_runtime_audit"):
            ws_manager.attach_runtime_audit(self)
        tracker = getattr(ws_manager, "tracker", None)
        if tracker is not None and hasattr(tracker, "add_event_listener"):
            tracker.add_event_listener(self.on_tracker_event)
        self._manifest.update(
            {
                "tracker_db_path": str(getattr(ws_manager.config, "tracker_db_path", "")),
                "bundle_status": getattr(ws_manager.ml_analyzer, "model_status", "unknown"),
                "bundle_version": getattr(ws_manager.ml_analyzer, "model_version", "unavailable"),
            }
        )
        _write_json(self.manifest_path, self._manifest)

    def detach_ws_manager(self):
        tracker = getattr(self._ws_manager, "tracker", None)
        if tracker is not None and hasattr(tracker, "remove_event_listener"):
            tracker.remove_event_listener(self.on_tracker_event)
        self._ws_manager = None

    def _base_event(self, kind: str) -> dict[str, Any]:
        return {
            "kind": kind,
            "logged_at_utc": _utc_now_iso(),
            "elapsed_sec": round(time.perf_counter() - self._started_perf, 6),
            "wall_ts": round(time.time(), 6),
        }

    def event(self, kind: str, **payload: Any):
        data = self._base_event(kind)
        data.update(payload)
        _append_ndjson(self.events_path, data, self._lock)

    def sample(self, kind: str, **payload: Any):
        data = self._base_event(kind)
        data.update(payload)
        _append_ndjson(self.samples_path, data, self._lock)

    def alert(self, code: str, *, severity: str = "warning", **payload: Any):
        data = self._base_event("alert")
        data.update({"code": code, "severity": severity})
        data.update(payload)
        _append_ndjson(self.alerts_path, data, self._lock)

    def api_probe(self, *, latency_ms: float, status_code: int, ok: bool, payload_count: int = 0, error: str = ""):
        self.sample(
            "api_probe",
            latency_ms=round(float(latency_ms), 6),
            status_code=int(status_code),
            ok=bool(ok),
            payload_count=int(payload_count),
            error=str(error or ""),
        )

    def on_ws_ingest(self, *, exchange: str, symbol: str, market_type: str, latency_ms: float, connected: bool):
        self.event(
            "ws_ingest",
            exchange=str(exchange),
            symbol=str(symbol),
            market_type=str(market_type),
            latency_ms=round(float(latency_ms), 6),
            connected=bool(connected),
        )

    def record_ws_ingest(
        self,
        exchange: str,
        symbol: str,
        market_type: str,
        latency_ms: float,
        source_delay_ms: float | None = None,
    ):
        self.on_ws_ingest(
            exchange=exchange,
            symbol=symbol,
            market_type=market_type,
            latency_ms=latency_ms,
            connected=True,
        )
        if source_delay_ms is not None:
            self.event(
                "ws_source_delay",
                exchange=str(exchange),
                symbol=str(symbol),
                market_type=str(market_type),
                source_delay_ms=round(float(source_delay_ms), 6),
            )

    def record_exchange_status(self, exchange_statuses: list[dict[str, Any]]):
        self.sample("exchange_status_batch", exchanges=list(exchange_statuses))

    def on_tracker_event(self, payload: dict[str, Any]):
        kind = str(payload.get("kind") or "")
        if kind == "tracker_record":
            pair_key = str(payload.get("pair_key") or "")
            self._last_record_perf_by_pair[pair_key] = time.perf_counter()
            delta_ts = float(payload.get("delta_ts") or 0.0)
            if delta_ts > max(self._record_interval_sec * 1.5, self._record_interval_sec + 1.0):
                self.alert(
                    "record_frequency_deviation",
                    pair_key=pair_key,
                    delta_ts=delta_ts,
                    expected_interval_sec=self._record_interval_sec,
                )
            if payload.get("gap_detected"):
                self.alert(
                    "gap_detected",
                    pair_key=pair_key,
                    delta_ts=delta_ts,
                    gap_threshold_sec=float(payload.get("gap_threshold_sec") or 0.0),
                    block_id=int(payload.get("block_id") or 0),
                    session_id=int(payload.get("session_id") or 0),
                )
            if not bool(payload.get("numeric_valid", True)):
                self.alert("invalid_record", severity="error", pair_key=pair_key, payload=payload)
            if not bool(payload.get("timestamp_monotonic", True)):
                self.alert("non_monotonic_timestamp", severity="error", pair_key=pair_key, payload=payload)
        elif kind == "tracker_flush":
            if float(payload.get("duration_ms") or 0.0) > 5_000.0:
                self.alert("slow_tracker_flush", duration_ms=float(payload.get("duration_ms") or 0.0))
        elif kind == "tracker_error":
            self.alert("tracker_error", severity="error", payload=payload)
        event_payload = dict(payload)
        event_payload.pop("kind", None)
        self.event(kind or "tracker_event", **event_payload)

    def record_tracker_record(
        self,
        *,
        pair_id: str,
        ts: float,
        entry: float,
        exit: float,
        session_id: int,
        block_id: int,
        record_delta_sec: float,
        gap_detected: bool,
        monotonic: bool,
        invalid_fields: list[str],
    ):
        self.on_tracker_event(
            {
                "kind": "tracker_record",
                "pair_key": str(pair_id),
                "record_ts": round(float(ts), 6),
                "entry": round(float(entry), 6),
                "exit": round(float(exit), 6),
                "session_id": int(session_id),
                "block_id": int(block_id),
                "delta_ts": round(float(record_delta_sec), 6),
                "gap_detected": bool(gap_detected),
                "gap_threshold_sec": float(self._gap_threshold_sec),
                "numeric_valid": not invalid_fields,
                "invalid_fields": list(invalid_fields),
                "timestamp_monotonic": bool(monotonic),
            }
        )

    def on_runtime_error(self, *, component: str, error: str, context: dict[str, Any] | None = None):
        self.alert(
            "runtime_error",
            severity="error",
            component=str(component),
            error=str(error),
            context=context or {},
        )

    def record_error(self, component: str, error: str, *, details: dict[str, Any] | None = None):
        self.on_runtime_error(component=component, error=error, context=details or {})

    def on_inference(self, *, pair_key: str, current_entry: float, history_points: int, result: dict[str, Any]):
        previous = self._last_probability_by_pair.get(pair_key)
        probability = float(result.get("inversion_probability") or 0.0)
        if previous is not None and abs(probability - previous) > 0.30:
            self.alert(
                "signal_probability_jump",
                pair_key=pair_key,
                previous_probability=round(previous, 6),
                current_probability=round(probability, 6),
                delta=round(abs(probability - previous), 6),
            )
        self._last_probability_by_pair[pair_key] = probability
        latency_ms = float(result.get("inference_latency_ms") or 0.0)
        if latency_ms > 50.0:
            self.alert("inference_latency_high", pair_key=pair_key, latency_ms=latency_ms)
        if str(result.get("drift_status") or "") == "drifted":
            self.alert(
                "drift_detected",
                pair_key=pair_key,
                drifted_features=list(result.get("drifted_features") or []),
            )
        e2e_latency_ms = 0.0
        last_record_perf = self._last_record_perf_by_pair.get(pair_key)
        if last_record_perf is not None:
            e2e_latency_ms = max(0.0, (time.perf_counter() - last_record_perf) * 1000.0)
        self.event(
            "inference",
            pair_key=pair_key,
            current_entry=round(float(current_entry), 6),
            history_points=int(history_points),
            e2e_latency_ms=round(e2e_latency_ms, 6),
            **result,
        )
        if str(result.get("signal_action") or "WAIT") != "WAIT":
            self.event(
                "signal",
                pair_key=pair_key,
                inversion_probability=round(probability, 6),
                eta_seconds=int(result.get("eta_seconds") or 0),
                signal_action=str(result.get("signal_action") or "WAIT"),
                context_strength=str(result.get("context_strength") or ""),
                inference_latency_ms=round(latency_ms, 6),
                e2e_latency_ms=round(e2e_latency_ms, 6),
                drift_status=str(result.get("drift_status") or "unknown"),
            )

    def record_inference(self, *, pair_id: str, result: dict[str, Any], end_to_end_ms: float | None = None):
        if end_to_end_ms is not None:
            self._last_record_perf_by_pair[str(pair_id)] = time.perf_counter() - (float(end_to_end_ms) / 1000.0)
        self.on_inference(
            pair_key=str(pair_id),
            current_entry=float(result.get("current_entry") or 0.0),
            history_points=int(result.get("history_points") or 0),
            result=result,
        )

    def sample_system(
        self,
        *,
        pid: int,
        db_path: Path,
        tracker_stats: dict[str, Any],
        exchange_statuses: list[dict[str, Any]],
        extra: dict[str, Any] | None = None,
    ):
        payload: dict[str, Any] = {
            "pid": int(pid),
            "db_path": str(db_path),
            "db_size_bytes": int(Path(db_path).stat().st_size) if Path(db_path).exists() else 0,
            "tracker_stats": dict(tracker_stats),
            "exchanges": list(exchange_statuses),
            "extra": dict(extra or {}),
        }
        if psutil is not None:
            try:
                process = psutil.Process(int(pid))
                memory = process.memory_info()
                payload.update(
                    {
                        "cpu_percent": float(psutil.cpu_percent(interval=None)),
                        "rss_bytes": int(memory.rss),
                        "vms_bytes": int(memory.vms),
                        "thread_count": int(process.num_threads()),
                    }
                )
            except Exception:  # pragma: no cover
                pass
        self.sample("system_sample", **payload)

    def sample_runtime(self):
        ws_manager = self._ws_manager
        if ws_manager is None:
            return
        try:
            status = ws_manager.get_status()
            tracker_stats = ws_manager.tracker.get_storage_stats()
            expected = sorted(list(getattr(ws_manager, "_exchange_instances", {}).keys()))
            connected = status.get("exchanges", {})
            connected_count = 0
            for exchange in expected:
                exchange_state = connected.get(exchange, {"spot": False, "futures": False})
                pair = (bool(exchange_state.get("spot")), bool(exchange_state.get("futures")))
                previous = self._last_exchange_state.get(exchange)
                if previous is not None and previous != pair:
                    if pair != (False, False) and previous == (False, False):
                        self.alert("exchange_reconnected", exchange=exchange, spot=pair[0], futures=pair[1])
                    elif pair == (False, False):
                        self.alert("exchange_disconnected", exchange=exchange)
                self._last_exchange_state[exchange] = pair
                if pair[0] or pair[1]:
                    connected_count += 1
                self.sample(
                    "exchange_status",
                    exchange=exchange,
                    spot_connected=pair[0],
                    futures_connected=pair[1],
                    expected=True,
                )
            summary = status.get("summary", {})
            self.sample(
                "runtime_status",
                expected_exchange_count=len(expected),
                connected_exchange_count=connected_count,
                connected_exchanges=connected,
                scanner_clients=int(status.get("scanner_clients") or 0),
                symbols_tracked=int(summary.get("symbols_tracked") or 0),
                current_opportunities=int(summary.get("opportunities") or 0),
                tracker_stats=tracker_stats,
            )
        except Exception:  # pragma: no cover
            pass

    def finalize_runtime_only(self) -> dict[str, Any]:
        summary = build_runtime_summary(self.output_dir)
        summary["manifest"] = dict(self._manifest)
        summary["finished_at_utc"] = _utc_now_iso()
        _write_json(self.summary_path, summary)
        self.runtime_report_path.write_text(build_runtime_markdown(summary), encoding="utf-8")
        return summary

    def finalize(self) -> dict[str, Path]:
        self.finalize_runtime_only()
        return {
            "events_path": self.events_path,
            "alerts_path": self.alerts_path,
            "samples_path": self.samples_path,
            "summary_path": self.summary_path,
        }


RuntimeAuditSession = RuntimeAuditCollector


def build_runtime_summary(output_dir: Path) -> dict[str, Any]:
    output_root = Path(output_dir)
    events = list(_iter_ndjson(output_root / "events.ndjson"))
    alerts = list(_iter_ndjson(output_root / "alerts.ndjson"))
    samples = list(_iter_ndjson(output_root / "samples.ndjson"))
    api_probes = list(_iter_ndjson(output_root / "api_probe.ndjson"))

    ws_by_exchange: dict[str, list[float]] = {}
    inference_latencies: list[float] = []
    end_to_end_latencies: list[float] = []
    signal_counts: Counter[str] = Counter()
    model_status_counts: Counter[str] = Counter()
    alert_counts: Counter[str] = Counter(str(item.get("code") or "") for item in alerts)

    for event in events:
        kind = str(event.get("kind") or "")
        if kind == "ws_ingest":
            exchange = str(event.get("exchange") or "unknown")
            ws_by_exchange.setdefault(exchange, []).append(float(event.get("latency_ms") or 0.0))
        elif kind == "inference":
            inference_latencies.append(float(event.get("inference_latency_ms") or 0.0))
            end_to_end_latencies.append(float(event.get("e2e_latency_ms") or 0.0))
            model_status_counts[str(event.get("model_status") or "unknown")] += 1
        elif kind == "signal":
            signal_counts[str(event.get("signal_action") or "WAIT")] += 1

    api_probe_latencies = [
        float(sample.get("latency_ms") or 0.0)
        for sample in samples
        if str(sample.get("kind") or "") == "api_probe"
    ]
    api_probe_latencies.extend(
        float(sample.get("latency_ms") or 0.0)
        for sample in api_probes
        if str(sample.get("kind") or "") == "api_probe"
    )
    system_db_sizes = [
        float(sample.get("db_size_bytes") or 0.0)
        for sample in samples
        if str(sample.get("kind") or "") == "system_sample"
    ]
    cpu_values = [
        float(sample.get("cpu_percent") or 0.0)
        for sample in samples
        if str(sample.get("kind") or "") == "system_sample" and sample.get("cpu_percent") is not None
    ]
    rss_values = [
        float(sample.get("rss_bytes") or 0.0)
        for sample in samples
        if str(sample.get("kind") or "") == "system_sample" and sample.get("rss_bytes") is not None
    ]

    return {
        "counts": {
            "events": len(events),
            "alerts": len(alerts),
            "samples": len(samples),
        },
        "signal_counts": dict(signal_counts),
        "model_status_counts": dict(model_status_counts),
        "alert_counts": dict(alert_counts),
        "ws_ingest_latency_ms": {
            exchange: _numeric_summary(values)
            for exchange, values in sorted(ws_by_exchange.items())
        },
        "inference_latency_ms": _numeric_summary(inference_latencies),
        "end_to_end_latency_ms": _numeric_summary(end_to_end_latencies),
        "api_probe_latency_ms": _numeric_summary(api_probe_latencies),
        "resource_summary": {
            "cpu_percent": _numeric_summary(cpu_values),
            "rss_bytes": _numeric_summary(rss_values),
            "db_size_bytes": _numeric_summary(system_db_sizes),
        },
    }


def build_runtime_markdown(summary: dict[str, Any]) -> str:
    counts = summary.get("counts", {})
    signal_counts = summary.get("signal_counts", {})
    inference = summary.get("inference_latency_ms", {})
    end_to_end = summary.get("end_to_end_latency_ms", {})
    alert_counts = summary.get("alert_counts", {})
    lines = [
        "# Runtime Audit",
        "",
        "## Overview",
        f"- Events: {counts.get('events', 0)}",
        f"- Alerts: {counts.get('alerts', 0)}",
        f"- Samples: {counts.get('samples', 0)}",
        f"- Signals: {signal_counts}",
        "",
        "## Latency",
        f"- Inference ms: min={inference.get('min', 0):.3f} max={inference.get('max', 0):.3f} p99={inference.get('p99', 0):.3f}",
        f"- End-to-end ms: min={end_to_end.get('min', 0):.3f} max={end_to_end.get('max', 0):.3f} p99={end_to_end.get('p99', 0):.3f}",
        "",
        "## Alerts",
        f"- Alert counts: {alert_counts}",
        "",
        "## Notes",
        "- Este relatório é separado do relatório principal do modelo para evitar poluição do fluxo normal.",
        "- O runner de 2 horas usa este pacote dedicado e não substitui `ml_diagnostic_report.md`.",
        "",
    ]
    return "\n".join(lines)


def _manual_feature_checks(bundle: Any, max_checks: int = 3) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    limit = min(int(bundle.X.shape[0]), max_checks)
    for sample_index in range(limit):
        tensor = bundle.X[sample_index].detach().cpu().tolist()
        if len(tensor) < 5:
            continue
        entries = [float(row[0]) for row in tensor]
        mean_value = sum(entries[:5]) / 5.0
        variance = sum((value - mean_value) ** 2 for value in entries[:5]) / 5.0
        std_value = math.sqrt(variance)
        zscore_value = 0.0 if std_value < 1e-6 else (entries[4] - mean_value) / std_value
        checks.append(
            {
                "sample_index": int(sample_index),
                "pair_id": str(bundle.pair_ids[sample_index]),
                "block_id": int(bundle.block_ids[sample_index]),
                "delta_entry_expected": round(entries[1] - entries[0], 6),
                "delta_entry_actual": round(float(tensor[1][2]), 6),
                "rolling_std_entry_expected": round(float(std_value), 6),
                "rolling_std_entry_actual": round(float(tensor[4][6]), 6),
                "zscore_entry_expected": round(float(zscore_value), 6),
                "zscore_entry_actual": round(float(tensor[4][8]), 6),
            }
        )
    return checks


def _manual_label_checks(bundle: Any, max_checks: int = 3) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    limit = min(int(bundle.y_class.shape[0]), max_checks)
    for sample_index in range(limit):
        checks.append(
            {
                "sample_index": int(sample_index),
                "pair_id": str(bundle.pair_ids[sample_index]),
                "block_id": int(bundle.block_ids[sample_index]),
                "session_id": int(bundle.session_ids[sample_index]),
                "y_class": float(bundle.y_class[sample_index].item()),
                "y_eta": float(bundle.y_eta[sample_index].item()),
            }
        )
    return checks


def _dataset_summary_payload(bundle: Any) -> dict[str, Any]:
    return {
        **dict(bundle.summary),
        "shape": list(bundle.X.shape),
        "feature_abs_sum": float(bundle.X.abs().sum().item()),
        "feature_has_nan": bool(getattr(bundle.X, "isnan")().any().item()),
        "feature_has_inf": bool(getattr(bundle.X, "isinf")().any().item()),
        "class_has_nan": bool(getattr(bundle.y_class, "isnan")().any().item()),
        "eta_has_nan": bool(getattr(bundle.y_eta, "isnan")().any().item()),
        "num_cross_block_windows": 0,
        "manual_feature_checks": _manual_feature_checks(bundle),
        "manual_label_checks": _manual_label_checks(bundle),
    }


def _build_manual_verification_markdown(
    *,
    snapshot_path: Path,
    dataset_summary: dict[str, Any],
    training_report: dict[str, Any],
) -> str:
    lines = [
        "# Manual Data Verification",
        "",
        f"- Snapshot: `{snapshot_path}`",
        f"- Samples: {dataset_summary.get('num_samples', 0)}",
        f"- Positive samples: {dataset_summary.get('num_positive_samples', 0)}",
        f"- Negative samples: {dataset_summary.get('num_negative_samples', 0)}",
        f"- Cross-block windows: {dataset_summary.get('num_cross_block_windows', 0)}",
        "",
        "## Feature checks",
    ]
    for item in dataset_summary.get("manual_feature_checks", []):
        lines.append(
            "- Sample {sample_index} / block {block_id}: delta_entry expected {delta_entry_expected}, actual {delta_entry_actual}; "
            "rolling_std expected {rolling_std_entry_expected}, actual {rolling_std_entry_actual}; "
            "zscore expected {zscore_entry_expected}, actual {zscore_entry_actual}".format(**item)
        )
    lines.extend(["", "## Label checks"])
    for item in dataset_summary.get("manual_label_checks", []):
        lines.append(
            f"- Sample {item['sample_index']} / block {item['block_id']}: y_class={item['y_class']} y_eta={item['y_eta']}"
        )
    lines.extend(["", "## Training", f"- Status: {training_report.get('model_status', 'unknown')}", f"- Artifacts: {training_report.get('artifacts', {})}", ""])
    return "\n".join(lines)


def _build_final_audit_markdown(
    *,
    runtime_summary: dict[str, Any],
    dataset_summary: dict[str, Any],
    training_report: dict[str, Any],
    output_dir: Path,
) -> str:
    inference = runtime_summary.get("inference_latency_ms", {})
    signal_counts = runtime_summary.get("signal_counts", {})
    lines = [
        "# Runtime Audit Package",
        "",
        "## Runtime Overview",
        f"- Events: {runtime_summary.get('counts', {}).get('events', 0)}",
        f"- Alerts: {runtime_summary.get('counts', {}).get('alerts', 0)}",
        f"- Signals: {signal_counts}",
        f"- Inference p99: {inference.get('p99', 0):.3f} ms",
        "",
        "## Collection & Integrity",
        f"- Dataset samples: {dataset_summary.get('num_samples', 0)}",
        f"- Cross-block windows: {dataset_summary.get('num_cross_block_windows', 0)}",
        f"- Shape: {dataset_summary.get('shape', [])}",
        "",
        "## Offline Training",
        f"- Training status: {training_report.get('model_status', 'unknown')}",
        f"- Metrics: {training_report.get('metrics', {})}",
        f"- Thresholds: {training_report.get('thresholds', {})}",
        "",
        "## Notes",
        "- Esta trilha de auditoria é dedicada e isolada do relatório principal do modelo.",
        f"- Output dir: `{output_dir}`",
        "",
        "## Artifacts",
        f"- Runtime summary: `{output_dir / 'summary.json'}`",
        f"- Dataset summary: `{output_dir / 'dataset_summary.json'}`",
        f"- Training report: `{output_dir / 'training_report.json'}`",
        "",
    ]
    return "\n".join(lines)


def finalize_runtime_audit_package(
    *,
    output_dir: Path,
    state_path: Path,
    run_training_fn=run_training_loop,
    duration_sec: int = 7_200,
    sequence_length: int = 15,
    prediction_horizon_sec: int = 14_400,
    run_status: str = "completed",
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    runtime_summary = build_runtime_summary(output_root)
    runtime_summary["finished_at_utc"] = _utc_now_iso()
    runtime_summary["duration_sec"] = int(duration_sec)
    runtime_summary["run_status"] = str(run_status)
    summary_path = _write_json(output_root / "summary.json", runtime_summary)
    (output_root / "runtime_audit.md").write_text(build_runtime_markdown(runtime_summary), encoding="utf-8")

    source_state = Path(state_path)
    snapshot_name = "sqlite_snapshot.sqlite" if source_state.suffix.lower() == ".sqlite" else source_state.name
    snapshot_path = output_root / snapshot_name
    training_root = output_root / "training_artifacts"
    training_root.mkdir(parents=True, exist_ok=True)
    try:
        create_sqlite_snapshot(source_state, snapshot_path)
        dataset_bundle = build_dataset_bundle(
            state_path=snapshot_path,
            sequence_length=sequence_length,
            prediction_horizon_sec=prediction_horizon_sec,
        )
        dataset_summary = _dataset_summary_payload(dataset_bundle)
        training_report = run_training_fn(
            state_file=snapshot_path,
            artifact_dir=training_root,
            sequence_length=sequence_length,
            prediction_horizon_sec=prediction_horizon_sec,
            audit_output=training_root / "best_lstm_model.audit.md",
        )
    except Exception as exc:
        dataset_summary = {
            "model_status": "dataset_failed",
            "error": str(exc),
            "num_samples": 0,
            "num_positive_samples": 0,
            "num_negative_samples": 0,
            "shape": [0, sequence_length, 10],
            "feature_abs_sum": 0.0,
            "feature_has_nan": False,
            "feature_has_inf": False,
            "class_has_nan": False,
            "eta_has_nan": False,
            "num_cross_block_windows": 0,
            "manual_feature_checks": [],
            "manual_label_checks": [],
        }
        training_report = {
            "model_status": "dataset_failed",
            "error": str(exc),
            "metrics": {},
            "thresholds": {},
            "artifacts": {},
        }
    dataset_summary_path = _write_json(output_root / "dataset_summary.json", dataset_summary)
    training_report_path = _write_json(output_root / "training_report.json", training_report)

    manual_verification_path = output_root / "manual_data_verification.md"
    manual_verification_path.write_text(
        _build_manual_verification_markdown(
            snapshot_path=snapshot_path,
            dataset_summary=dataset_summary,
            training_report=training_report,
        ),
        encoding="utf-8",
    )

    final_audit_path = output_root / "final_audit.md"
    final_audit_path.write_text(
        _build_final_audit_markdown(
            runtime_summary=runtime_summary,
            dataset_summary=dataset_summary,
            training_report=training_report,
            output_dir=output_root,
        ),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_root),
        "summary_path": str(summary_path),
        "sqlite_snapshot": str(snapshot_path),
        "dataset_summary_path": str(dataset_summary_path),
        "training_report_path": str(training_report_path),
        "manual_verification_path": str(manual_verification_path),
        "final_audit_path": str(final_audit_path),
    }


__all__ = [
    "RuntimeAuditCollector",
    "RuntimeAuditSession",
    "build_runtime_markdown",
    "build_runtime_summary",
    "create_sqlite_snapshot",
    "default_runtime_audit_dir",
    "finalize_runtime_audit_package",
]
