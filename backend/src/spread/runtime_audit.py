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

from .ml_analyzer import SpreadMLAnalyzer
from .ml_dataset import build_dataset_bundle
from .spread_tracker import SpreadRecord, build_recurring_context_from_episodes, compute_closed_episodes
from .train_model import run_clean_training_cycle


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


_PAIR_ALERT_COOLDOWN_SEC = 5 * 60


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


def _format_range(min_value: Any, max_value: Any) -> str:
    try:
        return f"{float(min_value):.2f}% à {float(max_value):.2f}%"
    except (TypeError, ValueError):
        return "--"


def summarize_dashboard_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        rows = []
    adjusting_count = 0
    actionable_without_range_count = 0
    strong_eta_divergent_count = 0
    above_band_mismatch_count = 0
    range_core_mismatch_count = 0
    reason_lane_message_mismatch_count = 0
    book_ages: list[float] = []
    book_age_asymmetries: list[float] = []
    invalid_quote_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        ml = row.get("mlContext") if isinstance(row.get("mlContext"), dict) else {}
        signal_action = str(ml.get("signal_action") or "WAIT")
        range_status = str(ml.get("range_status") or "unknown")
        entry_position = str(ml.get("entry_position_label") or "unknown")
        eta_alignment_status = str(ml.get("eta_alignment_status") or "unknown")
        action_lane = str(row.get("action_lane") or "blocked")
        signal_reason_code = str(row.get("signal_reason_code") or "")
        operator_message = str(row.get("operator_message") or "")
        try:
            buy_age = float(row.get("buyBookAge") or 0.0)
        except (TypeError, ValueError):
            buy_age = 0.0
        try:
            sell_age = float(row.get("sellBookAge") or 0.0)
        except (TypeError, ValueError):
            sell_age = 0.0
        if buy_age > 0.0:
            book_ages.append(buy_age)
        if sell_age > 0.0:
            book_ages.append(sell_age)
        if buy_age > 0.0 and sell_age > 0.0:
            book_age_asymmetries.append(abs(buy_age - sell_age))
        try:
            buy_price = float(row.get("buyPrice") or 0.0)
        except (TypeError, ValueError):
            buy_price = 0.0
        try:
            sell_price = float(row.get("sellPrice") or 0.0)
        except (TypeError, ValueError):
            sell_price = 0.0
        if buy_price <= 0.0 or sell_price <= 0.0:
            invalid_quote_rows += 1

        adjusting_count += 1 if str(ml.get("recommended_entry_range") or "") == "Ajustando..." else 0
        adjusting_count += 1 if str(ml.get("recommended_exit_range") or "") == "Ajustando..." else 0

        if signal_action in {"EXECUTE", "STRONG_EXECUTE"} and range_status == "insufficient_empirical_context":
            actionable_without_range_count += 1
        if signal_action == "STRONG_EXECUTE" and eta_alignment_status == "divergent":
            strong_eta_divergent_count += 1
        if entry_position == "above_band" and (
            signal_reason_code != "entry_above_recurring_band" or action_lane != "blocked"
        ):
            above_band_mismatch_count += 1

        expected_entry_range = _format_range(ml.get("entry_core_range_min"), ml.get("entry_core_range_max"))
        expected_exit_range = _format_range(ml.get("exit_core_range_min"), ml.get("exit_core_range_max"))
        if range_status != "insufficient_empirical_context":
            if expected_entry_range != str(ml.get("recommended_entry_range") or "--"):
                range_core_mismatch_count += 1
            if expected_exit_range != str(ml.get("recommended_exit_range") or "--"):
                range_core_mismatch_count += 1

        mismatch = False
        if signal_reason_code == "median_total_spread_below_threshold":
            mismatch = (
                action_lane != "blocked"
                or "abaixo do mínimo operacional" not in operator_message
            )
        elif range_status == "insufficient_empirical_context":
            mismatch = (
                action_lane != "blocked"
                or signal_reason_code != "insufficient_empirical_context"
                or "Faixa recorrente indisponível" not in operator_message
            )
        elif entry_position == "above_band":
            mismatch = (
                action_lane != "blocked"
                or signal_reason_code != "entry_above_recurring_band"
                or "acima da banda recorrente" not in operator_message
            )
        elif signal_reason_code == "execute_ready":
            mismatch = action_lane != "execute_now"
        elif signal_reason_code == "strong_execute_ready":
            mismatch = action_lane != "execute_now"
        elif signal_reason_code == "eta_divergent":
            mismatch = action_lane != "execute_now" or "diverge" not in operator_message
        if mismatch:
            reason_lane_message_mismatch_count += 1

    return {
        "payload_count": len(rows),
        "adjusting_count": adjusting_count,
        "actionable_without_range_count": actionable_without_range_count,
        "strong_eta_divergent_count": strong_eta_divergent_count,
        "above_band_mismatch_count": above_band_mismatch_count,
        "range_core_mismatch_count": range_core_mismatch_count,
        "reason_lane_message_mismatch_count": reason_lane_message_mismatch_count,
        "book_age_count": len(book_ages),
        "book_age_median_sec": round(_percentile(book_ages, 50.0), 6) if book_ages else 0.0,
        "book_age_p95_sec": round(_percentile(book_ages, 95.0), 6) if book_ages else 0.0,
        "book_age_max_sec": round(max(book_ages), 6) if book_ages else 0.0,
        "book_asymmetry_p95_sec": round(_percentile(book_age_asymmetries, 95.0), 6) if book_age_asymmetries else 0.0,
        "book_age_asymmetry_p95_sec": round(_percentile(book_age_asymmetries, 95.0), 6) if book_age_asymmetries else 0.0,
        "quote_row_count": len(rows),
        "empty_invalid_quote_count": invalid_quote_rows,
        "empty_invalid_quote_rate_pct": round((invalid_quote_rows / len(rows)) * 100.0, 6) if rows else 0.0,
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
        self._pair_alert_state: dict[tuple[str, str], dict[str, float | int]] = {}
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

    def _rate_limited_pair_alert(
        self,
        code: str,
        *,
        pair_key: str,
        severity: str = "warning",
        cooldown_sec: int = _PAIR_ALERT_COOLDOWN_SEC,
        **payload: Any,
    ) -> None:
        normalized_pair_key = str(pair_key or "")
        if not normalized_pair_key:
            self.alert(code, severity=severity, **payload)
            return
        now = time.time()
        state_key = (str(code), normalized_pair_key)
        state = dict(self._pair_alert_state.get(state_key) or {})
        last_ts = float(state.get("last_ts", 0.0) or 0.0)
        suppressed_count = int(state.get("suppressed_since_last", 0) or 0)
        if last_ts > 0.0 and (now - last_ts) < int(cooldown_sec):
            state["suppressed_since_last"] = suppressed_count + 1
            self._pair_alert_state[state_key] = state
            return
        enriched_payload = dict(payload)
        if suppressed_count > 0:
            enriched_payload["suppressed_since_last"] = suppressed_count
        self.alert(code, severity=severity, pair_key=normalized_pair_key, **enriched_payload)
        self._pair_alert_state[state_key] = {
            "last_ts": now,
            "suppressed_since_last": 0,
        }

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
                self._rate_limited_pair_alert(
                    "record_frequency_deviation",
                    pair_key=pair_key,
                    delta_ts=delta_ts,
                    expected_interval_sec=self._record_interval_sec,
                )
            if payload.get("gap_detected"):
                self._rate_limited_pair_alert(
                    "gap_detected",
                    pair_key=pair_key,
                    delta_ts=delta_ts,
                    gap_threshold_sec=float(payload.get("gap_threshold_sec") or 0.0),
                    block_id=int(payload.get("block_id") or 0),
                    session_id=int(payload.get("session_id") or 0),
                )
            if not bool(payload.get("numeric_valid", True)):
                self._rate_limited_pair_alert(
                    "invalid_record",
                    severity="error",
                    pair_key=pair_key,
                    payload=payload,
                )
            if not bool(payload.get("timestamp_monotonic", True)):
                self._rate_limited_pair_alert(
                    "non_monotonic_timestamp",
                    severity="error",
                    pair_key=pair_key,
                    payload=payload,
                )
            return
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
        sanitized_result = dict(result)
        sanitized_result.pop("current_entry", None)
        sanitized_result.pop("history_points", None)
        previous = self._last_probability_by_pair.get(pair_key)
        probability = float(sanitized_result.get("inversion_probability") or 0.0)
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
            **sanitized_result,
        )
        if str(sanitized_result.get("signal_action") or "WAIT") != "WAIT":
            self.event(
                "signal",
                pair_key=pair_key,
                current_entry=round(float(current_entry), 6),
                inversion_probability=round(probability, 6),
                eta_seconds=int(sanitized_result.get("eta_seconds") or 0),
                display_eta_seconds=int(sanitized_result.get("display_eta_seconds") or sanitized_result.get("eta_seconds") or 0),
                model_eta_seconds=int(sanitized_result.get("model_eta_seconds") or 0),
                signal_action=str(sanitized_result.get("signal_action") or "WAIT"),
                range_status=str(sanitized_result.get("range_status") or "unknown"),
                range_window=str(sanitized_result.get("range_window") or "none"),
                recommended_entry_range=str(sanitized_result.get("recommended_entry_range") or "--"),
                recommended_exit_range=str(sanitized_result.get("recommended_exit_range") or "--"),
                eta_alignment_status=str(sanitized_result.get("eta_alignment_status") or "unknown"),
                context_strength=str(sanitized_result.get("context_strength") or ""),
                entry_position_label=str(sanitized_result.get("entry_position_label") or "unknown"),
                signal_reason=str(sanitized_result.get("signal_reason") or ""),
                inference_latency_ms=round(latency_ms, 6),
                e2e_latency_ms=round(e2e_latency_ms, 6),
                drift_status=str(sanitized_result.get("drift_status") or "unknown"),
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


def _pair_key_from_row(row: sqlite3.Row) -> str:
    return f"{row['symbol']}|{row['buy_ex']}|{row['buy_mt']}|{row['sell_ex']}|{row['sell_mt']}"


def _latest_events_by_pair(events: list[dict[str, Any]], *, kind: str, model_status: str | None = None) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        if str(event.get("kind") or "") != kind:
            continue
        if model_status is not None and str(event.get("model_status") or "") != model_status:
            continue
        pair_key = str(event.get("pair_key") or "")
        if not pair_key:
            continue
        wall_ts = float(event.get("wall_ts") or 0.0)
        previous = latest.get(pair_key)
        if previous is None or wall_ts >= float(previous.get("wall_ts") or 0.0):
            latest[pair_key] = event
    return latest


def _load_snapshot_pair_maps(snapshot_path: Path) -> tuple[dict[str, int], dict[str, float], set[str]]:
    pair_to_id: dict[str, int] = {}
    pair_max_ts: dict[str, float] = {}
    eligible_pairs: set[str] = set()
    try:
        with sqlite3.connect(snapshot_path, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            pair_rows = list(
                conn.execute(
                    """
                    SELECT id, symbol, buy_ex, buy_mt, sell_ex, sell_mt
                    FROM tracker_pairs
                    """
                )
            )
            for row in pair_rows:
                pair_to_id[_pair_key_from_row(row)] = int(row["id"])
            for row in conn.execute(
                """
                SELECT p.symbol, p.buy_ex, p.buy_mt, p.sell_ex, p.sell_mt, MAX(r.ts) AS max_ts
                FROM tracker_records r
                JOIN tracker_pairs p ON p.id = r.pair_id
                GROUP BY r.pair_id
                """
            ):
                pair_max_ts[_pair_key_from_row(row)] = float(row["max_ts"] or 0.0)
            for row in conn.execute(
                """
                SELECT p.symbol, p.buy_ex, p.buy_mt, p.sell_ex, p.sell_mt
                FROM tracker_pair_blocks b
                JOIN tracker_pairs p ON p.id = b.pair_id
                WHERE b.is_open = 0 AND b.record_count >= 32
                GROUP BY b.pair_id
                """
            ):
                eligible_pairs.add(_pair_key_from_row(row))
    except sqlite3.Error:
        return {}, {}, set()
    return pair_to_id, pair_max_ts, eligible_pairs


def _build_context_coverage(events: list[dict[str, Any]], snapshot_path: Path | None) -> dict[str, Any]:
    latest_ready = _latest_events_by_pair(events, kind="inference", model_status="ready")
    if snapshot_path is None or not snapshot_path.is_file() or snapshot_path.suffix.lower() != ".sqlite":
        return {
            "eligible_ready_pairs": 0,
            "ready_short_pairs": 0,
            "ready_long_fallback_pairs": 0,
            "insufficient_empirical_context_pairs": 0,
            "insufficient_empirical_context_rate": 0.0,
            "support_short": _numeric_summary([]),
            "support_long": _numeric_summary([]),
            "near_miss_reasons": {},
        }
    _, _, eligible_pairs = _load_snapshot_pair_maps(snapshot_path)
    eligible_ready = {
        pair_key: payload
        for pair_key, payload in latest_ready.items()
        if pair_key in eligible_pairs
    }
    ready_short = 0
    ready_long = 0
    insufficient = 0
    short_support: list[float] = []
    long_support: list[float] = []
    near_miss_reasons: Counter[str] = Counter()
    for payload in eligible_ready.values():
        range_status = str(payload.get("range_status") or "unknown")
        if range_status == "ready_short":
            ready_short += 1
        elif range_status == "ready_long_fallback":
            ready_long += 1
        elif range_status == "insufficient_empirical_context":
            insufficient += 1
        short_support.append(float(payload.get("empirical_support_short") or 0.0))
        long_support.append(float(payload.get("empirical_support_long") or 0.0))
        if str(payload.get("signal_action") or "WAIT") != "STRONG_EXECUTE":
            if not bool(payload.get("strong_short_ready")):
                near_miss_reasons["short_ready_strong"] += 1
            elif not bool(payload.get("strong_long_ready")):
                near_miss_reasons["long_ready_strong"] += 1
            elif str(payload.get("entry_position_label") or "") != "inside_core":
                near_miss_reasons["inside_core"] += 1
            elif not bool(payload.get("entry_coherent_short_long")) or not bool(payload.get("exit_coherent_short_long")):
                near_miss_reasons["short_long_coherence"] += 1
            elif float(payload.get("inversion_probability") or 0.0) < float(payload.get("strong_threshold_used") or 0.0):
                near_miss_reasons["strong_threshold"] += 1
            elif str(payload.get("eta_alignment_status") or "") == "divergent":
                near_miss_reasons["eta_alignment"] += 1
    eligible_count = len(eligible_ready)
    return {
        "eligible_ready_pairs": eligible_count,
        "ready_short_pairs": ready_short,
        "ready_long_fallback_pairs": ready_long,
        "insufficient_empirical_context_pairs": insufficient,
        "insufficient_empirical_context_rate": round((insufficient / eligible_count), 4) if eligible_count else 0.0,
        "support_short": _numeric_summary(short_support),
        "support_long": _numeric_summary(long_support),
        "near_miss_reasons": dict(sorted(near_miss_reasons.items())),
    }


def _load_snapshot_histories(snapshot_path: Path) -> tuple[dict[str, list[dict[str, float]]], dict[str, float]]:
    histories: dict[str, list[dict[str, float]]] = {}
    pair_max_ts: dict[str, float] = {}
    try:
        with sqlite3.connect(snapshot_path, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute(
                """
                SELECT p.symbol, p.buy_ex, p.buy_mt, p.sell_ex, p.sell_mt,
                       r.ts, r.entry_spread_pct, r.exit_spread_pct, r.session_id, r.block_id
                FROM tracker_records r
                JOIN tracker_pairs p ON p.id = r.pair_id
                ORDER BY p.symbol, p.buy_ex, p.buy_mt, p.sell_ex, p.sell_mt, r.ts ASC
                """
            ):
                pair_key = _pair_key_from_row(row)
                record = {
                    "timestamp": float(row["ts"] or 0.0),
                    "entry_spread": float(row["entry_spread_pct"] or 0.0),
                    "exit_spread": float(row["exit_spread_pct"] or 0.0),
                    "session_id": int(row["session_id"] or 0),
                    "block_id": int(row["block_id"] or 0),
                }
                histories.setdefault(pair_key, []).append(record)
                pair_max_ts[pair_key] = float(record["timestamp"])
    except sqlite3.Error:
        return {}, {}
    return histories, pair_max_ts


def _load_snapshot_qualified_episodes(
    snapshot_path: Path,
    *,
    min_total_spread_pct: float,
) -> tuple[dict[str, list[Any]], dict[str, float]]:
    histories, pair_max_ts = _load_snapshot_histories(snapshot_path)
    qualified: dict[str, list[Any]] = {}
    for pair_key, history in histories.items():
        episodes = compute_closed_episodes(
            [
                SpreadRecord(
                    timestamp=float(record["timestamp"]),
                    entry_spread_pct=float(record["entry_spread"]),
                    exit_spread_pct=float(record["exit_spread"]),
                    session_id=int(record.get("session_id", 0) or 0),
                    block_id=int(record.get("block_id", 0) or 0),
                )
                for record in history
            ]
        )
        qualified[pair_key] = [
            episode
            for episode in episodes
            if float(getattr(episode, "total_spread", 0.0) or 0.0) >= float(min_total_spread_pct or 0.0)
        ]
    return qualified, pair_max_ts


def _confirm_signals_against_qualified_episodes(
    signal_events: list[dict[str, Any]],
    *,
    qualified_episodes_by_pair: dict[str, list[Any]],
    pair_max_ts: dict[str, float],
) -> dict[str, Any]:
    by_action: dict[str, Counter[str]] = {}
    by_eta_alignment: dict[str, Counter[str]] = {}
    signals: list[dict[str, Any]] = []
    confirmed = 0
    confirmable = 0
    awaiting = 0
    for event in signal_events:
        pair_key = str(event.get("pair_key") or "")
        signal_action = str(event.get("signal_action") or "WAIT")
        eta_alignment = str(event.get("eta_alignment_status") or "unknown")
        signal_ts = float(event.get("wall_ts") or 0.0)
        display_eta = int(event.get("display_eta_seconds") or event.get("eta_seconds") or 0)
        available_until_ts = float(pair_max_ts.get(pair_key, 0.0))
        status = "awaiting_confirmation"
        actual_duration_sec = None
        confirmed_within_eta = False
        if display_eta > 0 and available_until_ts >= signal_ts + display_eta:
            confirmable += 1
            qualified_episodes = qualified_episodes_by_pair.get(pair_key, [])
            first_episode = next(
                (
                    episode
                    for episode in qualified_episodes
                    if float(episode.end_ts) > signal_ts and float(episode.end_ts) <= signal_ts + display_eta
                ),
                None,
            )
            if first_episode is not None:
                status = "confirmed"
                confirmed += 1
                confirmed_within_eta = True
                actual_duration_sec = round(float(first_episode.end_ts) - signal_ts, 6)
            else:
                status = "not_confirmed"
        else:
            awaiting += 1
        by_action.setdefault(signal_action, Counter())[status] += 1
        by_eta_alignment.setdefault(eta_alignment, Counter())[status] += 1
        signals.append(
            {
                "pair_key": pair_key,
                "signal_action": signal_action,
                "signal_ts": signal_ts,
                "display_eta_seconds": display_eta,
                "model_eta_seconds": int(event.get("model_eta_seconds") or 0),
                "eta_alignment_status": eta_alignment,
                "status": status,
                "confirmed_within_eta": confirmed_within_eta,
                "actual_duration_sec": actual_duration_sec,
                "available_until_ts": available_until_ts,
                "range_status": str(event.get("range_status") or "unknown"),
                "range_window": str(event.get("range_window") or "none"),
                "recommended_entry_range": str(event.get("recommended_entry_range") or "--"),
                "recommended_exit_range": str(event.get("recommended_exit_range") or "--"),
                "current_entry": float(event.get("current_entry") or 0.0),
                "inversion_probability": float(event.get("inversion_probability") or 0.0),
                "context_strength": str(event.get("context_strength") or ""),
            }
        )
    return {
        "summary": {
            "total": len(signal_events),
            "confirmable_now": confirmable,
            "confirmed": confirmed,
            "not_confirmed": max(confirmable - confirmed, 0),
            "awaiting_confirmation": awaiting,
        },
        "by_action": {key: dict(value) for key, value in sorted(by_action.items())},
        "by_eta_alignment": {key: dict(value) for key, value in sorted(by_eta_alignment.items())},
        "signals": signals,
    }


def _build_signal_confirmations(
    events: list[dict[str, Any]],
    snapshot_path: Path | None,
    *,
    min_total_spread_pct: float = 1.0,
) -> dict[str, Any]:
    signal_events = [event for event in events if str(event.get("kind") or "") == "signal"]
    if snapshot_path is None or not snapshot_path.is_file() or snapshot_path.suffix.lower() != ".sqlite":
        return {"summary": {"total": len(signal_events), "confirmable_now": 0, "confirmed": 0, "awaiting_confirmation": len(signal_events)}, "by_action": {}, "by_eta_alignment": {}, "signals": []}
    try:
        qualified_episodes_by_pair, pair_max_ts = _load_snapshot_qualified_episodes(
            snapshot_path,
            min_total_spread_pct=min_total_spread_pct,
        )
    except sqlite3.Error:
        return {"summary": {"total": len(signal_events), "confirmable_now": 0, "confirmed": 0, "awaiting_confirmation": len(signal_events)}, "by_action": {}, "by_eta_alignment": {}, "signals": []}
    return _confirm_signals_against_qualified_episodes(
        signal_events,
        qualified_episodes_by_pair=qualified_episodes_by_pair,
        pair_max_ts=pair_max_ts,
    )


def _build_legacy_comparison(legacy_package_dir: Path | None) -> dict[str, Any]:
    if legacy_package_dir is None:
        return {"status": "not_requested"}
    legacy_dir = Path(legacy_package_dir)
    if not legacy_dir.is_dir():
        return {"status": "unavailable", "reason": "legacy package directory not found"}
    legacy_events = list(_iter_ndjson(legacy_dir / "events.ndjson"))
    legacy_signals = [event for event in legacy_events if str(event.get("kind") or "") == "signal"]
    if not legacy_signals:
        return {"status": "unavailable", "reason": "legacy package has no actionable signal events"}
    snapshot_path = legacy_dir / "sqlite_snapshot.sqlite"
    if not snapshot_path.is_file():
        return {"status": "unavailable", "reason": "legacy package has no sqlite snapshot"}
    context_ready = 0
    blocked_by_context = 0
    unresolved = 0
    compared = 0
    with sqlite3.connect(snapshot_path, timeout=30.0) as conn:
        conn.row_factory = sqlite3.Row
        pair_to_id, _, _ = _load_snapshot_pair_maps(snapshot_path)
        for event in legacy_signals:
            pair_key = str(event.get("pair_key") or "")
            signal_ts = float(event.get("wall_ts") or 0.0)
            current_entry = float(event.get("current_entry") or 0.0)
            pair_id = pair_to_id.get(pair_key)
            if pair_id is None or signal_ts <= 0.0:
                unresolved += 1
                continue
            pair_row = conn.execute("SELECT session_id, block_id, ts, entry_spread_pct, exit_spread_pct FROM tracker_records WHERE pair_id = ? AND ts <= ? ORDER BY ts ASC", (int(pair_id), signal_ts)).fetchall()
            if not pair_row:
                unresolved += 1
                continue
            records = [
                SpreadRecord(
                    timestamp=float(row["ts"]),
                    entry_spread_pct=float(row["entry_spread_pct"]),
                    exit_spread_pct=float(row["exit_spread_pct"]),
                    session_id=int(row["session_id"] or 0),
                    block_id=int(row["block_id"] or 0),
                )
                for row in pair_row
            ]
            context = build_recurring_context_from_episodes(
                compute_closed_episodes(records),
                current_entry=current_entry,
                now_ts=signal_ts,
            )
            compared += 1
            if str(context.get("range_status") or "") == "insufficient_empirical_context" or str(context.get("entry_position_label") or "") in {"below_band", "above_band"}:
                blocked_by_context += 1
            else:
                context_ready += 1
    return {
        "status": "ok",
        "legacy_actionable_signals": len(legacy_signals),
        "compared_signals": compared,
        "blocked_by_new_context": blocked_by_context,
        "still_context_ready": context_ready,
        "unresolved_signals": unresolved,
    }


def _training_threshold_from_report(training_report: dict[str, Any]) -> float:
    for payload in (
        training_report.get("preflight", {}),
        training_report.get("artifact_metadata", {}).get("training_config", {}),
        training_report.get("training_config", {}),
    ):
        value = payload.get("selected_threshold")
        if value is None:
            value = payload.get("min_total_spread_pct")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0.0:
            return numeric
    return 1.0


def _build_snapshot_retrain_readiness(snapshot_path: Path, *, min_total_spread_pct: float) -> dict[str, Any]:
    qualified_episodes_by_pair, _ = _load_snapshot_qualified_episodes(
        snapshot_path,
        min_total_spread_pct=min_total_spread_pct,
    )
    qualified_episode_count = sum(len(episodes) for episodes in qualified_episodes_by_pair.values())
    if qualified_episode_count >= 500:
        status = "preferred_met"
    elif qualified_episode_count >= 200:
        status = "minimum_met"
    else:
        status = "below_minimum"
    return {
        "qualified_episode_count_at_threshold": int(qualified_episode_count),
        "min_total_spread_pct": float(min_total_spread_pct),
        "retrain_readiness_status": status,
    }


def _psi_value(observed: list[float], expected: list[float]) -> float:
    epsilon = 1e-6
    total = 0.0
    for observed_value, expected_value in zip(observed, expected):
        observed_safe = max(float(observed_value), epsilon)
        expected_safe = max(float(expected_value), epsilon)
        total += (observed_safe - expected_safe) * math.log(observed_safe / expected_safe)
    return float(total)


def _build_snapshot_psi_summary(
    snapshot_path: Path,
    *,
    artifact_dir: Path,
    sequence_length: int,
    prediction_horizon_sec: int,
) -> dict[str, Any]:
    metadata_path = artifact_dir / "best_lstm_model.meta.json"
    if not metadata_path.is_file():
        return {"status": "unavailable", "reason": "artifact metadata missing"}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unavailable", "reason": f"artifact metadata unreadable: {exc}"}
    psi_reference = metadata.get("training_config", {}).get("psi_reference", {})
    feature_bins = psi_reference.get("feature_bins", {})
    min_samples_required = int(psi_reference.get("min_samples_required", 200) or 200)
    if not feature_bins:
        return {"status": "unavailable", "reason": "psi reference missing from metadata"}
    bundle = build_dataset_bundle(
        state_path=snapshot_path,
        sequence_length=sequence_length,
        prediction_horizon_sec=prediction_horizon_sec,
        min_total_spread_pct=float(metadata.get("training_config", {}).get("min_total_spread_pct", 1.0) or 1.0),
    )
    if int(bundle.summary.get("num_samples", 0)) < min_samples_required:
        return {
            "status": "insufficient_samples",
            "sample_count": int(bundle.summary.get("num_samples", 0)),
            "min_samples_required": min_samples_required,
            "features": {},
        }
    flattened = bundle.X.detach().cpu().numpy().reshape(-1, bundle.X.shape[-1])
    feature_names = list(metadata.get("feature_names", []))
    psi_by_feature: dict[str, Any] = {}
    for feature_index, feature_name in enumerate(feature_names):
        reference = feature_bins.get(feature_name)
        if not isinstance(reference, dict):
            continue
        edges = [
            -np.inf if edge is None and index == 0 else (np.inf if edge is None else float(edge))
            for index, edge in enumerate(reference.get("bin_edges", []))
        ]
        expected = [float(value) for value in reference.get("expected_proportions", [])]
        if len(edges) < 2 or len(expected) == 0:
            continue
        counts, _ = np.histogram(flattened[:, feature_index], bins=np.asarray(edges, dtype=float))
        observed = (counts / max(int(counts.sum()), 1)).tolist()
        psi_score = _psi_value(observed, expected)
        if psi_score > 0.20:
            status = "alert"
        elif psi_score >= 0.10:
            status = "investigate"
        else:
            status = "stable"
        psi_by_feature[feature_name] = {
            "psi": float(psi_score),
            "status": status,
        }
    return {
        "status": "ok",
        "sample_count": int(bundle.summary.get("num_samples", 0)),
        "min_samples_required": min_samples_required,
        "features": psi_by_feature,
    }


def _build_snapshot_replay(
    snapshot_path: Path,
    *,
    artifact_dir: Path,
    min_total_spread_pct: float,
) -> dict[str, Any]:
    model_path = artifact_dir / "best_lstm_model.pth"
    metadata_path = artifact_dir / "best_lstm_model.meta.json"
    if not model_path.is_file() or not metadata_path.is_file():
        return {"status": "unavailable", "reason": "artifact bundle missing"}
    analyzer = SpreadMLAnalyzer(
        artifact_dir=artifact_dir,
        allow_stale_artifacts=True,
        min_total_spread_pct=min_total_spread_pct,
    )
    if analyzer.model_status != "ready":
        return {"status": "unavailable", "reason": analyzer.model_status}
    histories, pair_max_ts = _load_snapshot_histories(snapshot_path)
    qualified_episodes_by_pair, _ = _load_snapshot_qualified_episodes(
        snapshot_path,
        min_total_spread_pct=min_total_spread_pct,
    )
    raw_actionable_states: list[dict[str, Any]] = []
    deduped_signals: list[dict[str, Any]] = []
    max_repeated_action_streak_sec = 0.0
    for pair_key, history in histories.items():
        active_until_ts = 0.0
        streak_start_ts: float | None = None
        for index in range(max(analyzer.sequence_length, 4), len(history) + 1):
            current_history = history[:index]
            current_ts = float(current_history[-1]["timestamp"])
            current_entry = float(current_history[-1]["entry_spread"])
            result = analyzer.analyze_pair(current_entry, current_history, pair_key=pair_key) or {}
            action = str(result.get("signal_action") or "WAIT")
            if action not in {"EXECUTE", "STRONG_EXECUTE"}:
                streak_start_ts = None
                if current_ts > active_until_ts:
                    active_until_ts = 0.0
                continue
            raw_state = {
                "pair_key": pair_key,
                "signal_action": action,
                "wall_ts": current_ts,
                "current_entry": current_entry,
                "display_eta_seconds": int(result.get("display_eta_seconds") or result.get("eta_seconds") or 0),
                "model_eta_seconds": int(result.get("model_eta_seconds") or 0),
                "eta_alignment_status": str(result.get("eta_alignment_status") or "unknown"),
                "range_status": str(result.get("range_status") or "unknown"),
                "range_window": str(result.get("range_window") or "none"),
                "recommended_entry_range": str(result.get("recommended_entry_range") or "--"),
                "recommended_exit_range": str(result.get("recommended_exit_range") or "--"),
                "inversion_probability": float(result.get("inversion_probability") or 0.0),
                "context_strength": str(result.get("context_strength") or ""),
            }
            raw_actionable_states.append(raw_state)
            if streak_start_ts is None:
                streak_start_ts = current_ts
            max_repeated_action_streak_sec = max(max_repeated_action_streak_sec, current_ts - streak_start_ts)
            if current_ts > active_until_ts:
                deduped_signals.append(dict(raw_state))
                active_until_ts = current_ts + float(raw_state["display_eta_seconds"] or 0)

    actionable_states_per_pair_per_hour: dict[str, float] = {}
    for pair_key, history in histories.items():
        if len(history) < 2:
            actionable_states_per_pair_per_hour[pair_key] = 0.0
            continue
        state_count = sum(1 for state in raw_actionable_states if str(state.get("pair_key") or "") == pair_key)
        elapsed_hours = max((float(history[-1]["timestamp"]) - float(history[0]["timestamp"])) / 3600.0, 1.0 / 3600.0)
        actionable_states_per_pair_per_hour[pair_key] = float(state_count / elapsed_hours)

    confirmations = _confirm_signals_against_qualified_episodes(
        deduped_signals,
        qualified_episodes_by_pair=qualified_episodes_by_pair,
        pair_max_ts=pair_max_ts,
    )
    return {
        "status": "ok",
        "raw_actionable_state_count": len(raw_actionable_states),
        "deduped_signal_count": len(deduped_signals),
        "actionable_states_per_pair_per_hour": actionable_states_per_pair_per_hour,
        "max_repeated_action_streak_sec": float(max_repeated_action_streak_sec),
        "confirmation_summary": dict(confirmations.get("summary", {})),
        "confirmations_by_action": dict(confirmations.get("by_action", {})),
        "confirmations_by_eta_alignment": dict(confirmations.get("by_eta_alignment", {})),
        "deduped_signals": deduped_signals,
    }


def build_runtime_summary(
    output_dir: Path,
    *,
    snapshot_path: Path | None = None,
    legacy_package_dir: Path | None = None,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    event_rows = events if events is not None else list(_iter_ndjson(output_root / "events.ndjson"))
    alerts = list(_iter_ndjson(output_root / "alerts.ndjson"))
    samples = list(_iter_ndjson(output_root / "samples.ndjson"))
    api_probes = list(_iter_ndjson(output_root / "api_probe.ndjson"))

    ws_by_exchange: dict[str, list[float]] = {}
    inference_latencies: list[float] = []
    end_to_end_latencies: list[float] = []
    signal_counts: Counter[str] = Counter()
    model_status_counts: Counter[str] = Counter()
    alert_counts: Counter[str] = Counter(str(item.get("code") or "") for item in alerts)

    for event in event_rows:
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

    dashboard_validation = {
        "adjusting_count": sum(int(sample.get("adjusting_count") or 0) for sample in api_probes),
        "actionable_without_range_count": sum(int(sample.get("actionable_without_range_count") or 0) for sample in api_probes),
        "strong_eta_divergent_count": sum(int(sample.get("strong_eta_divergent_count") or 0) for sample in api_probes),
        "above_band_mismatch_count": sum(int(sample.get("above_band_mismatch_count") or 0) for sample in api_probes),
        "range_core_mismatch_count": sum(int(sample.get("range_core_mismatch_count") or 0) for sample in api_probes),
        "reason_lane_message_mismatch_count": sum(int(sample.get("reason_lane_message_mismatch_count") or 0) for sample in api_probes),
        "book_age_median_sec": _numeric_summary(
            [float(sample.get("book_age_median_sec") or 0.0) for sample in api_probes if float(sample.get("book_age_median_sec") or 0.0) > 0.0]
        ),
        "book_age_p95_sec": _numeric_summary(
            [float(sample.get("book_age_p95_sec") or 0.0) for sample in api_probes if float(sample.get("book_age_p95_sec") or 0.0) > 0.0]
        ),
        "book_age_max_sec": _numeric_summary(
            [float(sample.get("book_age_max_sec") or 0.0) for sample in api_probes if float(sample.get("book_age_max_sec") or 0.0) > 0.0]
        ),
        "book_age_asymmetry_p95_sec": _numeric_summary(
            [float(sample.get("book_age_asymmetry_p95_sec") or 0.0) for sample in api_probes if float(sample.get("book_age_asymmetry_p95_sec") or 0.0) > 0.0]
        ),
        "empty_invalid_quote_rate_pct": _numeric_summary(
            [float(sample.get("empty_invalid_quote_rate_pct") or 0.0) for sample in api_probes]
        ),
        "probe_count": len(api_probes),
    }

    context_coverage = _build_context_coverage(event_rows, snapshot_path)
    legacy_comparison = _build_legacy_comparison(legacy_package_dir)

    return {
        "counts": {
            "events": len(event_rows),
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
        "dashboard_validation": dashboard_validation,
        "context_coverage": context_coverage,
        "legacy_comparison": legacy_comparison,
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
        "num_cross_block_windows": int(bundle.summary.get("num_cross_block_windows", 0) or 0),
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
    signal_confirmations: dict[str, Any],
    auxiliary_short_block_smoke: dict[str, Any],
    output_dir: Path,
) -> str:
    inference = runtime_summary.get("inference_latency_ms", {})
    signal_counts = runtime_summary.get("signal_counts", {})
    context_coverage = runtime_summary.get("context_coverage", {})
    dashboard_validation = runtime_summary.get("dashboard_validation", {})
    legacy_comparison = runtime_summary.get("legacy_comparison", {})
    lines = [
        "# Runtime Audit Package",
        "",
        "## Runtime Overview",
        f"- Events: {runtime_summary.get('counts', {}).get('events', 0)}",
        f"- Alerts: {runtime_summary.get('counts', {}).get('alerts', 0)}",
        f"- Signals: {signal_counts}",
        f"- Inference p99: {inference.get('p99', 0):.3f} ms",
        "",
        "## Context Coverage",
        f"- Eligible ready pairs: {context_coverage.get('eligible_ready_pairs', 0)}",
        f"- ready_short: {context_coverage.get('ready_short_pairs', 0)}",
        f"- ready_long_fallback: {context_coverage.get('ready_long_fallback_pairs', 0)}",
        f"- insufficient_empirical_context_rate: {context_coverage.get('insufficient_empirical_context_rate', 0):.4f}",
        f"- Near misses for STRONG_EXECUTE: {context_coverage.get('near_miss_reasons', {})}",
        "",
        "## Dashboard Validation",
        f"- Adjustando occurrences: {dashboard_validation.get('adjusting_count', 0)}",
        f"- Actionable without range: {dashboard_validation.get('actionable_without_range_count', 0)}",
        f"- Strong with eta divergent: {dashboard_validation.get('strong_eta_divergent_count', 0)}",
        f"- Above-band mismatches: {dashboard_validation.get('above_band_mismatch_count', 0)}",
        f"- Range/core mismatches: {dashboard_validation.get('range_core_mismatch_count', 0)}",
        f"- Reason/lane/message mismatches: {dashboard_validation.get('reason_lane_message_mismatch_count', 0)}",
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
        "## Signal Confirmation",
        f"- Summary: {signal_confirmations.get('summary', {})}",
        f"- By action: {signal_confirmations.get('by_action', {})}",
        f"- By eta alignment: {signal_confirmations.get('by_eta_alignment', {})}",
        "",
        "## Auxiliary Short-Block Smoke",
        f"- Status: {auxiliary_short_block_smoke.get('status', 'unavailable')}",
        f"- Dataset status: {auxiliary_short_block_smoke.get('dataset_status', 'unavailable')}",
        f"- All-negative expected: {auxiliary_short_block_smoke.get('all_negative_expected', False)}",
        "",
        "## Legacy Comparison",
        f"- Status: {legacy_comparison.get('status', 'not_requested')}",
        f"- Details: {legacy_comparison}",
        "",
        "## Notes",
        "- Esta trilha de auditoria é dedicada e isolada do relatório principal do modelo.",
        f"- Output dir: `{output_dir}`",
        "",
        "## Artifacts",
        f"- Runtime summary: `{output_dir / 'summary.json'}`",
        f"- Dataset summary: `{output_dir / 'dataset_summary.json'}`",
        f"- Training report: `{output_dir / 'training_report.json'}`",
        f"- Signal confirmations: `{output_dir / 'signal_confirmations.json'}`",
        "",
    ]
    return "\n".join(lines)


def _build_auxiliary_short_block_smoke(
    *,
    snapshot_path: Path,
    output_root: Path,
    run_training_fn: Any,
    sequence_length: int,
    prediction_horizon_sec: int,
) -> dict[str, Any]:
    smoke = {
        "status": "unavailable",
        "dataset_status": "unavailable",
        "all_negative_expected": True,
        "not_for_model_quality_decision": True,
        "auxiliary_short_block_smoke": True,
    }
    if snapshot_path.suffix.lower() != ".sqlite" or not snapshot_path.is_file():
        smoke["reason"] = "snapshot is not a sqlite database"
        return smoke
    try:
        with sqlite3.connect(snapshot_path, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT id, end_ts, start_ts, record_count
                FROM tracker_pair_blocks
                WHERE is_open = 0
                ORDER BY end_ts DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
    except sqlite3.Error as exc:
        smoke["reason"] = f"unable to inspect closed blocks: {exc}"
        return smoke
    if row is None:
        smoke["reason"] = "no closed block available for auxiliary smoke"
        return smoke
    block_id = int(row["id"])
    smoke.update(
        {
            "status": "selected",
            "selected_block_id": block_id,
            "selected_block_duration_sec": max(0.0, float(row["end_ts"] or 0.0) - float(row["start_ts"] or 0.0)),
            "selected_block_record_count": int(row["record_count"] or 0),
        }
    )
    try:
        bundle = build_dataset_bundle(
            state_path=snapshot_path,
            sequence_length=sequence_length,
            prediction_horizon_sec=prediction_horizon_sec,
            selected_block_ids=[block_id],
        )
        smoke["dataset_status"] = "built"
        smoke["dataset_summary"] = _dataset_summary_payload(bundle)
        smoke["all_negative_expected"] = bool(smoke["dataset_summary"].get("num_positive_samples", 0) == 0)
    except Exception as exc:
        smoke["dataset_status"] = "dataset_failed"
        smoke["dataset_error"] = str(exc)
        return smoke
    try:
        smoke_artifact_dir = output_root / "training_artifacts" / "auxiliary_short_block_smoke"
        smoke["training_report"] = run_training_fn(
            state_file=snapshot_path,
            artifact_dir=smoke_artifact_dir,
            sequence_length=sequence_length,
            prediction_horizon_sec=prediction_horizon_sec,
            selected_block_ids=[block_id],
            audit_output=smoke_artifact_dir / "best_lstm_model.audit.md",
        )
        smoke["status"] = "completed"
    except Exception as exc:
        smoke["status"] = "training_failed"
        smoke["training_error"] = str(exc)
    return smoke


def finalize_runtime_audit_package(
    *,
    output_dir: Path,
    state_path: Path,
    run_training_fn=run_clean_training_cycle,
    duration_sec: int = 7_200,
    sequence_length: int = 15,
    prediction_horizon_sec: int = 14_400,
    run_status: str = "completed",
    legacy_package_dir: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cached_events = list(_iter_ndjson(output_root / "events.ndjson"))

    runtime_summary = build_runtime_summary(output_root, events=cached_events)
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
    min_total_spread_pct = 1.0
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
        min_total_spread_pct = _training_threshold_from_report(training_report if isinstance(training_report, dict) else {})
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

    signal_confirmations = _build_signal_confirmations(
        cached_events,
        snapshot_path,
        min_total_spread_pct=min_total_spread_pct,
    )
    _write_json(output_root / "signal_confirmations.json", signal_confirmations)
    snapshot_replay = _build_snapshot_replay(
        snapshot_path,
        artifact_dir=training_root,
        min_total_spread_pct=min_total_spread_pct,
    )
    snapshot_replay_path = _write_json(output_root / "snapshot_replay_report.json", snapshot_replay)
    psi_monitoring = _build_snapshot_psi_summary(
        snapshot_path,
        artifact_dir=training_root,
        sequence_length=sequence_length,
        prediction_horizon_sec=prediction_horizon_sec,
    )
    _write_json(output_root / "psi_monitoring.json", psi_monitoring)
    retrain_readiness = _build_snapshot_retrain_readiness(
        snapshot_path,
        min_total_spread_pct=min_total_spread_pct,
    )
    _write_json(output_root / "retrain_readiness.json", retrain_readiness)
    auxiliary_short_block_smoke = _build_auxiliary_short_block_smoke(
        snapshot_path=snapshot_path,
        output_root=output_root,
        run_training_fn=run_training_fn,
        sequence_length=sequence_length,
        prediction_horizon_sec=prediction_horizon_sec,
    )
    if isinstance(training_report, dict):
        training_report["auxiliary_short_block_smoke"] = auxiliary_short_block_smoke
    dataset_summary_path = _write_json(output_root / "dataset_summary.json", dataset_summary)
    training_report_path = _write_json(output_root / "training_report.json", training_report)

    runtime_summary = build_runtime_summary(
        output_root,
        snapshot_path=snapshot_path,
        legacy_package_dir=legacy_package_dir,
        events=cached_events,
    )
    runtime_summary["finished_at_utc"] = _utc_now_iso()
    runtime_summary["duration_sec"] = int(duration_sec)
    runtime_summary["run_status"] = str(run_status)
    runtime_summary["signal_confirmation_summary"] = dict(signal_confirmations.get("summary", {}))
    runtime_summary["snapshot_replay"] = {
        "status": snapshot_replay.get("status", "unavailable"),
        "raw_actionable_state_count": int(snapshot_replay.get("raw_actionable_state_count", 0)),
        "deduped_signal_count": int(snapshot_replay.get("deduped_signal_count", 0)),
        "confirmation_summary": dict(snapshot_replay.get("confirmation_summary", {})),
    }
    runtime_summary["psi_monitoring"] = psi_monitoring
    runtime_summary["retrain_readiness"] = retrain_readiness
    summary_path = _write_json(output_root / "summary.json", runtime_summary)
    (output_root / "runtime_audit.md").write_text(build_runtime_markdown(runtime_summary), encoding="utf-8")

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
            signal_confirmations=signal_confirmations,
            auxiliary_short_block_smoke=auxiliary_short_block_smoke,
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
        "snapshot_replay_path": str(snapshot_replay_path),
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
    "summarize_dashboard_payload",
]
