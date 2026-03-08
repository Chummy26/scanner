"""
Runtime analyzer for ArbML.

Combines empirical context extraction with a trained LSTM artifact bundle.
The analyzer never emits unsafe signals when the artifact is missing, invalid,
or the history is too small.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .ml_dataset import canonicalize_record
from .ml_model import (
    ModelArtifactMetadata,
    SpreadSequenceLSTM,
    get_artifact_paths,
    load_artifact_bundle,
    validate_artifact_metadata,
)
from .spread_tracker import SpreadRecord, build_recurring_context_from_episodes, compute_closed_episodes

logger = logging.getLogger(__name__)


class SpreadMLAnalyzer:
    def __init__(
        self,
        sequence_length: int = 15,
        artifact_dir: Path | None = None,
        allow_stale_artifacts: bool = False,
        min_total_spread_pct: float = 0.0,
    ):
        self.sequence_length = int(sequence_length)
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.allow_stale_artifacts = bool(allow_stale_artifacts)
        self.min_total_spread_pct = max(0.0, float(min_total_spread_pct or 0.0))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[SpreadSequenceLSTM] = None
        self.metadata: Optional[ModelArtifactMetadata] = None
        self.model_status = "missing_artifact"
        self.model_version = "unavailable"
        self.signal_reason = "trained artifact bundle not found"
        self.last_drift_status = "unknown"
        self.last_drifted_features: list[str] = []
        self.tracker: Any | None = None

        model_path, meta_path = get_artifact_paths(self.artifact_dir)
        if not model_path.is_file() or not meta_path.is_file():
            logger.warning("[ML] Artifact bundle missing at %s", model_path.parent)
            return

        try:
            model, metadata = load_artifact_bundle(
                self.artifact_dir,
                map_location=self.device,
                enforce_current_contract=not self.allow_stale_artifacts,
            )
        except Exception as exc:
            try:
                _, meta_path = get_artifact_paths(self.artifact_dir)
                metadata_payload = json.loads(meta_path.read_text(encoding="utf-8"))
                metadata = ModelArtifactMetadata.from_dict(metadata_payload)
                issues = validate_artifact_metadata(metadata)
                if issues:
                    self.metadata = metadata
                    self.model_version = metadata.version
                    self.sequence_length = metadata.sequence_length
                    self.model_status = "stale_artifact"
                    self.signal_reason = "; ".join(issues)
                    logger.warning("[ML] Rejected stale artifact bundle %s: %s", metadata.version, self.signal_reason)
                    return
            except Exception:
                pass
            self.model_status = "load_failed"
            self.signal_reason = f"artifact load failed: {exc}"
            logger.warning("[ML] Failed to load artifact bundle: %s", exc)
            return

        self.model = model.to(self.device)
        self.model.eval()
        self.metadata = metadata
        self.model_status = "ready"
        self.model_version = metadata.version
        self.sequence_length = metadata.sequence_length
        self.signal_reason = "trained artifact ready"

    def attach_tracker(self, tracker: Any | None):
        self.tracker = tracker

    def _coerce_history(self, history: list[Any]) -> list[dict[str, float]]:
        canonical: list[dict[str, float]] = []
        for index, item in enumerate(history):
            if not isinstance(item, dict):
                continue
            canonical.append(canonicalize_record(item, fallback_ts=float(index)))
        canonical.sort(key=lambda record: record["timestamp"])
        return canonical

    @staticmethod
    def _normalize_pair_key(pair_key: Any) -> Optional[tuple[str, str, str, str, str]]:
        if pair_key is None:
            return None
        if isinstance(pair_key, tuple) and len(pair_key) == 5:
            return (
                str(pair_key[0]).upper(),
                str(pair_key[1]).lower(),
                str(pair_key[2]).lower(),
                str(pair_key[3]).lower(),
                str(pair_key[4]).lower(),
            )
        if isinstance(pair_key, str):
            parts = pair_key.split("|")
            if len(parts) == 5:
                return (
                    str(parts[0]).upper(),
                    str(parts[1]).lower(),
                    str(parts[2]).lower(),
                    str(parts[3]).lower(),
                    str(parts[4]).lower(),
                )
        return None

    def _range_context(
        self,
        *,
        current_entry: float,
        history: list[dict[str, float]] | None = None,
        pair_key: Any = None,
        now_ts: float | None = None,
    ) -> dict[str, Any]:
        normalized_key = self._normalize_pair_key(pair_key)
        if self.tracker is not None and normalized_key is not None:
            try:
                return dict(
                    self.tracker.get_pair_recurring_context(
                        *normalized_key,
                        current_entry=float(current_entry),
                        now_ts=float(now_ts) if now_ts is not None else (float(history[-1]["timestamp"]) if history else time.time()),
                    )
                )
            except Exception as exc:
                logger.debug("[ML] Falling back to local recurring context for %s: %s", normalized_key, exc)
        history = history or []
        records = [
            SpreadRecord(
                timestamp=float(record["timestamp"]),
                entry_spread_pct=float(record["entry_spread"]),
                exit_spread_pct=float(record["exit_spread"]),
                session_id=1,
                block_id=1,
            )
            for record in history
        ]
        episodes = compute_closed_episodes(records)
        return dict(
            build_recurring_context_from_episodes(
                episodes,
                current_entry=float(current_entry),
                now_ts=float(history[-1]["timestamp"]) if history else time.time(),
                min_total_spread_pct=self.min_total_spread_pct,
            )
        )

    @staticmethod
    def _rolling_std_zscore(buffer: deque[float], current: float) -> tuple[float, float]:
        if len(buffer) < 2:
            return 0.0, 0.0
        mean = sum(buffer) / len(buffer)
        variance = sum((v - mean) ** 2 for v in buffer) / len(buffer)
        std = math.sqrt(variance)
        if std < 1e-6:
            return 0.0, 0.0
        return std, (current - mean) / std

    def _build_feature_tensor(self, history: list[dict[str, float]]) -> torch.Tensor:
        if self.metadata is None:
            raise RuntimeError("Cannot build feature tensor: model metadata not loaded")
        recent = history[-self.metadata.sequence_length :]
        expected_names = self.metadata.feature_names
        features = []
        previous_entry: Optional[float] = None
        previous_exit: Optional[float] = None
        previous_delta_entry: Optional[float] = None
        previous_delta_exit: Optional[float] = None
        entry_buffer: deque[float] = deque(maxlen=5)
        exit_buffer: deque[float] = deque(maxlen=5)
        for record in recent:
            entry_spread = record["entry_spread"]
            exit_spread = record["exit_spread"]
            delta_entry = 0.0 if previous_entry is None else entry_spread - previous_entry
            delta_exit = 0.0 if previous_exit is None else exit_spread - previous_exit
            delta2_entry = 0.0 if previous_delta_entry is None else delta_entry - previous_delta_entry
            delta2_exit = 0.0 if previous_delta_exit is None else delta_exit - previous_delta_exit
            entry_buffer.append(entry_spread)
            exit_buffer.append(exit_spread)
            rolling_std_entry, zscore_entry = self._rolling_std_zscore(entry_buffer, entry_spread)
            rolling_std_exit, zscore_exit = self._rolling_std_zscore(exit_buffer, exit_spread)
            all_features = {
                "entry_spread": entry_spread,
                "exit_spread": exit_spread,
                "delta_entry": delta_entry,
                "delta_exit": delta_exit,
                "delta2_entry": delta2_entry,
                "delta2_exit": delta2_exit,
                "rolling_std_entry": rolling_std_entry,
                "rolling_std_exit": rolling_std_exit,
                "zscore_entry": zscore_entry,
                "zscore_exit": zscore_exit,
            }
            features.append([all_features[name] for name in expected_names])
            previous_entry = entry_spread
            previous_exit = exit_spread
            previous_delta_entry = delta_entry
            previous_delta_exit = delta_exit

        feature_array = np.asarray(features, dtype=np.float32)
        drift_status, drifted_features = self._check_drift(feature_array)
        self.last_drift_status = drift_status
        self.last_drifted_features = drifted_features
        mean = np.asarray(self.metadata.feature_mean, dtype=np.float32)
        std = np.asarray(self.metadata.feature_std, dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        normalized = (feature_array - mean) / std
        return torch.tensor(normalized[np.newaxis, ...], dtype=torch.float32, device=self.device)

    def _check_drift(self, feature_array: np.ndarray) -> tuple[str, list[str]]:
        if self.metadata is None:
            return "unknown", []
        train_mean = np.asarray(self.metadata.feature_mean, dtype=np.float32)
        train_std = np.asarray(self.metadata.feature_std, dtype=np.float32)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)
        runtime_mean = feature_array.mean(axis=0)
        z_scores = np.abs((runtime_mean - train_mean) / train_std)
        drifted = z_scores > 3.0
        if np.any(drifted):
            names = self.metadata.feature_names
            drifted_features = [names[i] for i in range(len(names)) if drifted[i]]
            logger.warning(
                "[ML] Concept drift detected — features %s diverged >3σ from training distribution (z=%s)",
                drifted_features,
                [f"{z_scores[i]:.2f}" for i in range(len(names)) if drifted[i]],
            )
            return "drifted", drifted_features
        return "stable", []
    def _predict(self, history: list[dict[str, float]]) -> tuple[float, int]:
        if self.model is None or self.metadata is None:
            raise RuntimeError("Cannot predict: model or metadata not loaded")
        tensor = self._build_feature_tensor(history)
        with torch.no_grad():
            logits, eta_raw = self.model(tensor)
        logit_value = float(logits.item())
        calibrated_prob = 1.0 / (1.0 + math.exp(-(logit_value * self.metadata.platt_scale + self.metadata.platt_bias)))
        eta_seconds = int(max(60.0, math.expm1(max(float(eta_raw.item()), 0.0))))
        horizon = int(self.metadata.training_config.get("prediction_horizon_sec", 14_400))
        return float(np.clip(calibrated_prob, 1e-6, 1.0 - 1e-6)), min(eta_seconds, horizon)

    @staticmethod
    def _format_eta(eta_seconds: int) -> str:
        if eta_seconds >= 3600:
            return f"{eta_seconds // 3600}h {(eta_seconds % 3600) // 60}m"
        return f"{eta_seconds // 60}m"

    @staticmethod
    def _eta_payload(model_eta_seconds: int, context: dict[str, Any]) -> dict[str, Any]:
        q10 = int(round(float(context.get("empirical_eta_q10_seconds") or 0.0)))
        q25 = int(round(float(context.get("empirical_eta_q25_seconds") or 0.0)))
        q75 = int(round(float(context.get("empirical_eta_q75_seconds") or 0.0)))
        q90 = int(round(float(context.get("empirical_eta_q90_seconds") or 0.0)))
        if str(context.get("range_status") or "") == "insufficient_empirical_context":
            return {
                "eta_alignment_status": "unknown",
                "display_eta_seconds": 0,
                "eta_source": "not_displayed_no_range",
                "estimated_time_to_close": "--",
            }
        if q10 <= 0 or q90 <= 0:
            return {
                "eta_alignment_status": "unknown",
                "display_eta_seconds": int(model_eta_seconds),
                "eta_source": "model_only",
                "estimated_time_to_close": SpreadMLAnalyzer._format_eta(int(model_eta_seconds)),
            }
        if q25 <= model_eta_seconds <= q75:
            status = "aligned"
        elif q10 <= model_eta_seconds <= q90:
            status = "loose_alignment"
        else:
            status = "divergent"
        return {
            "eta_alignment_status": status,
            "display_eta_seconds": int(model_eta_seconds),
            "eta_source": "model_only",
            "estimated_time_to_close": SpreadMLAnalyzer._format_eta(int(model_eta_seconds)),
        }

    def _safe_wait_payload(
        self,
        current_entry: float,
        history: list[dict[str, float]],
        model_status: str,
        signal_reason: str,
        pair_key: Any = None,
    ) -> dict[str, Any]:
        context = self._range_context(current_entry=current_entry, history=history, pair_key=pair_key)
        nlp = (
            "Modelo indisponível ou histórico insuficiente. "
            f"Faixa recorrente: {context['recommended_entry_range']}."
        )
        return {
            "is_weak": True,
            "is_normal": False,
            "is_strong": False,
            "recommended_entry_range": context["recommended_entry_range"],
            "recommended_exit_range": context["recommended_exit_range"],
            "success_probability": 0.0,
            "estimated_time_to_close": "--",
            "signal_action": "WAIT",
            "nlp_context": nlp,
            "ml_score": 0,
            "model_status": model_status,
            "model_version": self.model_version,
            "inversion_probability": 0.0,
            "eta_seconds": 0,
            "model_eta_seconds": 0,
            "display_eta_seconds": 0,
            "eta_source": "unavailable",
            "eta_alignment_status": "unknown",
            "history_points": len(history),
            "context_percentile": context["context_percentile_excursions"],
            "context_percentile_excursions": context["context_percentile_excursions"],
            "empirical_support": context["empirical_support"],
            "empirical_support_short": context["empirical_support_short"],
            "empirical_support_long": context["empirical_support_long"],
            "raw_empirical_support": context.get("raw_empirical_support", 0),
            "raw_empirical_support_short": context.get("raw_empirical_support_short", 0),
            "raw_empirical_support_long": context.get("raw_empirical_support_long", 0),
            "context_strength": context["context_strength"],
            "is_entry_inside_range": context["is_entry_inside_range"],
            "range_status": context["range_status"],
            "range_window": context["range_window"],
            "median_total_spread": float(context.get("median_total_spread", 0.0) or 0.0),
            "min_total_spread_threshold": float(context.get("min_total_spread_threshold", self.min_total_spread_pct) or 0.0),
            "entry_outer_range_min": context["entry_outer_range_min"],
            "entry_outer_range_max": context["entry_outer_range_max"],
            "entry_core_range_min": context["entry_core_range_min"],
            "entry_core_range_max": context["entry_core_range_max"],
            "exit_outer_range_min": context["exit_outer_range_min"],
            "exit_outer_range_max": context["exit_outer_range_max"],
            "exit_core_range_min": context["exit_core_range_min"],
            "exit_core_range_max": context["exit_core_range_max"],
            "entry_position_label": context["entry_position_label"],
            "entry_coherent_short_long": context["entry_coherent_short_long"],
            "exit_coherent_short_long": context["exit_coherent_short_long"],
            "empirical_eta_median_seconds": context["empirical_eta_median_seconds"],
            "empirical_eta_q10_seconds": context["empirical_eta_q10_seconds"],
            "empirical_eta_q25_seconds": context["empirical_eta_q25_seconds"],
            "empirical_eta_q75_seconds": context["empirical_eta_q75_seconds"],
            "empirical_eta_q90_seconds": context["empirical_eta_q90_seconds"],
            "execute_threshold_used": float(self.metadata.execute_threshold) if self.metadata is not None else 0.0,
            "strong_threshold_used": float(self.metadata.strong_threshold) if self.metadata is not None else 0.0,
            "signal_reason_code": model_status,
            "signal_reason": signal_reason,
            "drift_status": self.last_drift_status,
            "drifted_features": list(self.last_drifted_features),
            "inference_latency_ms": 0.0,
            "artifact_feature_count": len(self.metadata.feature_names) if self.metadata is not None else 0,
            "artifact_trained_at_utc": self.metadata.trained_at_utc if self.metadata is not None else "",
            "artifact_dataset_samples": int(self.metadata.dataset_summary.get("num_samples", 0)) if self.metadata is not None else 0,
        }

    def predict_history(self, history: list[Any]) -> dict[str, Any]:
        canonical_history = self._coerce_history(history)
        history_points = len(canonical_history)
        history_last_ts = float(canonical_history[-1]["timestamp"]) if canonical_history else 0.0
        if history_points < max(4, self.sequence_length):
            return {
                "prediction_status": "insufficient_history",
                "model_status": "insufficient_history",
                "signal_reason": "not enough structured history for inference",
                "history_points": history_points,
                "history_last_ts": history_last_ts,
                "canonical_history": canonical_history,
            }
        if self.model is None or self.metadata is None:
            return {
                "prediction_status": self.model_status,
                "model_status": self.model_status,
                "signal_reason": self.signal_reason,
                "history_points": history_points,
                "history_last_ts": history_last_ts,
                "canonical_history": canonical_history,
            }

        min_history_points = max(self.metadata.min_history_points, self.sequence_length)
        if history_points < min_history_points:
            return {
                "prediction_status": "insufficient_history",
                "model_status": "insufficient_history",
                "signal_reason": f"need at least {min_history_points} structured history points",
                "history_points": history_points,
                "history_last_ts": history_last_ts,
                "canonical_history": canonical_history,
            }

        start_time = time.perf_counter()
        inversion_probability, eta_seconds = self._predict(canonical_history)
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        if inference_time_ms > 50.0:
            logger.warning("[ML] Inference latency alert: %.2fms", inference_time_ms)
        return {
            "prediction_status": "ready",
            "model_status": "ready",
            "signal_reason": "trained artifact ready",
            "history_points": history_points,
            "history_last_ts": history_last_ts,
            "canonical_history": canonical_history,
            "inversion_probability": float(inversion_probability),
            "model_eta_seconds": int(eta_seconds),
            "inference_latency_ms": float(inference_time_ms),
            "drift_status": self.last_drift_status,
            "drifted_features": list(self.last_drifted_features),
            "artifact_feature_count": len(self.metadata.feature_names),
            "artifact_trained_at_utc": self.metadata.trained_at_utc,
            "artifact_dataset_samples": int(self.metadata.dataset_summary.get("num_samples", 0)),
        }

    def render_prediction(
        self,
        current_entry: float,
        prediction: dict[str, Any],
        *,
        pair_key: Any = None,
    ) -> dict[str, Any]:
        model_status = str(prediction.get("model_status") or self.model_status)
        signal_reason = str(prediction.get("signal_reason") or self.signal_reason)
        history_points = int(prediction.get("history_points", 0) or 0)
        history_last_ts = float(prediction.get("history_last_ts", 0.0) or 0.0)
        canonical_history = list(prediction.get("canonical_history") or [])
        if str(prediction.get("prediction_status") or "") != "ready":
            return self._safe_wait_payload(
                current_entry=current_entry,
                history=canonical_history,
                model_status=model_status,
                signal_reason=signal_reason,
                pair_key=pair_key,
            )

        inversion_probability = float(prediction["inversion_probability"])
        eta_seconds = int(prediction["model_eta_seconds"])
        inference_time_ms = float(prediction.get("inference_latency_ms", 0.0) or 0.0)
        context = self._range_context(
            current_entry=current_entry,
            history=canonical_history,
            pair_key=pair_key,
            now_ts=history_last_ts,
        )
        eta_payload = self._eta_payload(int(eta_seconds), context)
        ml_score = round(inversion_probability * 100)

        execute_threshold = self.metadata.execute_threshold
        strong_threshold = self.metadata.strong_threshold
        signal_action = "WAIT"
        signal_reason_code = "insufficient_empirical_context"
        signal_reason = "Faixa recorrente indisponível: ainda não há recorrência suficiente em blocos fechados recentes."
        if str(context["range_status"]) == "insufficient_empirical_context":
            if (
                float(context.get("min_total_spread_threshold", 0.0) or 0.0) > 0.0
                and float(context.get("median_total_spread", 0.0) or 0.0) < float(context.get("min_total_spread_threshold", 0.0) or 0.0)
                and (
                    bool(context.get("raw_short_ready"))
                    or bool(context.get("raw_long_ready"))
                    or int(context.get("raw_empirical_support", 0) or 0) > 0
                )
            ):
                signal_reason_code = "median_total_spread_below_threshold"
                signal_reason = "Deslocamento total mediano abaixo do mínimo operacional."
            else:
                signal_reason_code = "insufficient_empirical_context"
                signal_reason = "Faixa recorrente indisponível: ainda não há recorrência suficiente em blocos fechados recentes."
        elif str(context["entry_position_label"]) == "below_band":
            signal_reason_code = "entry_outside_recurring_band"
            signal_reason = "Spread atual abaixo da faixa recorrente de entrada."
        elif str(context["entry_position_label"]) == "above_band":
            signal_reason_code = "entry_above_recurring_band"
            signal_reason = "Spread atual acima da faixa recorrente de entrada; caso bloqueado nesta versão."
        elif inversion_probability < execute_threshold:
            signal_reason_code = "probability_below_threshold"
            signal_reason = "Faixa recorrente válida, mas a probabilidade calibrada ainda está abaixo do limiar de execução."
        elif (
            inversion_probability >= strong_threshold
            and bool(context["strong_short_ready"])
            and bool(context["strong_long_ready"])
            and str(context["context_strength"]) == "strong"
            and bool(context["entry_coherent_short_long"])
            and bool(context["exit_coherent_short_long"])
            and str(eta_payload["eta_alignment_status"]) != "divergent"
        ):
            signal_action = "STRONG_EXECUTE"
            signal_reason_code = "strong_execute_ready"
            signal_reason = "Contexto forte confirmado. Probabilidade alta e janela empírica alinhada."
        elif inversion_probability >= execute_threshold:
            signal_action = "EXECUTE"
            if str(eta_payload["eta_alignment_status"]) == "divergent":
                signal_reason_code = "eta_divergent"
                signal_reason = "Contexto recorrente válido, porém o ETA do modelo diverge do comportamento empírico recente."
            else:
                signal_reason_code = "execute_ready"
                signal_reason = "O modelo encontrou contexto favorável. Probabilidade e contexto passaram o limiar de execução."

        return {
            "is_weak": signal_action == "WAIT",
            "is_normal": signal_action == "EXECUTE",
            "is_strong": signal_action == "STRONG_EXECUTE",
            "recommended_entry_range": context["recommended_entry_range"],
            "recommended_exit_range": context["recommended_exit_range"],
            "success_probability": round(inversion_probability * 100.0, 1),
            "estimated_time_to_close": eta_payload["estimated_time_to_close"],
            "signal_action": signal_action,
            "nlp_context": signal_reason,
            "ml_score": int(ml_score),
            "model_status": "ready",
            "model_version": self.model_version,
            "inversion_probability": round(inversion_probability, 6),
            "eta_seconds": int(eta_payload["display_eta_seconds"]),
            "model_eta_seconds": int(eta_seconds),
            "display_eta_seconds": int(eta_payload["display_eta_seconds"]),
            "eta_source": eta_payload["eta_source"],
            "eta_alignment_status": eta_payload["eta_alignment_status"],
            "history_points": history_points,
            "context_percentile": context["context_percentile_excursions"],
            "context_percentile_excursions": context["context_percentile_excursions"],
            "empirical_support": context["empirical_support"],
            "empirical_support_short": context["empirical_support_short"],
            "empirical_support_long": context["empirical_support_long"],
            "raw_empirical_support": context.get("raw_empirical_support", 0),
            "raw_empirical_support_short": context.get("raw_empirical_support_short", 0),
            "raw_empirical_support_long": context.get("raw_empirical_support_long", 0),
            "context_strength": context["context_strength"],
            "is_entry_inside_range": context["is_entry_inside_range"],
            "range_status": context["range_status"],
            "range_window": context["range_window"],
            "median_total_spread": float(context.get("median_total_spread", 0.0) or 0.0),
            "min_total_spread_threshold": float(context.get("min_total_spread_threshold", self.min_total_spread_pct) or 0.0),
            "entry_outer_range_min": context["entry_outer_range_min"],
            "entry_outer_range_max": context["entry_outer_range_max"],
            "entry_core_range_min": context["entry_core_range_min"],
            "entry_core_range_max": context["entry_core_range_max"],
            "exit_outer_range_min": context["exit_outer_range_min"],
            "exit_outer_range_max": context["exit_outer_range_max"],
            "exit_core_range_min": context["exit_core_range_min"],
            "exit_core_range_max": context["exit_core_range_max"],
            "entry_position_label": context["entry_position_label"],
            "entry_coherent_short_long": context["entry_coherent_short_long"],
            "exit_coherent_short_long": context["exit_coherent_short_long"],
            "empirical_eta_median_seconds": context["empirical_eta_median_seconds"],
            "empirical_eta_q10_seconds": context["empirical_eta_q10_seconds"],
            "empirical_eta_q25_seconds": context["empirical_eta_q25_seconds"],
            "empirical_eta_q75_seconds": context["empirical_eta_q75_seconds"],
            "empirical_eta_q90_seconds": context["empirical_eta_q90_seconds"],
            "execute_threshold_used": float(execute_threshold),
            "strong_threshold_used": float(strong_threshold),
            "signal_reason_code": signal_reason_code,
            "signal_reason": signal_reason,
            "drift_status": str(prediction.get("drift_status") or self.last_drift_status),
            "drifted_features": list(prediction.get("drifted_features") or self.last_drifted_features),
            "inference_latency_ms": round(float(inference_time_ms), 3),
            "artifact_feature_count": int(prediction.get("artifact_feature_count", len(self.metadata.feature_names))),
            "artifact_trained_at_utc": str(prediction.get("artifact_trained_at_utc") or self.metadata.trained_at_utc),
            "artifact_dataset_samples": int(prediction.get("artifact_dataset_samples", self.metadata.dataset_summary.get("num_samples", 0))),
        }

    def analyze_pair(self, current_entry: float, history: list[Any], *, pair_key: Any = None) -> Optional[dict[str, Any]]:
        prediction = self.predict_history(history)
        return self.render_prediction(current_entry, prediction, pair_key=pair_key)


__all__ = ["SpreadMLAnalyzer", "SpreadSequenceLSTM"]
