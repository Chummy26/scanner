from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from .feature_contracts import (
    DEFAULT_FEATURE_CONTRACT_VERSION,
    feature_contract_version_for_names,
)
from .ml_dataset import (
    DatasetBundle,
    _build_pair_segments,
    _load_blocks_from_sqlite,
    _load_tracker_gap_threshold_sec,
    build_dataset_bundle,
    build_group_splits,
    compute_feature_stats,
    normalize_features,
)
from .training_certification import run_training_certification as _certify_data_for_training_impl
from .ml_model import (
    ARTIFACT_BASENAME,
    ModelArtifactMetadata,
    SpreadSequenceLSTM,
    current_feature_schema_hash,
    get_artifact_paths,
    get_default_artifact_dir,
    save_artifact_bundle,
)

logger = logging.getLogger("ml_trainer")
_DEFAULT_PRELIGHT_THRESHOLDS = [0.8, 1.0, 1.2]
_DEFAULT_LABEL_PERCENTILES = [60, 70, 80]


def _make_loader(bundle: DatasetBundle, batch_size: int, shuffle: bool) -> DataLoader:
    effective_batch = max(1, min(int(batch_size), max(1, bundle.summary["num_samples"])))
    dataset = TensorDataset(bundle.X, bundle.y_class, bundle.y_eta)
    use_pin = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=shuffle,
        pin_memory=use_pin,
        num_workers=4 if use_pin else 0,
        persistent_workers=True if use_pin else False,
        prefetch_factor=2 if use_pin else None,
    )


def _eta_loss(
    criterion: nn.SmoothL1Loss,
    eta_raw: torch.Tensor,
    target_eta_seconds: torch.Tensor,
    target_class: torch.Tensor,
) -> torch.Tensor:
    mask = target_class > 0.5
    if not torch.any(mask):
        return eta_raw.new_tensor(0.0)
    target = torch.log1p(target_eta_seconds[mask])
    pred = eta_raw.squeeze(1)[mask]
    return criterion(pred, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def _derive_focal_alpha(positive_rate_train: float) -> float:
    return float(min(0.95, max(0.25, 1.0 - float(positive_rate_train))))


def _normalize_threshold_values(thresholds: list[float] | None) -> list[float]:
    values = [float(value) for value in (thresholds or _DEFAULT_PRELIGHT_THRESHOLDS)]
    normalized: list[float] = []
    seen: set[float] = set()
    for value in values:
        if not math.isfinite(value):
            continue
        numeric = float(value)
        if numeric in seen:
            continue
        seen.add(numeric)
        normalized.append(numeric)
    return normalized or [float(value) for value in _DEFAULT_PRELIGHT_THRESHOLDS]


def _normalize_label_percentiles(label_percentiles: list[int] | None) -> list[int]:
    values = [int(value) for value in (label_percentiles or _DEFAULT_LABEL_PERCENTILES)]
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        numeric = max(0, min(100, int(value)))
        if numeric in seen:
            continue
        seen.add(numeric)
        normalized.append(numeric)
    return normalized or [int(value) for value in _DEFAULT_LABEL_PERCENTILES]


def _adaptive_label_requested(
    *,
    thresholds: list[float] | None,
    label_percentiles: list[int] | None,
) -> bool:
    return label_percentiles is not None or thresholds is None


def _iter_label_configs(
    *,
    thresholds: list[float] | None,
    label_percentiles: list[int] | None,
    label_episode_window_days: int,
) -> list[dict[str, Any]]:
    threshold_values = _normalize_threshold_values(thresholds)
    if not _adaptive_label_requested(thresholds=thresholds, label_percentiles=label_percentiles):
        return [
            {
                "key": f"{float(threshold):.1f}",
                "threshold": float(threshold),
                "cost_floor_pct": float(threshold),
                "label_percentile": None,
                "label_episode_window_days": None,
                "label_threshold_mode": "fixed_threshold",
            }
            for threshold in threshold_values
        ]
    percentile_values = _normalize_label_percentiles(label_percentiles)
    return [
        {
            "key": f"{float(threshold):.1f}_p{int(percentile)}",
            "threshold": float(threshold),
            "cost_floor_pct": float(threshold),
            "label_percentile": int(percentile),
            "label_episode_window_days": int(max(label_episode_window_days, 1)),
            "label_threshold_mode": "rolling_pair_percentile",
        }
        for threshold in threshold_values
        for percentile in percentile_values
    ]


def _label_config_payload(label_config: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(label_config or {})
    percentile_raw = config.get("label_percentile")
    percentile = None if percentile_raw is None else int(percentile_raw)
    window_days_raw = config.get("label_episode_window_days")
    window_days = None if window_days_raw is None else int(window_days_raw)
    return {
        "threshold": float(config.get("threshold", config.get("cost_floor_pct", 0.0)) or 0.0),
        "cost_floor_pct": float(config.get("cost_floor_pct", config.get("threshold", 0.0)) or 0.0),
        "label_percentile": percentile,
        "label_episode_window_days": window_days,
        "label_threshold_mode": str(config.get("label_threshold_mode") or "fixed_threshold"),
    }


def _label_config_rank(label_config: dict[str, Any] | None) -> tuple[float, int]:
    payload = _label_config_payload(label_config)
    percentile = payload["label_percentile"]
    return (
        float(payload["cost_floor_pct"]),
        int(percentile) if percentile is not None else -1,
    )


def _dataset_build_kwargs_for_label_config(label_config: dict[str, Any] | None) -> dict[str, Any]:
    payload = _label_config_payload(label_config)
    kwargs: dict[str, Any] = {"min_total_spread_pct": float(payload["cost_floor_pct"])}
    if payload["label_threshold_mode"] == "rolling_pair_percentile":
        kwargs.update(
            {
                "label_cost_floor_pct": float(payload["cost_floor_pct"]),
                "label_percentile": int(payload["label_percentile"]) if payload["label_percentile"] is not None else 70,
                "label_episode_window_days": int(payload["label_episode_window_days"]) if payload["label_episode_window_days"] is not None else 5,
            }
        )
    return kwargs


def _apply_platt_scaling(logits: np.ndarray, scale: float, bias: float) -> np.ndarray:
    calibrated = 1.0 / (1.0 + np.exp(-(logits * scale + bias)))
    return np.clip(calibrated, 1e-6, 1.0 - 1e-6)


def _fit_platt_scaler(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if logits.size == 0 or len(np.unique(labels)) < 2:
        return 1.0, 0.0
    model = LogisticRegression(solver="lbfgs", random_state=42, max_iter=200)
    model.fit(logits.reshape(-1, 1), labels.astype(int))
    return float(model.coef_[0][0]), float(model.intercept_[0])


def _safe_average_precision(y_true: np.ndarray, probs: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    positives = int(np.sum(y_true >= 0.5))
    if positives == 0:
        return 0.0
    if positives == y_true.size:
        return 1.0
    try:
        return float(average_precision_score(y_true, probs))
    except ValueError:
        return 0.0


def _safe_roc_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    if len(np.unique(y_true)) < 2:
        return 0.0
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return 0.0


def _calibration_report(
    y_true: np.ndarray,
    probs: np.ndarray,
    bins: int = 10,
) -> dict[str, Any]:
    if y_true.size == 0 or probs.size == 0:
        return {"ece": 0.0, "brier_score": 0.0, "log_loss": 0.0, "bins": []}

    clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    calibration_bins: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bins):
        lower = bin_edges[index]
        upper = bin_edges[index + 1]
        if index == bins - 1:
            mask = (clipped >= lower) & (clipped <= upper)
        else:
            mask = (clipped >= lower) & (clipped < upper)
        if not np.any(mask):
            calibration_bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": 0,
                    "avg_confidence": 0.0,
                    "observed_rate": 0.0,
                    "gap": 0.0,
                }
            )
            continue
        avg_confidence = float(np.mean(clipped[mask]))
        observed_rate = float(np.mean(y_true[mask]))
        gap = abs(avg_confidence - observed_rate)
        ece += gap * (float(np.sum(mask)) / float(y_true.size))
        calibration_bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(np.sum(mask)),
                "avg_confidence": avg_confidence,
                "observed_rate": observed_rate,
                "gap": float(gap),
            }
        )

    return {
        "ece": float(ece),
        "brier_score": float(brier_score_loss(y_true, clipped)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "bins": calibration_bins,
    }


def _high_confidence_calibration(calibration_report: dict[str, Any]) -> dict[str, Any]:
    bins = [
        dict(item)
        for item in calibration_report.get("bins", [])
        if float(item.get("lower", 0.0)) >= 0.70
    ]
    populated = [item for item in bins if int(item.get("count", 0)) > 0]
    weighted_count = sum(int(item.get("count", 0)) for item in populated)
    if not populated:
        return {
            "status": "insufficient_samples",
            "count": 0,
            "weighted_gap": 0.0,
            "max_bin_gap": 0.0,
            "bins": bins,
        }
    weighted_gap = (
        sum(float(item.get("gap", 0.0)) * int(item.get("count", 0)) for item in populated)
        / max(weighted_count, 1)
    )
    max_bin_gap = max(float(item.get("gap", 0.0)) for item in populated)
    significant_bins = [
        item
        for item in populated
        if int(item.get("count", 0)) >= 25 and float(item.get("gap", 0.0)) > 0.15
    ]
    if weighted_count < 25:
        status = "insufficient_samples"
    elif weighted_gap <= 0.10 and not significant_bins:
        status = "ok"
    else:
        status = "review_required"
    return {
        "status": status,
        "count": int(weighted_count),
        "weighted_gap": float(weighted_gap),
        "max_bin_gap": float(max_bin_gap),
        "bins": bins,
    }


def _classification_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    if y_true.size == 0 or probs.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "average_precision": 0.0,
            "predicted_positive_rate": 0.0,
            "positive_rate": 0.0,
            "threshold": float(threshold),
            "roc_auc": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "brier_score": 0.0,
            "log_loss": 0.0,
            "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        }

    clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    recall = float(recall_score(y_true, preds, zero_division=0))
    specificity = float(tn / max(tn + fp, 1))
    false_positive_rate = float(fp / max(fp + tn, 1))
    false_negative_rate = float(fn / max(fn + tp, 1))
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": recall,
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "average_precision": _safe_average_precision(y_true, probs),
        "predicted_positive_rate": float(preds.mean()) if preds.size else 0.0,
        "positive_rate": float(y_true.mean()) if y_true.size else 0.0,
        "threshold": float(threshold),
        "specificity": specificity,
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "brier_score": float(brier_score_loss(y_true, clipped)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    metrics["roc_auc"] = _safe_roc_auc(y_true, probs)
    return metrics


def _eta_metrics(
    y_true_class: np.ndarray,
    y_true_eta: np.ndarray,
    y_pred_eta: np.ndarray,
) -> dict[str, float]:
    mask = y_true_class >= 0.5
    if not np.any(mask):
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "median_absolute_error": 0.0,
            "p90_absolute_error": 0.0,
            "positive_samples": 0,
            "horizon_buckets": [],
        }
    errors = y_pred_eta[mask] - y_true_eta[mask]
    abs_errors = np.abs(errors)
    horizon_buckets = _eta_horizon_buckets(y_true_eta[mask], abs_errors)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "median_absolute_error": float(np.median(abs_errors)),
        "p90_absolute_error": float(np.percentile(abs_errors, 90)),
        "positive_samples": int(mask.sum()),
        "horizon_buckets": horizon_buckets,
    }


def _eta_horizon_buckets(y_true_eta: np.ndarray, abs_errors: np.ndarray) -> list[dict[str, float | int | str]]:
    buckets = [
        ("0-15m", 0, 900),
        ("15-60m", 900, 3600),
        ("1-2h", 3600, 7200),
        ("2-4h", 7200, 14_400),
        ("4h+", 14_400, float("inf")),
    ]
    results: list[dict[str, float | int | str]] = []
    for label, lower, upper in buckets:
        mask = (y_true_eta >= lower) & (y_true_eta < upper)
        count = int(np.sum(mask))
        if count == 0:
            results.append({"bucket": label, "count": 0, "mae": 0.0, "median_absolute_error": 0.0})
            continue
        results.append(
            {
                "bucket": label,
                "count": count,
                "mae": float(np.mean(abs_errors[mask])),
                "median_absolute_error": float(np.median(abs_errors[mask])),
            }
        )
    return results


def _select_thresholds(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    candidates = [round(value, 2) for value in np.arange(0.30, 0.86, 0.05)]
    sweep: list[dict[str, float]] = []
    for threshold in candidates:
        metrics = _classification_metrics(y_true, probs, threshold)
        sweep.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "predicted_positive_rate": metrics["predicted_positive_rate"],
                "false_positive_rate": metrics["false_positive_rate"],
            }
        )

    execute_candidates = [
        item for item in sweep if item["predicted_positive_rate"] > 0.0 and item["precision"] >= 0.50
    ] or [item for item in sweep if item["predicted_positive_rate"] > 0.0] or sweep
    execute_choice = max(
        execute_candidates,
        key=lambda item: (
            item["f1"] + 0.20 * item["precision"] + 0.15 * item["balanced_accuracy"] - 0.10 * item["false_positive_rate"],
            item["precision"],
            -item["false_positive_rate"],
        ),
    )

    strong_candidates = [
        item
        for item in sweep
        if item["threshold"] >= execute_choice["threshold"] + 0.05 and item["precision"] >= max(0.65, execute_choice["precision"])
    ] or [
        item for item in sweep if item["threshold"] >= execute_choice["threshold"] + 0.05
    ] or [execute_choice]
    strong_choice = max(
        strong_candidates,
        key=lambda item: (
            item["precision"] + 0.20 * item["balanced_accuracy"] + 0.10 * item["f1"],
            -item["false_positive_rate"],
            item["threshold"],
        ),
    )

    execute_threshold = float(execute_choice["threshold"])
    strong_threshold = float(max(strong_choice["threshold"], execute_threshold + 0.05))
    strong_threshold = min(0.95, strong_threshold)

    execute_metrics = _classification_metrics(y_true, probs, execute_threshold)
    strong_metrics = _classification_metrics(y_true, probs, strong_threshold)

    return {
        "execute_threshold": execute_threshold,
        "strong_threshold": strong_threshold,
        "sweep": sweep,
        "execute": {
            "threshold": execute_threshold,
            "objective": "maximize f1 with recall and balanced-accuracy support under a precision floor",
            "metrics": execute_metrics,
        },
        "strong": {
            "threshold": strong_threshold,
            "objective": "favor higher precision and specificity for strong signals",
            "metrics": strong_metrics,
        },
    }


def _label_diversity(labels: np.ndarray) -> int:
    if labels.size == 0:
        return 0
    return int(len(np.unique(labels.astype(int))))


def _partition_validation_chronologically(
    timestamps: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if timestamps.size == 0:
        empty = np.array([], dtype=int)
        return empty, empty, {
            "mode": "chronological",
            "calibration_count": 0,
            "selection_count": 0,
            "calibration_start_ts": 0.0,
            "calibration_end_ts": 0.0,
            "selection_start_ts": 0.0,
            "selection_end_ts": 0.0,
            "calibration_class_diversity": 0,
            "selection_class_diversity": 0,
        }

    order = np.argsort(timestamps)
    if order.size == 1:
        single_idx = order.astype(int)
        only_ts = float(timestamps[single_idx[0]])
        diversity = _label_diversity(labels[single_idx])
        return single_idx, single_idx, {
            "mode": "chronological_fallback",
            "calibration_count": 1,
            "selection_count": 1,
            "calibration_start_ts": only_ts,
            "calibration_end_ts": only_ts,
            "selection_start_ts": only_ts,
            "selection_end_ts": only_ts,
            "calibration_class_diversity": diversity,
            "selection_class_diversity": diversity,
        }

    best_split = 1
    best_score: tuple[float, float, float] | None = None
    total = float(order.size)
    for split_point in range(1, order.size):
        cal_idx = order[:split_point]
        sel_idx = order[split_point:]
        score = (
            float(_label_diversity(labels[cal_idx]) + _label_diversity(labels[sel_idx])),
            -abs((split_point / total) - 0.5),
            -abs(split_point - (order.size / 2.0)),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_split = split_point

    cal_idx = order[:best_split].astype(int)
    sel_idx = order[best_split:].astype(int)
    if sel_idx.size == 0:
        sel_idx = cal_idx.copy()
    return cal_idx, sel_idx, {
        "mode": "chronological",
        "calibration_count": int(cal_idx.size),
        "selection_count": int(sel_idx.size),
        "calibration_start_ts": float(timestamps[cal_idx[0]]) if cal_idx.size else 0.0,
        "calibration_end_ts": float(timestamps[cal_idx[-1]]) if cal_idx.size else 0.0,
        "selection_start_ts": float(timestamps[sel_idx[0]]) if sel_idx.size else 0.0,
        "selection_end_ts": float(timestamps[sel_idx[-1]]) if sel_idx.size else 0.0,
        "calibration_class_diversity": _label_diversity(labels[cal_idx]),
        "selection_class_diversity": _label_diversity(labels[sel_idx]),
    }


def _bucket_classification_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    values: np.ndarray,
    threshold: float,
    bins: list[tuple[str, float, float]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for label, lower, upper in bins:
        mask = (values >= lower) & (values < upper)
        count = int(np.sum(mask))
        if count == 0:
            results.append({"bucket": label, "count": 0})
            continue
        metrics = _classification_metrics(y_true[mask], probs[mask], threshold)
        results.append(
            {
                "bucket": label,
                "count": count,
                "positive_rate": float(np.mean(y_true[mask])),
                "avg_probability": float(np.mean(probs[mask])),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
    return results


def _subgroup_metrics(
    evaluation: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    y_true = evaluation["y_class"]
    probs = evaluation["probs"]
    entry_values = evaluation["last_entries"]
    volatility_values = evaluation["entry_volatility"]
    return {
        "entry_spread_buckets": _bucket_classification_metrics(
            y_true,
            probs,
            entry_values,
            threshold,
            [
                ("lt_0_10", -float("inf"), 0.10),
                ("0_10_to_0_30", 0.10, 0.30),
                ("0_30_to_0_60", 0.30, 0.60),
                ("ge_0_60", 0.60, float("inf")),
            ],
        ),
        "volatility_buckets": _bucket_classification_metrics(
            y_true,
            probs,
            volatility_values,
            threshold,
            [
                ("lt_0_05", -float("inf"), 0.05),
                ("0_05_to_0_10", 0.05, 0.10),
                ("0_10_to_0_20", 0.10, 0.20),
                ("ge_0_20", 0.20, float("inf")),
            ],
        ),
    }


def _temporal_walk_forward_report(
    evaluation: dict[str, Any],
    threshold: float,
    windows: int = 4,
) -> dict[str, Any]:
    timestamps = evaluation["timestamps"]
    if timestamps.size == 0:
        return {"num_windows": 0, "windows": [], "summary": {}}

    order = np.argsort(timestamps)
    y_true = evaluation["y_class"][order]
    probs = evaluation["probs"][order]
    ordered_timestamps = timestamps[order]

    chunk_size = max(1, int(math.ceil(len(ordered_timestamps) / windows)))
    window_reports: list[dict[str, Any]] = []
    for index in range(windows):
        start = index * chunk_size
        end = min(len(ordered_timestamps), start + chunk_size)
        if start >= end:
            continue
        metrics = _classification_metrics(y_true[start:end], probs[start:end], threshold)
        window_reports.append(
            {
                "window_index": index + 1,
                "start_ts": float(ordered_timestamps[start]),
                "end_ts": float(ordered_timestamps[end - 1]),
                "count": int(end - start),
                "metrics": metrics,
            }
        )

    if not window_reports:
        return {"num_windows": 0, "windows": [], "summary": {}}

    ap_values = [window["metrics"]["average_precision"] for window in window_reports]
    recall_values = [window["metrics"]["recall"] for window in window_reports]
    return {
        "num_windows": len(window_reports),
        "windows": window_reports,
        "summary": {
            "min_average_precision": float(min(ap_values)),
            "avg_average_precision": float(np.mean(ap_values)),
            "min_recall": float(min(recall_values)),
            "avg_recall": float(np.mean(recall_values)),
        },
    }


def _evaluate_split(
    model: SpreadSequenceLSTM,
    bundle: DatasetBundle,
    device: torch.device,
    scale: float,
    bias: float,
    threshold: float,
) -> dict[str, Any]:
    loader = _make_loader(bundle, batch_size=256, shuffle=False)
    logits_list: list[np.ndarray] = []
    eta_list: list[np.ndarray] = []
    y_class_list: list[np.ndarray] = []
    y_eta_list: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y_class, batch_y_eta in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, eta_raw = model(batch_x)
            logits_list.append(logits.squeeze(1).cpu().numpy())
            eta_list.append(torch.expm1(eta_raw.squeeze(1)).clamp(min=0.0).cpu().numpy())
            y_class_list.append(batch_y_class.numpy())
            y_eta_list.append(batch_y_eta.numpy())

    logits_np = np.concatenate(logits_list) if logits_list else np.array([], dtype=float)
    probs_np = _apply_platt_scaling(logits_np, scale, bias) if logits_np.size else np.array([], dtype=float)
    eta_np = np.concatenate(eta_list) if eta_list else np.array([], dtype=float)
    y_class_np = np.concatenate(y_class_list) if y_class_list else np.array([], dtype=float)
    y_eta_np = np.concatenate(y_eta_list) if y_eta_list else np.array([], dtype=float)
    timestamps_np = np.asarray(bundle.timestamps, dtype=float)
    last_entries_np = np.asarray(bundle.last_entries, dtype=float)
    entry_volatility_np = bundle.X[:, :, 0].detach().cpu().std(dim=1).numpy() if bundle.summary["num_samples"] else np.array([], dtype=float)

    return {
        "classification": _classification_metrics(y_class_np, probs_np, threshold),
        "eta": _eta_metrics(y_class_np, y_eta_np, eta_np),
        "logits": logits_np,
        "probs": probs_np,
        "y_class": y_class_np,
        "y_eta": y_eta_np,
        "eta_pred": eta_np,
        "timestamps": timestamps_np,
        "last_entries": last_entries_np,
        "entry_volatility": entry_volatility_np,
        "pair_ids": list(bundle.pair_ids),
    }


def _baseline_metrics(
    train_bundle: DatasetBundle,
    target_bundle: DatasetBundle,
) -> dict[str, dict[str, float]]:
    y_true = target_bundle.y_class.numpy()

    always_negative_probs = np.zeros_like(y_true, dtype=float)
    always_negative = _classification_metrics(y_true, always_negative_probs, threshold=0.5)

    percentile_threshold = float(np.quantile(np.array(train_bundle.last_entries, dtype=float), 0.75))
    percentile_probs = np.array(
        [1.0 if value >= percentile_threshold else 0.0 for value in target_bundle.last_entries],
        dtype=float,
    )
    percentile_rule = _classification_metrics(y_true, percentile_probs, threshold=0.5)
    percentile_rule["entry_threshold"] = percentile_threshold

    return {
        "always_negative": always_negative,
        "percentile_rule": percentile_rule,
    }


def _build_split_summary(splits: dict[str, DatasetBundle]) -> dict[str, Any]:
    train_pairs = set(splits["train"].pair_ids)
    val_pairs = set(splits["val"].pair_ids)
    test_pairs = set(splits["test"].pair_ids)
    train_blocks = set(splits["train"].block_ids)
    val_blocks = set(splits["val"].block_ids)
    test_blocks = set(splits["test"].block_ids)
    train_timestamps = splits["train"].timestamps
    val_timestamps = splits["val"].timestamps
    test_timestamps = splits["test"].timestamps
    shared_split_summary = splits["train"].summary.get("split_summary", {})
    pair_overlap_breakdown = {
        "train_val": len(train_pairs & val_pairs),
        "train_test": len(train_pairs & test_pairs),
        "val_test": len(val_pairs & test_pairs),
    }
    unique_overlap_pairs = (train_pairs & val_pairs) | (train_pairs & test_pairs) | (val_pairs & test_pairs)
    pair_overlap = len(unique_overlap_pairs)
    pair_overlap_pairwise_sum = sum(pair_overlap_breakdown.values())
    total_unique_pairs = len(train_pairs | val_pairs | test_pairs)
    return {
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "train_blocks": len(train_blocks),
        "val_blocks": len(val_blocks),
        "test_blocks": len(test_blocks),
        "pair_overlap": pair_overlap,
        "pair_overlap_pairwise_sum": pair_overlap_pairwise_sum,
        "pair_overlap_breakdown": pair_overlap_breakdown,
        "pair_overlap_rate": float(pair_overlap / total_unique_pairs) if total_unique_pairs else 0.0,
        "train_samples": splits["train"].summary["num_samples"],
        "val_samples": splits["val"].summary["num_samples"],
        "test_samples": splits["test"].summary["num_samples"],
        "train_start_ts": min(train_timestamps) if train_timestamps else 0.0,
        "train_end_ts": max(train_timestamps) if train_timestamps else 0.0,
        "val_start_ts": min(val_timestamps) if val_timestamps else 0.0,
        "val_end_ts": max(val_timestamps) if val_timestamps else 0.0,
        "test_start_ts": min(test_timestamps) if test_timestamps else 0.0,
        "test_end_ts": max(test_timestamps) if test_timestamps else 0.0,
        "train_positive_pairs": int(shared_split_summary.get("train_positive_pairs", 0)),
        "val_positive_pairs": int(shared_split_summary.get("val_positive_pairs", 0)),
        "test_positive_pairs": int(shared_split_summary.get("test_positive_pairs", 0)),
        "train_negative_pairs": int(shared_split_summary.get("train_negative_pairs", 0)),
        "val_negative_pairs": int(shared_split_summary.get("val_negative_pairs", 0)),
        "test_negative_pairs": int(shared_split_summary.get("test_negative_pairs", 0)),
        "embargo_samples": int(shared_split_summary.get("embargo_samples", 0)),
        "embargo_time_sec": int(shared_split_summary.get("embargo_time_sec", 0)),
        "split_mode": str(shared_split_summary.get("split_mode", "sample_chronological")),
        "split_mode_fallback_reason": str(shared_split_summary.get("split_mode_fallback_reason", "")),
        "train_session_ids": list(shared_split_summary.get("train_session_ids", [])),
        "val_session_ids": list(shared_split_summary.get("val_session_ids", [])),
        "test_session_ids": list(shared_split_summary.get("test_session_ids", [])),
        "train_end_label_ts": float(shared_split_summary.get("train_end_label_ts", 0.0)),
        "val_end_label_ts": float(shared_split_summary.get("val_end_label_ts", 0.0)),
        "purged_temporal_separation_ok": bool(shared_split_summary.get("purged_temporal_separation_ok", False)),
        "min_cross_split_gap_sec": float(shared_split_summary.get("min_cross_split_gap_sec", 0.0)),
        "train_positive_rate": float(splits["train"].summary.get("positive_rate", 0.0)),
        "val_positive_rate": float(splits["val"].summary.get("positive_rate", 0.0)),
        "test_positive_rate": float(splits["test"].summary.get("positive_rate", 0.0)),
        "global_temporal_order": bool(
            (max(train_timestamps) if train_timestamps else 0.0) <= (min(val_timestamps) if val_timestamps else float("inf"))
            and (max(val_timestamps) if val_timestamps else 0.0) <= (min(test_timestamps) if test_timestamps else float("inf"))
        ),
    }


def _split_positive_rates(splits: dict[str, DatasetBundle]) -> dict[str, float]:
    return {
        split_name: float(split_bundle.summary.get("positive_rate", 0.0))
        for split_name, split_bundle in splits.items()
    }


def _split_positive_counts(splits: dict[str, DatasetBundle]) -> dict[str, int]:
    return {
        split_name: int(split_bundle.summary.get("num_positive_samples", 0))
        for split_name, split_bundle in splits.items()
    }


def _stability_summary(split_positive_rates: dict[str, float]) -> dict[str, Any]:
    populated = [float(value) for value in split_positive_rates.values() if float(value) > 0.0]
    if not populated:
        max_delta = 0.0
        ratio = 0.0
    else:
        max_delta = max(populated) - min(populated)
        ratio = min(populated) / max(populated) if max(populated) > 0.0 else 0.0
    return {
        "split_positive_rates": {key: float(value) for key, value in split_positive_rates.items()},
        "max_positive_rate_delta": float(max_delta),
        "min_to_max_positive_rate_ratio": float(ratio),
        "stability_ok_primary": bool(max_delta <= 0.05 and ratio >= 0.5),
        "stability_ok_relaxed": bool(max_delta <= 0.08 and ratio >= 0.5),
    }


def _preflight_entry(
    *,
    threshold: float,
    label_config: dict[str, Any] | None,
    bundle: DatasetBundle | None,
    splits: dict[str, DatasetBundle] | None,
    error: str | None = None,
    min_train_positive_samples: int,
    min_val_positive_samples: int,
    min_test_positive_samples: int,
) -> dict[str, Any]:
    label_payload = _label_config_payload(label_config)

    def _failure_reasons(
        *,
        bundle_obj: DatasetBundle,
        guardrail_ok_value: bool,
        purging_ok_value: bool,
        stability_obj: dict[str, Any],
        positive_counts_obj: dict[str, int],
    ) -> list[str]:
        reasons: list[str] = []
        if int(bundle_obj.summary.get("num_samples", 0)) <= 0:
            reasons.append("no_samples")
        if int(bundle_obj.summary.get("num_positive_samples", 0)) <= 0:
            reasons.append("no_positive_samples")
        if int(bundle_obj.summary.get("skipped_windows_right_censored", 0)) > 0:
            reasons.append("right_censoring_present")
        if int(bundle_obj.summary.get("block_diagnostics", {}).get("irregular_block_count", 0)) > 0:
            reasons.append("intra_block_irregularity")
        if int(
            bundle_obj.summary.get("block_diagnostics", {})
            .get("feature_window_feasibility", {})
            .get("eligible_blocks_for_sequence_length", 0)
        ) <= 0:
            reasons.append("insufficient_eligible_blocks")
        if (
            int(bundle_obj.summary.get("num_samples", 0)) <= 0
            and int(
                bundle_obj.summary.get("block_diagnostics", {})
                .get("feature_window_feasibility", {})
                .get("eligible_sessions_with_any_eligible_block", 0)
            ) > 1
        ):
            reasons.append("fragmentation_across_short_sessions")
        if not guardrail_ok_value:
            if int(positive_counts_obj.get("train", 0)) < int(min_train_positive_samples):
                reasons.append("insufficient_train_positives")
            if int(positive_counts_obj.get("val", 0)) < int(min_val_positive_samples):
                reasons.append("insufficient_val_positives")
            if int(positive_counts_obj.get("test", 0)) < int(min_test_positive_samples):
                reasons.append("insufficient_test_positives")
        if not purging_ok_value:
            reasons.append("purging_failed")
        if not bool(stability_obj.get("stability_ok_primary", False)):
            reasons.append("temporal_instability")
        if bool(
            bundle_obj.summary.get("cross_session_merge_enabled", False)
        ) and int(bundle_obj.summary.get("cross_session_merges_applied", 0)) <= 0:
            reasons.append("cross_session_merge_no_effect")
        return list(dict.fromkeys(reasons))

    if bundle is None or splits is None:
        return {
            "threshold": float(threshold),
            **label_payload,
            "build_ok": False,
            "guardrail_ok": False,
            "purging_ok": False,
            "stability_ok": False,
            "stability_mode": "primary",
            "qualifies_for_training": False,
            "qualifies_for_training_relaxed": False,
            "failure_reasons": ["dataset_build_failed"],
            "error": str(error or "dataset_build_failed"),
        }

    split_summary = _build_split_summary(splits)
    positive_counts = _split_positive_counts(splits)
    positive_rates = _split_positive_rates(splits)
    stability = _stability_summary(positive_rates)
    guardrail_ok = bool(
        positive_counts["train"] >= int(min_train_positive_samples)
        and positive_counts["val"] >= int(min_val_positive_samples)
        and positive_counts["test"] >= int(min_test_positive_samples)
    )
    purging_ok = bool(split_summary["purged_temporal_separation_ok"])
    qualifies_primary = bool(guardrail_ok and purging_ok and stability["stability_ok_primary"])
    qualifies_relaxed = bool(guardrail_ok and purging_ok and stability["stability_ok_relaxed"])
    return {
        "threshold": float(threshold),
        **label_payload,
        "build_ok": True,
        "guardrail_ok": guardrail_ok,
        "purging_ok": purging_ok,
        "stability_ok": bool(stability["stability_ok_primary"]),
        "stability_ok_primary": bool(stability["stability_ok_primary"]),
        "stability_ok_relaxed": bool(stability["stability_ok_relaxed"]),
        "stability_mode": "primary" if stability["stability_ok_primary"] else ("relaxed" if stability["stability_ok_relaxed"] else "failed"),
        "qualifies_for_training": qualifies_primary,
        "qualifies_for_training_relaxed": qualifies_relaxed,
        "num_samples": int(bundle.summary.get("num_samples", 0)),
        "num_positive_samples": int(bundle.summary.get("num_positive_samples", 0)),
        "num_negative_samples": int(bundle.summary.get("num_negative_samples", 0)),
        "positive_rate": float(bundle.summary.get("num_positive_samples", 0) / max(bundle.summary.get("num_samples", 1), 1)),
        "split_positive_counts": positive_counts,
        "split_positive_rates": stability["split_positive_rates"],
        "max_positive_rate_delta": float(stability["max_positive_rate_delta"]),
        "min_to_max_positive_rate_ratio": float(stability["min_to_max_positive_rate_ratio"]),
        "split_summary": split_summary,
        "label_audit": dict(bundle.summary.get("label_audit", {})),
        "label_thresholds": dict(bundle.summary.get("label_thresholds", {})),
        "block_diagnostics": dict(bundle.summary.get("block_diagnostics", {})),
        "episode_diagnostics": dict(bundle.summary.get("episode_diagnostics", {})),
        "cross_session_merge_enabled": bool(bundle.summary.get("cross_session_merge_enabled", False)),
        "cross_session_merges_applied": int(bundle.summary.get("cross_session_merges_applied", 0)),
        "cross_session_merge_diagnostics": dict(bundle.summary.get("cross_session_merge_diagnostics", {})),
        "failure_reasons": _failure_reasons(
            bundle_obj=bundle,
            guardrail_ok_value=guardrail_ok,
            purging_ok_value=purging_ok,
            stability_obj=stability,
            positive_counts_obj=positive_counts,
        ),
        "error": str(error or ""),
    }


def run_threshold_preflight(
    *,
    state_file: Path | None = None,
    sequence_length: int = 15,
    prediction_horizon_sec: int = 14_400,
    thresholds: list[float] | None = None,
    label_percentiles: list[int] | None = None,
    label_episode_window_days: int = 5,
    selected_session_ids: list[int] | None = None,
    selected_block_ids: list[int] | None = None,
    allow_cross_session_merge: bool = False,
    max_session_gap_sec: float | None = None,
    regime_shift_score_threshold: float | None = 3.0,
    min_train_positive_samples: int = 500,
    min_val_positive_samples: int = 50,
    min_test_positive_samples: int = 50,
    output_path: Path | None = None,
    _preloaded_blocks: tuple[list, float, dict] | None = None,
    _precomputed_pair_segments: tuple[list[dict[str, Any]], dict[str, Any]] | None = None,
    _precomputed_segment_features: dict[int, list[list[float]]] | None = None,
    _scaffold_cache: dict[tuple[int, int, bool], Any] | None = None,
    window_stride: int = 1,
) -> dict[str, Any]:
    default_state = Path(__file__).resolve().parent.parent.parent / "out" / "config" / "tracker_history.sqlite"
    state_path = Path(state_file) if state_file is not None else default_state
    threshold_values = _normalize_threshold_values(thresholds)
    adaptive_mode = _adaptive_label_requested(thresholds=thresholds, label_percentiles=label_percentiles)
    effective_label_percentiles = _normalize_label_percentiles(label_percentiles) if adaptive_mode else []
    label_configs = _iter_label_configs(
        thresholds=threshold_values,
        label_percentiles=label_percentiles,
        label_episode_window_days=label_episode_window_days,
    )
    preflight: dict[str, Any] = {
        "state_path": str(state_path),
        "sequence_length": int(sequence_length),
        "prediction_horizon_sec": int(prediction_horizon_sec),
        "thresholds": {},
        "cost_floors": [float(value) for value in threshold_values],
        "label_percentiles": [int(value) for value in effective_label_percentiles],
        "label_episode_window_days": int(max(label_episode_window_days, 1)),
        "label_threshold_mode": "rolling_pair_percentile" if adaptive_mode else "fixed_threshold",
        "block_diagnostics": {},
        "cross_session_merge_enabled": bool(allow_cross_session_merge),
        "max_session_gap_sec": None if max_session_gap_sec is None else float(max_session_gap_sec),
        "regime_shift_score_threshold": None if regime_shift_score_threshold is None else float(regime_shift_score_threshold),
        "selected_threshold": None,
        "selected_label_config": None,
        "selection_mode": "none",
        "qualifies_for_training": False,
    }
    primary_candidates: list[dict[str, Any]] = []
    relaxed_candidates: list[dict[str, Any]] = []
    if _preloaded_blocks is None:
        _preloaded_blocks = _load_blocks_from_sqlite(
            state_path,
            selected_block_ids=selected_block_ids,
            selected_session_ids=selected_session_ids,
            selected_only=True,
            closed_only=True,
        )
    # Use pre-computed segments if available, otherwise build them
    if _precomputed_pair_segments is not None:
        _cached_segments = _precomputed_pair_segments
    else:
        _effective_max_session_gap_sec = max_session_gap_sec
        if bool(allow_cross_session_merge) and _effective_max_session_gap_sec is None:
            _effective_max_session_gap_sec = _load_tracker_gap_threshold_sec(state_path)
        _cached_segments = _build_pair_segments(
            _preloaded_blocks[0],
            allow_cross_session_merge=bool(allow_cross_session_merge),
            max_session_gap_sec=_effective_max_session_gap_sec,
            regime_shift_score_threshold=regime_shift_score_threshold,
        )
    # Feature cache: features depend only on segment records/episodes,
    # not on threshold/prediction_horizon. Compute once, reuse across configs.
    if _precomputed_segment_features is None:
        _precomputed_segment_features = {}
    for label_config in label_configs:
        threshold = float(label_config["threshold"])
        try:
            bundle = build_dataset_bundle(
                state_path=state_path,
                sequence_length=sequence_length,
                prediction_horizon_sec=prediction_horizon_sec,
                selected_session_ids=selected_session_ids,
                selected_block_ids=selected_block_ids,
                allow_cross_session_merge=allow_cross_session_merge,
                max_session_gap_sec=max_session_gap_sec,
                regime_shift_score_threshold=regime_shift_score_threshold,
                _preloaded_blocks=_preloaded_blocks,
                _precomputed_pair_segments=_cached_segments,
                _precomputed_segment_features=_precomputed_segment_features,
                _scaffold_cache=_scaffold_cache,
                window_stride=window_stride,
                **_dataset_build_kwargs_for_label_config(label_config),
            )
            splits = build_group_splits(bundle, prediction_horizon_sec=prediction_horizon_sec)
            entry = _preflight_entry(
                threshold=threshold,
                label_config=label_config,
                bundle=bundle,
                splits=splits,
                min_train_positive_samples=min_train_positive_samples,
                min_val_positive_samples=min_val_positive_samples,
                min_test_positive_samples=min_test_positive_samples,
            )
        except Exception as exc:
            entry = _preflight_entry(
                threshold=threshold,
                label_config=label_config,
                bundle=None,
                splits=None,
                error=str(exc),
                min_train_positive_samples=min_train_positive_samples,
                min_val_positive_samples=min_val_positive_samples,
                min_test_positive_samples=min_test_positive_samples,
            )
        key = str(label_config["key"])
        preflight["thresholds"][key] = entry
        if (not preflight["block_diagnostics"]) and entry.get("block_diagnostics"):
            preflight["block_diagnostics"] = dict(entry["block_diagnostics"])
        if entry.get("qualifies_for_training"):
            primary_candidates.append(_label_config_payload(label_config))
        if entry.get("qualifies_for_training_relaxed"):
            relaxed_candidates.append(_label_config_payload(label_config))

    if primary_candidates:
        selected_label_config = max(primary_candidates, key=_label_config_rank)
        preflight["selected_threshold"] = float(selected_label_config["cost_floor_pct"])
        preflight["selected_label_config"] = dict(selected_label_config)
        preflight["selection_mode"] = "primary"
        preflight["qualifies_for_training"] = True
    elif relaxed_candidates:
        selected_label_config = max(relaxed_candidates, key=_label_config_rank)
        preflight["selected_threshold"] = float(selected_label_config["cost_floor_pct"])
        preflight["selected_label_config"] = dict(selected_label_config)
        preflight["selection_mode"] = "relaxed"
        preflight["qualifies_for_training"] = True

    if output_path is not None:
        output_target = Path(output_path)
        _write_json(output_target, preflight)
        preflight["output_path"] = str(output_target)
    return preflight


def _archive_existing_artifacts(artifact_root: Path) -> str:
    legacy_candidates = [
        artifact_root / f"{ARTIFACT_BASENAME}.pth",
        artifact_root / f"{ARTIFACT_BASENAME}.meta.json",
        artifact_root / f"{ARTIFACT_BASENAME}.report.json",
        artifact_root / f"{ARTIFACT_BASENAME}.audit.md",
    ]
    existing = [path for path in legacy_candidates if path.exists()]
    if not existing:
        return ""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    legacy_dir = artifact_root / "legacy_artifacts" / timestamp
    legacy_dir.mkdir(parents=True, exist_ok=True)
    for source_path in existing:
        shutil.copy2(source_path, legacy_dir / source_path.name)
    return str(legacy_dir)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _build_dataset_fingerprint(
    *,
    state_path: Path,
    bundle: DatasetBundle,
    min_total_spread_pct: float,
    label_cost_floor_pct: float | None,
    label_percentile: int | None,
    label_episode_window_days: int | None,
    label_threshold_mode: str | None,
    selected_session_ids: list[int] | None,
    selected_block_ids: list[int] | None,
    allow_cross_session_merge: bool,
    max_session_gap_sec: float | None,
    regime_shift_score_threshold: float | None,
) -> str:
    digest = hashlib.sha1()
    header = {
        "state_path": str(state_path.resolve()),
        "num_samples": int(bundle.summary.get("num_samples", 0)),
        "num_positive_samples": int(bundle.summary.get("num_positive_samples", 0)),
        "num_pairs": int(bundle.summary.get("num_pairs", 0)),
        "blocks_used": int(bundle.summary.get("blocks_used", 0)),
        "sessions_used": int(bundle.summary.get("sessions_used", 0)),
        "min_total_spread_pct": float(min_total_spread_pct),
        "label_cost_floor_pct": float(bundle.summary.get("label_cost_floor_pct", label_cost_floor_pct if label_cost_floor_pct is not None else min_total_spread_pct)),
        "label_percentile": (
            None
            if bundle.summary.get("label_percentile", label_percentile) is None
            else float(bundle.summary.get("label_percentile", label_percentile))
        ),
        "label_episode_window_days": (
            None
            if bundle.summary.get("label_episode_window_days", label_episode_window_days) is None
            else int(bundle.summary.get("label_episode_window_days", label_episode_window_days))
        ),
        "label_threshold_mode": str(bundle.summary.get("label_threshold_mode", label_threshold_mode or "fixed_threshold")),
        "selected_session_ids": [int(value) for value in selected_session_ids or []],
        "selected_block_ids": [int(value) for value in selected_block_ids or []],
        "allow_cross_session_merge": bool(allow_cross_session_merge),
        "max_session_gap_sec": None if max_session_gap_sec is None else float(max_session_gap_sec),
        "regime_shift_score_threshold": None if regime_shift_score_threshold is None else float(regime_shift_score_threshold),
        "feature_names": list(bundle.feature_names),
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for pair_id, session_id, block_id, ts_value, label_end_ts, y_class_value, y_eta_value, label_threshold_value in zip(
        bundle.pair_ids,
        bundle.session_ids,
        bundle.block_ids,
        bundle.timestamps,
        bundle.label_end_timestamps,
        bundle.y_class.tolist(),
        bundle.y_eta.tolist(),
        bundle.label_thresholds,
    ):
        digest.update(
            (
                f"{pair_id}|{int(session_id)}|{int(block_id)}|"
                f"{float(ts_value):.6f}|{float(label_end_ts):.6f}|"
                f"{float(y_class_value):.6f}|{float(y_eta_value):.6f}|"
                f"{float(label_threshold_value):.6f}"
            ).encode()
        )
    return digest.hexdigest()


def certify_data_for_training(
    *,
    state_file: Path | None = None,
    artifact_dir: Path | None = None,
    selected_session_ids: list[int] | None = None,
    selected_block_ids: list[int] | None = None,
    thresholds: list[float] | None = None,
    label_percentiles: list[int] | None = None,
    sequence_length: int = 15,
    prediction_horizon_sec: int = 14_400,
    allow_cross_session_merge: bool = False,
    max_session_gap_sec: float | None = None,
    regime_shift_score_threshold: float | None = 3.0,
    certification_mode: str = "full",
    max_certification_duration_sec: int = 300,
    label_episode_window_days: int = 5,
    allow_legacy_sessions: bool = False,
    runtime_audit_dir: Path | None = None,
    run_reconnection_stress: bool = False,
    _preloaded_blocks: tuple[list, float, dict] | None = None,
    _precomputed_segments_by_merge: dict[bool, tuple[list[dict[str, Any]], dict[str, Any]]] | None = None,
    _precomputed_features_by_merge: dict[bool, dict[int, list[list[float]]]] | None = None,
    _scaffold_cache: dict[tuple[int, int, bool], Any] | None = None,
    window_stride: int = 1,
) -> dict[str, Any]:
    default_state = Path(__file__).resolve().parent.parent.parent / "out" / "config" / "tracker_history.sqlite"
    state_path = Path(state_file) if state_file is not None else default_state
    artifact_root = Path(artifact_dir) if artifact_dir is not None else get_default_artifact_dir()
    artifact_root.mkdir(parents=True, exist_ok=True)
    return _certify_data_for_training_impl(
        state_file=state_path,
        artifact_dir=artifact_root,
        selected_session_ids=selected_session_ids,
        selected_block_ids=selected_block_ids,
        thresholds=thresholds,
        label_percentiles=label_percentiles,
        sequence_length=sequence_length,
        prediction_horizon_sec=prediction_horizon_sec,
        allow_cross_session_merge=allow_cross_session_merge,
        max_session_gap_sec=max_session_gap_sec,
        regime_shift_score_threshold=regime_shift_score_threshold,
        certification_mode=certification_mode,
        max_certification_duration_sec=max_certification_duration_sec,
        label_episode_window_days=label_episode_window_days,
        allow_legacy_sessions=allow_legacy_sessions,
        runtime_audit_dir=runtime_audit_dir,
        run_reconnection_stress=run_reconnection_stress,
        preflight_fn=run_threshold_preflight,
        dataset_fingerprint_fn=_build_dataset_fingerprint,
        _preloaded_blocks=_preloaded_blocks,
        _precomputed_segments_by_merge=_precomputed_segments_by_merge,
        _precomputed_features_by_merge=_precomputed_features_by_merge,
        _scaffold_cache=_scaffold_cache,
        window_stride=window_stride,
    )


def _build_psi_reference(
    X: torch.Tensor,
    feature_names: list[str],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    if X.numel() == 0:
        return {"feature_bins": {}, "min_samples_required": 200}
    flattened = X.detach().cpu().numpy().reshape(-1, X.shape[-1])
    feature_bins: dict[str, Any] = {}
    for index, feature_name in enumerate(feature_names):
        column = flattened[:, index]
        quantiles = np.quantile(column, np.linspace(0.0, 1.0, bins + 1))
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        counts, _ = np.histogram(column, bins=quantiles)
        expected = (counts / max(int(counts.sum()), 1)).tolist()
        edges = [
            None if not math.isfinite(float(value)) else float(value)
            for value in quantiles.tolist()
        ]
        feature_bins[feature_name] = {
            "bin_edges": edges,
            "expected_proportions": [float(value) for value in expected],
        }
    return {
        "feature_bins": feature_bins,
        "min_samples_required": 200,
    }
def write_audit_report(report: dict[str, Any], audit_path: Path) -> Path:
    dataset_summary = report["dataset_summary"]
    split_summary = report["split_summary"]
    metrics = report["metrics"]["test"]
    baselines = report["baselines"]
    calibration = report["calibration"]["test"]
    val_threshold_calibration = report["calibration"].get("val_threshold", {})
    high_confidence_calibration = calibration.get("high_confidence", {})
    temporal_summary = report["temporal_walk_forward"]["test"]["summary"]
    validation_partition = report["validation_partition"]
    feature_monitoring = report["feature_monitoring"]
    label_audit = report.get("label_audit", {})
    checks = {
        "dataset_non_degenerate": bool(dataset_summary["num_samples"] > 0 and dataset_summary["feature_abs_sum"] > 0.0),
        "windows_respect_session_boundaries": int(dataset_summary.get("num_cross_session_windows", 0)) == 0,
        "global_temporal_order": bool(split_summary["global_temporal_order"]),
        "purged_temporal_separation_ok": bool(split_summary.get("purged_temporal_separation_ok", False)),
        "beats_negative_baseline_recall": bool(metrics["recall"] > baselines["always_negative"]["recall"]),
        "beats_negative_baseline_ap": bool(
            metrics["average_precision"] >= baselines["always_negative"]["average_precision"]
        ),
        "nonzero_positive_predictions": bool(metrics["predicted_positive_rate"] > 0.0),
        "calibration_within_guardrail": bool(calibration["ece"] <= 0.12),
        "high_confidence_calibration_ok": bool(
            high_confidence_calibration.get("status") in {"ok", "insufficient_samples"}
        ),
        "temporal_ap_above_baseline": bool(
            temporal_summary.get("min_average_precision", 0.0) >= baselines["always_negative"]["average_precision"]
        ),
        "strong_precision_guardrail": bool(report["strong_signal_metrics"]["test"]["precision"] >= 0.80),
    }

    findings: list[str] = []
    if not checks["dataset_non_degenerate"]:
        findings.append("- Crítico: o dataset continua degenerado ou vazio.")
    if not checks["windows_respect_session_boundaries"]:
        findings.append("- Crítico: existem janelas cruzando boundaries de sessão/merge, violando a continuidade do treino.")
    if not checks["global_temporal_order"]:
        findings.append("- Crítico: treino, validação e teste ainda se sobrepõem no tempo global.")
    if not checks["purged_temporal_separation_ok"]:
        findings.append("- Crítico: o split temporal não respeitou a separação purgada pelo horizonte do label.")
    if not checks["beats_negative_baseline_recall"]:
        findings.append("- Alto: o modelo não superou o baseline sempre-negativo em recall.")
    if not checks["beats_negative_baseline_ap"]:
        findings.append("- Alto: o modelo não superou o baseline sempre-negativo em average precision.")
    if not checks["nonzero_positive_predictions"]:
        findings.append("- Alto: o modelo colapsou para zero previsões positivas no threshold operacional.")
    if not checks["temporal_ap_above_baseline"]:
        findings.append("- Alto: o bloco temporal mais fraco caiu abaixo do baseline sempre-negativo em PR-AUC.")
    if not checks["calibration_within_guardrail"]:
        findings.append("- Médio: a calibração ficou acima do guardrail de ECE e exige threshold mais conservador.")
    if not checks["high_confidence_calibration_ok"]:
        findings.append("- Médio: os bins de alta confiança ficaram descalibrados e exigem revisão manual antes da promoção.")
    if not checks["strong_precision_guardrail"]:
        findings.append("- Médio: o threshold de STRONG_EXECUTE não atingiu a precisão mínima esperada para sinal forte.")
    if not findings:
        findings.append("- Médio: os gates offline passaram, mas drift temporal e qualidade do histórico ainda precisam de monitoramento contínuo.")

    lines = [
        "# ArbML LSTM Audit",
        "",
        "## Findings",
        *findings,
        "",
        "## Dataset",
        f"- Samples: {dataset_summary['num_samples']}",
        f"- Positive samples: {dataset_summary['num_positive_samples']}",
        f"- Negative samples: {dataset_summary['num_negative_samples']}",
        f"- Pair count: {dataset_summary['num_pairs']}",
        f"- Session count: {dataset_summary.get('num_sessions', 0)}",
        f"- Block count: {dataset_summary.get('num_blocks', 0)}",
        f"- Blocks used: {dataset_summary.get('blocks_used', 0)}",
        f"- Skipped short blocks: {dataset_summary.get('skipped_blocks_too_short', 0)}",
        f"- Cross-block windows: {dataset_summary.get('num_cross_block_windows', 0)}",
        f"- Cross-session windows: {dataset_summary.get('num_cross_session_windows', 0)}",
        f"- Feature abs sum: {dataset_summary['feature_abs_sum']:.4f}",
        "",
        "## Split Integrity",
        f"- Train pairs: {split_summary['train_pairs']}",
        f"- Validation pairs: {split_summary['val_pairs']}",
        f"- Test pairs: {split_summary['test_pairs']}",
        f"- Train blocks: {split_summary.get('train_blocks', 0)}",
        f"- Validation blocks: {split_summary.get('val_blocks', 0)}",
        f"- Test blocks: {split_summary.get('test_blocks', 0)}",
        f"- Unique overlapping pairs: {split_summary['pair_overlap']}",
        f"- Pairwise overlap sum: {split_summary['pair_overlap_pairwise_sum']}",
        f"- Unique pair overlap rate: {split_summary['pair_overlap_rate']:.4f}",
        f"- Embargo samples: {split_summary['embargo_samples']}",
        f"- Embargo time sec: {split_summary.get('embargo_time_sec', 0)}",
        f"- Global temporal order: {split_summary['global_temporal_order']}",
        f"- Purged temporal separation ok: {split_summary.get('purged_temporal_separation_ok', False)}",
        f"- Min cross-split gap sec: {split_summary.get('min_cross_split_gap_sec', 0.0):.1f}",
        f"- Split mode fallback reason: {split_summary.get('split_mode_fallback_reason', '')}",
        f"- Train time range: {split_summary['train_start_ts']:.0f} -> {split_summary['train_end_ts']:.0f}",
        f"- Validation time range: {split_summary['val_start_ts']:.0f} -> {split_summary['val_end_ts']:.0f}",
        f"- Test time range: {split_summary['test_start_ts']:.0f} -> {split_summary['test_end_ts']:.0f}",
        "",
        "## Test Metrics",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        f"- ROC-AUC: {metrics['roc_auc']:.4f}",
        f"- PR-AUC: {metrics['average_precision']:.4f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        f"- Specificity: {metrics['specificity']:.4f}",
        f"- False positive rate: {metrics['false_positive_rate']:.4f}",
        f"- False negative rate: {metrics['false_negative_rate']:.4f}",
        f"- ETA MAE (s): {report['eta_metrics']['test']['mae']:.2f}",
        f"- ETA RMSE (s): {report['eta_metrics']['test']['rmse']:.2f}",
        f"- ETA median AE (s): {report['eta_metrics']['test']['median_absolute_error']:.2f}",
        f"- ETA p90 AE (s): {report['eta_metrics']['test']['p90_absolute_error']:.2f}",
        "",
        "## Baselines",
        f"- Always-negative recall: {baselines['always_negative']['recall']:.4f}",
        f"- Always-negative PR-AUC: {baselines['always_negative']['average_precision']:.4f}",
        f"- Percentile-rule recall: {baselines['percentile_rule']['recall']:.4f}",
        f"- Percentile-rule PR-AUC: {baselines['percentile_rule']['average_precision']:.4f}",
        "",
        "## Thresholds",
        f"- Execute threshold: {report['thresholds']['execute_threshold']:.2f}",
        f"- Strong threshold: {report['thresholds']['strong_threshold']:.2f}",
        f"- Validation precision @ execute: {report['threshold_selection']['execute']['metrics']['precision']:.4f}",
        f"- Validation recall @ execute: {report['threshold_selection']['execute']['metrics']['recall']:.4f}",
        f"- Validation precision @ strong: {report['threshold_selection']['strong']['metrics']['precision']:.4f}",
        f"- Test precision @ strong: {report['strong_signal_metrics']['test']['precision']:.4f}",
        f"- Test recall @ strong: {report['strong_signal_metrics']['test']['recall']:.4f}",
        "",
        "## Calibration",
        f"- Validation ECE: {val_threshold_calibration.get('ece', 0.0):.4f}",
        f"- Test ECE: {report['calibration']['test']['ece']:.4f}",
        f"- Test Brier score: {calibration['brier_score']:.4f}",
        f"- Test log loss: {calibration['log_loss']:.4f}",
        f"- High-confidence status: {high_confidence_calibration.get('status', 'unavailable')}",
        f"- High-confidence weighted gap: {high_confidence_calibration.get('weighted_gap', 0.0):.4f}",
        "",
        "## Validation Partition",
        f"- Mode: {validation_partition['mode']}",
        f"- Calibration samples: {validation_partition['calibration_count']}",
        f"- Selection samples: {validation_partition['selection_count']}",
        f"- Calibration range: {validation_partition['calibration_start_ts']:.0f} -> {validation_partition['calibration_end_ts']:.0f}",
        f"- Selection range: {validation_partition['selection_start_ts']:.0f} -> {validation_partition['selection_end_ts']:.0f}",
        f"- Calibration class diversity: {validation_partition['calibration_class_diversity']}",
        f"- Selection class diversity: {validation_partition['selection_class_diversity']}",
        "",
        "## Temporal Audit",
        f"- Windows: {report['temporal_walk_forward']['test']['num_windows']}",
        f"- Min PR-AUC across windows: {temporal_summary.get('min_average_precision', 0.0):.4f}",
        f"- Avg PR-AUC across windows: {temporal_summary.get('avg_average_precision', 0.0):.4f}",
        f"- Min recall across windows: {temporal_summary.get('min_recall', 0.0):.4f}",
        "",
        "## Feature Monitoring",
        f"- Feature count: {feature_monitoring['feature_count']}",
        f"- Mean abs runtime z-score (test): {feature_monitoring['mean_abs_zscore_test']:.4f}",
        f"- Max train variance: {feature_monitoring['max_variance']:.4f}",
        f"- Min train variance: {feature_monitoring['min_variance']:.4f}",
        "",
        "## Label Audit",
        f"- Labeling method: {dataset_summary.get('labeling_method', 'legacy')}",
        f"- Timeout only: {dataset_summary.get('labeling_timeout_only', False)}",
        f"- Timeout windows without future episode: {label_audit.get('timeouts_without_future_episode', 0)}",
        f"- Timeout windows with only sub-threshold episodes: {label_audit.get('timeouts_with_only_sub_threshold_episode', 0)}",
        f"- Positive entry bucket lt_0_30: {label_audit.get('positive_entry_spread_buckets', {}).get('lt_0_30', 0)}",
        f"- Positive entry bucket 0_30_to_0_50: {label_audit.get('positive_entry_spread_buckets', {}).get('0_30_to_0_50', 0)}",
        f"- Positive entry bucket 0_50_to_1_00: {label_audit.get('positive_entry_spread_buckets', {}).get('0_50_to_1_00', 0)}",
        f"- Positive entry bucket 1_00_to_2_00: {label_audit.get('positive_entry_spread_buckets', {}).get('1_00_to_2_00', 0)}",
        f"- Positive entry bucket ge_2_00: {label_audit.get('positive_entry_spread_buckets', {}).get('ge_2_00', 0)}",
        f"- Timeout peak bucket 0_80_to_1_00: {label_audit.get('timeout_peak_future_total_spread_buckets', {}).get('0_80_to_1_00', 0)}",
        f"- Future total spread p50: {label_audit.get('future_episode_total_spread_quantiles', {}).get('p50', 0.0):.4f}",
        f"- Future total spread p90: {label_audit.get('future_episode_total_spread_quantiles', {}).get('p90', 0.0):.4f}",
        "",
        "## Subgroups",
        f"- Entry buckets analysed: {len(report['subgroup_metrics']['test']['entry_spread_buckets'])}",
        f"- Volatility buckets analysed: {len(report['subgroup_metrics']['test']['volatility_buckets'])}",
        "",
        "## Runtime Gates",
        f"- Model status: {report['model_status']}",
        f"- Artifact version: {report['artifact_metadata']['version']}",
        f"- Execute threshold: {report['thresholds']['execute_threshold']:.2f}",
        f"- Strong threshold: {report['thresholds']['strong_threshold']:.2f}",
        "",
        "## Residual Risks",
        "- A auditoria offline mede janelas cronológicas, mas não substitui monitoramento contínuo de drift em produção.",
        "- A cabeça de ETA é mais sensível a mudança de regime do que a cabeça de classificação.",
        "- O bootstrap do servidor continua caro porque discovery e feeds não fazem parte desta refatoração de ML.",
    ]

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit_path


def run_training_loop(
    state_file: Path | None = None,
    artifact_dir: Path | None = None,
    sequence_length: int = 15,
    prediction_horizon_sec: int = 14_400,
    min_total_spread_pct: float = 0.50,
    label_cost_floor_pct: float | None = None,
    label_percentile: int | None = None,
    label_episode_window_days: int | None = None,
    selected_session_ids: list[int] | None = None,
    selected_block_ids: list[int] | None = None,
    allow_cross_session_merge: bool = False,
    max_session_gap_sec: float | None = None,
    regime_shift_score_threshold: float | None = 3.0,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.35,
    batch_size: int = 1024,
    window_stride: int = 5,
    max_epochs: int = 80,
    patience: int = 5,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    focal_alpha: float | None = None,
    focal_gamma: float = 2.0,
    min_train_positive_samples: int = 500,
    min_val_positive_samples: int = 50,
    min_test_positive_samples: int = 50,
    seed: int = 42,
    audit_output: Path | None = None,
    certification_context: dict[str, Any] | None = None,
    _preloaded_blocks: tuple[list, float, dict] | None = None,
    _precomputed_pair_segments: tuple[list[dict[str, Any]], dict[str, Any]] | None = None,
    _precomputed_segment_features: dict[int, list[list[float]]] | None = None,
    _scaffold_cache: dict[tuple[int, int, bool], Any] | None = None,
) -> dict[str, Any]:
    logger.info("Initializing robust ArbML training pipeline...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    default_state = Path(__file__).resolve().parent.parent.parent / "out" / "config" / "tracker_history.sqlite"
    state_path = Path(state_file) if state_file is not None else default_state
    artifact_root = Path(artifact_dir) if artifact_dir is not None else get_default_artifact_dir()
    if audit_output is None:
        audit_path = (
            artifact_root / f"{ARTIFACT_BASENAME}.audit.md"
            if artifact_dir is not None
            else Path(__file__).resolve().parent.parent.parent / "ml_diagnostic_report.md"
        )
    else:
        audit_path = Path(audit_output)

    bundle = build_dataset_bundle(
        state_path=state_path,
        sequence_length=sequence_length,
        prediction_horizon_sec=prediction_horizon_sec,
        min_total_spread_pct=min_total_spread_pct,
        label_cost_floor_pct=label_cost_floor_pct,
        label_percentile=label_percentile,
        label_episode_window_days=label_episode_window_days,
        selected_session_ids=selected_session_ids,
        selected_block_ids=selected_block_ids,
        allow_cross_session_merge=allow_cross_session_merge,
        max_session_gap_sec=max_session_gap_sec,
        regime_shift_score_threshold=regime_shift_score_threshold,
        window_stride=window_stride,
        _preloaded_blocks=_preloaded_blocks,
        _precomputed_pair_segments=_precomputed_pair_segments,
        _precomputed_segment_features=_precomputed_segment_features,
        _scaffold_cache=_scaffold_cache,
    )
    if bundle.summary["num_samples"] < 30:
        raise ValueError("Dataset construction failed or too small for robust training.")

    splits = build_group_splits(bundle, prediction_horizon_sec=prediction_horizon_sec)
    split_summary = _build_split_summary(splits)
    if not split_summary["global_temporal_order"]:
        raise ValueError("Global temporal separation failed in split construction.")
    if not split_summary["purged_temporal_separation_ok"]:
        raise ValueError("Temporal split did not preserve purged separation between boundaries.")
    if min(splits["train"].summary["num_samples"], splits["val"].summary["num_samples"], splits["test"].summary["num_samples"]) == 0:
        raise ValueError("At least one split is empty; need more disjoint pairs.")

    split_positive_counts = {
        "train": int(splits["train"].summary["num_positive_samples"]),
        "val": int(splits["val"].summary["num_positive_samples"]),
        "test": int(splits["test"].summary["num_positive_samples"]),
    }
    if split_positive_counts["train"] < int(min_train_positive_samples):
        raise ValueError(
            f"Train split has only {split_positive_counts['train']} positive samples; minimum required is {int(min_train_positive_samples)}. "
            "Revise min_total_spread_pct or prediction_horizon_sec before retraining."
        )
    if split_positive_counts["val"] < int(min_val_positive_samples):
        raise ValueError(
            f"Validation split has only {split_positive_counts['val']} positive samples; minimum required is {int(min_val_positive_samples)}. "
            "Revise min_total_spread_pct or prediction_horizon_sec before retraining."
        )
    if split_positive_counts["test"] < int(min_test_positive_samples):
        raise ValueError(
            f"Test split has only {split_positive_counts['test']} positive samples; minimum required is {int(min_test_positive_samples)}. "
            "Revise min_total_spread_pct or prediction_horizon_sec before retraining."
        )

    raw_train_X = splits["train"].X.detach().clone()
    feature_mean, feature_std = compute_feature_stats(splits["train"].X)
    for split_bundle in splits.values():
        split_bundle.X = normalize_features(split_bundle.X, feature_mean, feature_std)

    positive_rate_train = float(splits["train"].y_class.mean().item()) if splits["train"].summary["num_samples"] else 0.0
    focal_alpha_effective = float(focal_alpha) if focal_alpha is not None else _derive_focal_alpha(positive_rate_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    logger.info("Training device: %s%s (AMP=%s)", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "", use_amp)
    model = SpreadSequenceLSTM(
        input_sz=splits["train"].X.shape[-1],
        hidden_sz=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_attention=True,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6,
    )

    criterion_prob = FocalLoss(alpha=focal_alpha_effective, gamma=focal_gamma)
    criterion_eta = nn.SmoothL1Loss()

    train_loader = _make_loader(splits["train"], batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(splits["val"], batch_size=batch_size, shuffle=False)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_history: list[dict[str, float | int]] = []
    training_started_at = _time.time()

    for epoch in range(max_epochs):
        model.train()
        train_losses: list[float] = []
        gradient_norms: list[float] = []
        for batch_x, batch_y_class, batch_y_eta in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y_class = batch_y_class.to(device, non_blocking=True)
            batch_y_eta = batch_y_eta.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, eta_raw = model(batch_x)
                loss_prob = criterion_prob(logits.squeeze(1), batch_y_class)
                loss_eta = _eta_loss(criterion_eta, eta_raw, batch_y_eta, batch_y_class)
                loss = loss_prob + 0.25 * loss_eta
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
                optimizer.step()
            train_losses.append(float(loss.item()))
            gradient_norms.append(gradient_norm)

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch_x, batch_y_class, batch_y_eta in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y_class = batch_y_class.to(device, non_blocking=True)
                batch_y_eta = batch_y_eta.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, eta_raw = model(batch_x)
                    loss_prob = criterion_prob(logits.squeeze(1), batch_y_class)
                    loss_eta = _eta_loss(criterion_eta, eta_raw, batch_y_eta, batch_y_class)
                    val_losses.append(float((loss_prob + 0.25 * loss_eta).item()))

        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %s/%s | train_loss=%.4f | val_loss=%.4f | lr=%.2e",
            epoch + 1,
            max_epochs,
            avg_train_loss,
            avg_val_loss,
            current_lr,
        )
        scheduler.step(avg_val_loss)
        epoch_history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": float(current_lr),
                "avg_gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
                "max_gradient_norm": float(np.max(gradient_norms)) if gradient_norms else 0.0,
            }
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered at epoch %s", epoch + 1)
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    val_eval_uncalibrated = _evaluate_split(model, splits["val"], device, scale=1.0, bias=0.0, threshold=0.5)
    val_logits = val_eval_uncalibrated["logits"]
    val_labels = val_eval_uncalibrated["y_class"]
    cal_idx, sel_idx, validation_partition = _partition_validation_chronologically(
        val_eval_uncalibrated["timestamps"],
        val_labels,
    )
    platt_scale, platt_bias = _fit_platt_scaler(val_logits[cal_idx], val_labels[cal_idx])
    sel_probs = _apply_platt_scaling(val_logits[sel_idx], platt_scale, platt_bias)
    threshold_selection = _select_thresholds(val_labels[sel_idx], sel_probs)
    execute_threshold = threshold_selection["execute_threshold"]
    strong_threshold = threshold_selection["strong_threshold"]
    val_threshold_bundle = splits["val"].subset(sel_idx.astype(int).tolist(), "val_threshold")
    evaluations = {
        "train": _evaluate_split(model, splits["train"], device, platt_scale, platt_bias, execute_threshold),
        "val_threshold": _evaluate_split(model, val_threshold_bundle, device, platt_scale, platt_bias, execute_threshold),
        "test": _evaluate_split(model, splits["test"], device, platt_scale, platt_bias, execute_threshold),
    }
    baselines = _baseline_metrics(splits["train"], splits["test"])

    dataset_summary = dict(bundle.summary)
    dataset_summary["feature_abs_sum"] = float(bundle.X.abs().sum().item())
    dataset_summary["positive_rate_train"] = positive_rate_train
    dataset_summary["focal_alpha_effective"] = focal_alpha_effective
    label_audit = dict(dataset_summary.get("label_audit", {}))
    test_x = splits["test"].X.detach().cpu().numpy() if splits["test"].summary["num_samples"] else np.empty((0, 0, 0), dtype=float)
    feature_std_np = feature_std.detach().cpu().numpy()
    mean_abs_zscore_test = 0.0
    if test_x.size:
        test_feature_means = test_x.reshape(-1, test_x.shape[-1]).mean(axis=0)
        mean_abs_zscore_test = float(np.mean(np.abs(test_feature_means)))
    feature_monitoring = {
        "feature_count": int(len(bundle.feature_names)),
        "feature_names": list(bundle.feature_names),
        "min_variance": float(np.min(np.square(feature_std_np))) if feature_std_np.size else 0.0,
        "max_variance": float(np.max(np.square(feature_std_np))) if feature_std_np.size else 0.0,
        "mean_abs_zscore_test": mean_abs_zscore_test,
        "psi_reference": _build_psi_reference(raw_train_X, list(bundle.feature_names)),
    }
    calibration = {
        split_name: _calibration_report(evaluation["y_class"], evaluation["probs"])
        for split_name, evaluation in evaluations.items()
        if split_name in {"val_threshold", "test"}
    }
    for calibration_payload in calibration.values():
        calibration_payload["high_confidence"] = _high_confidence_calibration(calibration_payload)
    subgroup_metrics = {
        split_name: _subgroup_metrics(evaluation, execute_threshold)
        for split_name, evaluation in evaluations.items()
    }
    temporal_walk_forward = {
        split_name: _temporal_walk_forward_report(evaluation, execute_threshold)
        for split_name, evaluation in evaluations.items()
    }
    strong_signal_metrics = {
        split_name: _classification_metrics(evaluation["y_class"], evaluation["probs"], strong_threshold)
        for split_name, evaluation in evaluations.items()
    }

    report = {
        "model_status": "trained",
        "device": str(device),
        "dataset_summary": dataset_summary,
        "split_summary": split_summary,
        "metrics": {
            split_name: evaluation["classification"]
            for split_name, evaluation in evaluations.items()
        },
        "eta_metrics": {
            split_name: evaluation["eta"]
            for split_name, evaluation in evaluations.items()
        },
        "baselines": baselines,
        "thresholds": {
            "execute_threshold": execute_threshold,
            "strong_threshold": strong_threshold,
        },
        "threshold_selection": threshold_selection,
        "strong_signal_metrics": strong_signal_metrics,
        "calibration": calibration,
        "subgroup_metrics": subgroup_metrics,
        "temporal_walk_forward": temporal_walk_forward,
        "validation_partition": validation_partition,
        "feature_monitoring": feature_monitoring,
        "label_audit": label_audit,
        "training": {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "max_epochs": max_epochs,
            "patience": patience,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "focal_alpha": focal_alpha_effective,
            "positive_rate_train": positive_rate_train,
            "focal_alpha_requested": float(focal_alpha) if focal_alpha is not None else None,
            "focal_alpha_effective": focal_alpha_effective,
            "focal_gamma": focal_gamma,
            "platt_scale": platt_scale,
            "platt_bias": platt_bias,
            "train_seconds": float(_time.time() - training_started_at),
            "epoch_history": epoch_history,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        },
    }

    training_config_payload = {
        "prediction_horizon_sec": prediction_horizon_sec,
        "min_total_spread_pct": float(min_total_spread_pct),
        "selected_threshold": float(min_total_spread_pct),
        "label_threshold_mode": str(dataset_summary.get("label_threshold_mode", "fixed_threshold")),
        "label_cost_floor_pct": float(
            dataset_summary.get(
                "label_cost_floor_pct",
                label_cost_floor_pct if label_cost_floor_pct is not None else min_total_spread_pct,
            )
        ),
        "label_percentile": dataset_summary.get("label_percentile"),
        "label_episode_window_days": dataset_summary.get("label_episode_window_days"),
        "selected_label_config": {
            "threshold": float(min_total_spread_pct),
            "cost_floor_pct": float(
                dataset_summary.get(
                    "label_cost_floor_pct",
                    label_cost_floor_pct if label_cost_floor_pct is not None else min_total_spread_pct,
                )
            ),
            "label_percentile": dataset_summary.get("label_percentile"),
            "label_episode_window_days": dataset_summary.get("label_episode_window_days"),
            "label_threshold_mode": str(dataset_summary.get("label_threshold_mode", "fixed_threshold")),
        },
        "label_thresholds": dict(dataset_summary.get("label_thresholds", {})),
        "labeling_method": str(dataset_summary.get("labeling_method", "legacy")),
        "labeling_timeout_only": bool(dataset_summary.get("labeling_timeout_only", False)),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "focal_alpha": focal_alpha_effective,
        "positive_rate_train": positive_rate_train,
        "focal_alpha_requested": float(focal_alpha) if focal_alpha is not None else None,
        "focal_alpha_effective": focal_alpha_effective,
        "focal_gamma": focal_gamma,
        "min_train_positive_samples": int(min_train_positive_samples),
        "min_val_positive_samples": int(min_val_positive_samples),
        "min_test_positive_samples": int(min_test_positive_samples),
        "threshold_selection": threshold_selection["execute"]["objective"],
        "validation_partition_mode": validation_partition["mode"],
        "selected_session_ids": list(selected_session_ids or []),
        "selected_block_ids": list(selected_block_ids or []),
        "allow_cross_session_merge": bool(allow_cross_session_merge),
        "max_session_gap_sec": None if max_session_gap_sec is None else float(max_session_gap_sec),
        "regime_shift_score_threshold": None if regime_shift_score_threshold is None else float(regime_shift_score_threshold),
        "psi_reference": feature_monitoring["psi_reference"],
        "feature_contract_version": feature_contract_version_for_names(list(bundle.feature_names)) or DEFAULT_FEATURE_CONTRACT_VERSION,
    }
    if certification_context:
        training_config_payload["certification_id"] = str(certification_context.get("certification_id") or "")
        training_config_payload["certification_verdict"] = str(certification_context.get("verdict") or "")
        training_config_payload["certification_failure_reasons"] = list(certification_context.get("failure_reasons") or [])
        training_config_payload["certification_warnings"] = list(certification_context.get("warnings") or [])
        training_config_payload["preflight_off_vs_on_summary"] = dict(certification_context.get("preflight_off_vs_on_summary") or {})
        training_config_payload["runtime_audit_package_path"] = str(certification_context.get("runtime_audit_package_path") or "")

    metadata = ModelArtifactMetadata(
        version="arbml-lstm-v3-seed{}-t{}-d{}".format(
            seed,
            int(_time.time()),
            hashlib.sha1(
                f"{bundle.summary['num_samples']}:{bundle.summary['num_pairs']}:{bundle.summary['num_positive_samples']}".encode()
            ).hexdigest()[:8],
        ),
        sequence_length=sequence_length,
        input_size=int(splits["train"].X.shape[-1]),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        feature_names=list(bundle.feature_names),
        feature_mean=[float(value) for value in feature_mean.tolist()],
        feature_std=[float(value) for value in feature_std.tolist()],
        platt_scale=float(platt_scale),
        platt_bias=float(platt_bias),
        execute_threshold=execute_threshold,
        strong_threshold=strong_threshold,
        min_history_points=sequence_length,
        min_empirical_events=2,
        validation_metrics=report["metrics"]["val_threshold"],
        test_metrics=report["metrics"]["test"],
        baselines=baselines,
        dataset_summary=dataset_summary,
        split_summary=split_summary,
        training_config=training_config_payload,
        feature_contract_version=feature_contract_version_for_names(list(bundle.feature_names)) or DEFAULT_FEATURE_CONTRACT_VERSION,
        use_attention=True,
        trained_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_fingerprint=_build_dataset_fingerprint(
            state_path=state_path,
            bundle=bundle,
            min_total_spread_pct=min_total_spread_pct,
            label_cost_floor_pct=label_cost_floor_pct,
            label_percentile=label_percentile,
            label_episode_window_days=label_episode_window_days,
            label_threshold_mode=str(dataset_summary.get("label_threshold_mode", "fixed_threshold")),
            selected_session_ids=selected_session_ids,
            selected_block_ids=selected_block_ids,
            allow_cross_session_merge=allow_cross_session_merge,
            max_session_gap_sec=max_session_gap_sec,
            regime_shift_score_threshold=regime_shift_score_threshold,
        ),
        feature_schema_hash=current_feature_schema_hash(),
    )
    report["artifact_metadata"] = metadata.to_dict()
    report["training_config"] = dict(training_config_payload)

    model_path, meta_path = save_artifact_bundle(model.cpu(), metadata, artifact_root)
    report_path = artifact_root / f"{ARTIFACT_BASENAME}.report.json"
    _write_json(report_path, report)
    audit_report_path = write_audit_report(report, audit_path)

    report["artifacts"] = {
        "model_path": str(model_path),
        "metadata_path": str(meta_path),
        "report_path": str(report_path),
        "audit_path": str(audit_report_path),
    }
    _write_json(report_path, report)
    logger.info("Training complete. Artifacts saved to %s", artifact_root)
    return report


def run_clean_training_cycle(
    *,
    state_file: Path | None = None,
    artifact_dir: Path | None = None,
    sequence_length: int = 15,
    prediction_horizon_sec: int = 14_400,
    thresholds: list[float] | None = None,
    label_percentiles: list[int] | None = None,
    selected_session_ids: list[int] | None = None,
    selected_block_ids: list[int] | None = None,
    allow_cross_session_merge: bool = False,
    max_session_gap_sec: float | None = None,
    regime_shift_score_threshold: float | None = 3.0,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.35,
    batch_size: int = 1024,
    window_stride: int = 5,
    max_epochs: int = 80,
    patience: int = 5,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    focal_alpha: float | None = None,
    focal_gamma: float = 2.0,
    min_train_positive_samples: int = 500,
    min_val_positive_samples: int = 50,
    min_test_positive_samples: int = 50,
    seed: int = 42,
    audit_output: Path | None = None,
    certification_mode: str = "full",
    max_certification_duration_sec: int = 300,
    label_episode_window_days: int = 5,
    allow_legacy_sessions: bool = False,
    runtime_audit_dir: Path | None = None,
    run_reconnection_stress: bool = False,
) -> dict[str, Any]:
    artifact_root = Path(artifact_dir) if artifact_dir is not None else get_default_artifact_dir()
    artifact_root.mkdir(parents=True, exist_ok=True)
    default_state = Path(__file__).resolve().parent.parent.parent / "out" / "config" / "tracker_history.sqlite"
    _resolved_state = Path(state_file) if state_file is not None else default_state
    logger.info("Loading blocks from %s (single load for full pipeline)...", _resolved_state)
    _cached_blocks = _load_blocks_from_sqlite(
        _resolved_state,
        selected_block_ids=selected_block_ids,
        selected_session_ids=selected_session_ids,
        selected_only=True,
        closed_only=True,
    )
    logger.info("Loaded %d blocks from %d sessions.", _cached_blocks[2].get("num_blocks", 0), _cached_blocks[2].get("num_sessions", 0))
    # Pre-compute segments + feature cache per merge mode (reused across all phases).
    # Features depend only on segment records/episodes, NOT on threshold/horizon.
    # Cache keyed by merge mode: Gate 10 tests both True/False — different segments.
    _shared_segments: dict[bool, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    _shared_feat_cache: dict[bool, dict[int, list[list[float]]]] = {}
    _shared_scaffold_cache: dict[tuple[int, int, bool, int], Any] = {}  # (seq_len, horizon, merge, stride) → _WindowScaffold
    for _merge_mode in ([False, True] if allow_cross_session_merge else [False]):
        _eff_gap = max_session_gap_sec
        if _merge_mode and _eff_gap is None:
            _eff_gap = _load_tracker_gap_threshold_sec(_resolved_state)
        _shared_segments[_merge_mode] = _build_pair_segments(
            _cached_blocks[0],
            allow_cross_session_merge=_merge_mode,
            max_session_gap_sec=_eff_gap if _merge_mode else max_session_gap_sec,
            regime_shift_score_threshold=regime_shift_score_threshold,
        )
        _shared_feat_cache[_merge_mode] = {}
    logger.info("Pre-computed segments for %d merge modes.", len(_shared_segments))
    # Scale certification timeout proportionally to dataset size.
    _num_blocks = int(_cached_blocks[2].get("num_blocks", 0))
    _scaled_timeout = max(int(max_certification_duration_sec), 300 + _num_blocks // 500)
    if _scaled_timeout != max_certification_duration_sec:
        logger.info("Scaled certification timeout %ds -> %ds for %d blocks.", max_certification_duration_sec, _scaled_timeout, _num_blocks)
    certification = certify_data_for_training(
        state_file=state_file,
        artifact_dir=artifact_root,
        sequence_length=sequence_length,
        prediction_horizon_sec=prediction_horizon_sec,
        thresholds=thresholds,
        label_percentiles=label_percentiles,
        selected_session_ids=selected_session_ids,
        selected_block_ids=selected_block_ids,
        allow_cross_session_merge=allow_cross_session_merge,
        max_session_gap_sec=max_session_gap_sec,
        regime_shift_score_threshold=regime_shift_score_threshold,
        certification_mode=certification_mode,
        max_certification_duration_sec=_scaled_timeout,
        label_episode_window_days=label_episode_window_days,
        allow_legacy_sessions=allow_legacy_sessions,
        runtime_audit_dir=runtime_audit_dir,
        run_reconnection_stress=run_reconnection_stress,
        _preloaded_blocks=_cached_blocks,
        _precomputed_segments_by_merge=_shared_segments,
        _precomputed_features_by_merge=_shared_feat_cache,
        _scaffold_cache=_shared_scaffold_cache,
        window_stride=window_stride,
    )
    if not bool(certification.get("certified")):
        raise ValueError(
            "Training data certification failed: "
            + ", ".join(certification.get("failure_reasons", []) or ["unknown_certification_failure"])
        )
    effective_scope = dict(certification.get("effective_session_scope") or {})
    effective_session_ids = [
        int(value)
        for value in (effective_scope.get("session_ids") or effective_scope.get("effective_session_ids") or selected_session_ids or [])
        if int(value) > 0
    ]
    preflight_output = artifact_root / "threshold_preflight.json"
    preflight = run_threshold_preflight(
        state_file=state_file,
        sequence_length=sequence_length,
        prediction_horizon_sec=prediction_horizon_sec,
        thresholds=thresholds,
        label_percentiles=label_percentiles,
        label_episode_window_days=label_episode_window_days,
        selected_session_ids=effective_session_ids or selected_session_ids,
        selected_block_ids=selected_block_ids,
        allow_cross_session_merge=allow_cross_session_merge,
        max_session_gap_sec=max_session_gap_sec,
        regime_shift_score_threshold=regime_shift_score_threshold,
        min_train_positive_samples=min_train_positive_samples,
        min_val_positive_samples=min_val_positive_samples,
        min_test_positive_samples=min_test_positive_samples,
        output_path=preflight_output,
        _preloaded_blocks=_cached_blocks,
        _precomputed_pair_segments=_shared_segments.get(allow_cross_session_merge),
        _precomputed_segment_features=_shared_feat_cache.get(allow_cross_session_merge),
        _scaffold_cache=_shared_scaffold_cache,
        window_stride=window_stride,
    )
    if not preflight.get("qualifies_for_training"):
        raise ValueError("No threshold qualified for clean training in preflight.")
    selected_threshold = float(preflight["selected_threshold"])
    selected_label_config = _label_config_payload(preflight.get("selected_label_config"))
    legacy_archive_dir = _archive_existing_artifacts(artifact_root)
    report = run_training_loop(
        state_file=state_file,
        artifact_dir=artifact_root,
        sequence_length=sequence_length,
        prediction_horizon_sec=prediction_horizon_sec,
        min_total_spread_pct=selected_threshold,
        label_cost_floor_pct=(
            float(selected_label_config["cost_floor_pct"])
            if selected_label_config.get("label_threshold_mode") == "rolling_pair_percentile"
            else None
        ),
        label_percentile=(
            int(selected_label_config["label_percentile"])
            if selected_label_config.get("label_threshold_mode") == "rolling_pair_percentile"
            and selected_label_config.get("label_percentile") is not None
            else None
        ),
        label_episode_window_days=(
            int(selected_label_config["label_episode_window_days"])
            if selected_label_config.get("label_threshold_mode") == "rolling_pair_percentile"
            and selected_label_config.get("label_episode_window_days") is not None
            else None
        ),
        selected_session_ids=effective_session_ids or selected_session_ids,
        selected_block_ids=selected_block_ids,
        allow_cross_session_merge=allow_cross_session_merge,
        max_session_gap_sec=max_session_gap_sec,
        regime_shift_score_threshold=regime_shift_score_threshold,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        batch_size=batch_size,
        window_stride=window_stride,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        min_train_positive_samples=min_train_positive_samples,
        min_val_positive_samples=min_val_positive_samples,
        min_test_positive_samples=min_test_positive_samples,
        seed=seed,
        audit_output=audit_output,
        certification_context=certification,
        _preloaded_blocks=_cached_blocks,
        _precomputed_pair_segments=_shared_segments.get(allow_cross_session_merge),
        _precomputed_segment_features=_shared_feat_cache.get(allow_cross_session_merge),
        _scaffold_cache=_shared_scaffold_cache,
    )
    report["data_certification"] = certification
    report["certification_id"] = str(certification.get("certification_id") or "")
    report["preflight"] = preflight
    report["legacy_archive_dir"] = legacy_archive_dir
    report["training"]["selected_threshold"] = selected_threshold
    report["training"]["selected_label_config"] = dict(selected_label_config)
    report["training"]["certification_id"] = str(certification.get("certification_id") or "")
    report.setdefault("training_config", {})
    report["training_config"]["selected_threshold"] = selected_threshold
    report["training_config"]["selected_label_config"] = dict(selected_label_config)
    report["training_config"]["certification_id"] = str(certification.get("certification_id") or "")
    report["training_config"]["certification_verdict"] = str(certification.get("verdict") or "")
    report["training_config"]["runtime_audit_package_path"] = str(certification.get("runtime_audit_package_path") or "")
    report["training_config"]["preflight_off_vs_on_summary"] = certification.get("preflight_off_vs_on_summary", {})
    report_path = artifact_root / f"{ARTIFACT_BASENAME}.report.json"
    _write_json(report_path, report)

    _, metadata_path = get_artifact_paths(artifact_root)
    if metadata_path.is_file():
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_payload.setdefault("training_config", {})
        metadata_payload["training_config"]["certification_id"] = certification.get("certification_id", "")
        metadata_payload["training_config"]["certification_verdict"] = certification.get("verdict", "")
        metadata_payload["training_config"]["certification_failure_reasons"] = certification.get("failure_reasons", [])
        metadata_payload["training_config"]["certification_warnings"] = certification.get("warnings", [])
        metadata_payload["training_config"]["runtime_audit_package_path"] = certification.get("runtime_audit_package_path", "")
        metadata_payload["training_config"]["effective_session_scope"] = certification.get("effective_session_scope", {})
        metadata_payload["training_config"]["preflight_off_vs_on_summary"] = certification.get("preflight_off_vs_on_summary", {})
        metadata_payload["training_config"]["preflight"] = preflight
        metadata_payload["training_config"]["selected_threshold"] = selected_threshold
        metadata_payload["training_config"]["selected_label_config"] = dict(selected_label_config)
        metadata_payload["training_config"]["legacy_archive_dir"] = legacy_archive_dir
        metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")
    report.setdefault("artifact_metadata", {}).setdefault("training_config", {})
    report["artifact_metadata"]["training_config"]["selected_threshold"] = selected_threshold
    report["artifact_metadata"]["training_config"]["selected_label_config"] = dict(selected_label_config)
    report["artifact_metadata"]["training_config"]["certification_id"] = certification.get("certification_id", "")
    report["artifact_metadata"]["training_config"]["certification_verdict"] = certification.get("verdict", "")
    report["artifact_metadata"]["training_config"]["certification_failure_reasons"] = certification.get("failure_reasons", [])
    report["artifact_metadata"]["training_config"]["certification_warnings"] = certification.get("warnings", [])
    report["artifact_metadata"]["training_config"]["runtime_audit_package_path"] = certification.get("runtime_audit_package_path", "")
    report["artifact_metadata"]["training_config"]["effective_session_scope"] = certification.get("effective_session_scope", {})
    report["artifact_metadata"]["training_config"]["preflight_off_vs_on_summary"] = certification.get("preflight_off_vs_on_summary", {})
    _write_json(report_path, report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_training_loop()
