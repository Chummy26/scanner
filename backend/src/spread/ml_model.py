from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .feature_contracts import (
    DEFAULT_FEATURE_CONTRACT_VERSION,
    FEATURE_NAMES,
    feature_contract_version_for_names,
    feature_schema_hash,
    resolve_feature_contract,
)

logger = logging.getLogger(__name__)

ARTIFACT_BASENAME = "best_lstm_model"


def get_default_artifact_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "out" / "config"


def get_artifact_paths(
    artifact_dir: Path | None = None,
    base_name: str = ARTIFACT_BASENAME,
) -> tuple[Path, Path]:
    target_dir = Path(artifact_dir) if artifact_dir is not None else get_default_artifact_dir()
    return (
        target_dir / f"{base_name}.pth",
        target_dir / f"{base_name}.meta.json",
    )


@dataclass(slots=True)
class ModelArtifactMetadata:
    version: str
    sequence_length: int
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    feature_names: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    platt_scale: float
    platt_bias: float
    execute_threshold: float
    strong_threshold: float
    min_history_points: int
    min_empirical_events: int
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    baselines: dict[str, Any]
    dataset_summary: dict[str, Any]
    split_summary: dict[str, Any]
    training_config: dict[str, Any]
    feature_contract_version: str = DEFAULT_FEATURE_CONTRACT_VERSION
    use_attention: bool = True
    trained_at_utc: str = ""
    dataset_fingerprint: str = ""
    feature_schema_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelArtifactMetadata":
        payload.setdefault("use_attention", False)
        payload.setdefault(
            "feature_contract_version",
            feature_contract_version_for_names(list(payload.get("feature_names") or [])) or DEFAULT_FEATURE_CONTRACT_VERSION,
        )
        return cls(**payload)


def current_feature_schema_hash() -> str:
    return feature_schema_hash(list(FEATURE_NAMES))


def validate_artifact_metadata(metadata: ModelArtifactMetadata) -> list[str]:
    issues: list[str] = []
    try:
        resolved_version, expected_feature_names = resolve_feature_contract(
            list(metadata.feature_names),
            version=str(metadata.feature_contract_version or "").strip() or None,
        )
    except ValueError:
        issues.append("feature schema mismatch with supported contracts")
        expected_feature_names = list(metadata.feature_names)
        resolved_version = ""
    if metadata.input_size != len(expected_feature_names):
        issues.append(
            f"input size mismatch: artifact={metadata.input_size}, schema={len(expected_feature_names)}"
        )
    if list(metadata.feature_names) != list(expected_feature_names):
        issues.append("feature schema mismatch with current code feature contract")
    expected_hash = feature_schema_hash(list(expected_feature_names))
    if metadata.feature_schema_hash and metadata.feature_schema_hash != expected_hash:
        issues.append("feature schema hash mismatch")
    if resolved_version and str(metadata.feature_contract_version or "") != str(resolved_version):
        issues.append("feature contract version mismatch")
    if not metadata.use_attention:
        issues.append("artifact disabled temporal attention")
    return issues


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        scores = self.attention(lstm_output)
        weights = torch.softmax(scores, dim=1)
        context = (lstm_output * weights).sum(dim=1)
        return context


class SpreadSequenceLSTM(nn.Module):
    def __init__(
        self,
        input_sz: int = len(FEATURE_NAMES),
        hidden_sz: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.input_size = int(input_sz)
        self.hidden_size = int(hidden_sz)
        self.num_layers = int(num_layers)
        self.dropout_rate = float(dropout)
        self.use_attention = bool(use_attention)

        lstm_dropout = self.dropout_rate if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        if self.use_attention:
            self.temporal_attention = TemporalAttention(self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        head_hidden = max(8, self.hidden_size // 2)
        self.prob_out = nn.Sequential(
            nn.Linear(self.hidden_size, head_hidden),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(head_hidden, 1),
        )
        self.eta_out = nn.Sequential(
            nn.Linear(self.hidden_size, head_hidden),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        if self.use_attention:
            pooled = self.temporal_attention(out)
        else:
            pooled = out[:, -1, :]
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        prob_logits = self.prob_out(pooled)
        eta_raw = self.eta_out(pooled)
        return prob_logits, eta_raw


def build_model_from_metadata(metadata: ModelArtifactMetadata) -> SpreadSequenceLSTM:
    return SpreadSequenceLSTM(
        input_sz=metadata.input_size,
        hidden_sz=metadata.hidden_size,
        num_layers=metadata.num_layers,
        dropout=metadata.dropout,
        use_attention=metadata.use_attention,
    )


def _load_state_dict(model_path: Path, map_location: torch.device | str) -> dict[str, Any]:
    try:
        return torch.load(model_path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=map_location)


def save_artifact_bundle(
    model: SpreadSequenceLSTM,
    metadata: ModelArtifactMetadata,
    artifact_dir: Path | None = None,
    base_name: str = ARTIFACT_BASENAME,
) -> tuple[Path, Path]:
    model_path, meta_path = get_artifact_paths(artifact_dir, base_name=base_name)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    meta_path.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return model_path, meta_path


def load_artifact_bundle(
    artifact_dir: Path | None = None,
    base_name: str = ARTIFACT_BASENAME,
    map_location: torch.device | str = "cpu",
    *,
    enforce_current_contract: bool = False,
) -> tuple[SpreadSequenceLSTM, ModelArtifactMetadata]:
    model_path, meta_path = get_artifact_paths(artifact_dir, base_name=base_name)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata = ModelArtifactMetadata.from_dict(payload)
    if enforce_current_contract:
        issues = validate_artifact_metadata(metadata)
        if issues:
            raise ValueError("; ".join(issues))
    model = build_model_from_metadata(metadata)
    state_dict = _load_state_dict(model_path, map_location)
    model.load_state_dict(state_dict)
    logger.info("[ML] Loaded artifact bundle %s", metadata.version)
    return model, metadata
