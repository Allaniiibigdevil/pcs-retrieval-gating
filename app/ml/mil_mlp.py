from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from app.decision.gating_features import FEATURE_NAMES, MAX_REPRESENTATIVE_DOCS, MilBag


MODEL_TYPE = "representative_max_mil_mlp"
MODEL_FORMAT_VERSION = 1
MODEL_DTYPE = np.float32


@dataclass
class MilMlpGatingModel:
    input_weights: np.ndarray
    input_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: float
    feature_names: list[str]
    calibration_scale: float = 1.0
    calibration_bias: float = 0.0
    calibrated: bool = False

    def __post_init__(self) -> None:
        self.input_weights = np.asarray(self.input_weights, dtype=MODEL_DTYPE)
        self.input_bias = np.asarray(self.input_bias, dtype=MODEL_DTYPE)
        self.output_weights = np.asarray(self.output_weights, dtype=MODEL_DTYPE)
        self.output_bias = float(self.output_bias)
        self.feature_names = list(self.feature_names)
        self.calibration_scale = float(self.calibration_scale)
        self.calibration_bias = float(self.calibration_bias)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("gating model feature schema mismatch; retrain the model")
        hidden_dim = self.input_weights.shape[0] if self.input_weights.ndim == 2 else 0
        if self.input_weights.shape != (hidden_dim, len(FEATURE_NAMES)) or hidden_dim == 0:
            raise ValueError("invalid MIL MLP input weight shape")
        if self.input_bias.shape != (hidden_dim,):
            raise ValueError("invalid MIL MLP input bias shape")
        if self.output_weights.shape != (hidden_dim,):
            raise ValueError("invalid MIL MLP output weight shape")
        arrays = (self.input_weights, self.input_bias, self.output_weights)
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError("MIL MLP parameters must be finite")
        scalars = (self.output_bias, self.calibration_scale, self.calibration_bias)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("MIL MLP scalar parameters must be finite")
        if self.calibration_scale <= 0:
            raise ValueError("calibration scale must be positive")

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=MODEL_DTYPE)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(f"features must have shape (n, {len(self.feature_names)})")
        hidden = np.maximum(values @ self.input_weights.T + self.input_bias, 0.0)
        return hidden @ self.output_weights + self.output_bias

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = self.calibration_scale * self.predict_logits(features) + self.calibration_bias
        logits = np.clip(logits, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            np.savez(
                file,
                model_type=np.asarray(MODEL_TYPE),
                format_version=np.asarray(MODEL_FORMAT_VERSION, dtype=np.int64),
                feature_names=np.asarray(self.feature_names, dtype=np.str_),
                input_weights=self.input_weights,
                input_bias=self.input_bias,
                output_weights=self.output_weights,
                output_bias=np.asarray(self.output_bias, dtype=MODEL_DTYPE),
                calibration_scale=np.asarray(self.calibration_scale, dtype=MODEL_DTYPE),
                calibration_bias=np.asarray(self.calibration_bias, dtype=MODEL_DTYPE),
                calibrated=np.asarray(self.calibrated, dtype=np.bool_),
            )

    @classmethod
    def load(cls, path: str | Path) -> "MilMlpGatingModel":
        with np.load(Path(path), allow_pickle=False) as artifact:
            if str(artifact["model_type"].item()) != MODEL_TYPE:
                raise ValueError("gating model type is not representative max-MIL MLP")
            if int(artifact["format_version"].item()) != MODEL_FORMAT_VERSION:
                raise ValueError("unsupported MIL MLP artifact version")
            return cls(
                input_weights=artifact["input_weights"],
                input_bias=artifact["input_bias"],
                output_weights=artifact["output_weights"],
                output_bias=float(artifact["output_bias"].item()),
                feature_names=[str(value) for value in artifact["feature_names"].tolist()],
                calibration_scale=float(artifact["calibration_scale"].item()),
                calibration_bias=float(artifact["calibration_bias"].item()),
                calibrated=bool(artifact["calibrated"].item()),
            )


class _TorchMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.input(features))).squeeze(-1)


def train_mil_mlp(
    bags: Sequence[MilBag],
    *,
    calibration_bags: Sequence[MilBag] | None = None,
    hidden_dim: int = 16,
    learning_rate: float = 0.01,
    epochs: int = 500,
    l2: float = 1e-4,
    batch_size: int = 32,
    negative_instance_weight: float = 0.25,
    seed: int = 13,
) -> MilMlpGatingModel:
    _validate_training_bags(bags)
    if calibration_bags:
        _validate_training_bags(calibration_bags)
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be greater than 0")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if negative_instance_weight < 0:
        raise ValueError("negative_instance_weight cannot be negative")
    torch.manual_seed(seed)
    torch_model = _TorchMlp(len(FEATURE_NAMES), hidden_dim)
    optimizer = torch.optim.AdamW(torch_model.parameters(), lr=learning_rate, weight_decay=l2)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)

    for _ in range(epochs):
        order = torch.randperm(len(bags), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            batch = [bags[index] for index in order[start : start + batch_size]]
            bag_logits: list[torch.Tensor] = []
            labels: list[float] = []
            negative_losses: list[torch.Tensor] = []
            for bag in batch:
                doc_logits = torch_model(torch.as_tensor(bag.features, dtype=torch.float32))
                bag_logits.append(torch.max(doc_logits))
                labels.append(bag.label)
                if bag.label == 0.0:
                    negative_losses.append(criterion(doc_logits, torch.zeros_like(doc_logits)))
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(torch.stack(bag_logits), torch.as_tensor(labels, dtype=torch.float32))
            if negative_losses:
                loss = loss + negative_instance_weight * torch.stack(negative_losses).mean()
            loss.backward()
            optimizer.step()

    model = MilMlpGatingModel(
        input_weights=torch_model.input.weight.detach().cpu().numpy(),
        input_bias=torch_model.input.bias.detach().cpu().numpy(),
        output_weights=torch_model.output.weight.detach().cpu().numpy().reshape(-1),
        output_bias=float(torch_model.output.bias.detach().cpu().item()),
        feature_names=list(FEATURE_NAMES),
    )
    if calibration_bags:
        _fit_calibrator(model, calibration_bags, seed)
    return model


def _validate_training_bags(bags: Sequence[MilBag]) -> None:
    if not bags:
        raise ValueError("at least one MIL bag is required")
    for bag in bags:
        features = np.asarray(bag.features)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"bag features must have shape (n, {len(FEATURE_NAMES)})")
        if not 1 <= features.shape[0] <= MAX_REPRESENTATIVE_DOCS:
            raise ValueError(f"MIL bags must contain 1 to {MAX_REPRESENTATIVE_DOCS} documents")
        if not np.isfinite(features).all():
            raise ValueError("MIL bag features must be finite")
        if bag.label not in (0.0, 1.0):
            raise ValueError("MIL bag labels must be 0 or 1")


def _fit_calibrator(model: MilMlpGatingModel, bags: Sequence[MilBag], seed: int) -> None:
    labels = np.asarray([bag.label for bag in bags], dtype=np.float32)
    if len(np.unique(labels)) < 2:
        return
    logits = np.asarray(
        [float(np.max(model.predict_logits(bag.features))) for bag in bags],
        dtype=np.float32,
    )
    torch.manual_seed(seed)
    raw_scale = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
    bias = nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([raw_scale, bias], lr=0.03)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.nn.functional.softplus(raw_scale)
        loss = criterion(scale * torch.as_tensor(logits) + bias, torch.as_tensor(labels))
        loss.backward()
        optimizer.step()
    model.calibration_scale = float(torch.nn.functional.softplus(raw_scale).detach().item())
    model.calibration_bias = float(bias.detach().item())
    model.calibrated = True
