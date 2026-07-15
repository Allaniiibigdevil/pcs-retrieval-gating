from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from app.decision.gating_features import FEATURE_NAMES, MilBag


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

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
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
        output_path.write_text(
            json.dumps(
                {
                    "model_type": "representative_max_mil_mlp",
                    "feature_names": self.feature_names,
                    "input_weights": self.input_weights.tolist(),
                    "input_bias": self.input_bias.tolist(),
                    "output_weights": self.output_weights.tolist(),
                    "output_bias": self.output_bias,
                    "calibration_scale": self.calibration_scale,
                    "calibration_bias": self.calibration_bias,
                    "calibrated": self.calibrated,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "MilMlpGatingModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != "representative_max_mil_mlp":
            raise ValueError("gating model type is not representative max-MIL MLP")
        if list(payload.get("feature_names", [])) != FEATURE_NAMES:
            raise ValueError("gating model feature schema mismatch; retrain the model")
        return cls(
            input_weights=np.asarray(payload["input_weights"], dtype=np.float64),
            input_bias=np.asarray(payload["input_bias"], dtype=np.float64),
            output_weights=np.asarray(payload["output_weights"], dtype=np.float64),
            output_bias=float(payload["output_bias"]),
            feature_names=list(payload["feature_names"]),
            calibration_scale=float(payload.get("calibration_scale", 1.0)),
            calibration_bias=float(payload.get("calibration_bias", 0.0)),
            calibrated=bool(payload.get("calibrated", False)),
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
    if not bags:
        raise ValueError("at least one MIL bag is required")
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
        input_weights=torch_model.input.weight.detach().cpu().numpy().astype(np.float64),
        input_bias=torch_model.input.bias.detach().cpu().numpy().astype(np.float64),
        output_weights=torch_model.output.weight.detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(np.float64),
        output_bias=float(torch_model.output.bias.detach().cpu().item()),
        feature_names=list(FEATURE_NAMES),
    )
    if calibration_bags:
        _fit_calibrator(model, calibration_bags, seed)
    return model


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
