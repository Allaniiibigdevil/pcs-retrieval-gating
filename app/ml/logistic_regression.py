from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.decision.gating_features import FEATURE_NAMES


@dataclass
class LogisticRegressionGatingModel:
    weights: np.ndarray
    bias: float
    feature_names: list[str]

    @classmethod
    def fresh(cls) -> "LogisticRegressionGatingModel":
        return cls(
            weights=np.zeros(len(FEATURE_NAMES), dtype=np.float64),
            bias=0.0,
            feature_names=list(FEATURE_NAMES),
        )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = np.clip(features @ self.weights + self.bias, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "weights": self.weights.tolist(),
                    "bias": self.bias,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegressionGatingModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        feature_names = list(payload.get("feature_names", FEATURE_NAMES))
        weights = np.array(payload["weights"], dtype=np.float64)
        if feature_names != FEATURE_NAMES or len(weights) != len(FEATURE_NAMES):
            raise ValueError(
                "gating model feature schema does not match current FEATURE_NAMES; "
                "retrain the model with the current training data schema"
            )
        return cls(
            weights=weights,
            bias=float(payload["bias"]),
            feature_names=feature_names,
        )


def train_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float = 0.1,
    epochs: int = 1000,
    l2: float = 0.0,
    batch_size: int = 32,
    seed: int = 13,
) -> LogisticRegressionGatingModel:
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"features must have shape (n, {len(FEATURE_NAMES)})")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must have shape (n,)")

    if features.shape[0] == 0:
        raise ValueError("features and labels must contain at least one sample")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    torch.manual_seed(seed)
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    label_tensor = torch.as_tensor(labels, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(feature_tensor, label_tensor)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, features.shape[0]),
        shuffle=True,
        generator=generator,
    )

    torch_model = nn.Linear(len(FEATURE_NAMES), 1)
    with torch.no_grad():
        torch_model.weight.zero_()
        torch_model.bias.zero_()

    optimizer = torch.optim.AdamW(
        torch_model.parameters(),
        lr=learning_rate,
        weight_decay=l2,
    )
    criterion = nn.BCEWithLogitsLoss()

    torch_model.train()
    for _ in range(epochs):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = torch_model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    torch_model.eval()
    with torch.no_grad():
        weights = torch_model.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
        bias = float(torch_model.bias.detach().cpu().item())
    return LogisticRegressionGatingModel(
        weights=weights,
        bias=bias,
        feature_names=list(FEATURE_NAMES),
    )
