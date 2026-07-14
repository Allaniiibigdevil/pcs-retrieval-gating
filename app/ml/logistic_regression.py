from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from app.decision.gating_features import FEATURE_NAMES


@dataclass
class LogisticRegressionGatingModel:
    weights: np.ndarray
    bias: float
    feature_names: list[str]
    feature_means: np.ndarray
    feature_stds: np.ndarray

    @classmethod
    def fresh(
        cls,
        *,
        feature_means: np.ndarray | None = None,
        feature_stds: np.ndarray | None = None,
    ) -> "LogisticRegressionGatingModel":
        return cls(
            weights=np.zeros(len(FEATURE_NAMES), dtype=np.float64),
            bias=0.0,
            feature_names=list(FEATURE_NAMES),
            feature_means=(
                np.zeros(len(FEATURE_NAMES), dtype=np.float64)
                if feature_means is None
                else feature_means
            ),
            feature_stds=(
                np.ones(len(FEATURE_NAMES), dtype=np.float64)
                if feature_stds is None
                else feature_stds
            ),
        )

    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        return (features - self.feature_means) / self.feature_stds

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        transformed = self._transform_features(features)
        logits = np.clip(transformed @ self.weights + self.bias, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "feature_means": self.feature_means.tolist(),
                    "feature_stds": self.feature_stds.tolist(),
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
        feature_means = np.array(payload.get("feature_means", []), dtype=np.float64)
        feature_stds = np.array(payload.get("feature_stds", []), dtype=np.float64)
        if (
            feature_names != FEATURE_NAMES
            or len(weights) != len(FEATURE_NAMES)
            or len(feature_means) != len(FEATURE_NAMES)
            or len(feature_stds) != len(FEATURE_NAMES)
        ):
            raise ValueError(
                "gating model feature schema does not match current FEATURE_NAMES; "
                "retrain the model with the current training data schema"
            )
        return cls(
            weights=weights,
            bias=float(payload["bias"]),
            feature_names=feature_names,
            feature_means=feature_means,
            feature_stds=np.where(feature_stds == 0.0, 1.0, feature_stds),
        )


def train_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float = 0.1,
    epochs: int = 1000,
    l2: float = 0.0,
) -> LogisticRegressionGatingModel:
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"features must have shape (n, {len(FEATURE_NAMES)})")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must have shape (n,)")

    feature_means = features.mean(axis=0)
    feature_stds = features.std(axis=0)
    feature_stds = np.where(feature_stds == 0.0, 1.0, feature_stds)
    transformed_features = (features - feature_means) / feature_stds

    model = LogisticRegressionGatingModel.fresh(
        feature_means=feature_means,
        feature_stds=feature_stds,
    )
    n_samples = features.shape[0]
    for _ in range(epochs):
        probs = model.predict_proba(features)
        errors = probs - labels
        grad_w = (transformed_features.T @ errors) / n_samples + l2 * model.weights
        grad_b = float(errors.mean())
        model.weights -= learning_rate * grad_w
        model.bias -= learning_rate * grad_b
    return model
