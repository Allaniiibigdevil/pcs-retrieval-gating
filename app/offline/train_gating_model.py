from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.decision.gating_features import GatingFeatureRow, rows_to_numpy
from app.ml.logistic_regression import train_logistic_regression


def load_rows(path: Path) -> list[GatingFeatureRow]:
    rows: list[GatingFeatureRow] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows.append(GatingFeatureRow(**payload))
    return rows


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Train the logistic-regression gating model from JSONL labels."
    )
    parser.add_argument("--input", default=settings.GATING_TRAINING_DATA_PATH)
    parser.add_argument("--output", default=settings.GATING_MODEL_PATH)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--l2", type=float, default=0.0)
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    features, labels = rows_to_numpy(rows)
    model = train_logistic_regression(
        features,
        labels,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
    )
    model.save(args.output)
    print(f"trained rows={len(labels)} output={args.output}")


if __name__ == "__main__":
    main()
