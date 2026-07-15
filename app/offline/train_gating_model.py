from __future__ import annotations

import argparse
from pathlib import Path
import random

from app.config import get_settings
from app.decision.gating_features import GatingCase, MilBag, cases_to_mil_bags
from app.ml.mil_mlp import train_mil_mlp


def load_cases(path: Path) -> list[GatingCase]:
    cases: list[GatingCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                cases.append(GatingCase.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid gating case at {path}:{line_no}") from exc
    return cases


def split_bags_by_query(
    bags: list[MilBag], calibration_fraction: float, seed: int
) -> tuple[list[MilBag], list[MilBag]]:
    query_keys = sorted({bag.query_key for bag in bags})
    if calibration_fraction <= 0 or len(query_keys) < 2:
        return bags, []
    if calibration_fraction >= 1:
        raise ValueError("calibration_fraction must be smaller than 1")
    random.Random(seed).shuffle(query_keys)
    count = min(max(1, round(len(query_keys) * calibration_fraction)), len(query_keys) - 1)
    calibration_keys = set(query_keys[:count])
    return (
        [bag for bag in bags if bag.query_key not in calibration_keys],
        [bag for bag in bags if bag.query_key in calibration_keys],
    )


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train nine-representative max-MIL MLP.")
    parser.add_argument("--input", default=settings.GATING_CASES_PATH)
    parser.add_argument("--output", default=settings.GATING_MODEL_PATH)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--negative-instance-weight", type=float, default=0.25)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    bags = cases_to_mil_bags(load_cases(Path(args.input)))
    training, calibration = split_bags_by_query(bags, args.calibration_fraction, args.seed)
    model = train_mil_mlp(
        training,
        calibration_bags=calibration,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
        batch_size=args.batch_size,
        negative_instance_weight=args.negative_instance_weight,
        seed=args.seed,
    )
    model.save(args.output)
    print(
        f"trained bags={len(training)} calibration_bags={len(calibration)} "
        f"calibrated={model.calibrated} output={args.output}"
    )


if __name__ == "__main__":
    main()
