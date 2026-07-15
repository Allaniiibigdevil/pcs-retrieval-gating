import numpy as np

from app.decision.gating_features import FEATURE_NAMES, MilBag
from app.ml.mil_mlp import MilMlpGatingModel, train_mil_mlp


def _xor_bags() -> list[MilBag]:
    bags: list[MilBag] = []
    for index, (left, right, label) in enumerate(
        [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
    ):
        features = np.zeros((1, len(FEATURE_NAMES)), dtype=float)
        features[0, :2] = [left, right]
        bags.append(MilBag(f"q{index}", "source", features, label))
    return bags


def test_mil_mlp_learns_nonlinear_boundary() -> None:
    bags = _xor_bags()
    model = train_mil_mlp(
        bags,
        hidden_dim=8,
        learning_rate=0.03,
        epochs=800,
        batch_size=4,
        seed=7,
    )
    probabilities = [float(model.predict_proba(bag.features)[0]) for bag in bags]
    assert probabilities[0] < 0.2
    assert probabilities[1] > 0.8
    assert probabilities[2] > 0.8
    assert probabilities[3] < 0.2


def test_calibrated_model_round_trips(tmp_path) -> None:
    bags = _xor_bags()
    model = train_mil_mlp(
        bags,
        calibration_bags=bags,
        hidden_dim=8,
        learning_rate=0.03,
        epochs=500,
        batch_size=4,
        seed=11,
    )
    path = tmp_path / "model.npz"
    model.save(path)
    assert path.read_bytes().startswith(b"PK")
    loaded = MilMlpGatingModel.load(path)
    assert loaded.calibrated is True
    assert loaded.calibration_scale > 0
    np.testing.assert_allclose(
        loaded.predict_proba(bags[0].features),
        model.predict_proba(bags[0].features),
        rtol=1e-5,
    )
