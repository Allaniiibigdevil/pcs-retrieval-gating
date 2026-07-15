import pytest
from pydantic import ValidationError

from app.schemas.decision import DecideRequest


def test_decide_request_rejects_source_hint_fields() -> None:
    with pytest.raises(ValidationError):
        DecideRequest.model_validate({"task": "去年京都的红色寺庙", "source_type": "photo"})
