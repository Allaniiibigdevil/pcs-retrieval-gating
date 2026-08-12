import pytest

from app.api import embedding as embedding_api
from app.schemas.embedding import Request


class FakeEmbeddingService:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result or [[0.1, 0.2, 0.3]]
        self.error = error
        self.calls: list[list[str]] = []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.asyncio
async def test_embedding_success(monkeypatch) -> None:
    service = FakeEmbeddingService()
    monkeypatch.setattr(embedding_api, "embedding_service", service)

    response = await embedding_api.embedding(
        Request(texts="hello", is_normalize=1)
    )

    assert service.calls == [["hello"]]
    assert response.model_dump() == {
        "result": {
            "code": "0",
            "des": "success",
            "content": [
                {
                    "embeddings": [[0.1, 0.2, 0.3]],
                }
            ],
        }
    }


@pytest.mark.asyncio
async def test_embedding_failure_returns_code_2(monkeypatch) -> None:
    service = FakeEmbeddingService(error=RuntimeError("boom"))
    monkeypatch.setattr(embedding_api, "embedding_service", service)

    response = await embedding_api.embedding(
        Request(texts="hello", is_normalize=0)
    )

    assert response.model_dump() == {
        "result": {
            "code": "2",
            "des": "embedding failed",
            "content": None,
        }
    }


@pytest.mark.asyncio
async def test_embedding_empty_text_returns_code_2(monkeypatch) -> None:
    service = FakeEmbeddingService()
    monkeypatch.setattr(embedding_api, "embedding_service", service)

    response = await embedding_api.embedding(Request(texts="   ", is_normalize=0))

    assert service.calls == []
    assert response.model_dump() == {
        "result": {
            "code": "2",
            "des": "texts must not be empty",
            "content": None,
        }
    }
