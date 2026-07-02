import hashlib
import math
import random
from typing import Protocol

from app.config import get_settings
from app.schemas.doc import SourceDoc


class EmbeddingService(Protocol):
    async def embed(self, text: str) -> list[float]:
        ...


class MockEmbeddingService:
    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or get_settings().EMBEDDING_DIM

    async def embed(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        vector = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def build_embedding_text(doc: SourceDoc) -> str:
    keywords = ", ".join(doc.keywords)
    return f"system: {doc.system_id}\nsummary: {doc.summary}\nkeywords: {keywords}"


def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    if settings.EMBEDDING_PROVIDER != "mock":
        raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")
    return MockEmbeddingService(settings.EMBEDDING_DIM)
