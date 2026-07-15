import hashlib
import math
import random
from typing import Protocol

from app.config import get_settings
from app.schemas.doc import SourceDoc


class EmbeddingService(Protocol):
    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class MockEmbeddingService:
    def __init__(self, dim: int | None = None, query_instruction: str | None = None) -> None:
        settings = get_settings()
        self.dim = dim or settings.EMBEDDING_DIM
        self.query_instruction = (
            query_instruction
            if query_instruction is not None
            else settings.EMBEDDING_QUERY_INSTRUCTION
        )

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([self.query_instruction + text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        vector = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class BGEEmbeddingService:
    def __init__(
        self,
        model_path: str | None = None,
        query_instruction: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model_path = model_path or settings.EMBEDDING_MODEL_PATH
        self.query_instruction = (
            query_instruction
            if query_instruction is not None
            else settings.EMBEDDING_QUERY_INSTRUCTION
        )
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_path)
        return self._model

    async def embed_query(self, text: str) -> list[float]:
        return (await self._encode([self.query_instruction + text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._encode(texts)

    async def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32").tolist()


def build_embedding_text(doc: SourceDoc) -> str:
    keywords = ", ".join(doc.keywords)
    parts = [f"summary: {doc.summary}"]
    if keywords:
        parts.append(f"keywords: {keywords}")
    return "\n".join(parts)


def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    if settings.EMBEDDING_PROVIDER == "mock":
        return MockEmbeddingService(
            settings.EMBEDDING_DIM,
            query_instruction=settings.EMBEDDING_QUERY_INSTRUCTION,
        )
    if settings.EMBEDDING_PROVIDER == "bge":
        return BGEEmbeddingService(
            settings.EMBEDDING_MODEL_PATH,
            query_instruction=settings.EMBEDDING_QUERY_INSTRUCTION,
        )
    raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")
