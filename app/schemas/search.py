from typing import Any

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    doc_id: str
    system_id: str

    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    bm25_score: float | None = None
    bm25_rank: int | None = None

    vector_score: float | None = None
    vector_rank: int | None = None

    reranker_score: float | None = None
    reranker_rank: int | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 50
    source_id: str | None = None


class SearchResponse(BaseModel):
    query: str
    source_id: str | None = None
    hits: list[SearchHit]
