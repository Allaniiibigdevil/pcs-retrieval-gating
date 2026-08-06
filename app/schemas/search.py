from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1)
    top_k: int = Field(default=50, ge=1, le=1000)
    source_id: str = Field(min_length=1)


class SearchResponse(BaseModel):
    query: str
    source_id: str
    hits: list[SearchHit]
