from pydantic import BaseModel, Field

from app.config import get_settings


class DecideRequest(BaseModel):
    task_id: str | None = None
    task: str
    top_k_docs: int = Field(
        default_factory=lambda: get_settings().DEFAULT_TOP_K_DOCS,
        ge=1,
        le=1000,
    )


class EvidenceDoc(BaseModel):
    doc_id: str
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)
    highlight: dict[str, list[str]] = Field(default_factory=dict)
    bm25_score: float | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float
    rrf_rank: int


class SystemDecision(BaseModel):
    system_id: str
    selected: bool
    rrf_score: float
    evidence_docs: list[EvidenceDoc]


class DecideResponse(BaseModel):
    task_id: str | None = None
    task: str
    selected_systems: list[str] = Field(default_factory=list)
    decisions: list[SystemDecision]
    latency_ms: dict[str, float] = Field(default_factory=dict)
