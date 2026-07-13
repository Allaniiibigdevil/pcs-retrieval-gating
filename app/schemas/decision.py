from pydantic import BaseModel, Field


class DecideRequest(BaseModel):
    task_id: str | None = None
    task: str
    top_k_docs: int = 50
    max_systems: int = 5


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


class SystemDecision(BaseModel):
    system_id: str
    selected: bool
    confidence: float
    evidence_docs: list[EvidenceDoc]


class DecideResponse(BaseModel):
    task_id: str | None = None
    task: str
    selected_systems: list[str] = Field(default_factory=list)
    decisions: list[SystemDecision]
    latency_ms: dict[str, float] = Field(default_factory=dict)
