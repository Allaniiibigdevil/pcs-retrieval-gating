from pydantic import BaseModel, ConfigDict, Field


class DecideRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    task_id: str | None = None
    task: str = Field(min_length=1)


class EvidenceDoc(BaseModel):
    doc_id: str
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)
    matched_queries: list[str] = Field(default_factory=list)
    highlight: dict[str, list[str]] = Field(default_factory=dict)
    bm25_score: float | None = None
    bm25_score_norm: float | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None


class SystemDecision(BaseModel):
    system_id: str
    selected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_docs: list[EvidenceDoc]


class DecideResponse(BaseModel):
    task_id: str | None = None
    task: str
    rewritten_queries: list[str] = Field(default_factory=list)
    selected_systems: list[str] = Field(default_factory=list)
    decisions: list[SystemDecision]
    latency_ms: dict[str, float] = Field(default_factory=dict)
