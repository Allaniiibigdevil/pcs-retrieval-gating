from pydantic import BaseModel, ConfigDict, Field


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    task: str = Field(min_length=1)
    top_k_docs: int = Field(default=20, ge=1, le=1000)
    max_systems: int = Field(default=5, ge=0)


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
    gating_score: float | None = None


class SystemDecision(BaseModel):
    system_id: str
    selected: bool
    confidence: float
    confidence_kind: str = "heuristic"
    threshold: float
    trigger_doc_id: str | None = None
    evidence_docs: list[EvidenceDoc]


class DecideResponse(BaseModel):
    task_id: str | None = None
    task: str
    selected_systems: list[str] = Field(default_factory=list)
    decisions: list[SystemDecision]
    latency_ms: dict[str, float] = Field(default_factory=dict)
