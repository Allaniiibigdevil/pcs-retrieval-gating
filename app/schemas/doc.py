from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SourceDoc(BaseModel):
    doc_id: str
    system_id: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime | None = None
