from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceDoc(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    doc_id: str = Field(min_length=1)
    system_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime | None = None

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            keyword = value.strip()
            if not keyword or keyword in seen:
                continue
            seen.add(keyword)
            normalized.append(keyword)
        return normalized
