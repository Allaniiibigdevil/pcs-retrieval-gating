from pydantic import BaseModel, ConfigDict, Field


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1)
    doc: str = Field(min_length=1)


class RerankResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float
