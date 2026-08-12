from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class Request(BaseModel):
    texts: str = ""
    is_normalize: int = 0


class Content(BaseModel):
    embeddings: Optional[List[List[float]]] = None

    model_config = ConfigDict(extra="forbid")


class Result(BaseModel):
    code: str
    des: str
    content: Optional[List[Content]] = None

    model_config = ConfigDict(extra="forbid")


class Response(BaseModel):
    result: Optional[Result] = None

    model_config = ConfigDict(extra="forbid")
