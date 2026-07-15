import numpy as np
import pytest

from app.embedding.embedding_service import BGEEmbeddingService


class RecordingModel:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        self.inputs.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32)


@pytest.mark.asyncio
async def test_bge_instruction_is_added_only_to_query() -> None:
    service = BGEEmbeddingService(
        model_path="unused",
        query_instruction="为这个句子生成表示以用于检索相关文章：",
    )
    model = RecordingModel()
    service._model = model

    await service.embed_query("去年京都的红色寺庙")
    await service.embed_documents(["summary: 京都旅行照片\nkeywords: 京都, 寺庙"])

    assert model.inputs == [
        ["为这个句子生成表示以用于检索相关文章：去年京都的红色寺庙"],
        ["summary: 京都旅行照片\nkeywords: 京都, 寺庙"],
    ]
