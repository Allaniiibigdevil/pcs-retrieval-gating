import threading

import numpy as np
import pytest

from app.embedding.embedding_service import BGEEmbeddingService, build_embedding_text
from app.schemas.doc import SourceDoc


class FakeModel:
    def __init__(self) -> None:
        self.thread_id: int | None = None

    def encode(self, texts, **kwargs):
        del kwargs
        self.thread_id = threading.get_ident()
        return np.ones((len(texts), 2), dtype="float32")


@pytest.mark.asyncio
async def test_bge_encode_runs_in_a_worker_thread() -> None:
    service = BGEEmbeddingService("unused")
    fake_model = FakeModel()
    service._model = fake_model
    event_loop_thread = threading.get_ident()

    vectors = await service.embed_batch(["one", "two"])

    assert fake_model.thread_id is not None
    assert fake_model.thread_id != event_loop_thread
    assert vectors == [[1.0, 1.0], [1.0, 1.0]]


def test_embedding_text_contains_content_but_not_source_identity() -> None:
    doc = SourceDoc(
        doc_id="doc-1",
        system_id="notepad",
        summary="海鲜过敏记录",
        keywords=["海鲜", "过敏"],
    )

    text = build_embedding_text(doc)

    assert "海鲜过敏记录" in text
    assert "海鲜, 过敏" in text
    assert "notepad" not in text
