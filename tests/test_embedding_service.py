import threading

import numpy as np
import pytest

from app.embedding.embedding_service import BGEEmbeddingService, build_embedding_text
from app.schemas.doc import SourceDoc


class FakeModel:
    def __init__(self) -> None:
        self.thread_id: int | None = None
        self.kwargs: dict = {}

    def encode(self, texts, **kwargs):
        self.thread_id = threading.get_ident()
        self.kwargs = dict(kwargs)
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
    assert fake_model.kwargs["show_progress_bar"] is False
    assert vectors == [[1.0, 1.0], [1.0, 1.0]]


def test_embedding_text_matches_legacy_format() -> None:
    doc = SourceDoc(
        doc_id="doc-1",
        system_id="notepad",
        summary="海鲜过敏记录",
        keywords=["海鲜", "过敏"],
    )

    text = build_embedding_text(doc)

    assert "system: notepad" in text
    assert "summary: 海鲜过敏记录" in text
    assert "keywords: 海鲜, 过敏" in text
