import math

import pytest

from app.embedding.embedding_service import MockEmbeddingService
from app.offline.local_index_builder import LocalIndexBuilder
from app.retrieval.local_faiss_retriever import LocalFaissRetriever
from app.schemas.doc import SourceDoc
from app.storage.local_artifact_store import LocalArtifactStore


@pytest.mark.asyncio
async def test_local_index_builder_outputs_searchable_artifacts(tmp_path) -> None:
    docs = [
        SourceDoc(
            doc_id="memo_doc_001",
            system_id="memo_system",
            summary="The memo records a shanghai business trip meeting plan.",
            keywords=["memo", "shanghai", "trip", "meeting"],
        ),
        SourceDoc(
            doc_id="album_doc_001",
            system_id="album_system",
            summary="The album contains meeting photos from the shanghai trip.",
            keywords=["album", "photo", "shanghai", "trip", "meeting"],
        ),
    ]
    artifact_store = LocalArtifactStore(tmp_path)
    embedding_service = MockEmbeddingService(dim=16)

    result = await LocalIndexBuilder(
        artifact_store=artifact_store,
        embedding_service=embedding_service,
    ).build(docs)

    assert result.doc_count == 2
    assert result.embedding_dim == 16
    assert artifact_store.docs_path.exists()
    assert artifact_store.faiss_path.exists()
    assert artifact_store.faiss_doc_ids_path.exists()
    assert artifact_store.manifest_path.exists()

    vector_hits = await LocalFaissRetriever(
        artifact_store,
        embedding_service,
        min_score_threshold=-1.0,
    ).search(
        "shanghai trip meeting",
        top_k=2,
    )
    assert len(vector_hits) == 2
    assert all(hit.vector_rank is not None for hit in vector_hits)
    assert all(
        hit.vector_score is not None and math.isfinite(hit.vector_score) for hit in vector_hits
    )

    filtered_hits = await LocalFaissRetriever(
        artifact_store,
        embedding_service,
        min_score_threshold=1.0,
    ).search("shanghai trip meeting", top_k=2)
    assert filtered_hits == []

    mismatched_retriever = LocalFaissRetriever(
        artifact_store,
        MockEmbeddingService(dim=8),
        min_score_threshold=-1.0,
    )
    with pytest.raises(
        RuntimeError,
        match=r"index dimension is 16, query embedding dimension is 8",
    ):
        await mismatched_retriever.search("shanghai trip meeting", top_k=2)
