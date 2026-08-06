import numpy as np
import pytest

from app.storage.local_artifact_store import LocalArtifactStore


def test_store_writes_and_loads_distinct_source_faiss_artifacts(tmp_path) -> None:
    faiss = pytest.importorskip("faiss")
    store = LocalArtifactStore(artifact_dir=tmp_path / "shared")

    first_index = faiss.IndexFlatIP(2)
    first_index.add(np.asarray([[1.0, 0.0]], dtype="float32"))
    second_index = faiss.IndexFlatIP(2)
    second_index.add(np.asarray([[0.0, 1.0]], dtype="float32"))

    first_path = tmp_path / "photo" / "faiss.index"
    first_ids_path = tmp_path / "photo" / "faiss_doc_ids.json"
    second_path = tmp_path / "notepad" / "faiss.index"
    second_ids_path = tmp_path / "notepad" / "faiss_doc_ids.json"

    store.save_faiss(
        first_index,
        ["photo-1"],
        index_path=first_path,
        doc_ids_path=first_ids_path,
    )
    store.save_faiss(
        second_index,
        ["notepad-1"],
        index_path=second_path,
        doc_ids_path=second_ids_path,
    )

    loaded_first, first_ids = store.load_faiss(
        index_path=first_path,
        doc_ids_path=first_ids_path,
    )
    loaded_second, second_ids = store.load_faiss(
        index_path=second_path,
        doc_ids_path=second_ids_path,
    )

    assert first_path != second_path
    assert first_ids_path != second_ids_path
    assert int(loaded_first.ntotal) == 1
    assert int(loaded_second.ntotal) == 1
    assert first_ids == ["photo-1"]
    assert second_ids == ["notepad-1"]
