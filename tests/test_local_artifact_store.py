import numpy as np
import pytest

from app.storage.local_artifact_store import LocalArtifactStore


def test_store_writes_and_loads_distinct_source_faiss_artifacts(tmp_path) -> None:
    faiss = pytest.importorskip("faiss")
    store = LocalArtifactStore(artifact_dir=tmp_path / "shared")

    photo_index = faiss.IndexFlatIP(2)
    photo_index.add(np.asarray([[1.0, 0.0]], dtype="float32"))
    notepad_index = faiss.IndexFlatIP(2)
    notepad_index.add(np.asarray([[0.0, 1.0]], dtype="float32"))

    photo_path = tmp_path / "photo" / "faiss.index"
    photo_ids_path = tmp_path / "photo" / "faiss_doc_ids.json"
    notepad_path = tmp_path / "notepad" / "faiss.index"
    notepad_ids_path = tmp_path / "notepad" / "faiss_doc_ids.json"

    store.save_faiss(
        photo_index,
        ["photo-1"],
        index_path=photo_path,
        doc_ids_path=photo_ids_path,
    )
    store.save_faiss(
        notepad_index,
        ["notepad-1"],
        index_path=notepad_path,
        doc_ids_path=notepad_ids_path,
    )

    loaded_photo, photo_ids = store.load_faiss(
        index_path=photo_path,
        doc_ids_path=photo_ids_path,
    )
    loaded_notepad, notepad_ids = store.load_faiss(
        index_path=notepad_path,
        doc_ids_path=notepad_ids_path,
    )

    assert photo_path != notepad_path
    assert photo_ids_path != notepad_ids_path
    assert int(loaded_photo.ntotal) == 1
    assert int(loaded_notepad.ntotal) == 1
    assert photo_ids == ["photo-1"]
    assert notepad_ids == ["notepad-1"]
