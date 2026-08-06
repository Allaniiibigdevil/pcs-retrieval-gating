from types import SimpleNamespace

from app.storage.local_artifact_store import LocalArtifactStore


def test_default_store_uses_configured_faiss_paths(monkeypatch, tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    shared_dir = tmp_path / "shared-faiss"
    monkeypatch.setattr(
        "app.storage.local_artifact_store.get_settings",
        lambda: SimpleNamespace(
            LOCAL_ARTIFACT_DIR=str(artifact_dir),
            LOCAL_FAISS_INDEX_PATH=str(shared_dir / "shared.index"),
            LOCAL_FAISS_DOC_IDS_PATH=str(shared_dir / "shared_doc_ids.json"),
        ),
    )

    store = LocalArtifactStore()

    assert store.artifact_dir == artifact_dir
    assert store.faiss_path == shared_dir / "shared.index"
    assert store.faiss_doc_ids_path == shared_dir / "shared_doc_ids.json"

    store.ensure_faiss_dirs()
    assert shared_dir.is_dir()


def test_explicit_artifact_dir_keeps_legacy_faiss_layout(monkeypatch, tmp_path) -> None:
    configured_dir = tmp_path / "configured"
    monkeypatch.setattr(
        "app.storage.local_artifact_store.get_settings",
        lambda: SimpleNamespace(
            LOCAL_ARTIFACT_DIR=str(configured_dir),
            LOCAL_FAISS_INDEX_PATH=str(configured_dir / "configured.index"),
            LOCAL_FAISS_DOC_IDS_PATH=str(configured_dir / "configured_doc_ids.json"),
        ),
    )
    explicit_dir = tmp_path / "explicit"

    store = LocalArtifactStore(explicit_dir)

    assert store.faiss_path == explicit_dir / "faiss.index"
    assert store.faiss_doc_ids_path == explicit_dir / "faiss_doc_ids.json"
