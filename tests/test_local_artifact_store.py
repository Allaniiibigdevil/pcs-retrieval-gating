from types import SimpleNamespace

from app.storage.local_artifact_store import LocalArtifactStore


def _settings(artifact_dir, shared_dir):
    return SimpleNamespace(
        LOCAL_ARTIFACT_DIR=str(artifact_dir),
        LOCAL_FAISS_INDEX_PATH=str(shared_dir / "shared.index"),
        LOCAL_FAISS_DOC_IDS_PATH=str(shared_dir / "shared_doc_ids.json"),
    )


def test_default_store_uses_configured_shared_faiss_paths(monkeypatch, tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    shared_dir = tmp_path / "shared-faiss"
    monkeypatch.setattr(
        "app.storage.local_artifact_store.get_settings",
        lambda: _settings(artifact_dir, shared_dir),
    )

    store = LocalArtifactStore()

    assert store.artifact_dir == artifact_dir
    assert store.faiss_path == shared_dir / "shared.index"
    assert store.faiss_doc_ids_path == shared_dir / "shared_doc_ids.json"


def test_artifact_directory_does_not_override_shared_faiss_paths(monkeypatch, tmp_path) -> None:
    configured_dir = tmp_path / "configured"
    shared_dir = tmp_path / "shared"
    monkeypatch.setattr(
        "app.storage.local_artifact_store.get_settings",
        lambda: _settings(configured_dir, shared_dir),
    )

    store = LocalArtifactStore(artifact_dir=tmp_path / "other-artifacts")

    assert store.docs_path == tmp_path / "other-artifacts" / "docs.jsonl"
    assert store.faiss_path == shared_dir / "shared.index"
    assert store.faiss_doc_ids_path == shared_dir / "shared_doc_ids.json"
