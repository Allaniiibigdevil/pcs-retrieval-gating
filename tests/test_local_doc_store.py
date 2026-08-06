import pytest

from app.schemas.doc import SourceDoc
from app.storage.local_doc_store import LocalDocStore


def test_source_doc_strips_fields_and_deduplicates_keywords() -> None:
    doc = SourceDoc(
        doc_id="  doc-1  ",
        system_id="  memo  ",
        summary="  summary  ",
        keywords=[" allergy ", "allergy", "", " seafood "],
    )

    assert doc.doc_id == "doc-1"
    assert doc.system_id == "memo"
    assert doc.summary == "summary"
    assert doc.keywords == ["allergy", "seafood"]


def test_local_doc_store_rejects_duplicate_doc_ids(tmp_path) -> None:
    store = LocalDocStore(tmp_path / "docs.jsonl")
    docs = [
        SourceDoc(doc_id="same", system_id="memo", summary="first"),
        SourceDoc(doc_id="same", system_id="photo", summary="second"),
    ]

    with pytest.raises(ValueError, match="Duplicate doc_id"):
        store.save_all(docs)
