import json

import numpy as np

from app.offline.local_es_vector_indexer import LocalElasticsearchVectorIndexer
from app.schemas.doc import SourceDoc


class RecordingVectorIndexer(LocalElasticsearchVectorIndexer):
    def __init__(self) -> None:
        super().__init__(index_name="vectors-test", base_url="http://localhost:9200")
        self.requests: list[tuple[str, str, dict | None]] = []
        self.raw_requests: list[tuple[str, str, str | bytes | None, str]] = []

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        self.requests.append((method, path, body))
        return {}

    def _request_raw(
        self,
        method: str,
        path: str,
        body: str | bytes | None = None,
        content_type: str = "application/json",
    ) -> dict:
        self.raw_requests.append((method, path, body, content_type))
        return {"errors": False}


def _doc(doc_id: str, system_id: str) -> SourceDoc:
    return SourceDoc(
        doc_id=doc_id,
        system_id=system_id,
        summary=f"summary {system_id}",
        keywords=[system_id],
    )


def test_rebuild_creates_one_dense_vector_index_for_all_sources() -> None:
    indexer = RecordingVectorIndexer()
    docs = [_doc("same-id", "memo"), _doc("same-id", "photo")]
    embeddings = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype="float32",
    )

    indexer.rebuild(docs, embeddings)

    put_request = next(item for item in indexer.requests if item[:2] == ("PUT", "/vectors-test"))
    mapping = put_request[2]
    assert mapping is not None
    assert mapping["mappings"]["properties"]["embedding"] == {
        "type": "dense_vector",
        "dims": 4,
    }
    assert mapping["mappings"]["properties"]["system_id"] == {"type": "keyword"}

    bulk_request = next(item for item in indexer.raw_requests if item[1] == "/_bulk")
    body = bulk_request[2]
    assert isinstance(body, str)
    lines = [json.loads(line) for line in body.strip().splitlines()]

    assert lines[0]["index"]["_id"] == "memo:same-id"
    assert lines[1]["system_id"] == "memo"
    assert lines[1]["embedding"] == [1.0, 0.0, 0.0, 0.0]
    assert lines[2]["index"]["_id"] == "photo:same-id"
    assert lines[3]["system_id"] == "photo"
    assert lines[3]["embedding"] == [0.0, 1.0, 0.0, 0.0]
