import json
from urllib import error, request

import numpy as np

from app.config import get_settings
from app.schemas.doc import SourceDoc


class ElasticsearchVectorRequestError(RuntimeError):
    def __init__(self, method: str, path: str, status_code: int, response_body: str) -> None:
        self.method = method
        self.path = path
        self.status_code = status_code
        self.response_body = response_body
        detail = response_body or "<empty response body>"
        super().__init__(
            f"Elasticsearch {method} {path} failed with HTTP {status_code}: {detail}"
        )


class LocalElasticsearchVectorIndexer:
    """Rebuild one unified ES 7.10 dense_vector index from all source documents."""

    def __init__(
        self,
        index_name: str | None = None,
        base_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.base_url = (base_url or settings.LOCAL_ES_URL).rstrip("/")
        self.index_name = index_name or settings.LOCAL_ES_VECTOR_INDEX
        if not self.index_name.strip():
            raise ValueError("vector index_name must not be blank")

    def rebuild(self, docs: list[SourceDoc], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix")
        if len(docs) != int(embeddings.shape[0]):
            raise ValueError("Document count and embedding count must match")
        if not docs:
            raise ValueError("Cannot build vector index from an empty document set")

        self._delete_index_if_exists()
        self._request("PUT", f"/{self.index_name}", self._mapping(int(embeddings.shape[1])))
        self._bulk_index(docs, embeddings)
        self._request("POST", f"/{self.index_name}/_refresh")

    def _mapping(self, embedding_dim: int) -> dict:
        return {
            "settings": {
                "number_of_shards": self.settings.LOCAL_ES_SHARDS,
                "number_of_replicas": self.settings.LOCAL_ES_REPLICAS,
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "system_id": {"type": "keyword"},
                    "summary": {"type": "text", "index": False},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": embedding_dim,
                    },
                }
            },
        }

    def _bulk_index(self, docs: list[SourceDoc], embeddings: np.ndarray) -> None:
        batch_size = self.settings.LOCAL_ES_BULK_BATCH_SIZE
        total = len(docs)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            lines: list[str] = []
            for doc, vector in zip(docs[start:end], embeddings[start:end]):
                es_id = f"{doc.system_id}:{doc.doc_id}"
                lines.append(
                    json.dumps({"index": {"_index": self.index_name, "_id": es_id}})
                )
                lines.append(
                    json.dumps(
                        {
                            "doc_id": doc.doc_id,
                            "system_id": doc.system_id,
                            "summary": doc.summary,
                            "embedding": vector.tolist(),
                        },
                        ensure_ascii=False,
                    )
                )

            body = "\n".join(lines) + "\n"
            response = self._request_raw(
                "POST",
                "/_bulk",
                body,
                content_type="application/x-ndjson",
            )
            if response.get("errors"):
                failed_items = [
                    item
                    for item in response.get("items", [])
                    if item.get("index", {}).get("error") is not None
                ]
                raise RuntimeError(
                    "Elasticsearch vector bulk indexing failed "
                    f"for docs {start}:{end}: {failed_items[:3]}"
                )

    def _delete_index_if_exists(self) -> None:
        try:
            self._request("DELETE", f"/{self.index_name}")
        except ElasticsearchVectorRequestError as exc:
            if exc.status_code != 404:
                raise

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        return self._request_raw(method, path, data)

    def _request_raw(
        self,
        method: str,
        path: str,
        body: str | bytes | None = None,
        content_type: str = "application/json",
    ) -> dict:
        data = body.encode("utf-8") if isinstance(body, str) else body
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": content_type},
        )
        try:
            with request.urlopen(req, timeout=self.settings.LOCAL_ES_TIMEOUT_SECONDS) as response:
                payload = response.read().decode("utf-8")
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace").strip()
            raise ElasticsearchVectorRequestError(
                method=method,
                path=path,
                status_code=exc.code,
                response_body=response_body,
            ) from exc
        return json.loads(payload) if payload else {}
