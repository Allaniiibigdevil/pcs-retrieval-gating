import json
from urllib import error, request

from app.config import get_settings
from app.schemas.doc import SourceDoc


class ElasticsearchRequestError(RuntimeError):
    def __init__(self, method: str, path: str, status_code: int, response_body: str) -> None:
        self.method = method
        self.path = path
        self.status_code = status_code
        self.response_body = response_body
        detail = response_body or "<empty response body>"
        super().__init__(
            f"Elasticsearch {method} {path} failed with HTTP {status_code}: {detail}"
        )


class LocalElasticsearchIndexer:
    def __init__(self, base_url: str | None = None, index_name: str | None = None) -> None:
        settings = get_settings()
        self.settings = settings
        self.base_url = (base_url or settings.LOCAL_ES_URL).rstrip("/")
        self.index_name = index_name or settings.LOCAL_ES_INDEX

    def rebuild(self, docs: list[SourceDoc]) -> None:
        self._delete_index_if_exists()
        self._request("PUT", f"/{self.index_name}", self._mapping())
        self._bulk_index(docs)
        self._request("POST", f"/{self.index_name}/_refresh")

    def _mapping(self) -> dict:
        analyzer = self.settings.LOCAL_ES_ANALYZER
        search_analyzer = self.settings.LOCAL_ES_SEARCH_ANALYZER
        text_field = {
            "type": "text",
            "analyzer": analyzer,
            "search_analyzer": search_analyzer,
        }
        return {
            "settings": {
                "number_of_shards": self.settings.LOCAL_ES_SHARDS,
                "number_of_replicas": self.settings.LOCAL_ES_REPLICAS,
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "system_id": {"type": "keyword"},
                    "summary": text_field,
                    "keywords": text_field,
                    "metadata": {"enabled": False},
                }
            },
        }

    def _bulk_index(self, docs: list[SourceDoc]) -> None:
        lines: list[str] = []
        for doc in docs:
            lines.append(json.dumps({"index": {"_index": self.index_name, "_id": doc.doc_id}}))
            lines.append(json.dumps(self._source(doc), ensure_ascii=False))
        body = "\n".join(lines) + "\n"
        response = self._request_raw("POST", "/_bulk", body, content_type="application/x-ndjson")
        if response.get("errors"):
            failed_items = [
                item
                for item in response.get("items", [])
                if item.get("index", {}).get("error") is not None
            ]
            raise RuntimeError(f"Elasticsearch bulk indexing failed: {failed_items[:3]}")

    def _source(self, doc: SourceDoc) -> dict:
        return {
            "doc_id": doc.doc_id,
            "system_id": doc.system_id,
            "summary": doc.summary,
            "keywords": doc.keywords,
            "metadata": doc.metadata,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        }

    def _delete_index_if_exists(self) -> None:
        try:
            self._request("DELETE", f"/{self.index_name}")
        except ElasticsearchRequestError as exc:
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
            raise ElasticsearchRequestError(
                method=method,
                path=path,
                status_code=exc.code,
                response_body=response_body,
            ) from exc
        return json.loads(payload) if payload else {}
