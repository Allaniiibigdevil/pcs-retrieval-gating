import json
from collections.abc import Iterable
from urllib import error, request

from app.config import get_settings
from app.schemas.doc import SourceDoc
from app.utils.progress import render_progress


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
    def __init__(
        self,
        index_name: str,
        base_url: str | None = None,
        *,
        show_progress: bool = False,
    ) -> None:
        if not index_name.strip():
            raise ValueError("index_name must not be blank")
        settings = get_settings()
        self.settings = settings
        self.base_url = (base_url or settings.LOCAL_ES_URL).rstrip("/")
        self.index_name = index_name
        self.show_progress = show_progress

    def rebuild(self, docs: list[SourceDoc]) -> None:
        self.rebuild_iter(docs, total=len(docs))

    def rebuild_iter(self, docs: Iterable[SourceDoc], *, total: int | None = None) -> None:
        self._delete_index_if_exists()
        self._request("PUT", f"/{self.index_name}", self._mapping())

        batch_size = self.settings.LOCAL_ES_BULK_BATCH_SIZE
        batch: list[SourceDoc] = []
        indexed = 0
        for doc in docs:
            batch.append(doc)
            if len(batch) >= batch_size:
                self._bulk_index_batch(batch, start=indexed, total=total)
                indexed += len(batch)
                batch.clear()
        if batch:
            self._bulk_index_batch(batch, start=indexed, total=total)
            indexed += len(batch)

        if indexed == 0 and self.show_progress:
            render_progress("ES indexing", 0, 0, detail=self.index_name)
        self._request("POST", f"/{self.index_name}/_refresh")

    def _mapping(self) -> dict:
        text_field = {
            "type": "text",
            "analyzer": self.settings.LOCAL_ES_ANALYZER,
            "search_analyzer": self.settings.LOCAL_ES_SEARCH_ANALYZER,
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
                    "updated_at": {"type": "date"},
                }
            },
        }

    def _bulk_index_batch(
        self,
        batch: list[SourceDoc],
        *,
        start: int,
        total: int | None,
    ) -> None:
        lines: list[str] = []
        for doc in batch:
            lines.append(
                json.dumps({"index": {"_index": self.index_name, "_id": doc.doc_id}})
            )
            lines.append(json.dumps(self._source(doc), ensure_ascii=False))
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
                "Elasticsearch bulk indexing failed "
                f"for docs {start}:{start + len(batch)}: {failed_items[:3]}"
            )
        if self.show_progress:
            current = start + len(batch)
            render_progress(
                "ES indexing",
                current,
                total if total is not None else current,
                detail=self.index_name,
            )

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
