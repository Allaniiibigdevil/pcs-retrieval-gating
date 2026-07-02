# Retrieval Gating Service

Python FastAPI MVP for turning document-level retrieval evidence into system-level retrieval decisions.

The important product boundary is:

```text
user task -> relevant document evidence -> subsystem decision
```

Documents are evidence. The final output is which source subsystems should be retrieved, not a document ranking.

## Structure

```text
app/
  api/            FastAPI routers
  schemas/        Pydantic request/response models
  embedding/      Mock embedding abstraction
  indexing/       Elasticsearch and GaussDB write adapters
  retrieval/      Elasticsearch and vector search adapters, candidate merge
  decision/       normalization, evidence, aggregation, decision engine
  utils/          timing and helpers
tests/            unit tests for core decision logic
```

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Environment Variables

See `.env.example` for all settings. The main values are:

```env
ES_URL=http://localhost:9200
ES_INDEX_NAME=source_docs
GAUSSDB_DSN=postgresql://user:password@localhost:5432/retrieval
GAUSSDB_VECTOR_TABLE=source_doc_vectors
EMBEDDING_PROVIDER=mock
EMBEDDING_DIM=384
DEFAULT_TOP_K_DOCS=50
DEFAULT_MAX_SYSTEMS=5
RETRIEVE_THRESHOLD=0.80
MAYBE_RETRIEVE_THRESHOLD=0.55
```

## Upsert Examples

Memo system:

```bash
curl -X POST http://localhost:8000/v1/docs/upsert ^
  -H "Content-Type: application/json" ^
  -d "{\"doc_id\":\"memo_doc_001\",\"system_id\":\"memo_system\",\"summary\":\"用户在备忘录中记录了上海出差计划，包括会议时间、客户名称和待办事项。\",\"keywords\":[\"备忘录\",\"出差\",\"上海\",\"会议\",\"待办事项\"],\"metadata\":{\"source\":\"memo_app\",\"doc_type\":\"note_summary\"},\"updated_at\":\"2026-07-02T10:00:00+08:00\"}"
```

Album system:

```bash
curl -X POST http://localhost:8000/v1/docs/upsert ^
  -H "Content-Type: application/json" ^
  -d "{\"doc_id\":\"album_doc_001\",\"system_id\":\"album_system\",\"summary\":\"相册中包含用户在上海出差期间拍摄的会议白板、客户合影和酒店照片。\",\"keywords\":[\"相册\",\"照片\",\"上海\",\"出差\",\"会议白板\",\"客户合影\"],\"metadata\":{\"source\":\"photo_app\",\"doc_type\":\"album_summary\"},\"updated_at\":\"2026-07-02T10:05:00+08:00\"}"
```

## Decision Example

```bash
curl -X POST http://localhost:8000/v1/decide ^
  -H "Content-Type: application/json" ^
  -d "{\"task_id\":\"task_001\",\"task\":\"我想查一下上海出差相关的会议照片和备忘录\",\"top_k_docs\":50,\"max_systems\":5}"
```

## MVP Notes

- Uses deterministic mock embeddings by default.
- Does not use LLM query rewrite.
- Does not use rerank or cross-encoder rerank.
- Current scoring is intentionally simple and replaceable.
- Elasticsearch and GaussDB details are isolated in adapters.
- Core decision logic can run in unit tests without ES or GaussDB.

## Tests

```bash
pytest
```
