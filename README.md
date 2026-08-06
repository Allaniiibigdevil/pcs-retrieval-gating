# Retrieval Gating Service · Score Fusion

本分支实现本地多数据源检索门控。目标不是返回最终答案，而是根据用户任务判断应该继续检索哪些 `system_id`。

```text
用户任务
  -> query rewrite
  -> 每个 source 并行执行自己的 ES 与 FAISS 检索
  -> 同一 source 内合并多 query、多通道候选
  -> 文档级 score fusion
  -> source 级 max-pool
  -> selected_systems
```

## 核心约束

- 每个 source 使用独立 Elasticsearch index。
- 每个 source 使用独立 FAISS index 和独立 doc-id mapping。
- `docs.jsonl` 与 embedding 模型实例可以共享，但在线 FAISS 检索不会加载全局索引后再过滤。
- 所有 source 配置都必须在 `config/sources.json` 中显式声明，不从全局 `.env` 回退。
- 当前分支不使用 FAISS 最低分阈值或自适应阈值。
- `doc_id` 在整个文档集合中必须全局唯一。

## 配置

服务级配置放在 `.env`：

```env
APP_NAME=retrieval-gating-service
LOCAL_RAW_DOCS_PATH=data/raw/docs.jsonl
LOCAL_ARTIFACT_DIR=data/artifacts
LOCAL_SOURCE_CONFIG_PATH=config/sources.json
LOCAL_ES_URL=http://127.0.0.1:9200
LOCAL_ES_ANALYZER=standard
LOCAL_ES_SEARCH_ANALYZER=standard
EMBEDDING_PROVIDER=bge
EMBEDDING_MODEL_PATH=BAAI/bge-small-zh-v1.5
EMBEDDING_DIM=512
REWRITTEN_QUERY_ES_WEIGHT=1.0
```

每个 source 的索引和评分参数放在 `config/sources.json`：

```json
{
  "sources": {
    "photo": {
      "enabled": true,
      "es_index": "pcs_retrieval_photo",
      "faiss_index_path": "data/artifacts/photo/faiss.index",
      "faiss_doc_ids_path": "data/artifacts/photo/faiss_doc_ids.json",
      "es_top_k": 50,
      "faiss_top_k": 50,
      "evidence_docs_per_system": 3,
      "selection_threshold": 0.55,
      "es_score_weight": 0.45,
      "agreement_weight": 0.20,
      "semantic_match_threshold": 0.30,
      "lexical_match_threshold": 0.25
    }
  }
}
```

不同 source 不允许复用 ES index、FAISS index path 或 FAISS doc-id mapping path。

## 离线建索引

```bash
uv run python -m app.offline.build_index --docs examples/docs.jsonl
```

同时重建每个 source 的 ES index：

```bash
uv run python -m app.offline.build_index \
  --docs examples/docs.jsonl \
  --index-es
```

构建流程：

1. 校验 `doc_id` 全局唯一，并确认所有 `system_id` 已配置。
2. 一次批量生成全部文档 embedding。
3. 按 source 切分向量。
4. 为每个 source 分别创建 `IndexFlatIP`。
5. 写入该 source 的 `faiss.index` 和 `faiss_doc_ids.json`。
6. 写入共享 `docs.jsonl` 与 `manifest.json`。
7. 使用 `--index-es` 时，为每个 source 重建独立 ES index；空 source 也会重建为空索引，避免残留旧数据。

示例输出结构：

```text
data/artifacts/
  docs.jsonl
  manifest.json
  photo/
    faiss.index
    faiss_doc_ids.json
  notepad/
    faiss.index
    faiss_doc_ids.json
```

## 在线检索

`LocalFaissRetriever` 只加载当前 source 配置的 FAISS 文件。加载时会校验：

- index 向量数与 doc-id mapping 数量一致；
- mapping 中没有重复 doc-id；
- mapping 中每个文档都存在于 `docs.jsonl`；
- mapping 中所有文档都属于当前 source；
- mapping 与 `docs.jsonl` 中该 source 的文档集合完全一致。

任一条件不满足都会拒绝检索，要求重新建索引。

ES 使用：

```text
multi_match.type = cross_fields
operator = or
minimum_should_match = 1<2
summary boost = 1.0
keywords boost = 1.0
```

调试接口必须显式传 `source_id`：

```bash
curl -X POST http://127.0.0.1:8000/v1/search/vector \
  -H "Content-Type: application/json" \
  -d '{"source_id":"notepad","query":"海鲜过敏","top_k":5}'
```

## Query rewrite 与多 query 聚合

`rewrite_queries()` 当前是 identity 占位实现。`prepare_queries()` 始终把原始 task 放在第 0 条，然后追加非空、去重后的 rewrite。

所有 ES 与 FAISS query 并行执行。单条 query 失败不会中断同通道其他 query；只有某个 source 的 ES 和 FAISS 全部失败时，该 source 才算失败。

不同 query 的 ES raw `_score` 不直接比较。每个 query 内先归一化：

```text
raw_norm_q(d) = bm25_score_q(d) / max_x bm25_score_q(x)
query_weight_q = 1.0                       if q is original task
query_weight_q = REWRITTEN_QUERY_ES_WEIGHT otherwise
lexical_q(d) = raw_norm_q(d) * query_weight_q
lexical(d) = max_q lexical_q(d)
semantic(d) = max_q clamp(vector_score_q(d), 0, 1)
```

响应会保留：

- `bm25_score`：原始 ES `_score`；
- `bm25_score_norm`：query 内归一化并乘 rewrite 权重后的词法分；
- `matched_queries`：命中过该文档的所有 query；
- 合并后的 matched keywords 与 highlight。

## Score fusion

每个 source 使用自己的参数：

```text
base = max(
  semantic,
  es_score_weight * lexical
)

agreement = agreement_weight * sqrt(semantic * lexical)
  if semantic >= semantic_match_threshold
  and lexical >= lexical_match_threshold
  else 0

doc_strength = clamp(base + agreement, 0, 1)
source_confidence = max(doc_strength of source documents)
selected = source_confidence >= selection_threshold
```

ES-only 文档的最高基础贡献受 `es_score_weight` 限制。FAISS 不设硬阈值，低分向量只会以其实际分数参与融合。

## 启动

```bash
uv sync --extra dev
uv run uvicorn app.main:app --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

前端：

```text
http://127.0.0.1:8000/frontend/
```

决策接口：

```bash
curl -X POST http://127.0.0.1:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task_001","task":"我可以吃海鲜吗？"}'
```

## 检查

```bash
uv run pytest -p no:cacheprovider
uv run ruff check --no-cache .
uv run python -m compileall app tests
```
