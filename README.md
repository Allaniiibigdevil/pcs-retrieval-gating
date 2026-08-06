# Retrieval Gating Service · Reranker

本分支实现本地多数据源检索门控。目标不是返回最终答案，而是根据用户任务判断应该继续检索哪些 `system_id`。

```text
用户任务
  -> query rewrite
  -> 每个 source 并行执行自己的 ES 与 FAISS 检索
  -> 每个 source 自适应筛选向量候选
  -> 合并所有 source 候选
  -> GTE cross-encoder 一次批量 rerank
  -> source 级 max-pool
  -> selected_systems
```

## 核心约束

- 每个 source 使用独立 Elasticsearch index。
- 每个 source 使用独立 FAISS index 和独立 doc-id mapping。
- `docs.jsonl`、embedding 模型实例和 reranker 模型实例可以共享。
- 在线 FAISS 检索不会加载全局索引后再按 source 过滤。
- 所有 source 参数必须在 `config/sources.json` 中显式声明，不从全局 `.env` 回退。
- 最终文档分数只由 reranker 产生；BM25 与 FAISS 只负责召回和调试。
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
RERANKER_MODEL_PATH=Alibaba-NLP/gte-multilingual-reranker-base
RERANKER_LOCAL_FILES_ONLY=true
RERANKER_DEVICE=auto
RERANKER_BATCH_SIZE=8
RERANKER_MAX_LENGTH=512
```

每个 source 的索引和决策参数放在 `config/sources.json`：

```json
{
  "sources": {
    "notepad": {
      "enabled": true,
      "es_index": "pcs_retrieval_notepad",
      "faiss_index_path": "data/artifacts/notepad/faiss.index",
      "faiss_doc_ids_path": "data/artifacts/notepad/faiss_doc_ids.json",
      "es_top_k": 50,
      "faiss_top_k": 50,
      "evidence_docs_per_system": 3,
      "faiss_preferred_score_threshold": 0.60,
      "faiss_min_score_threshold": 0.30,
      "faiss_target_hits": 10,
      "reranker_score_threshold": 0.55
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
7. 使用 `--index-es` 时，为每个 source 重建独立 ES index；空 source 也会重建为空索引。

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

## 在线 FAISS 检索

`LocalFaissRetriever` 只加载当前 source 配置的 FAISS 文件。加载时会校验：

- index 向量数与 doc-id mapping 数量一致；
- mapping 中没有重复 doc-id；
- mapping 中每个文档都存在于 `docs.jsonl`；
- mapping 中所有文档都属于当前 source；
- mapping 与 `docs.jsonl` 中该 source 的文档集合完全一致。

任一条件不满足都会拒绝检索，要求重新建索引。

调试接口必须显式传 `source_id`：

```bash
curl -X POST http://127.0.0.1:8000/v1/search/vector \
  -H "Content-Type: application/json" \
  -d '{"source_id":"notepad","query":"海鲜过敏","top_k":5}'
```

## Query rewrite 与粗召回

`rewrite_queries()` 当前是 identity 占位实现。`prepare_queries()` 始终把原始 task 放在第 0 条，然后追加非空、去重后的 rewrite。

所有 source、所有 query、ES 与 FAISS 检索并行执行。单条 query 失败不会中断同通道其他 query。

ES 使用：

```text
multi_match.type = cross_fields
operator = or
minimum_should_match = 1<2
summary boost = 1.0
keywords boost = 1.0
```

同一文档跨 rewrite 聚合时：

- ES 不比较不同 query 的 raw `_score`，优先使用更好的 query 内 rank；
- FAISS 保留最高 vector score；
- 合并 `matched_queries`、`matched_query_indexes`、matched keywords 和 highlight。

## 自适应向量候选

每个 source 只对自己的 FAISS hits 做筛选：

```text
eligible = hits where vector_score >= faiss_min_score_threshold

if count(score >= faiss_preferred_score_threshold) >= faiss_target_hits:
    effective_threshold = faiss_preferred_score_threshold
else:
    effective_threshold = max(
        faiss_min_score_threshold,
        score_of_target_rank_or_last_available
    )

selected_vector_hits = eligible where score >= effective_threshold
```

不会根据阈值重复查询 FAISS，也不会把不同 source 的向量候选混在一起计算阈值。

## Reranker

所有 source 的候选合并后，一次批量送入 `gte-multilingual-reranker-base`：

```text
passage(doc) = "关键词：" + keywords + "\n摘要：" + summary
raw_logit(doc) = GTE(task, passage(doc))
reranker_score(doc) = sigmoid(raw_logit(doc))
```

输入不包含 `system_id`、BM25 分数、vector 分数或粗召回 rank。

最终 source 决策：

```text
source_score = max(reranker_score of source documents)
selected = source_score >= source.reranker_score_threshold
```

reranker score 必须是有限的 0～1 值。该值未经业务标注校准，不能直接解释为真实概率。

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
