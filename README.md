# Retrieval Gating Service

这是一个本地优先的检索门控原型。目标不是做最终文档排序，而是根据用户任务判断应该检索哪些 `system_id`。

```text
用户任务
  -> query rewrite
  -> 多 query 并行执行 ES / FAISS 检索
  -> 同通道按 doc_id 聚合
  -> ES + FAISS 证据融合
  -> 聚合到 system_id
  -> 输出 selected_systems
```

当前分支使用本地 Elasticsearch 做词法检索、本地 FAISS 做向量检索，embedding 默认使用 `BAAI/bge-small-zh-v1.5`。

## 查询与召回

`rewrite_queries()` 当前是占位实现，默认返回原始 task。后续可以返回多个改写 query。所有 ES 和 FAISS 查询会并行执行；单个 query 失败不会中断其他查询，只有两个通道全部失败时才返回错误。

ES 查询参数与 `rrf` / `reranker` 分支保持一致：

```text
multi_match.type = cross_fields
operator = or
minimum_should_match = 1<2
summary boost = 1.0
keywords boost = 1.0
```

ES 和 FAISS 的 top-k 分开配置：

```env
ES_TOP_K_DOCS=50
FAISS_TOP_K_DOCS=50
EVIDENCE_DOCS_PER_SYSTEM=3
```

当前 score-fusion 分支**不对 FAISS 结果设置最低分或自适应阈值**。原因是最终决策直接使用向量分，并且 agreement 另有最低语义门槛；过早删除中等向量分会让原本的双路证据退化成 ES-only。

## 多 query 聚合

不同 rewrite 的 ES `_score` 不直接横向比较。每个 query 内先归一化：

```text
bm25_score_norm_q(d) = bm25_score_q(d) / max_x bm25_score_q(x)
```

同一文档跨 rewrite 聚合：

```text
lexical(d) = max_q bm25_score_norm_q(d)
semantic(d) = max_q clamp(vector_score_q(d), 0, 1)
```

原始 ES `_score` 保留在 `bm25_score`，归一化结果保存在 `bm25_score_norm`。响应中的 `matched_queries` 会列出命中过该文档的 rewrite query。

## 评分公式

```text
base_score = max(
  semantic,
  ES_SCORE_WEIGHT * lexical
)

agreement_score = AGREEMENT_WEIGHT * sqrt(semantic * lexical)
  if semantic >= SEMANTIC_MATCH_THRESHOLD
  and lexical >= LEXICAL_MATCH_THRESHOLD
  else 0

doc_strength = clamp(base_score + agreement_score, 0, 1)
system_confidence = max(doc_strength of docs in the system)
selected = system_confidence >= SYSTEM_SELECTION_THRESHOLD
```

默认配置：

```env
SYSTEM_SELECTION_THRESHOLD=0.60
ES_SCORE_WEIGHT=0.55
AGREEMENT_WEIGHT=0.20
SEMANTIC_MATCH_THRESHOLD=0.30
LEXICAL_MATCH_THRESHOLD=0.30
```

## 离线建索引

```bash
uv run python -m app.offline.build_index --docs examples/docs.jsonl
```

同时重建本地 ES 索引：

```bash
uv run python -m app.offline.build_index --docs examples/docs.jsonl --index-es
```

FAISS 使用归一化 BGE embedding 和 `IndexFlatIP`。

## 启动

```bash
uv run uvicorn app.main:app --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

前端控制台：

```text
http://127.0.0.1:8000/frontend/
```

## 决策接口

```bash
curl -X POST http://127.0.0.1:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task_001","task":"我可以吃海鲜吗？"}'
```

响应包含：

- `rewritten_queries`：本次实际执行的 query 列表；
- `selected_systems`：最终建议检索的系统；
- `decisions`：所有候选系统的 confidence 和证据；
- `evidence_docs[].bm25_score`：原始 ES `_score`；
- `evidence_docs[].bm25_score_norm`：query 内归一化词法分；
- `evidence_docs[].matched_queries`：命中过该文档的 rewrite query。

## 检查

```bash
uv run pytest -p no:cacheprovider
uv run ruff check --no-cache .
uv run python -m compileall app tests
```
