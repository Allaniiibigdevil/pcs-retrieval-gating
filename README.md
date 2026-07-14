# Retrieval Gating Service

这是一个本地优先的检索门控原型服务。它的目标不是给文档做最终排序，而是根据当前用户任务判断应该检索哪些子系统。

```text
用户任务
  -> ES / FAISS 找到相关文档证据
  -> 聚合到 system_id
  -> 输出 selected_systems
```

当前分支只维护本地模式：使用本地 Elasticsearch 做词法检索，使用本地 FAISS 做向量检索，不实现远端 Elasticsearch 和 GaussDB 路径。

## 当前实现

本地链路：

```text
离线灌入 SourceDoc
  -> 构建本地 Elasticsearch 词法索引
  -> 使用 BGE 生成文档向量
  -> 构建本地 FAISS 索引
  -> 在线 FastAPI 查询本地 ES 并加载本地 FAISS artifacts
  -> /v1/decide 输出需要检索的子系统列表
```

本地模式特征：

- 不提供实时文档写入接口。
- 文档通过离线命令灌入和建索引。
- 词法检索使用本地 Elasticsearch。
- 向量检索使用本地 FAISS artifact。
- embedding 默认使用 `BAAI/bge-small-zh-v1.5`。
- `/v1/decide` 的核心输出是 `selected_systems`。

## 目录结构

```text
app/
  api/            FastAPI 路由
  decision/       query 处理、证据增强、系统聚合、决策引擎
  embedding/      mock embedding 和 BGE embedding
  offline/        本地离线灌入和索引构建命令
  retrieval/      本地 ES、本地 FAISS、候选合并、retriever factory
  schemas/        Pydantic 请求 / 响应模型
  storage/        本地 JSONL 文档存储和 artifact 存储
  utils/          计时等工具

data/
  raw/            本地原始文档，git 忽略
  artifacts/      生成的 FAISS artifacts，git 忽略

examples/         示例 SourceDoc 输入
tests/            单元测试
```

## 使用 uv 准备环境

推荐 Python 3.11：

```bash
uv python install 3.11
uv venv --python 3.11 --clear
uv sync --extra dev
```

如果 Windows 上全局 cache 或 Python 安装目录有权限问题，可以放到项目目录内：

```powershell
$env:UV_CACHE_DIR='D:\Code\Python\pcs-retrieval-gating\.uv-cache'
$env:UV_PYTHON_INSTALL_DIR='D:\Code\Python\pcs-retrieval-gating\.uv-python'
uv sync --extra dev
```

## 配置

`.env.example` 中包含主要配置：

```env
APP_MODE=local
EMBEDDING_PROVIDER=bge
EMBEDDING_MODEL_PATH=BAAI/bge-small-zh-v1.5
EMBEDDING_DIM=512
LOCAL_RAW_DOCS_PATH=data/raw/docs.jsonl
LOCAL_ARTIFACT_DIR=data/artifacts
LOCAL_ES_URL=http://127.0.0.1:9200
LOCAL_ES_INDEX=pcs_retrieval_docs
LOCAL_ES_ANALYZER=standard
LOCAL_ES_SEARCH_ANALYZER=standard
LOCAL_ES_INDEX_ON_BUILD=false

SYSTEM_SELECTION_THRESHOLD=0.60
ES_SCORE_WEIGHT=0.55
AGREEMENT_WEIGHT=0.20
SEMANTIC_MATCH_THRESHOLD=0.30
LEXICAL_MATCH_THRESHOLD=0.30
```

## 查询处理

ES 词法检索和向量检索都保留用户原始 query 语义。应用层只去掉首尾空白并合并多余空白，不删除停用词、不做同义词替换、不做大小写归一化。

原因：

- ES 是词法检索，分词、大小写归一化、同义词、停用词和领域词配置应由 ES analyzer 统一承接，避免应用层改写 query 导致 ES `_score` 难以复现。
- BGE embedding 是语义检索，应该保留原始 query 的语义连贯性。
- 文档 embedding 使用原始 `summary` 和 `keywords` 构造，不做停用词删除。

线上词法检索和优先证据关键词匹配由本地 ES 提供：ES 检索会请求 `keywords` / `summary` highlight，并优先使用 `keywords` highlight 生成 `matched_keywords`。ES analyzer 默认使用 `standard`，如果本地 ES 安装了 IK 或自定义同义词 / 停用词 / 领域词 analyzer，可以通过 `LOCAL_ES_ANALYZER` 和 `LOCAL_ES_SEARCH_ANALYZER` 切换。

ES 证据匹配由 ES analyzer 决定。修改 ES analyzer 词表后，建议用 `--index-es` 重新构建 ES 索引。

## 评分机制

当前评分是 MVP 规则，不是训练出来的模型。ES 词法 `_score` 会暂存到兼容字段 `bm25_score`，只在当前 query 的候选集合内归一化：

```text
vector_score_norm = clamp(vector_score, 0, 1)
es_score_norm = es_score / max_es_score_in_candidates
```

单文档强度：

```text
base_score = max(vector_score_norm, ES_SCORE_WEIGHT * es_score_norm)
agreement_score = AGREEMENT_WEIGHT * sqrt(vector_score_norm * es_score_norm) if (
  vector_score_norm >= SEMANTIC_MATCH_THRESHOLD
  and es_score_norm >= LEXICAL_MATCH_THRESHOLD
) else 0
doc_strength = clamp(
  base_score + agreement_score,
  0,
  1
)
```

说明：

- `vector_score_norm`：向量相似度，负数按 0 处理，正数按 0 到 1 使用。
- `es_score_norm`：ES 分在当前 query 候选集内的相对强度，最高分为 1，不用于跨 query 比较。
- `base_score`：取向量分和加权 ES 分的较大值，任意一路强命中都能作为基础证据。
- `ES_SCORE_WEIGHT`：限制 ES 单路第一名的最高基础贡献，避免其因归一化为 1 而自动入选。
- `agreement_score`：同一文档被两路命中且均达到最低门槛时，按两路分数几何平均给予连续奖励。
- `AGREEMENT_WEIGHT`：一致性奖励的系数。
- 每个 `system_id` 的 `confidence` 使用该系统下最强证据文档的 `doc_strength`。
- `confidence >= SYSTEM_SELECTION_THRESHOLD` 时，该系统进入 `selected_systems`。

## 离线灌入和建索引

直接使用示例文档构建本地 FAISS artifacts：

```bash
uv run python -m app.offline.build_index --docs examples/docs.jsonl
```

如果本地 ES 已启动，并希望同步重建 ES 词法索引：

```bash
uv run python -m app.offline.build_index --docs examples/docs.jsonl --index-es
```

也可以先灌入到本地 raw store，再构建索引：

```bash
uv run python -m app.offline.ingest examples/docs.jsonl
uv run python -m app.offline.build_index
```

构建完成后会生成：

```text
data/artifacts/docs.jsonl
data/artifacts/faiss.index
data/artifacts/faiss_doc_ids.json
data/artifacts/manifest.json
```

## 启动服务

```bash
uv run uvicorn app.main:app --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

预期返回：

```json
{"status":"ok"}
```

前端控制台：

```text
http://127.0.0.1:8000/frontend/
```

## 调试检索接口

ES 词法检索：

```bash
curl -X POST http://127.0.0.1:8000/v1/search/bm25 ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"我可以吃海鲜吗\",\"top_k\":5}"
```

兼容接口：

```text
/v1/search/es
```

在 `local` 模式下，`/v1/search/bm25` 和 `/v1/search/es` 都会走本地 ES 词法检索。

向量检索：

```bash
curl -X POST http://127.0.0.1:8000/v1/search/vector ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"我可以吃海鲜吗\",\"top_k\":5}"
```

## 子系统选择接口

```bash
curl -X POST http://127.0.0.1:8000/v1/decide ^
  -H "Content-Type: application/json" ^
  -d "{\"task_id\":\"task_001\",\"task\":\"我可以吃海鲜吗？\",\"top_k_docs\":50,\"max_systems\":5}"
```

返回示例：

```json
{
  "task_id": "task_001",
  "task": "我可以吃海鲜吗？",
  "selected_systems": ["notepad"],
  "decisions": [
    {
      "system_id": "notepad",
      "selected": true,
      "confidence": 0.84,
      "evidence_docs": [
        {
          "doc_id": "3",
          "summary": "记录了用户对海鲜过敏",
          "keywords": ["海鲜过敏", "饮食禁忌"],
          "matched_keywords": ["海鲜过敏"],
          "highlight": {
            "summary": ["记录了用户对<em>海鲜</em>过敏"],
            "keywords": ["<em>海鲜过敏</em>"]
          },
          "bm25_score": 1.2,
          "vector_score": 0.84,
          "bm25_rank": 1,
          "vector_rank": 1
        }
      ]
    }
  ],
  "latency_ms": {}
}
```

字段含义：

- `selected_systems`：最终建议检索的子系统名称列表，这是主要输出。
- `decisions`：候选子系统的解释信息，用于调试和观察。
- `selected`：该候选系统是否进入 `selected_systems`。
- `confidence`：该系统最强证据文档的强度分。
- `evidence_docs`：该系统下用于解释的最多 3 个证据文档。

## 测试和检查

运行单元测试：

```bash
uv run pytest -p no:cacheprovider
```

运行 Ruff：

```bash
uv run ruff check --no-cache .
```

编译检查：

```powershell
$env:PYTHONPYCACHEPREFIX='D:\Code\Python\pcs-retrieval-gating\.pycache-tmp'
uv run python -m compileall app tests
```

## MVP 说明

- 本地模式使用本地 ES，不使用 GaussDB。
- 本地模式不提供实时文档写入接口。
- 文档更新后需要重新运行离线索引构建。
- 词法检索使用本地 ES analyzer；关键词证据优先来自 ES highlight，并在决策证据中返回 `highlight` 供前端红色高亮命中的摘要片段和关键词。
- 向量检索使用原始 query，不做停用词删除。
- 当前 scoring 是 MVP 规则，后续可以替换成更可控的打分模型。
- 最终输出目标是子系统选择，不是文档排序。

## 逻辑回归门控数据采集与训练

每次调用 `/v1/decide` 时，服务会按候选 `system_id` 追加写入一行 JSONL 训练样本，默认路径是 `data/gating/training_samples.jsonl`。样本包含原始 `query`、`task_id`、`system_id`、待人工标注的 `label` 以及以下特征：

- `vector_top1`
- `vector_best_rank_score`
- `es_top1_norm`
- `es_best_rank_score`
- `same_doc_hit_by_both`
- `same_system_hit_by_both`

人工标注时将 JSONL 中的 `label` 从 `null` 改为 `1`（应选该系统）或 `0`（不应选该系统），然后运行：

```bash
uv run python -m app.offline.train_gating_model \
  --input data/gating/training_samples.jsonl \
  --output data/gating/logistic_regression_model.json \
  --batch-size 32
```

训练实现使用 PyTorch `nn.Linear` + `BCEWithLogitsLoss`，默认按 mini-batch（`--batch-size 32`）shuffle 训练，并通过 `--seed` 固定随机性；如果样本量很小，实际 batch 会自动裁剪到样本数。

训练后如需用逻辑回归替换固定打分，将配置改为：

```env
GATING_SCORER=logistic_regression
GATING_MODEL_PATH=data/gating/logistic_regression_model.json
```

保留 `GATING_SCORER=fixed` 时，系统继续使用原有固定规则打分，但仍会写出训练样本；可用 `GATING_FEATURE_LOG_ENABLED=false` 关闭样本采集。

未召回的系统也会作为样本写入。默认 `GATING_INCLUDE_UNRECALLED_SYSTEMS=true` 时，服务会从本地索引 artifact 中读取完整 `system_id` 集合；没有被 ES 或向量检索召回的系统会以全 0 特征写入 JSONL，方便人工标注为负样本。这样训练集既包含被召回候选的排序/强度学习样本，也包含“完全没命中时不该选”的负样本。

`top_k_docs` 会影响训练样本：如果某个系统没有进入 ES 或 vector 的 top-k，它在样本中会表现为未召回或只被一路召回。通常 top10 之后的单条文档很难靠固定规则直接入选，但仍可能影响 top1、最佳 rank、双路命中等聚合特征。实际建议是线上采集阶段先保持一个略大的 top-k（例如 30-50）以避免早期漏采弱正例；标注和训练稳定后，再根据召回覆盖率、正例在 rank 分布中的位置、延迟和模型效果下调 top-k。
