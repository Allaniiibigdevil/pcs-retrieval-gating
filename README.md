# Retrieval Gating Service

这是一个本地优先的检索门控原型服务。它的目标不是给文档做最终排序，而是根据当前用户任务判断应该检索哪些子系统。

```text
用户任务
  -> ES / FAISS 全局召回文档
  -> FAISS 相似度阈值过滤
  -> RRF 融合两个文档排名
  -> 全局 Top-N 文档映射到 system_id
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

DEFAULT_TOP_K_DOCS=50
FAISS_SCORE_THRESHOLD=0.60
RRF_K=20
RRF_TOP_N_DOCS=10
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

当前实现不混合 BM25 `_score` 和向量相似度，只使用两路候选的排名做 Reciprocal Rank Fusion。ES 与 FAISS 各自执行全局 Top-K 召回，FAISS 文档在进入融合前必须满足：

```text
vector_score >= FAISS_SCORE_THRESHOLD
```

同一个 `doc_id` 在两路候选中合并后，文档 RRF 分数为：

```text
rrf(doc) =
  I(doc in ES) / (RRF_K + es_rank)
  + I(doc in FAISS) / (RRF_K + vector_rank)
```

说明：

- 某一路没有召回该文档时，该路贡献为 0。
- `RRF_K` 控制排名位置差异；当前默认值 20，适合较浅的候选列表。
- 全局按 `rrf(doc)` 排序后取 `RRF_TOP_N_DOCS` 篇文档，默认取前 10 篇。
- 只要一个 system 至少有一篇文档进入全局 RRF Top-N，就进入 `selected_systems`。
- system 的展示分数是该 system 最佳文档的 RRF 分数，不对多篇文档求和。
- RRF 分数不是相关概率；空结果依赖 ES 无命中以及 FAISS 阈值过滤后无候选。

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

`faiss.index` 的维度与构建时使用的 embedding 模型绑定。修改
`EMBEDDING_MODEL_PATH` 后必须重新运行建索引命令并重启服务；否则 query embedding
与旧索引维度不一致。可以先查看 `data/artifacts/manifest.json` 中记录的
`embedding_model` 和 `embedding_dim`。

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
  -d "{\"task_id\":\"task_001\",\"task\":\"我可以吃海鲜吗？\",\"top_k_docs\":50}"
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
      "rrf_score": 0.095238,
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
          "vector_rank": 1,
          "rrf_score": 0.095238
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
- `selected`：该 system 是否至少有一篇文档进入全局 RRF Top-N。
- `rrf_score`：该 system 最佳候选文档的 RRF 分数，不是概率。
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
- 当前 scoring 使用 RRF 排名融合，不对 BM25 和向量原始分做加权。
- 最终输出目标是子系统选择，不是文档排序。
