# Retrieval Gating Service

这是一个本地优先的检索门控原型服务。它的目标不是给文档做最终排序，而是根据当前用户任务判断应该检索哪些子系统。

```text
用户任务
  -> BM25 / FAISS 找到相关文档证据
  -> 聚合到 system_id
  -> 输出 selected_systems
```

当前分支只维护本地模式，不实现 Elasticsearch 和 GaussDB 路径。

## 当前实现

本地链路：

```text
离线灌入 SourceDoc
  -> 使用 jieba + 字符 n-gram 构建本地 BM25 索引
  -> 使用 BGE 生成文档向量
  -> 构建本地 FAISS 索引
  -> 在线 FastAPI 加载本地 artifacts
  -> /v1/decide 输出需要检索的子系统列表
```

本地模式特征：

- 不提供实时文档写入接口。
- 文档通过离线命令灌入和建索引。
- BM25 使用本地 artifact。
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
  retrieval/      本地 BM25、本地 FAISS、候选合并、retriever factory
  schemas/        Pydantic 请求 / 响应模型
  storage/        本地 JSONL 文档存储和 artifact 存储
  utils/          日志、计时等工具

data/
  raw/            本地原始文档，git 忽略
  artifacts/      生成的 BM25 / FAISS artifacts，git 忽略

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

SYSTEM_SELECTION_THRESHOLD=0.55
VECTOR_SCORE_WEIGHT=0.65
BM25_SCORE_WEIGHT=0.35
KEYWORD_BOOST_PER_MATCH=0.02
KEYWORD_BOOST_MAX=0.10
```

## 查询处理

BM25 和向量检索使用不同的 query：

```text
BM25 检索：使用 normalized query
向量检索：使用原始 query
```

原因：

- BM25 是词法检索，适合做基础归一化和分词。
- BGE embedding 是语义检索，应该保留原始 query 的语义连贯性。
- 文档 embedding 使用原始 `summary` 和 `keywords` 构造，不做停用词删除。

## 评分机制

当前评分是 MVP 规则，不是训练出来的模型。BM25 原始分会在当前 query 的候选集合内归一化：

```text
vector_score_norm = clamp(vector_score, 0, 1)
bm25_score_norm = bm25_score / max_bm25_score_in_candidates
```

单文档强度：

```text
keyword_boost = min(KEYWORD_BOOST_MAX, KEYWORD_BOOST_PER_MATCH * matched_keyword_count)
doc_strength = clamp(
  VECTOR_SCORE_WEIGHT * vector_score_norm
  + BM25_SCORE_WEIGHT * bm25_score_norm
  + keyword_boost,
  0,
  1
)
```

说明：

- `vector_score_norm`：向量相似度，负数按 0 处理，正数按 0 到 1 使用。
- `bm25_score_norm`：当前 query 候选集合内的 BM25 归一化分，最高 BM25 文档为 1。
- `VECTOR_SCORE_WEIGHT`：语义向量信号的权重。
- `BM25_SCORE_WEIGHT`：BM25 词法命中信号的权重。
- `keyword_boost`：命中文档关键词时的少量加分。
- 每个 `system_id` 的 `confidence` 使用该系统下最强证据文档的 `doc_strength`。
- `confidence >= SYSTEM_SELECTION_THRESHOLD` 时，该系统进入 `selected_systems`。

## 离线灌入和建索引

直接使用示例文档构建本地索引：

```bash
uv run python -m app.offline.build_index --docs examples/docs.jsonl
```

也可以先灌入到本地 raw store，再构建索引：

```bash
uv run python -m app.offline.ingest examples/docs.jsonl
uv run python -m app.offline.build_index
```

构建完成后会生成：

```text
data/artifacts/docs.jsonl
data/artifacts/bm25.pkl
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

BM25 检索：

```bash
curl -X POST http://127.0.0.1:8000/v1/search/bm25 ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"我可以吃海鲜吗\",\"top_k\":5}"
```

兼容接口：

```text
/v1/search/es
```

在 `local` 模式下，`/v1/search/es` 是 BM25 检索的 alias。

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
          "matched_keywords": ["海鲜过敏"],
          "bm25_score": 1.2,
          "vector_score": 0.84,
          "bm25_rank": 1,
          "vector_rank": 1
        }
      ],
      "reason": "命中相关关键词：海鲜过敏"
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

- 本地模式不使用 ES / GaussDB。
- 本地模式不提供实时文档写入接口。
- 文档更新后需要重新运行离线索引构建。
- BM25 使用 jieba 分词，并补充中文字符 n-gram。
- 向量检索使用原始 query，不做停用词删除。
- 当前 scoring 是 MVP 规则，后续可以替换成更可控的打分模型。
- 最终输出目标是子系统选择，不是文档排序。
