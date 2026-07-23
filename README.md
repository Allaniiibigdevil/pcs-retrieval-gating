# Retrieval Gating Service

这是一个本地优先的检索门控原型服务。它的目标不是给文档做最终排序，而是根据当前用户任务判断应该检索哪些子系统。

```text
用户任务
  -> ES / FAISS 全局召回文档
  -> 每条改写 query 各执行一次 ES / FAISS 召回
  -> 在合并后的 FAISS hits 内自适应放宽向量候选阈值
  -> GTE cross-encoder 对候选文档统一精排
  -> 相关文档按阈值映射到 system_id
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
- 文档相关度由本地 `gte-multilingual-reranker-base` 精排。
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

ES_TOP_K_DOCS=50
FAISS_TOP_K_DOCS=50
FAISS_PREFERRED_SCORE_THRESHOLD=0.60
FAISS_MIN_SCORE_THRESHOLD=0.30
FAISS_TARGET_HITS=10
EVIDENCE_DOCS_PER_SYSTEM=3

RERANKER_MODEL_PATH=Alibaba-NLP/gte-multilingual-reranker-base
RERANKER_LOCAL_FILES_ONLY=true
RERANKER_DEVICE=auto
RERANKER_BATCH_SIZE=8
RERANKER_MAX_LENGTH=512
RERANKER_SCORE_THRESHOLD=0.50
```

## 查询处理

`app/decision/query_rewriter.py` 当前默认返回 `[task]`，其中留有 TODO，可替换为
自定义 query 改写并返回一条或多条 query。空白和重复 query 会被删除；如果结果
为空则回退到原始 task。

所有改写 query 的 ES 与 FAISS 检索并行执行。ES query 只合并多余空白，不删除
停用词；FAISS 保留改写 query 的语义。相同文档被多条 query 召回时，ES 保留最佳
查询排名，FAISS 保留最高向量分，再输出统一的渠道排名。

原因：

- ES 是词法检索，分词、大小写归一化、停用词和领域词配置应由 ES analyzer
  统一承接，避免额外文本预处理导致 ES `_score` 难以复现。
- BGE embedding 是语义检索，应该保留每条改写 query 的语义连贯性。
- 文档 embedding 使用原始 `summary` 和 `keywords` 构造，不做停用词删除。

线上词法检索和优先证据关键词匹配由本地 ES 提供：ES 检索会请求 `keywords` / `summary` highlight，并优先使用 `keywords` highlight 生成 `matched_keywords`。ES analyzer 默认使用 `standard`，如果本地 ES 安装了 IK，可以通过 `LOCAL_ES_ANALYZER` 和 `LOCAL_ES_SEARCH_ANALYZER` 切换。

ES 查询使用 `cross_fields` 将 `summary` 和 `keywords` 作为组合字段匹配，并设置 `minimum_should_match="1<2"`：分析后只有一个词时要求命中该词，分析后有两个及以上词时至少命中两个词。

## 评分机制

当前实现不再使用 RRF，也不混合 BM25 `_score` 和向量相似度。ES 与 FAISS
只负责产生候选；最终文档相关度由 GTE cross-encoder 计算。

### 每条改写 query 一次 FAISS 查询

每条改写 query 只执行一次 Top-K 查询。合并去重后，先过滤掉低于安全下限的文档：

```text
V_raw = {doc in FAISS Top-K | vector_score >= FAISS_MIN_SCORE_THRESHOLD}
```

设 `N = FAISS_TARGET_HITS`。如果优选阈值已经得到至少 N 篇文档，就使用优选
阈值；否则把有效阈值降到第 N 名向量候选的分数，但不低于安全下限：

```text
T_eff = FAISS_PREFERRED_SCORE_THRESHOLD
        if count(vector_score >= FAISS_PREFERRED_SCORE_THRESHOLD) >= N
        else max(FAISS_MIN_SCORE_THRESHOLD, score_of_Nth_available_vector_hit)

V = {doc in V_raw | vector_score >= T_eff}
```

如果安全下限以上本来就不足 N 篇，则保留全部可用向量候选，不会突破安全下限。
这是对所有改写 query 的 FAISS 返回结果做逻辑截断，不会根据阈值反复调用
FAISS。ES 候选 `E` 和自适应向量候选 `V` 按 `doc_id` 合并，全部进入 reranker。

### GTE 精排与 system 选择

默认从项目根目录加载：

```text
Alibaba-NLP/gte-multilingual-reranker-base
```

模型目录已加入 `.gitignore`。模型按第一次 `/v1/decide` 请求懒加载；
`RERANKER_DEVICE=auto` 会优先使用 CUDA，否则使用 CPU。官方模型支持最长 8192
tokens，当前为控制时延默认截断到 512，可通过 `RERANKER_MAX_LENGTH` 调整。

每篇候选文档的输入只包含内容，不包含 `system_id`、BM25 分数、向量分数或两路排名：

```text
passage(doc) = "关键词：" + keywords + "\n摘要：" + summary
raw_logit(doc) = GTE(query, passage(doc))
reranker_score(doc) = sigmoid(raw_logit(doc))
```

按 `reranker_score` 做最终文档排名。system 使用其最佳文档做 max-pool：

```text
system_score(system) = max(reranker_score(doc) for doc in system)
selected(system) = system_score(system) >= RERANKER_SCORE_THRESHOLD
```

因此，一个 system 只需一篇文档达到阈值即可选中；文档数量不会被累加成优势，也不
强制返回固定数量的 system。sigmoid 分数便于统一到 0～1，但未经业务数据校准，不能
解释为真实相关概率，阈值仍需用标注 case 调整。

这里刻意不加入“BM25 高、向量低就降权”的手工规则。cross-encoder 已经负责判断
关键词是否只是浅层重合；提前降权可能误伤姓名、编号、日期等词法精确但向量分不高的
真实相关文档。两路原始分数仍保留在 API、前端和日志中，便于后续收集 hard negative。

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

输入框默认按 Enter 发送，使用 Ctrl+Enter 插入换行。应用日志会输出到启动
Uvicorn 的控制台，包括每次决策的两路命中数、合并候选数、选中系统和分阶段耗时。

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
  -d "{\"task_id\":\"task_001\",\"task\":\"我可以吃海鲜吗？\"}"
```

`/v1/decide` 不接受请求级 Top-K 覆盖；ES 和 FAISS 的召回数量分别由后端
`ES_TOP_K_DOCS`、`FAISS_TOP_K_DOCS` 配置控制。调试检索接口 `/v1/search/*`
仍支持请求级 `top_k`。

返回示例：

```json
{
  "task_id": "task_001",
  "task": "我可以吃海鲜吗？",
  "rewritten_queries": ["我可以吃海鲜吗？", "海鲜过敏记录"],
  "selected_systems": ["notepad"],
  "decisions": [
    {
      "system_id": "notepad",
      "selected": true,
      "reranker_score": 0.93,
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
          "reranker_score": 0.93,
          "reranker_rank": 1
        }
      ]
    }
  ],
  "latency_ms": {}
}
```

字段含义：

- `selected_systems`：最终建议检索的子系统名称列表，这是主要输出。
- `rewritten_queries`：本次实际并行检索使用的去重后 query 列表。
- `decisions`：候选子系统的解释信息，用于调试和观察。
- `selected`：该 system 是否至少有一篇文档达到 reranker 阈值。
- `reranker_score`：该 system 最佳候选文档的 sigmoid reranker 分数，不是已校准概率。
- `evidence_docs`：该系统下用于解释的证据文档，数量由
  `EVIDENCE_DOCS_PER_SYSTEM` 控制，默认最多 3 篇。
- `reranker_rank`：文档在本次 cross-encoder 精排候选中的全局排名；前端与 BM25、FAISS 排名一起展示。

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
- 向量检索使用改写后的 query，不做停用词删除。
- 当前 scoring 只使用 GTE cross-encoder，不对 BM25 和向量原始分做加权。
- 最终输出目标是子系统选择，不是文档排序。
