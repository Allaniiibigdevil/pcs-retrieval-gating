# Retrieval Gating Service

这是一个本地优先的个人内容检索门控原型。它只根据各数据源上报的内容判断哪些 system 可能相关，不使用 system 类型、能力、成本或延迟进行路由。

```text
用户任务
  -> ES / FAISS 找到相关文档证据
  -> 每个 system 选择 ES Top-3 / FAISS Top-3 / RRF Top-3
  -> 文档 MLP + max-MIL
  -> 输出 selected_systems
```

`doc_id` 在全部 systems 中全局唯一；同一文档进入 ES 和 FAISS 时使用同一个 `doc_id`。`system_id` 只用于分组、阈值和输出，不进入 embedding 或 MLP 特征。

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
  ml/             九元素 Max-MIL MLP 的训练、推理和 NPZ 权重读写
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
EMBEDDING_PROVIDER=bge
EMBEDDING_MODEL_PATH=BAAI/bge-small-zh-v1.5
EMBEDDING_DIM=512
LOCAL_RAW_DOCS_PATH=data/raw/docs.jsonl
LOCAL_ARTIFACT_DIR=data/artifacts
LOCAL_ES_URL=http://127.0.0.1:9200
LOCAL_ES_INDEX=pcs_retrieval_docs
LOCAL_ES_ANALYZER=standard
LOCAL_ES_SEARCH_ANALYZER=standard

SYSTEM_SELECTION_THRESHOLD=0.60
SYSTEM_SELECTION_THRESHOLDS={}
ES_SCORE_WEIGHT=0.55
AGREEMENT_WEIGHT=0.20
SEMANTIC_MATCH_THRESHOLD=0.30
LEXICAL_MATCH_THRESHOLD=0.30

GATING_SCORER=fixed
GATING_MODEL_PATH=data/gating/nine_representative_mil_mlp.npz
GATING_REQUIRE_CALIBRATION=true
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

服务只保留两个评分模式：`fixed` 用于尚无标注模型时的冷启动，`mil_mlp` 是正式的九元素 Max-MIL 路径。冷启动模式下，ES 词法 `_score` 会暂存到字段 `bm25_score`，只在当前 query 的候选集合内归一化：

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

`mil_mlp` 模式下，每个 system 从候选并集中分别取 ES Top-3、FAISS Top-3 和 RRF Top-3，按全局唯一 `doc_id` 去重后得到 1～9 篇代表文档。共享小型 MLP 分别打分，system 分数取最大值；RRF 仅用于代表文档选择和特征，不直接作为阈值。模型在按 query 隔离的留出集上做单调 Platt 校准。

模型权重保存为 NumPy `.npz` 二进制文件，包含 MLP 参数、特征 schema、格式版本和校准参数。加载时固定使用 `allow_pickle=False`，不执行 pickle 对象。训练样本仍使用 JSONL，因为它需要人工查看和填写 bag label，不属于模型权重。

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
  -d "{\"task_id\":\"task_001\",\"task\":\"我可以吃海鲜吗？\",\"top_k_docs\":20,\"max_systems\":5}"
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
      "confidence_kind": "calibrated_probability",
      "threshold": 0.6,
      "trigger_doc_id": "3",
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
          "gating_score": 0.84
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
- `threshold`：该 system 实际使用的选择阈值，可由 `SYSTEM_SELECTION_THRESHOLDS` 覆盖全局阈值。
- `trigger_doc_id`：经 Max pooling 后触发该 system 的代表文档。
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
- `fixed` 只承担无模型时的冷启动；完成训练和校准后使用 `mil_mlp`。
- 最终输出目标是子系统选择，不是文档排序。

## 九元素 Max-MIL 数据采集与训练

每次调用 `/v1/decide` 时，服务会为九元素代表集合中的每篇文档追加一行 JSONL。`task_id` 应唯一标识一次 query 样本；没有 `task_id` 时才使用 query 文本作为分组键。对同一个 `(task_id 或 query, system_id)` bag，将所有行的 `label` 统一改为 `1`（至少一篇相关）或 `0`（全部无关），然后运行：

```bash
uv run python -m app.offline.train_gating_model \
  --input data/gating/training_samples.jsonl \
  --output data/gating/nine_representative_mil_mlp.npz \
  --batch-size 32
```

训练使用共享 MLP，对 bag 内文档 logit 取 max 后与 system label 计算 BCE。对于负 system，额外把所有代表文档作为负例计算辅助 loss。不要把正 system 的全部文档分别标成正例。

训练后将配置改为：

```env
GATING_SCORER=mil_mlp
GATING_MODEL_PATH=data/gating/nine_representative_mil_mlp.npz
GATING_REQUIRE_CALIBRATION=true
```

`fixed` 仅用于冷启动。未进入 ES/FAISS 候选并集的 system 不会伪造全零样本；这类漏召回应由 Candidate Source Recall@K 单独评估。
