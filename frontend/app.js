const defaultSettings = {
  apiBase: window.location.origin.startsWith("http") ? window.location.origin : "http://127.0.0.1:8000",
  esTopKDocs: 50,
  faissTopKDocs: 50,
  faissPreferredThreshold: 0.6,
  faissMinThreshold: 0.3,
  faissTargetHits: 10,
  evidenceDocsPerSystem: 3,
  rerankerModelPath: "Alibaba-NLP/gte-multilingual-reranker-base",
  rerankerScoreThreshold: 0.5,
  rerankerBatchSize: 8,
  rerankerMaxLength: 512,
};

const state = {
  docs: [],
  settings: loadSettings(),
};

const views = document.querySelectorAll(".view");
const navItems = document.querySelectorAll(".nav-item");
const messageList = document.querySelector("#messageList");
const systemList = document.querySelector("#systemList");
const selectedSummary = document.querySelector("#selectedSummary");
const latencyText = document.querySelector("#latencyText");
const docsGrid = document.querySelector("#docsGrid");
const docsFilterInput = document.querySelector("#docsFilterInput");
const jsonlExample = document.querySelector("#jsonlExample");
const decideForm = document.querySelector("#decideForm");
const taskInput = document.querySelector("#taskInput");

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem("rg.settingsDraft.v8") || "{}");
    return { ...defaultSettings, ...saved };
  } catch {
    return { ...defaultSettings };
  }
}

function saveSettingsDraft() {
  localStorage.setItem("rg.settingsDraft.v8", JSON.stringify(state.settings));
}

function switchView(name) {
  views.forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  navItems.forEach((item) => item.classList.toggle("active", item.dataset.view === name));
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(4);
}

function formatRank(value) {
  const rank = Number(value);
  return Number.isInteger(rank) && rank > 0 ? `#${rank}` : "-";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderHighlightedText(value, fallback = "") {
  const text = String(value ?? fallback ?? "");
  return escapeHtml(text)
    .replaceAll("&lt;em&gt;", '<span style="color: #dc2626; font-weight: 700;">')
    .replaceAll("&lt;/em&gt;", "</span>");
}

function highlightPlainText(value, terms = []) {
  const text = String(value ?? "");
  const uniqueTerms = [...new Set((terms || []).filter(Boolean).map(String))].sort((a, b) => b.length - a.length);
  if (!text || !uniqueTerms.length) {
    return escapeHtml(text);
  }

  const pattern = uniqueTerms.map(escapeRegExp).join("|");
  return escapeHtml(text).replace(new RegExp(`(${pattern})`, "gi"), '<span style="color: #dc2626; font-weight: 700;">$1</span>');
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function renderSummaryWithHighlight(doc) {
  const summaryHighlights = doc.highlight?.summary || doc.metadata?.highlight?.summary || [];
  if (summaryHighlights.length) {
    return summaryHighlights.map((item) => renderHighlightedText(item)).join(' <span class="fragment-gap">...</span> ');
  }
  return highlightPlainText(doc.summary || "", doc.matched_keywords || doc.metadata?.matched_keywords || []);
}

function renderKeywordBadges(keywords = [], matchedKeywords = [], highlight = {}) {
  const matched = new Set(matchedKeywords || []);
  const highlightedKeywords = new Map(
    (highlight.keywords || []).map((item) => [String(item).replaceAll("<em>", "").replaceAll("</em>", ""), item]),
  );

  return (keywords || [])
    .map((item) => {
      const highlighted = highlightedKeywords.get(item);
      const isMatched = matched.has(item) || highlighted;
      return `<span class="badge keyword-badge ${isMatched ? "matched" : ""}">${
        highlighted ? renderHighlightedText(highlighted) : highlightPlainText(item, isMatched ? [item] : [])
      }</span>`;
    })
    .join("");
}

function createQueryHistoryEntry(task) {
  messageList.querySelector(".query-history-empty")?.remove();
  const node = document.createElement("article");
  node.className = "query-history-item pending";
  node.innerHTML = `
    <div class="query-history-label">用户问题</div>
    <div class="query-history-question">${escapeHtml(task)}</div>
    <div class="query-history-status">生成中...</div>
    <div class="query-history-queries"></div>
  `;
  messageList.prepend(node);
  messageList.scrollTop = 0;
  return node;
}

function completeQueryHistoryEntry(node, data) {
  const selected = data.selected_systems || [];
  const queries = data.rewritten_queries?.length
    ? data.rewritten_queries
    : [data.task].filter(Boolean);
  node.classList.remove("pending", "error");
  node.querySelector(".query-history-status").textContent = selected.length
    ? `建议检索：${selected.join(", ")}`
    : "没有选中子系统";
  node.querySelector(".query-history-queries").innerHTML = `
    <div class="query-history-label">改写 Query</div>
    <ol>${queries.map((query) => `<li>${escapeHtml(query)}</li>`).join("")}</ol>
  `;
}

function failQueryHistoryEntry(node, error) {
  node.classList.remove("pending");
  node.classList.add("error");
  node.querySelector(".query-history-status").textContent = `请求失败：${error.message}`;
}

async function postDecide(task) {
  const response = await fetch(`${state.settings.apiBase.replace(/\/$/, "")}/v1/decide`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

function renderDecision(data) {
  const selected = data.selected_systems || [];
  selectedSummary.textContent = selected.length ? `选中：${selected.join(", ")}` : "未选中系统";
  selectedSummary.classList.toggle("has-selection", selected.length > 0);

  const totalMs = Number(data.latency_ms?.total || 0);
  latencyText.textContent = totalMs ? `${totalMs.toFixed(1)} ms` : "-";

  if (!data.decisions?.length) {
    systemList.innerHTML = `<div class="empty-state">暂无候选系统</div>`;
    return;
  }

  systemList.innerHTML = data.decisions
    .map((item) => {
      const evidence = (item.evidence_docs || [])
        .map(
          (doc) => `
            <article class="doc-card evidence-doc-card">
              <div class="doc-topline">
                <div class="doc-id">${escapeHtml(doc.doc_id)}</div>
                <span class="badge system-badge">${escapeHtml(item.system_id)}</span>
              </div>
              <div class="doc-summary">${renderSummaryWithHighlight(doc)}</div>
              <div class="keyword-list">
                ${renderKeywordBadges(doc.keywords || doc.matched_keywords || [], doc.matched_keywords || [], doc.highlight || {})}
              </div>
              <div class="doc-meta">
                <div class="reranker-result">
                  <span>最终 Reranker 分数 <strong>${formatNumber(doc.reranker_score)}</strong></span>
                  <span>最终 Reranker 排名 <strong>${formatRank(doc.reranker_rank)}</strong></span>
                </div>
                <details class="retrieval-debug">
                  <summary>粗召回信号（不参与最终排名）</summary>
                  <div class="retrieval-scores">
                    <span>BM25 分数 <strong>${formatNumber(doc.bm25_score)}</strong></span>
                    <span>FAISS 分数 <strong>${formatNumber(doc.vector_score)}</strong></span>
                  </div>
                  <div class="retrieval-ranks">
                    <span>BM25 排名 <strong>${doc.bm25_rank ? formatRank(doc.bm25_rank) : "未召回"}</strong></span>
                    <span>FAISS 排名 <strong>${doc.vector_rank ? formatRank(doc.vector_rank) : "未召回"}</strong></span>
                  </div>
                </details>
              </div>
            </article>
          `,
        )
        .join("");

      return `
        <article class="system-card ${item.selected ? "selected" : ""}">
          <div class="system-topline">
            <div>
              <div class="system-name">${escapeHtml(item.system_id)}</div>
              <div class="system-score">最佳 Reranker 分数 ${formatNumber(item.reranker_score)}</div>
            </div>
            <span class="badge ${item.selected ? "selected" : ""}">${item.selected ? "selected" : "candidate"}</span>
          </div>
          <div class="evidence">
            <strong>证据</strong>
            <div class="evidence-list">${evidence || "<div>无</div>"}</div>
          </div>
        </article>
      `;
    })
    .join("");
}

function parseDocsText(text) {
  const trimmed = text.trim();
  if (!trimmed) {
    return [];
  }

  return trimmed
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function renderDocs() {
  const keyword = docsFilterInput.value.trim().toLowerCase();
  const docs = state.docs.filter((doc) => {
    const text = [
      doc.doc_id,
      doc.system_id,
      doc.summary,
      ...(doc.keywords || []),
      JSON.stringify(doc.metadata || {}),
    ]
      .join(" ")
      .toLowerCase();
    return !keyword || text.includes(keyword);
  });

  if (!docs.length) {
    docsGrid.innerHTML = `<div class="empty-state">暂无 docs</div>`;
    return;
  }

  docsGrid.innerHTML = docs
    .map(
      (doc) => `
        <article class="doc-card">
          <div class="doc-topline">
            <div class="doc-id">${escapeHtml(doc.doc_id)}</div>
            <span class="badge system-badge">${escapeHtml(doc.system_id)}</span>
          </div>
          <div class="doc-summary">${escapeHtml(doc.summary || "")}</div>
          <div class="keyword-list">
            ${renderKeywordBadges(doc.keywords || [], [], doc.metadata?.highlight || {})}
          </div>
          <div class="doc-meta">${escapeHtml(JSON.stringify(doc.metadata || {}))}</div>
        </article>
      `,
    )
    .join("");
}

function fillSettingsForm() {
  document.querySelector("#apiBaseInput").value = state.settings.apiBase;
  document.querySelector("#esTopKInput").value = state.settings.esTopKDocs;
  document.querySelector("#faissTopKInput").value = state.settings.faissTopKDocs;
  document.querySelector("#faissPreferredThresholdInput").value = state.settings.faissPreferredThreshold;
  document.querySelector("#faissMinThresholdInput").value = state.settings.faissMinThreshold;
  document.querySelector("#faissTargetHitsInput").value = state.settings.faissTargetHits;
  document.querySelector("#evidenceDocsPerSystemInput").value = state.settings.evidenceDocsPerSystem;
  document.querySelector("#rerankerModelPathInput").value = state.settings.rerankerModelPath;
  document.querySelector("#rerankerScoreThresholdInput").value = state.settings.rerankerScoreThreshold;
  document.querySelector("#rerankerBatchSizeInput").value = state.settings.rerankerBatchSize;
  document.querySelector("#rerankerMaxLengthInput").value = state.settings.rerankerMaxLength;
  renderEnvPreview();
}

function readSettingsForm() {
  state.settings = {
    apiBase: document.querySelector("#apiBaseInput").value.trim() || defaultSettings.apiBase,
    esTopKDocs: Number(document.querySelector("#esTopKInput").value || defaultSettings.esTopKDocs),
    faissTopKDocs: Number(document.querySelector("#faissTopKInput").value || defaultSettings.faissTopKDocs),
    faissPreferredThreshold: Number(
      document.querySelector("#faissPreferredThresholdInput").value || defaultSettings.faissPreferredThreshold,
    ),
    faissMinThreshold: Number(
      document.querySelector("#faissMinThresholdInput").value || defaultSettings.faissMinThreshold,
    ),
    faissTargetHits: Number(
      document.querySelector("#faissTargetHitsInput").value || defaultSettings.faissTargetHits,
    ),
    evidenceDocsPerSystem: Number(
      document.querySelector("#evidenceDocsPerSystemInput").value || defaultSettings.evidenceDocsPerSystem,
    ),
    rerankerModelPath:
      document.querySelector("#rerankerModelPathInput").value.trim() || defaultSettings.rerankerModelPath,
    rerankerScoreThreshold: Number(
      document.querySelector("#rerankerScoreThresholdInput").value || defaultSettings.rerankerScoreThreshold,
    ),
    rerankerBatchSize: Number(
      document.querySelector("#rerankerBatchSizeInput").value || defaultSettings.rerankerBatchSize,
    ),
    rerankerMaxLength: Number(
      document.querySelector("#rerankerMaxLengthInput").value || defaultSettings.rerankerMaxLength,
    ),
  };
  saveSettingsDraft();
  renderEnvPreview();
}

function buildEnvPreview() {
  return [
    `ES_TOP_K_DOCS=${state.settings.esTopKDocs}`,
    `FAISS_TOP_K_DOCS=${state.settings.faissTopKDocs}`,
    `FAISS_PREFERRED_SCORE_THRESHOLD=${state.settings.faissPreferredThreshold}`,
    `FAISS_MIN_SCORE_THRESHOLD=${state.settings.faissMinThreshold}`,
    `FAISS_TARGET_HITS=${state.settings.faissTargetHits}`,
    `EVIDENCE_DOCS_PER_SYSTEM=${state.settings.evidenceDocsPerSystem}`,
    `RERANKER_MODEL_PATH=${state.settings.rerankerModelPath}`,
    "RERANKER_LOCAL_FILES_ONLY=true",
    "RERANKER_DEVICE=auto",
    `RERANKER_BATCH_SIZE=${state.settings.rerankerBatchSize}`,
    `RERANKER_MAX_LENGTH=${state.settings.rerankerMaxLength}`,
    `RERANKER_SCORE_THRESHOLD=${state.settings.rerankerScoreThreshold}`,
  ].join("\n");
}

function renderEnvPreview() {
  document.querySelector("#envPreview").textContent = buildEnvPreview();
}

async function checkHealth() {
  const apiStatus = document.querySelector("#apiStatus");
  apiStatus.textContent = "检查中";
  try {
    const response = await fetch(`${state.settings.apiBase.replace(/\/$/, "")}/health`);
    apiStatus.textContent = response.ok ? "可用" : "异常";
  } catch {
    apiStatus.textContent = "不可用";
  }
}

navItems.forEach((item) => {
  item.addEventListener("click", () => switchView(item.dataset.view));
});

decideForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const task = taskInput.value.trim();
  if (!task) {
    return;
  }

  const historyEntry = createQueryHistoryEntry(task);
  taskInput.value = "";

  try {
    const data = await postDecide(task);
    completeQueryHistoryEntry(historyEntry, data);
    renderDecision(data);
  } catch (error) {
    failQueryHistoryEntry(historyEntry, error);
    systemList.innerHTML = `<div class="error-state">${escapeHtml(error.message)}</div>`;
  }
});

taskInput.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.isComposing || event.keyCode === 229) {
    return;
  }

  if (event.ctrlKey) {
    event.preventDefault();
    taskInput.setRangeText(
      "\n",
      taskInput.selectionStart,
      taskInput.selectionEnd,
      "end",
    );
    return;
  }

  event.preventDefault();
  decideForm.requestSubmit();
});

document.querySelector("#docsFileInput").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  const text = await file.text();
  try {
    state.docs = parseDocsText(text);
    jsonlExample.open = false;
    renderDocs();
  } catch (error) {
    docsGrid.innerHTML = `<div class="error-state">JSONL 解析失败：${escapeHtml(error.message)}</div>`;
  }
});

document.querySelector("#loadExampleButton").addEventListener("click", async () => {
  try {
    const response = await fetch("./sample-docs.jsonl");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    state.docs = parseDocsText(await response.text());
    jsonlExample.open = false;
    renderDocs();
  } catch (error) {
    docsGrid.innerHTML = `<div class="error-state">加载示例失败：${escapeHtml(error.message)}</div>`;
  }
});

docsFilterInput.addEventListener("input", renderDocs);

document
  .querySelectorAll(
    "#apiBaseInput, #esTopKInput, #faissTopKInput, #faissPreferredThresholdInput, #faissMinThresholdInput, #faissTargetHitsInput, #evidenceDocsPerSystemInput, #rerankerModelPathInput, #rerankerScoreThresholdInput, #rerankerBatchSizeInput, #rerankerMaxLengthInput",
  )
  .forEach((input) => {
    input.addEventListener("input", readSettingsForm);
  });

document.querySelector("#copyEnvButton").addEventListener("click", async () => {
  readSettingsForm();
  await navigator.clipboard.writeText(buildEnvPreview());
  document.querySelector("#copyEnvButton").textContent = "已复制";
  window.setTimeout(() => {
    document.querySelector("#copyEnvButton").textContent = "复制 .env 片段";
  }, 1500);
});

document.querySelector("#healthButton").addEventListener("click", checkHealth);

fillSettingsForm();
renderDocs();
systemList.innerHTML = `<div class="empty-state">暂无候选系统</div>`;
messageList.innerHTML = `<div class="empty-state query-history-empty">暂无查询记录</div>`;
checkHealth();
