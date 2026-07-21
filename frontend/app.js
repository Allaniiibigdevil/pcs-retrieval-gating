const defaultSettings = {
  apiBase: window.location.origin.startsWith("http") ? window.location.origin : "http://127.0.0.1:8000",
  topKDocs: 50,
  faissScoreThreshold: 0.6,
  rrfK: 20,
  rrfTopNDocs: 10,
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
    const saved = JSON.parse(localStorage.getItem("rg.settingsDraft.v5") || "{}");
    return { ...defaultSettings, ...saved };
  } catch {
    return { ...defaultSettings };
  }
}

function saveSettingsDraft() {
  localStorage.setItem("rg.settingsDraft.v5", JSON.stringify(state.settings));
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

function addMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text;
  messageList.appendChild(node);
  messageList.scrollTop = messageList.scrollHeight;
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

  const totalMs = Object.values(data.latency_ms || {}).reduce((sum, value) => sum + Number(value || 0), 0);
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
                <div class="retrieval-scores">
                  <span>BM25 分数 <strong>${formatNumber(doc.bm25_score)}</strong></span>
                  <span>FAISS 分数 <strong>${formatNumber(doc.vector_score)}</strong></span>
                </div>
                <div class="retrieval-ranks">
                  <span>BM25 排名 <strong>${doc.bm25_rank ? `#${doc.bm25_rank}` : "未召回"}</strong></span>
                  <span>FAISS 排名 <strong>${doc.vector_rank ? `#${doc.vector_rank}` : "未召回"}</strong></span>
                  <span>RRF 排名 <strong>#${doc.rrf_rank}</strong></span>
                </div>
              </div>
            </article>
          `,
        )
        .join("");

      return `
        <article class="system-card ${item.selected ? "selected" : ""}">
          <div class="system-topline">
            <div class="system-name">${escapeHtml(item.system_id)}</div>
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
  document.querySelector("#topKInput").value = state.settings.topKDocs;
  document.querySelector("#faissScoreThresholdInput").value = state.settings.faissScoreThreshold;
  document.querySelector("#rrfKInput").value = state.settings.rrfK;
  document.querySelector("#rrfTopNDocsInput").value = state.settings.rrfTopNDocs;
  renderEnvPreview();
}

function readSettingsForm() {
  state.settings = {
    apiBase: document.querySelector("#apiBaseInput").value.trim() || defaultSettings.apiBase,
    topKDocs: Number(document.querySelector("#topKInput").value || defaultSettings.topKDocs),
    faissScoreThreshold: Number(
      document.querySelector("#faissScoreThresholdInput").value || defaultSettings.faissScoreThreshold,
    ),
    rrfK: Number(document.querySelector("#rrfKInput").value || defaultSettings.rrfK),
    rrfTopNDocs: Number(
      document.querySelector("#rrfTopNDocsInput").value || defaultSettings.rrfTopNDocs,
    ),
  };
  saveSettingsDraft();
  renderEnvPreview();
}

function buildEnvPreview() {
  return [
    `DEFAULT_TOP_K_DOCS=${state.settings.topKDocs}`,
    `FAISS_SCORE_THRESHOLD=${state.settings.faissScoreThreshold}`,
    `RRF_K=${state.settings.rrfK}`,
    `RRF_TOP_N_DOCS=${state.settings.rrfTopNDocs}`,
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

  addMessage("user", task);
  taskInput.value = "";
  addMessage("assistant", "生成中...");

  try {
    const data = await postDecide(task);
    messageList.lastElementChild.textContent = data.selected_systems?.length
      ? `建议检索：${data.selected_systems.join(", ")}`
      : "没有选中子系统";
    renderDecision(data);
  } catch (error) {
    messageList.lastElementChild.textContent = `请求失败：${error.message}`;
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
    "#apiBaseInput, #topKInput, #faissScoreThresholdInput, #rrfKInput, #rrfTopNDocsInput",
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
addMessage("assistant", "输入任务后生成候选系统。");
checkHealth();
