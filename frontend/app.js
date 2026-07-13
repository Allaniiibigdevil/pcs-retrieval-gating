const defaultSettings = {
  apiBase: window.location.origin.startsWith("http") ? window.location.origin : "http://127.0.0.1:8000",
  topKDocs: 50,
  maxSystems: 5,
  threshold: 0.6,
  esWeight: 0.55,
  agreementWeight: 0.2,
  semanticThreshold: 0.3,
  lexicalThreshold: 0.3,
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

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem("rg.settingsDraft.v4") || "{}");
    return { ...defaultSettings, ...saved };
  } catch {
    return { ...defaultSettings };
  }
}

function saveSettingsDraft() {
  localStorage.setItem("rg.settingsDraft.v4", JSON.stringify(state.settings));
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
    .replaceAll("&lt;em&gt;", '<mark class="hit-highlight">')
    .replaceAll("&lt;/em&gt;", "</mark>");
}

function highlightPlainText(value, terms = []) {
  const text = String(value ?? "");
  const uniqueTerms = [...new Set((terms || []).filter(Boolean).map(String))].sort((a, b) => b.length - a.length);
  if (!text || !uniqueTerms.length) {
    return escapeHtml(text);
  }

  const pattern = uniqueTerms.map(escapeRegExp).join("|");
  return escapeHtml(text).replace(new RegExp(`(${pattern})`, "gi"), '<mark class="hit-highlight">$1</mark>');
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
    body: JSON.stringify({
      task,
      top_k_docs: Number(state.settings.topKDocs),
      max_systems: Number(state.settings.maxSystems),
    }),
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
      const width = Math.max(0, Math.min(100, Number(item.confidence || 0) * 100));
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
              <div class="doc-meta">BM25 ${formatNumber(doc.bm25_score)} / Vector ${formatNumber(doc.vector_score)}</div>
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
          <div class="confidence">
            <div class="meter"><span style="width:${width}%"></span></div>
            <div class="score-line">
              <strong>${formatNumber(item.confidence)}</strong>
              <span>confidence</span>
            </div>
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
  document.querySelector("#maxSystemsInput").value = state.settings.maxSystems;
  document.querySelector("#thresholdInput").value = state.settings.threshold;
  document.querySelector("#esWeightInput").value = state.settings.esWeight;
  document.querySelector("#agreementWeightInput").value = state.settings.agreementWeight;
  document.querySelector("#semanticThresholdInput").value = state.settings.semanticThreshold;
  document.querySelector("#lexicalThresholdInput").value = state.settings.lexicalThreshold;
  renderEnvPreview();
}

function readSettingsForm() {
  state.settings = {
    apiBase: document.querySelector("#apiBaseInput").value.trim() || defaultSettings.apiBase,
    topKDocs: Number(document.querySelector("#topKInput").value || defaultSettings.topKDocs),
    maxSystems: Number(document.querySelector("#maxSystemsInput").value || defaultSettings.maxSystems),
    threshold: Number(document.querySelector("#thresholdInput").value || defaultSettings.threshold),
    esWeight: Number(document.querySelector("#esWeightInput").value || defaultSettings.esWeight),
    agreementWeight: Number(
      document.querySelector("#agreementWeightInput").value || defaultSettings.agreementWeight,
    ),
    semanticThreshold: Number(
      document.querySelector("#semanticThresholdInput").value || defaultSettings.semanticThreshold,
    ),
    lexicalThreshold: Number(document.querySelector("#lexicalThresholdInput").value || defaultSettings.lexicalThreshold),
  };
  saveSettingsDraft();
  renderEnvPreview();
}

function buildEnvPreview() {
  return [
    `DEFAULT_TOP_K_DOCS=${state.settings.topKDocs}`,
    `DEFAULT_MAX_SYSTEMS=${state.settings.maxSystems}`,
    `SYSTEM_SELECTION_THRESHOLD=${state.settings.threshold}`,
    `ES_SCORE_WEIGHT=${state.settings.esWeight}`,
    `AGREEMENT_WEIGHT=${state.settings.agreementWeight}`,
    `SEMANTIC_MATCH_THRESHOLD=${state.settings.semanticThreshold}`,
    `LEXICAL_MATCH_THRESHOLD=${state.settings.lexicalThreshold}`,
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

document.querySelector("#decideForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.querySelector("#taskInput");
  const task = input.value.trim();
  if (!task) {
    return;
  }

  addMessage("user", task);
  input.value = "";
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
    "#apiBaseInput, #topKInput, #maxSystemsInput, #thresholdInput, #esWeightInput, #agreementWeightInput, #semanticThresholdInput, #lexicalThresholdInput",
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
