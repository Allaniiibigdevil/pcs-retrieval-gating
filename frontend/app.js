const defaultSettings = {
  apiBase: window.location.origin.startsWith("http") ? window.location.origin : "http://127.0.0.1:8000",
  topKDocs: 50,
  maxSystems: 5,
  threshold: 0.75,
  bm25Weight: 0.75,
  keywordBoost: 0.02,
  keywordBoostMax: 0.1,
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

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem("rg.settingsDraft") || "{}");
    return { ...defaultSettings, ...saved };
  } catch {
    return { ...defaultSettings };
  }
}

function saveSettingsDraft() {
  localStorage.setItem("rg.settingsDraft", JSON.stringify(state.settings));
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
            <div class="evidence-item">
              <strong>${escapeHtml(doc.doc_id)}</strong>
              <div>${escapeHtml(doc.summary || "")}</div>
              <div>BM25 ${formatNumber(doc.bm25_score)} / Vector ${formatNumber(doc.vector_score)}</div>
            </div>
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
            <div class="reason">${escapeHtml(item.reason)}</div>
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

  if (trimmed.startsWith("[")) {
    return JSON.parse(trimmed);
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
            <span class="badge">${escapeHtml(doc.system_id)}</span>
          </div>
          <div class="doc-summary">${escapeHtml(doc.summary || "")}</div>
          <div class="keyword-list">
            ${(doc.keywords || []).map((item) => `<span class="badge">${escapeHtml(item)}</span>`).join("")}
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
  document.querySelector("#bm25WeightInput").value = state.settings.bm25Weight;
  document.querySelector("#keywordBoostInput").value = state.settings.keywordBoost;
  document.querySelector("#keywordBoostMaxInput").value = state.settings.keywordBoostMax;
  renderEnvPreview();
}

function readSettingsForm() {
  state.settings = {
    apiBase: document.querySelector("#apiBaseInput").value.trim() || defaultSettings.apiBase,
    topKDocs: Number(document.querySelector("#topKInput").value || defaultSettings.topKDocs),
    maxSystems: Number(document.querySelector("#maxSystemsInput").value || defaultSettings.maxSystems),
    threshold: Number(document.querySelector("#thresholdInput").value || defaultSettings.threshold),
    bm25Weight: Number(document.querySelector("#bm25WeightInput").value || defaultSettings.bm25Weight),
    keywordBoost: Number(document.querySelector("#keywordBoostInput").value || defaultSettings.keywordBoost),
    keywordBoostMax: Number(document.querySelector("#keywordBoostMaxInput").value || defaultSettings.keywordBoostMax),
  };
  saveSettingsDraft();
  renderEnvPreview();
}

function buildEnvPreview() {
  return [
    `DEFAULT_TOP_K_DOCS=${state.settings.topKDocs}`,
    `DEFAULT_MAX_SYSTEMS=${state.settings.maxSystems}`,
    `SYSTEM_SELECTION_THRESHOLD=${state.settings.threshold}`,
    `BM25_RANK_WEIGHT=${state.settings.bm25Weight}`,
    `KEYWORD_BOOST_PER_MATCH=${state.settings.keywordBoost}`,
    `KEYWORD_BOOST_MAX=${state.settings.keywordBoostMax}`,
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
    renderDocs();
  } catch (error) {
    docsGrid.innerHTML = `<div class="error-state">解析失败：${escapeHtml(error.message)}</div>`;
  }
});

document.querySelector("#loadExampleButton").addEventListener("click", async () => {
  try {
    const response = await fetch("./sample-docs.jsonl");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    state.docs = parseDocsText(await response.text());
    renderDocs();
  } catch (error) {
    docsGrid.innerHTML = `<div class="error-state">加载示例失败：${escapeHtml(error.message)}</div>`;
  }
});

docsFilterInput.addEventListener("input", renderDocs);

document
  .querySelectorAll(
    "#apiBaseInput, #topKInput, #maxSystemsInput, #thresholdInput, #bm25WeightInput, #keywordBoostInput, #keywordBoostMaxInput",
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
