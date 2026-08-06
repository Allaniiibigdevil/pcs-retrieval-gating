const state = {
  apiBase:
    localStorage.getItem("rg.apiBase") ||
    (window.location.origin.startsWith("http")
      ? window.location.origin
      : "http://127.0.0.1:8000"),
  docs: [],
  latestRequestId: 0,
};

const views = document.querySelectorAll(".view");
const navItems = document.querySelectorAll(".nav-item");
const messageList = document.querySelector("#messageList");
const systemList = document.querySelector("#systemList");
const selectedSummary = document.querySelector("#selectedSummary");
const latencyText = document.querySelector("#latencyText");
const docsGrid = document.querySelector("#docsGrid");
const docsFilterInput = document.querySelector("#docsFilterInput");
const apiBaseInput = document.querySelector("#apiBaseInput");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value) {
  return value === null || value === undefined || Number.isNaN(Number(value))
    ? "-"
    : Number(value).toFixed(4);
}

function switchView(name) {
  views.forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  navItems.forEach((item) => item.classList.toggle("active", item.dataset.view === name));
}

function renderHighlighted(value) {
  return escapeHtml(value)
    .replaceAll("&lt;em&gt;", '<mark class="hit">')
    .replaceAll("&lt;/em&gt;", "</mark>");
}

function renderSummary(doc) {
  const fragments = doc.highlight?.summary || [];
  return fragments.length
    ? fragments.map(renderHighlighted).join(' <span class="muted">…</span> ')
    : escapeHtml(doc.summary || "");
}

function renderKeywords(doc) {
  const matched = new Set(doc.matched_keywords || []);
  return (doc.keywords || [])
    .map(
      (keyword) =>
        `<span class="tag ${matched.has(keyword) ? "matched" : ""}">${escapeHtml(keyword)}</span>`,
    )
    .join("");
}

function addHistory(task, status = "处理中…") {
  document.querySelector(".history-empty")?.remove();
  const item = document.createElement("article");
  item.className = "history-item";
  item.innerHTML = `<strong>${escapeHtml(task)}</strong><span>${escapeHtml(status)}</span>`;
  messageList.prepend(item);
  return item;
}

async function postDecide(task) {
  const response = await fetch(`${state.apiBase.replace(/\/$/, "")}/v1/decide`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task }),
  });
  if (!response.ok) {
    throw new Error((await response.text()) || `HTTP ${response.status}`);
  }
  return response.json();
}

function renderDecision(data) {
  const selected = data.selected_systems || [];
  selectedSummary.textContent = selected.length
    ? `选中：${selected.join(", ")}`
    : "未选中系统";
  selectedSummary.classList.toggle("selected", selected.length > 0);
  const total = Number(data.latency_ms?.total || 0);
  latencyText.textContent = total ? `${total.toFixed(1)} ms` : "-";

  if (!data.decisions?.length) {
    systemList.innerHTML = '<div class="empty-state">没有召回到候选系统</div>';
    return;
  }

  systemList.innerHTML = data.decisions
    .map((decision) => {
      const evidence = (decision.evidence_docs || [])
        .map(
          (doc) => `<article class="doc-card">
            <div class="card-row"><strong>${escapeHtml(doc.doc_id)}</strong><span>${escapeHtml(decision.system_id)}</span></div>
            <p>${renderSummary(doc)}</p>
            <div class="tags">${renderKeywords(doc)}</div>
            <div class="metrics">
              <span>ES raw ${formatNumber(doc.bm25_score)}</span>
              <span>ES norm ${formatNumber(doc.bm25_score_norm)}</span>
              <span>Vector ${formatNumber(doc.vector_score)}</span>
            </div>
            ${(doc.matched_queries || []).length ? `<div class="queries">queries: ${(doc.matched_queries || []).map(escapeHtml).join(" · ")}</div>` : ""}
          </article>`,
        )
        .join("");
      return `<article class="system-card ${decision.selected ? "selected" : ""}">
        <div class="card-row">
          <h3>${escapeHtml(decision.system_id)}</h3>
          <span class="status">${decision.selected ? "selected" : "candidate"}</span>
        </div>
        <div class="confidence">confidence ${formatNumber(decision.confidence)}</div>
        <div class="evidence-list">${evidence}</div>
      </article>`;
    })
    .join("");
}

function parseDocs(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function renderDocs() {
  const filter = docsFilterInput.value.trim().toLowerCase();
  const docs = state.docs.filter((doc) =>
    [doc.doc_id, doc.system_id, doc.summary, ...(doc.keywords || [])]
      .join(" ")
      .toLowerCase()
      .includes(filter),
  );
  docsGrid.innerHTML = docs.length
    ? docs
        .map(
          (doc) => `<article class="doc-card">
            <div class="card-row"><strong>${escapeHtml(doc.doc_id)}</strong><span>${escapeHtml(doc.system_id)}</span></div>
            <p>${escapeHtml(doc.summary || "")}</p>
            <div class="tags">${(doc.keywords || []).map((item) => `<span class="tag">${escapeHtml(item)}</span>`).join("")}</div>
          </article>`,
        )
        .join("")
    : '<div class="empty-state">暂无文档</div>';
}

async function checkHealth() {
  const status = document.querySelector("#apiStatus");
  status.textContent = "检查中";
  try {
    const response = await fetch(`${state.apiBase.replace(/\/$/, "")}/health`);
    status.textContent = response.ok ? "可用" : "异常";
  } catch {
    status.textContent = "不可用";
  }
}

navItems.forEach((item) => item.addEventListener("click", () => switchView(item.dataset.view)));

document.querySelector("#decideForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.querySelector("#taskInput");
  const task = input.value.trim();
  if (!task) return;
  const requestId = ++state.latestRequestId;
  const history = addHistory(task);
  input.value = "";
  try {
    const data = await postDecide(task);
    history.querySelector("span").textContent = data.selected_systems?.length
      ? `建议检索：${data.selected_systems.join(", ")}`
      : "没有选中系统";
    if (requestId === state.latestRequestId) renderDecision(data);
  } catch (error) {
    history.querySelector("span").textContent = `失败：${error.message}`;
  }
});

document.querySelector("#docsFileInput").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    state.docs = parseDocs(await file.text());
    renderDocs();
  } catch (error) {
    docsGrid.innerHTML = `<div class="error-state">JSONL 解析失败：${escapeHtml(error.message)}</div>`;
  }
});

document.querySelector("#loadExampleButton").addEventListener("click", async () => {
  try {
    const response = await fetch("./sample-docs.jsonl");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.docs = parseDocs(await response.text());
    renderDocs();
  } catch (error) {
    docsGrid.innerHTML = `<div class="error-state">加载失败：${escapeHtml(error.message)}</div>`;
  }
});

docsFilterInput.addEventListener("input", renderDocs);
apiBaseInput.value = state.apiBase;
apiBaseInput.addEventListener("change", () => {
  state.apiBase = apiBaseInput.value.trim() || "http://127.0.0.1:8000";
  localStorage.setItem("rg.apiBase", state.apiBase);
  checkHealth();
});
document.querySelector("#healthButton").addEventListener("click", checkHealth);

messageList.innerHTML = '<div class="empty-state history-empty">暂无查询记录</div>';
systemList.innerHTML = '<div class="empty-state">暂无结果</div>';
renderDocs();
checkHealth();
