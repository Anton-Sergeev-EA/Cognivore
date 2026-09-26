// Cognivore web UI -- vanilla JS, no build step, no framework.
// Talks to the FastAPI backend: /api/chat/stream (SSE) for answers,
// /api/insights/* for the knowledge map and gap radar. All visible text
// comes from i18n.js (t()), the map is drawn by cortex.js (Cortex).

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const formEl = $("chat-form");
const inputEl = $("chat-input");
const sendBtn = $("send-btn");
const heroEl = $("hero");

const state = {
  health: null,
  connOk: null,
  map: null,
  gaps: [],
  busy: false,
  turns: [], // { el, answer, insight, traceCount }
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Status ──────────────────────────────────────────────────────────────

function shortEmbedder(id) {
  if (!id) return "…";
  if (id.startsWith("fastembed:")) return id.slice("fastembed:".length).split("/").pop();
  if (id.startsWith("hashing")) return "hashing (offline)";
  return id;
}

function renderStatus() {
  const h = state.health;
  const modeCard = $("mode-card");
  const topMode = $("topbar-mode");
  if (h) {
    $("status-backend").textContent = h.llm_model || h.llm_backend.replace(/Backend$/, "");
    $("status-embedder").textContent = shortEmbedder(h.embedder);
    $("status-native").textContent = t(h.native_index ? "statusIndexNative" : "statusIndexPython");
    const tools = $("status-tools");
    tools.replaceChildren(...h.tools.map((name) => el("span", "tool-chip", name)));
    $("kb-count").textContent = t("kbChunks", { count: h.knowledge_base_chunks });
    const badge = $("mode-badge");
    badge.textContent = t(h.demo_mode ? "demoBadge" : "liveBadge");
    badge.className = "badge" + (h.demo_mode ? " demo" : "");
    $("mode-hint").textContent = h.demo_mode ? t("demoHint") : h.llm_model || "";
    modeCard.hidden = false;
    topMode.textContent = h.demo_mode ? t("demoBadge") : h.llm_model || t("liveBadge");
    topMode.hidden = false;
  } else {
    $("status-backend").textContent = t("statusUnreachable");
    modeCard.hidden = true;
    topMode.hidden = true;
  }
  const dot = $("conn-dot");
  if (state.connOk !== null) {
    dot.classList.toggle("online", state.connOk);
    dot.classList.toggle("offline", !state.connOk);
    dot.title = t(state.connOk ? "connOnline" : "connOffline");
  }
}

async function loadStatus() {
  try {
    const res = await fetch("/api/health");
    state.health = await res.json();
    state.connOk = true;
  } catch (err) {
    state.connOk = false;
  }
  renderStatus();
}

// ── Knowledge map & gap radar ──────────────────────────────────────────

async function loadMap() {
  try {
    const res = await fetch("/api/insights/map");
    if (!res.ok) return;
    state.map = await res.json();
    Cortex.setData(state.map);
    renderTopics();
  } catch (err) {
    /* the map is an enhancement; chat keeps working without it */
  }
}

function renderTopics() {
  const list = $("topic-list");
  list.replaceChildren();
  const map = state.map;
  if (!map) return;
  $("map-stats").textContent = map.total_chunks
    ? t("mapStats", {
        chunks: t("kbChunks", { count: map.total_chunks }),
        topics: t("mapTopics", { count: map.clusters.length }),
      })
    : "";
  for (const c of [...map.clusters].sort((a, b) => b.size - a.size)) {
    const chip = el("span", "topic");
    chip.tabIndex = 0;
    const dot = el("span", "dot");
    dot.style.background = Cortex.colorFor(c.id);
    chip.append(dot, el("span", "", c.keywords.join(" · ") || "—"));
    chip.append(el("span", "n", String(c.size)));
    chip.title = t("topicSize", { count: c.size });
    const focus = (on) => {
      chip.classList.toggle("active", on);
      Cortex.focusCluster(on ? c.id : null);
    };
    chip.addEventListener("mouseenter", () => focus(true));
    chip.addEventListener("mouseleave", () => focus(false));
    chip.addEventListener("focus", () => focus(true));
    chip.addEventListener("blur", () => focus(false));
    list.appendChild(chip);
  }
}

async function loadGaps() {
  try {
    const res = await fetch("/api/insights/gaps");
    if (!res.ok) return;
    state.gaps = await res.json();
    renderGaps();
  } catch (err) {
    /* optional panel */
  }
}

function renderGaps() {
  const list = $("gap-list");
  list.replaceChildren();
  if (!state.gaps.length) {
    list.appendChild(el("div", "gap-empty", t("gapsEmpty")));
    return;
  }
  list.appendChild(el("div", "gap-hint", t("gapsHint")));
  for (const gap of state.gaps) {
    const item = el("div", "gap-item");
    const body = el("div");
    body.append(el("div", "q", gap.query), el("div", "meta", t("gapAsked", { count: gap.count })));
    const resolve = el("button", "icon-btn", "✓");
    resolve.type = "button";
    resolve.title = t("gapResolve");
    resolve.setAttribute("aria-label", t("gapResolve"));
    resolve.addEventListener("click", async () => {
      await fetch(`/api/insights/gaps?query=${encodeURIComponent(gap.query)}`, { method: "DELETE" });
      loadGaps();
    });
    item.append(body, resolve);
    list.appendChild(item);
  }
}

async function openChunk(id) {
  try {
    const res = await fetch(`/api/knowledge_base/${id}`);
    if (!res.ok) return;
    const chunk = await res.json();
    $("chunk-source").textContent = chunk.source;
    $("chunk-text").textContent = chunk.text;
    $("chunk-dialog").showModal();
  } catch (err) {
    /* ignore */
  }
}

// ── Hero / examples ────────────────────────────────────────────────────

function renderExamples() {
  const box = $("examples");
  box.replaceChildren();
  for (const example of t("examples")) {
    const chip = el("button", "example-chip", example);
    chip.type = "button";
    chip.addEventListener("click", () => {
      if (!state.busy) sendMessage(example);
    });
    box.appendChild(chip);
  }
}

function hideHero() {
  if (heroEl && heroEl.isConnected) heroEl.remove();
}

// ── Chat turns ─────────────────────────────────────────────────────────

function newTurn() {
  const turnEl = el("div", "turn");
  const trace = el("details", "trace");
  trace.hidden = true;
  const summary = el("summary");
  const steps = el("div", "steps");
  trace.append(summary, steps);
  const answer = el("div", "answer");
  const thinking = el("div", "thinking");
  thinking.innerHTML = '<span class="thinking-dots"><span></span><span></span><span></span></span>';
  thinking.appendChild(el("span", "", t("thinking")));
  answer.appendChild(thinking);
  turnEl.append(trace, answer);
  messagesEl.appendChild(turnEl);
  const turn = { el: turnEl, trace, summary, steps, answer, thinking, textEl: null, text: "", insight: null, traceCount: 0 };
  state.turns.push(turn);
  scrollToBottom();
  return turn;
}

function renderTraceSummary(turn) {
  turn.summary.textContent = t("traceTitle", { count: turn.traceCount });
}

function addTraceStep(turn, step) {
  turn.trace.hidden = false;
  turn.traceCount += 1;
  renderTraceSummary(turn);
  const line = el("div", "step");
  if (step.action) {
    const observation = step.observation || "";
    line.append(el("span", "tool", step.action), " ", el("span", "args", JSON.stringify(step.action_input)), " → ");
    const short = el("span", "short", observation.slice(0, 160) + (observation.length > 160 ? "…" : ""));
    const full = el("span", "full", observation);
    line.append(short, full);
    if (observation.length > 160) {
      line.classList.add("expandable");
      line.title = t("traceShowFull");
      line.addEventListener("click", () => line.classList.toggle("expanded"));
    }
  } else {
    line.textContent = step.thought;
  }
  turn.steps.appendChild(line);
}

function ensureTextEl(turn) {
  if (turn.thinking.isConnected) turn.thinking.remove();
  if (!turn.textEl) {
    turn.textEl = el("div", "answer-text");
    turn.answer.prepend(turn.textEl);
  }
  return turn.textEl;
}

function renderAnswerText(turn) {
  const textEl = ensureTextEl(turn);
  const grounding = turn.insight && turn.insight.grounding;
  if (!grounding) {
    textEl.textContent = turn.text;
    return;
  }
  // Rebuild the answer as sentence spans, using the server's character
  // offsets, so weakly supported sentences can be marked in place.
  textEl.replaceChildren();
  let cursor = 0;
  for (const s of grounding.sentences) {
    if (s.start > cursor) textEl.append(turn.text.slice(cursor, s.start));
    const span = el("span", "", turn.text.slice(s.start, s.end));
    if (s.support < 0.35) {
      span.className = "s-weak";
      span.title = t("weakSentence");
    } else if (s.source_rank) {
      span.className = "s-strong";
      span.dataset.rank = String(s.source_rank);
      span.title = t("strongSentence", { rank: s.source_rank });
    }
    textEl.append(span);
    cursor = s.end;
  }
  if (cursor < turn.text.length) textEl.append(turn.text.slice(cursor));
}

function meter(labelText, value, level) {
  const box = el("div");
  const label = el("div", "meter-label");
  const right = el("b", `txt-${level}`, `${formatPercent(value)} · ${t("grounding" + level[0].toUpperCase() + level.slice(1))}`);
  label.append(el("span", "", labelText), right);
  const bar = el("div", "meter");
  const fill = el("span", `lvl-${level}`);
  bar.appendChild(fill);
  box.append(label, bar);
  requestAnimationFrame(() => requestAnimationFrame(() => (fill.style.width = `${Math.round(value * 100)}%`)));
  return box;
}

function confidenceLevel(value) {
  return value >= 0.5 ? "high" : value >= 0.3 ? "medium" : "low";
}

function highlightRank(turn, rank, on) {
  turn.textEl?.querySelectorAll(`.s-strong[data-rank="${rank}"]`).forEach((s) => s.classList.toggle("hl", on));
}

function renderInsight(turn) {
  const insight = turn.insight;
  turn.answer.querySelector(".insight")?.remove();
  renderAnswerText(turn);
  if (!insight || !insight.used_knowledge_base) return;

  const box = el("div", "insight");
  const meters = el("div", "meters");
  if (insight.grounding) {
    const m = meter(t("groundingTitle"), insight.grounding.score, insight.grounding.level);
    m.title = t("groundingExplain");
    meters.appendChild(m);
  }
  const c = meter(t("confidenceTitle"), insight.confidence, confidenceLevel(insight.confidence));
  meters.appendChild(c);
  box.appendChild(meters);

  if (insight.gap) box.appendChild(el("div", "gap-warning", t("gapWarning")));

  if (insight.hits.length) {
    const sources = el("div", "sources");
    sources.appendChild(el("div", "section-label", t("sourcesTitle")));
    for (const hit of insight.hits) {
      const card = el("button", "source");
      card.type = "button";
      card.title = t("showOnMap");
      const body = el("div");
      body.style.minWidth = "0";
      body.append(el("div", "src-name", hit.source), el("div", "src-text", hit.preview));
      const score = el("div", "scorebar");
      score.title = `${t("scoreExplain")}\n${t("scoreSemantic")}: ${hit.vector_score.toFixed(2)} · ${t("scoreLexical")}: ${hit.lexical_score.toFixed(2)}`;
      const bar = el("div", "bar");
      const sem = el("span", "sem");
      sem.style.width = `${Math.max(0, 65 * hit.vector_score)}%`;
      const lex = el("span", "lex");
      lex.style.width = `${Math.max(0, 35 * hit.lexical_score)}%`;
      bar.append(sem, lex);
      score.append(el("span", "", formatPercent(Math.max(0, Math.min(1, hit.score)))), bar);
      card.append(el("span", "rank", String(hit.rank)), body, score);
      card.addEventListener("mouseenter", () => {
        highlightRank(turn, hit.rank, true);
        Cortex.pulse(hit.id);
      });
      card.addEventListener("mouseleave", () => highlightRank(turn, hit.rank, false));
      card.addEventListener("click", () => {
        Cortex.pulse(hit.id);
        openChunk(hit.id);
      });
      sources.appendChild(card);
    }
    box.appendChild(sources);
  }
  turn.answer.appendChild(box);
}

function setBusy(busy) {
  state.busy = busy;
  sendBtn.disabled = busy;
}

function sendMessage(text) {
  hideHero();
  setBusy(true);
  messagesEl.appendChild(el("div", "msg user", text));
  const turn = newTurn();
  const source = new EventSource(`/api/chat/stream?message=${encodeURIComponent(text)}`);

  source.addEventListener("trace", (event) => {
    addTraceStep(turn, JSON.parse(event.data));
    scrollToBottom();
  });
  source.addEventListener("answer_chunk", (event) => {
    turn.text += event.data;
    ensureTextEl(turn).textContent = turn.text;
    scrollToBottom();
  });
  source.addEventListener("answer_final", (event) => {
    turn.text = event.data;
    ensureTextEl(turn).textContent = turn.text;
  });
  source.addEventListener("insight", (event) => {
    turn.insight = JSON.parse(event.data);
    renderInsight(turn);
    if (turn.insight.used_knowledge_base) {
      Cortex.showQuery(turn.insight.query_point, turn.insight.hits, t("mapQuery"));
    }
    if (turn.insight.gap) loadGaps();
    scrollToBottom();
  });
  source.addEventListener("done", () => {
    source.close();
    ensureTextEl(turn);
    setBusy(false);
  });
  source.onerror = () => {
    source.close();
    const textEl = ensureTextEl(turn);
    if (!turn.text) {
      textEl.textContent = t("chatConnectionError");
      textEl.classList.add("error");
    }
    setBusy(false);
  };
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text || state.busy) return;
  inputEl.value = "";
  autosize();
  sendMessage(text);
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

function autosize() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
}
inputEl.addEventListener("input", autosize);

// ── Ingestion & media ──────────────────────────────────────────────────

function logKb(text, cls) {
  const log = $("kb-log");
  const item = el("li", cls || "", text);
  log.prepend(item);
  while (log.children.length > 4) log.lastElementChild.remove();
  return item;
}

async function ingestFiles(files) {
  for (const file of files) {
    const item = logKb(t("kbIngesting", { file: file.name }));
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/ingest/file", { method: "POST", body: form });
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      item.textContent = t("kbIngested", { file: file.name, added: t("kbChunks", { count: data.chunks_added }) });
      item.className = "ok";
    } catch (err) {
      item.textContent = t("kbIngestFailed", { file: file.name });
      item.className = "err";
    }
  }
  loadStatus();
  loadMap();
}

async function processMedia(file, endpoint, resultKey) {
  hideHero();
  const key = resultKey === "transcript" ? "mediaUploadedAudio" : "mediaUploadedVideo";
  messagesEl.appendChild(el("div", "msg user", t(key, { file: file.name })));
  const turn = newTurn();
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch(endpoint, { method: "POST", body: form });
    const data = await res.json();
    turn.text = data[resultKey] || data.detail || t("mediaNoResult");
  } catch (err) {
    turn.text = t("mediaProcessingFailed");
    ensureTextEl(turn).classList.add("error");
  }
  ensureTextEl(turn).textContent = turn.text;
  scrollToBottom();
}

function wireDropzone(labelId, inputId, onFiles) {
  const label = $(labelId);
  const input = $(inputId);
  input.addEventListener("change", (event) => {
    const files = [...event.target.files];
    if (files.length) onFiles(files);
    input.value = "";
  });
  ["dragenter", "dragover"].forEach((evt) =>
    label.addEventListener(evt, (event) => {
      event.preventDefault();
      label.classList.add("dragover");
    })
  );
  ["dragleave", "dragend"].forEach((evt) => label.addEventListener(evt, () => label.classList.remove("dragover")));
  label.addEventListener("drop", (event) => {
    event.preventDefault();
    label.classList.remove("dragover");
    const files = [...event.dataTransfer.files];
    if (files.length) onFiles(files);
  });
}

wireDropzone("file-input-label", "file-input", ingestFiles);
wireDropzone("audio-input-label", "audio-input", (files) => processMedia(files[0], "/api/media/audio", "transcript"));
wireDropzone("video-input-label", "video-input", (files) => processMedia(files[0], "/api/media/video", "analysis"));

// ── Drawers (narrow screens) ───────────────────────────────────────────

const scrim = $("scrim");
function closeDrawers() {
  $("sidebar").classList.remove("open");
  $("cortex").classList.remove("open");
  scrim.hidden = true;
}
function openDrawer(id) {
  closeDrawers();
  $(id).classList.add("open");
  scrim.hidden = false;
}
$("sidebar-open").addEventListener("click", () => openDrawer("sidebar"));
$("sidebar-close").addEventListener("click", closeDrawers);
$("cortex-open").addEventListener("click", () => openDrawer("cortex"));
$("cortex-close").addEventListener("click", closeDrawers);
scrim.addEventListener("click", closeDrawers);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeDrawers();
});

// ── Language / theme changes ───────────────────────────────────────────

window.onLanguageChange = () => {
  renderStatus();
  renderExamples();
  renderTopics();
  renderGaps();
  initThemeSwitcher($("theme-switch"));
  Cortex.setQueryLabel(t("mapQuery"));
  for (const turn of state.turns) {
    if (turn.traceCount) renderTraceSummary(turn);
    if (turn.insight) renderInsight(turn);
  }
};
window.onThemeChange = () => {
  Cortex.refreshTheme();
  renderTopics();
};

// ── Boot ───────────────────────────────────────────────────────────────

initLanguageSwitcher($("lang-select"));
initThemeSwitcher($("theme-switch"));
Cortex.init({
  canvas: $("map-canvas"),
  tooltip: $("map-tooltip"),
  empty: $("map-empty"),
  onPointClick: openChunk,
});
renderExamples();
renderGaps();
loadStatus();
loadMap();
loadGaps();
setInterval(loadStatus, 20000);
inputEl.focus();
