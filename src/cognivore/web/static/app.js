// Cognivore chat UI -- vanilla JS, no build step, no framework.
// Talks to the FastAPI backend via /api/chat/stream (SSE).
// Text strings come from i18n.js (t(), setLang()) rather than being
// hardcoded here, so the UI works the same in any supported language.

const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const connDotEl = document.getElementById("conn-dot");

let lastHealth = null;
let lastConnOk = null;

function renderStatus() {
  if (lastHealth) {
    const backendLabel = lastHealth.llm_model
      ? `${lastHealth.llm_backend} (${lastHealth.llm_model})`
      : lastHealth.llm_backend;
    document.getElementById("status-backend").textContent = backendLabel;
    document.getElementById("status-native").textContent = lastHealth.native_index
      ? t("statusNativeYes")
      : t("statusNativeNo");
    document.getElementById("status-tools").textContent = lastHealth.tools.join(", ");
  } else {
    document.getElementById("status-backend").textContent = t("statusUnreachable");
  }
  if (lastConnOk !== null) {
    connDotEl.classList.toggle("online", lastConnOk);
    connDotEl.classList.toggle("offline", !lastConnOk);
    connDotEl.title = t(lastConnOk ? "connOnline" : "connOffline");
  }
}

async function loadStatus() {
  try {
    const res = await fetch("/api/health");
    lastHealth = await res.json();
    lastConnOk = true;
  } catch (err) {
    lastConnOk = false;
  }
  renderStatus();
}

// Re-render the parts of the UI that come from live data (not just the
// static data-i18n text) whenever the language changes, and regenerate
// the theme buttons' tooltips since those are built dynamically too.
window.onLanguageChange = () => {
  renderStatus();
  initThemeSwitcher(document.getElementById("theme-switch"));
};

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function addThinkingIndicator() {
  const el = document.createElement("div");
  el.className = "msg assistant thinking";
  el.innerHTML = '<span class="thinking-dots"><span></span><span></span><span></span></span>';
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function addTraceStep(step) {
  let container = messagesEl.lastElementChild;
  if (!container || !container.classList.contains("trace")) {
    container = document.createElement("div");
    container.className = "trace";
    messagesEl.appendChild(container);
  }
  const line = document.createElement("div");
  line.className = "step";
  if (step.action) {
    const observation = step.observation || "";
    const short = observation.slice(0, 200);
    const callText = `<b>${escapeHtml(step.action)}</b>(${escapeHtml(JSON.stringify(step.action_input))}) &rarr; `;
    line.innerHTML =
      `${callText}<span class="short">${escapeHtml(short)}${observation.length > 200 ? "…" : ""}</span>` +
      `<span class="full">${escapeHtml(observation)}</span>`;
    // Click a trace line to see the full, untruncated observation --
    // useful when demoing (people ask "what exactly did it find?") and a
    // small bit of live interactivity in an otherwise read-only trace.
    if (observation.length > 200) {
      line.addEventListener("click", () => line.classList.toggle("expanded"));
    }
  } else {
    line.textContent = step.thought;
  }
  container.appendChild(line);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function sendMessage(text) {
  addMessage("user", text);
  const thinkingEl = addThinkingIndicator();
  let answerEl = null;
  const url = `/api/chat/stream?message=${encodeURIComponent(text)}`;
  const source = new EventSource(url);

  const clearThinking = () => {
    if (thinkingEl.isConnected) thinkingEl.remove();
  };

  source.addEventListener("trace", (event) => {
    clearThinking();
    addTraceStep(JSON.parse(event.data));
  });
  source.addEventListener("answer_chunk", (event) => {
    clearThinking();
    if (!answerEl) answerEl = addMessage("assistant", "");
    answerEl.textContent += event.data;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
  source.addEventListener("done", () => {
    clearThinking();
    source.close();
  });
  source.onerror = () => {
    clearThinking();
    if (!answerEl) {
      answerEl = addMessage("assistant", "");
    }
    if (!answerEl.textContent) {
      answerEl.textContent = t("chatConnectionError");
      answerEl.classList.add("error");
    }
    source.close();
  };
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  sendMessage(text);
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

async function ingestFile(file) {
  const statusEl = document.getElementById("kb-status");
  statusEl.textContent = t("kbIngesting", { file: file.name });
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/api/ingest/file", { method: "POST", body: form });
    const data = await res.json();
    statusEl.textContent = t("kbIngested", { added: data.chunks_added, total: data.total_chunks });
  } catch (err) {
    statusEl.textContent = t("kbIngestFailed");
  }
}

async function processMedia(file, endpoint, resultLabel) {
  const uploadedKey = resultLabel === "transcript" ? "mediaUploadedAudio" : "mediaUploadedVideo";
  addMessage("user", t(uploadedKey, { file: file.name }));
  const placeholder = addMessage("assistant", t("mediaProcessing"));
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch(endpoint, { method: "POST", body: form });
    const data = await res.json();
    placeholder.textContent = data[resultLabel] || data.detail || t("mediaNoResult");
  } catch (err) {
    placeholder.textContent = t("mediaProcessingFailed");
    placeholder.classList.add("error");
  }
}

function wireDropzone(labelId, inputId, onFile) {
  const label = document.getElementById(labelId);
  const input = document.getElementById(inputId);

  input.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) onFile(file);
  });

  // The label already opens the native file picker on click (standard
  // <label for>); what's missing without this is the actually-drag-and-
  // drop part the dropzone text promises -- a plain <label>/<div> doesn't
  // receive dropped files on its own, it just needs these three events.
  ["dragenter", "dragover"].forEach((evt) =>
    label.addEventListener(evt, (event) => {
      event.preventDefault();
      label.classList.add("dragover");
    })
  );
  ["dragleave", "dragend"].forEach((evt) =>
    label.addEventListener(evt, () => label.classList.remove("dragover"))
  );
  label.addEventListener("drop", (event) => {
    event.preventDefault();
    label.classList.remove("dragover");
    const file = event.dataTransfer.files[0];
    if (file) onFile(file);
  });
}

wireDropzone("file-input-label", "file-input", ingestFile);
wireDropzone("audio-input-label", "audio-input", (file) =>
  processMedia(file, "/api/media/audio", "transcript")
);
wireDropzone("video-input-label", "video-input", (file) =>
  processMedia(file, "/api/media/video", "analysis")
);

initLanguageSwitcher(document.getElementById("lang-select"));
initThemeSwitcher(document.getElementById("theme-switch"));
loadStatus();
setInterval(loadStatus, 20000);
