// Cognivore chat UI -- vanilla JS, no build step, no framework.
// Talks to the FastAPI backend via /api/chat/stream (SSE).
// Text strings come from i18n.js (t(), setLang()) rather than being
// hardcoded here, so the UI works the same in any supported language.

const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");

let lastHealth = null;

function renderStatus() {
  if (!lastHealth) return;
  const backendLabel = lastHealth.llm_model
    ? `${lastHealth.llm_backend} (${lastHealth.llm_model})`
    : lastHealth.llm_backend;
  document.getElementById("status-backend").textContent = backendLabel;
  document.getElementById("status-native").textContent = lastHealth.native_index
    ? t("statusNativeYes")
    : t("statusNativeNo");
  document.getElementById("status-tools").textContent = lastHealth.tools.join(", ");
}

async function loadStatus() {
  try {
    const res = await fetch("/api/health");
    lastHealth = await res.json();
    renderStatus();
  } catch (err) {
    document.getElementById("status-backend").textContent = t("statusUnreachable");
  }
}

// Re-render the parts of the UI that come from live data (not just the
// static data-i18n text) whenever the language changes.
window.onLanguageChange = () => {
  renderStatus();
};

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
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
    line.innerHTML = `<b>${escapeHtml(step.action)}</b>(${escapeHtml(JSON.stringify(step.action_input))}) &rarr; ${escapeHtml((step.observation || "").slice(0, 200))}`;
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
  const answerEl = addMessage("assistant", "");
  const url = `/api/chat/stream?message=${encodeURIComponent(text)}`;
  const source = new EventSource(url);

  source.addEventListener("trace", (event) => {
    addTraceStep(JSON.parse(event.data));
  });
  source.addEventListener("answer_chunk", (event) => {
    answerEl.textContent += event.data;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
  source.addEventListener("done", () => {
    source.close();
  });
  source.onerror = () => {
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

document.getElementById("file-input").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
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
});

async function uploadMedia(inputId, endpoint, resultLabel) {
  document.getElementById(inputId).addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
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
  });
}

uploadMedia("audio-input", "/api/media/audio", "transcript");
uploadMedia("video-input", "/api/media/video", "analysis");

initLanguageSwitcher(document.getElementById("lang-select"));
loadStatus();
