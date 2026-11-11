const upload = document.querySelector("#upload");
const demo = document.querySelector("#demo");
const status = document.querySelector("#status");
const panel = document.querySelector("#review");
const title = document.querySelector("#title");
const findings = document.querySelector("#findings");
const publish = document.querySelector("#publish");
const download = document.querySelector("#download");
const labels = {
  changed: "Changed requirement",
  added: "Added requirement",
  removed: "Removed requirement",
  ambiguous: "Possible match — check both sources",
};
let current;
let busy = false;

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === "string" ? error.detail : `Request failed (${response.status}).`);
  }
  return response;
}

function element(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}

function citation(side, value) {
  const node = document.createElement("section");
  const source = value ? `${side} · ${value.source}:${value.line}` : side;
  node.append(element("h4", source), element("blockquote", value?.text ?? "No matching line"));
  return node;
}

function render(review, focusIndex, focusChoice) {
  current = review;
  panel.hidden = false;
  title.textContent = `${review.before_name} → ${review.after_name}`;
  findings.replaceChildren();
  if (!review.findings.length) findings.append(element("p", "No requirement changes found."));
  review.findings.forEach((finding, index) => {
    const card = document.createElement("article");
    card.append(element("h3", labels[finding.kind]));
    if (finding.signals.length) card.append(element("p", finding.signals.join(" · ")));
    const sources = document.createElement("div");
    sources.className = "sources";
    sources.append(citation("Before", finding.before), citation("After", finding.after));
    card.append(sources);
    if (finding.alternatives.length) {
      card.append(element("h4", "Other possible matches"));
      const alternatives = document.createElement("div");
      alternatives.className = "sources";
      for (const candidate of finding.alternatives) alternatives.append(citation("After", candidate));
      card.append(alternatives);
    }
    if (!review.published) {
      const choices = document.createElement("div");
      choices.className = "choices";
      for (const [decision, label] of [["accepted", "Accept"], ["rejected", "Reject"], ["pending", "Reset"]]) {
        const button = element("button", label);
        button.type = "button";
        button.id = `choice-${index}-${decision}`;
        button.setAttribute("aria-pressed", String(review.decisions[index] === decision));
        button.addEventListener("click", () => perform(async () => {
          const response = await request(`/api/reviews/${review.id}/findings/${index}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ decision }),
          });
          render(await response.json(), index, decision);
        }));
        choices.append(button);
      }
      card.append(choices);
    }
    findings.append(card);
  });
  publish.hidden = review.published;
  publish.disabled = review.decisions.includes("pending");
  download.hidden = !review.published;
  download.href = `/api/reviews/${review.id}/report`;
  const pending = review.decisions.filter((decision) => decision === "pending").length;
  status.textContent = review.published ? "Report published." : `${pending} findings to review.`;
  if (focusIndex !== undefined) document.getElementById(`choice-${focusIndex}-${focusChoice}`)?.focus();
}

async function perform(operation) {
  if (busy) return;
  busy = true;
  panel.setAttribute("aria-busy", "true");
  status.textContent = "Working…";
  try {
    await operation();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    busy = false;
    panel.removeAttribute("aria-busy");
  }
}

async function compare(data) {
  const response = await request("/api/reviews", { method: "POST", body: data });
  const review = await response.json();
  const url = new URL(location.href);
  url.searchParams.set("id", review.id);
  history.replaceState(null, "", url);
  render(review);
  title.setAttribute("tabindex", "-1");
  title.focus();
}

upload.addEventListener("submit", (event) => {
  event.preventDefault();
  perform(() => compare(new FormData(upload)));
});

demo.addEventListener("click", () => perform(async () => {
  const data = new FormData();
  for (const side of ["before", "after"]) {
    const name = `demo-${side}.md`;
    const response = await request(`/${name}`);
    data.append(side, await response.blob(), name);
  }
  await compare(data);
}));

publish.addEventListener("click", () => perform(async () => {
  const response = await request(`/api/reviews/${current.id}/publish`, { method: "POST" });
  render(await response.json());
  download.focus();
}));

const saved = new URL(location.href).searchParams.get("id");
if (saved) perform(async () => {
  const response = await request(`/api/reviews/${encodeURIComponent(saved)}`);
  render(await response.json());
});
