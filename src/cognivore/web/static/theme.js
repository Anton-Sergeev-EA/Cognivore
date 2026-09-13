// Cognivore theme switcher. Kept in its own file (like i18n.js) so the
// blocking inline snippet in <head> can set the theme attribute before
// first paint (avoiding a flash of the wrong theme) without needing this
// whole file to load synchronously -- only the tiny inline copy of
// detectInitialTheme() needs to run that early.

const THEMES = ["dark", "light", "aurora"];
const THEME_STORAGE_KEY = "cognivore-theme";
const THEME_ICONS = { dark: "\u{1F319}", light: "☀️", aurora: "\u{1F30C}" };
const THEME_LABEL_KEYS = { dark: "themeDark", light: "themeLight", aurora: "themeAurora" };

function detectInitialTheme() {
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved && THEMES.includes(saved)) return saved;
  } catch (e) {
    /* localStorage unavailable -- fall through to system preference */
  }
  const prefersLight =
    typeof matchMedia === "function" && matchMedia("(prefers-color-scheme: light)").matches;
  return prefersLight ? "light" : "dark";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
}

function setTheme(theme) {
  if (!THEMES.includes(theme)) return;
  applyTheme(theme);
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch (e) {
    /* ignore -- per-viewer convenience only */
  }
  document.querySelectorAll("#theme-switch [data-theme-btn]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.themeBtn === theme);
  });
}

function initThemeSwitcher(container) {
  container.innerHTML = "";
  const current = document.documentElement.getAttribute("data-theme") || detectInitialTheme();
  for (const theme of THEMES) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.dataset.themeBtn = theme;
    btn.textContent = THEME_ICONS[theme];
    btn.className = "theme-btn" + (theme === current ? " active" : "");
    btn.title = typeof t === "function" ? t(THEME_LABEL_KEYS[theme]) : theme;
    btn.addEventListener("click", () => setTheme(theme));
    container.appendChild(btn);
  }
}
