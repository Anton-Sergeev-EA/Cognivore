// Cognivore UI translations. Vanilla JS, no build step -- kept as a single
// dictionary object so the whole UI can be relabeled without touching
// index.html/app.js logic. Names, emails and code stay untranslated on
// purpose; only UI copy is.

const SUPPORTED_LANGS = ["en", "ru", "de", "fr", "it", "es"];
const LANG_LABELS = {
  en: "English",
  ru: "Русский",
  de: "Deutsch",
  fr: "Français",
  it: "Italiano",
  es: "Español",
};
const DEFAULT_LANG = "en";
const LANG_STORAGE_KEY = "cognivore-lang";

const TRANSLATIONS = {
  en: {
    tagline: "Local-first agent · RAG · audio · video",
    statusTitle: "Status",
    statusBackendLabel: "backend",
    statusNativeLabel: "native index",
    statusToolsLabel: "tools",
    statusUnreachable: "unreachable",
    statusNativeYes: "yes (C++)",
    statusNativeNo: "no (NumPy fallback)",
    kbTitle: "Knowledge base",
    kbDropzone: "Drop a .txt/.md file or click to ingest",
    kbIngesting: "Ingesting {file}…",
    kbIngested: "+{added} chunks (total {total})",
    kbIngestFailed: "ingest failed",
    mediaTitle: "Media tools",
    mediaAudio: "Transcribe audio",
    mediaVideo: "Analyze video",
    mediaUploadedAudio: "[uploaded audio: {file}]",
    mediaUploadedVideo: "[uploaded video: {file}]",
    mediaProcessing: "Processing…",
    mediaProcessingFailed: "Processing failed.",
    mediaNoResult: "(no result)",
    chatPlaceholder: "Ask Cognivore anything…",
    chatSend: "Send",
    chatConnectionError: "(connection error -- is the server running?)",
    contactTitle: "Contact",
    themeDark: "Dark theme",
    themeLight: "Light theme",
    themeAurora: "Aurora theme",
    connOnline: "Connected",
    connOffline: "Disconnected",
  },
  ru: {
    tagline: "Локальный агент · RAG · аудио · видео",
    statusTitle: "Статус",
    statusBackendLabel: "модель",
    statusNativeLabel: "нативный индекс",
    statusToolsLabel: "инструменты",
    statusUnreachable: "недоступно",
    statusNativeYes: "да (C++)",
    statusNativeNo: "нет (резервный вариант на NumPy)",
    kbTitle: "База знаний",
    kbDropzone: "Перетащите .txt/.md файл или нажмите, чтобы загрузить",
    kbIngesting: "Загрузка {file}…",
    kbIngested: "+{added} фрагментов (всего {total})",
    kbIngestFailed: "ошибка загрузки",
    mediaTitle: "Медиа-инструменты",
    mediaAudio: "Распознать аудио",
    mediaVideo: "Анализ видео",
    mediaUploadedAudio: "[загружено аудио: {file}]",
    mediaUploadedVideo: "[загружено видео: {file}]",
    mediaProcessing: "Обработка…",
    mediaProcessingFailed: "Ошибка обработки.",
    mediaNoResult: "(нет результата)",
    chatPlaceholder: "Задайте Cognivore любой вопрос…",
    chatSend: "Отправить",
    chatConnectionError: "(ошибка соединения — сервер запущен?)",
    contactTitle: "Контакты",
    themeDark: "Тёмная тема",
    themeLight: "Светлая тема",
    themeAurora: "Тема «Аврора»",
    connOnline: "Подключено",
    connOffline: "Нет соединения",
  },
  de: {
    tagline: "Lokaler Agent · RAG · Audio · Video",
    statusTitle: "Status",
    statusBackendLabel: "Backend",
    statusNativeLabel: "nativer Index",
    statusToolsLabel: "Werkzeuge",
    statusUnreachable: "nicht erreichbar",
    statusNativeYes: "ja (C++)",
    statusNativeNo: "nein (NumPy-Fallback)",
    kbTitle: "Wissensdatenbank",
    kbDropzone: "Ziehen Sie eine .txt/.md-Datei hierher oder klicken Sie zum Hochladen",
    kbIngesting: "Lade {file} hoch…",
    kbIngested: "+{added} Abschnitte (insgesamt {total})",
    kbIngestFailed: "Upload fehlgeschlagen",
    mediaTitle: "Medien-Werkzeuge",
    mediaAudio: "Audio transkribieren",
    mediaVideo: "Video analysieren",
    mediaUploadedAudio: "[Audio hochgeladen: {file}]",
    mediaUploadedVideo: "[Video hochgeladen: {file}]",
    mediaProcessing: "Verarbeitung…",
    mediaProcessingFailed: "Verarbeitung fehlgeschlagen.",
    mediaNoResult: "(kein Ergebnis)",
    chatPlaceholder: "Fragen Sie Cognivore etwas…",
    chatSend: "Senden",
    chatConnectionError: "(Verbindungsfehler — läuft der Server?)",
    contactTitle: "Kontakt",
    themeDark: "Dunkles Design",
    themeLight: "Helles Design",
    themeAurora: "Aurora-Design",
    connOnline: "Verbunden",
    connOffline: "Nicht verbunden",
  },
  fr: {
    tagline: "Agent local · RAG · audio · vidéo",
    statusTitle: "Statut",
    statusBackendLabel: "moteur",
    statusNativeLabel: "index natif",
    statusToolsLabel: "outils",
    statusUnreachable: "inaccessible",
    statusNativeYes: "oui (C++)",
    statusNativeNo: "non (repli NumPy)",
    kbTitle: "Base de connaissances",
    kbDropzone: "Déposez un fichier .txt/.md ou cliquez pour l'ajouter",
    kbIngesting: "Import de {file}…",
    kbIngested: "+{added} extraits (total {total})",
    kbIngestFailed: "échec de l'import",
    mediaTitle: "Outils multimédias",
    mediaAudio: "Transcrire l'audio",
    mediaVideo: "Analyser la vidéo",
    mediaUploadedAudio: "[audio importé : {file}]",
    mediaUploadedVideo: "[vidéo importée : {file}]",
    mediaProcessing: "Traitement…",
    mediaProcessingFailed: "Échec du traitement.",
    mediaNoResult: "(aucun résultat)",
    chatPlaceholder: "Posez une question à Cognivore…",
    chatSend: "Envoyer",
    chatConnectionError: "(erreur de connexion — le serveur est-il lancé ?)",
    contactTitle: "Contact",
    themeDark: "Thème sombre",
    themeLight: "Thème clair",
    themeAurora: "Thème Aurora",
    connOnline: "Connecté",
    connOffline: "Déconnecté",
  },
  it: {
    tagline: "Agente locale · RAG · audio · video",
    statusTitle: "Stato",
    statusBackendLabel: "backend",
    statusNativeLabel: "indice nativo",
    statusToolsLabel: "strumenti",
    statusUnreachable: "non raggiungibile",
    statusNativeYes: "sì (C++)",
    statusNativeNo: "no (fallback NumPy)",
    kbTitle: "Base di conoscenza",
    kbDropzone: "Rilascia un file .txt/.md o clicca per caricarlo",
    kbIngesting: "Caricamento di {file}…",
    kbIngested: "+{added} frammenti (totale {total})",
    kbIngestFailed: "caricamento non riuscito",
    mediaTitle: "Strumenti multimediali",
    mediaAudio: "Trascrivi audio",
    mediaVideo: "Analizza video",
    mediaUploadedAudio: "[audio caricato: {file}]",
    mediaUploadedVideo: "[video caricato: {file}]",
    mediaProcessing: "Elaborazione…",
    mediaProcessingFailed: "Elaborazione non riuscita.",
    mediaNoResult: "(nessun risultato)",
    chatPlaceholder: "Chiedi qualcosa a Cognivore…",
    chatSend: "Invia",
    chatConnectionError: "(errore di connessione — il server è attivo?)",
    contactTitle: "Contatti",
    themeDark: "Tema scuro",
    themeLight: "Tema chiaro",
    themeAurora: "Tema Aurora",
    connOnline: "Connesso",
    connOffline: "Disconnesso",
  },
  es: {
    tagline: "Agente local · RAG · audio · video",
    statusTitle: "Estado",
    statusBackendLabel: "backend",
    statusNativeLabel: "índice nativo",
    statusToolsLabel: "herramientas",
    statusUnreachable: "inaccesible",
    statusNativeYes: "sí (C++)",
    statusNativeNo: "no (alternativa NumPy)",
    kbTitle: "Base de conocimiento",
    kbDropzone: "Suelta un archivo .txt/.md o haz clic para cargarlo",
    kbIngesting: "Cargando {file}…",
    kbIngested: "+{added} fragmentos (total {total})",
    kbIngestFailed: "error al cargar",
    mediaTitle: "Herramientas multimedia",
    mediaAudio: "Transcribir audio",
    mediaVideo: "Analizar video",
    mediaUploadedAudio: "[audio cargado: {file}]",
    mediaUploadedVideo: "[video cargado: {file}]",
    mediaProcessing: "Procesando…",
    mediaProcessingFailed: "Error al procesar.",
    mediaNoResult: "(sin resultado)",
    chatPlaceholder: "Pregúntale algo a Cognivore…",
    chatSend: "Enviar",
    chatConnectionError: "(error de conexión: ¿el servidor está en marcha?)",
    contactTitle: "Contacto",
    themeDark: "Tema oscuro",
    themeLight: "Tema claro",
    themeAurora: "Tema Aurora",
    connOnline: "Conectado",
    connOffline: "Desconectado",
  },
};

function detectInitialLang() {
  try {
    const saved = localStorage.getItem(LANG_STORAGE_KEY);
    if (saved && SUPPORTED_LANGS.includes(saved)) return saved;
  } catch (e) {
    /* localStorage unavailable (private mode etc.) -- fall through */
  }
  const nav = (navigator.language || "en").slice(0, 2).toLowerCase();
  return SUPPORTED_LANGS.includes(nav) ? nav : DEFAULT_LANG;
}

let currentLang = detectInitialLang();

function t(key, vars) {
  const dict = TRANSLATIONS[currentLang] || TRANSLATIONS[DEFAULT_LANG];
  let str = dict[key] ?? TRANSLATIONS[DEFAULT_LANG][key] ?? key;
  if (vars) {
    for (const [k, v] of Object.entries(vars)) {
      str = str.replace(`{${k}}`, v);
    }
  }
  return str;
}

function applyStaticTranslations() {
  document.documentElement.lang = currentLang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  document.querySelectorAll("[data-i18n-title]").forEach((el) => {
    el.title = t(el.dataset.i18nTitle);
  });
}

function setLang(lang) {
  if (!SUPPORTED_LANGS.includes(lang)) return;
  currentLang = lang;
  try {
    localStorage.setItem(LANG_STORAGE_KEY, lang);
  } catch (e) {
    /* ignore -- per-viewer convenience only */
  }
  applyStaticTranslations();
  if (typeof window.onLanguageChange === "function") {
    window.onLanguageChange(lang);
  }
}

function initLanguageSwitcher(selectEl) {
  selectEl.innerHTML = "";
  for (const code of SUPPORTED_LANGS) {
    const opt = document.createElement("option");
    opt.value = code;
    opt.textContent = LANG_LABELS[code];
    selectEl.appendChild(opt);
  }
  selectEl.value = currentLang;
  selectEl.addEventListener("change", () => setLang(selectEl.value));
  applyStaticTranslations();
}
