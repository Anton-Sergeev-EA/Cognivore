// Cognivore UI translations. Vanilla JS, no build step -- kept as a single
// dictionary object so the whole UI can be relabeled without touching
// index.html/app.js logic. Names, emails and code stay untranslated on
// purpose; only UI copy is.

const SUPPORTED_LANGS = ["en", "ru", "de", "fr", "it", "es", "zh", "ja", "hi"];
const LANG_LABELS = {
  en: "English",
  ru: "Русский",
  de: "Deutsch",
  fr: "Français",
  it: "Italiano",
  es: "Español",
  zh: "中文",
  ja: "日本語",
  hi: "हिन्दी",
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
  zh: {
    tagline: "本地优先智能体 · RAG · 音频 · 视频",
    statusTitle: "状态",
    statusBackendLabel: "后端",
    statusNativeLabel: "原生索引",
    statusToolsLabel: "工具",
    statusUnreachable: "无法连接",
    statusNativeYes: "是（C++）",
    statusNativeNo: "否（NumPy 回退方案）",
    kbTitle: "知识库",
    kbDropzone: "拖放 .txt/.md 文件，或点击上传",
    kbIngesting: "正在导入 {file}…",
    kbIngested: "+{added} 个片段（共 {total} 个）",
    kbIngestFailed: "导入失败",
    mediaTitle: "媒体工具",
    mediaAudio: "转录音频",
    mediaVideo: "分析视频",
    mediaUploadedAudio: "[已上传音频：{file}]",
    mediaUploadedVideo: "[已上传视频：{file}]",
    mediaProcessing: "处理中…",
    mediaProcessingFailed: "处理失败。",
    mediaNoResult: "（无结果）",
    chatPlaceholder: "向 Cognivore 提问…",
    chatSend: "发送",
    chatConnectionError: "（连接错误——服务器正在运行吗？）",
    contactTitle: "联系方式",
    themeDark: "深色主题",
    themeLight: "浅色主题",
    themeAurora: "极光主题",
    connOnline: "已连接",
    connOffline: "已断开",
  },
  ja: {
    tagline: "ローカル優先エージェント · RAG · 音声 · 動画",
    statusTitle: "ステータス",
    statusBackendLabel: "バックエンド",
    statusNativeLabel: "ネイティブインデックス",
    statusToolsLabel: "ツール",
    statusUnreachable: "接続不可",
    statusNativeYes: "あり（C++）",
    statusNativeNo: "なし（NumPy フォールバック）",
    kbTitle: "ナレッジベース",
    kbDropzone: ".txt/.md ファイルをドロップ、またはクリックして取り込み",
    kbIngesting: "{file} を取り込み中…",
    kbIngested: "+{added} 件のチャンク（合計 {total} 件）",
    kbIngestFailed: "取り込みに失敗しました",
    mediaTitle: "メディアツール",
    mediaAudio: "音声を文字起こし",
    mediaVideo: "動画を解析",
    mediaUploadedAudio: "[音声をアップロード：{file}]",
    mediaUploadedVideo: "[動画をアップロード：{file}]",
    mediaProcessing: "処理中…",
    mediaProcessingFailed: "処理に失敗しました。",
    mediaNoResult: "（結果なし）",
    chatPlaceholder: "Cognivore に質問する…",
    chatSend: "送信",
    chatConnectionError: "（接続エラー — サーバーは起動していますか？）",
    contactTitle: "お問い合わせ",
    themeDark: "ダークテーマ",
    themeLight: "ライトテーマ",
    themeAurora: "オーロラテーマ",
    connOnline: "接続済み",
    connOffline: "未接続",
  },
  hi: {
    tagline: "लोकल-फ़र्स्ट एजेंट · RAG · ऑडियो · वीडियो",
    statusTitle: "स्थिति",
    statusBackendLabel: "बैकएंड",
    statusNativeLabel: "नेटिव इंडेक्स",
    statusToolsLabel: "टूल्स",
    statusUnreachable: "पहुँच योग्य नहीं",
    statusNativeYes: "हाँ (C++)",
    statusNativeNo: "नहीं (NumPy फ़ॉलबैक)",
    kbTitle: "नॉलेज बेस",
    kbDropzone: ".txt/.md फ़ाइल ड्रॉप करें या अपलोड के लिए क्लिक करें",
    kbIngesting: "{file} अपलोड हो रही है…",
    kbIngested: "+{added} खंड (कुल {total})",
    kbIngestFailed: "अपलोड विफल",
    mediaTitle: "मीडिया टूल्स",
    mediaAudio: "ऑडियो ट्रांसक्राइब करें",
    mediaVideo: "वीडियो का विश्लेषण करें",
    mediaUploadedAudio: "[ऑडियो अपलोड किया गया: {file}]",
    mediaUploadedVideo: "[वीडियो अपलोड किया गया: {file}]",
    mediaProcessing: "प्रोसेस हो रहा है…",
    mediaProcessingFailed: "प्रोसेसिंग विफल।",
    mediaNoResult: "(कोई परिणाम नहीं)",
    chatPlaceholder: "Cognivore से कुछ भी पूछें…",
    chatSend: "भेजें",
    chatConnectionError: "(कनेक्शन त्रुटि — क्या सर्वर चल रहा है?)",
    contactTitle: "संपर्क",
    themeDark: "डार्क थीम",
    themeLight: "लाइट थीम",
    themeAurora: "ऑरोरा थीम",
    connOnline: "कनेक्टेड",
    connOffline: "डिस्कनेक्टेड",
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
