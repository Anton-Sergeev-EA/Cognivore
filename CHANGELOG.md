# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this
project uses [Semantic Versioning](https://semver.org/).

## [0.2.0] - Unreleased

### Added

- **Explainability layer** (`cognivore.ml`, NumPy only):
  - a knowledge map of the whole knowledge base -- PCA to 2-D, k-means++
    topics with the number of topics chosen by silhouette score, topic
    names by class-based TF-IDF -- served at `GET /api/insights/map` and
    cached per store version;
  - sentence-level answer grounding (keyword recall + embedding cosine per
    sentence, with character offsets for in-place highlighting);
  - retrieval confidence with CJK-aware query coverage;
  - a persistent knowledge-gap radar (`GET`/`DELETE /api/insights/gaps`)
    that merges rephrased questions and ranks them by frequency.
- An `insight` object on `POST /api/chat` and a final `insight` SSE event
  (plus `answer_final` with the exact answer text) on `/api/chat/stream`.
- `GET /api/knowledge_base/{id}` to read a single chunk.
- A redesigned web UI: animated canvas knowledge map with beams from the
  question to its sources, grounding and confidence meters, weakly
  supported sentences underlined, source cards with a meaning-vs-words
  score breakdown, gap radar, onboarding screen with example questions,
  multi-file upload, responsive drawers for tablet and phone, and
  accessibility improvements (keyboard focus, ARIA labels, reduced-motion
  support).
- Demo knowledge bases in every UI language: a fictional company handbook
  each in Chinese, Spanish, Hindi, French, German, Japanese and Italian,
  seeded alongside the English and Russian ones, with UI example questions
  that each language's handbook answers; `tests/test_languages.py` checks
  every example and a gap question end to end in all nine languages.
- Query-focused snippets (`cognivore.rag.snippets`): the search tool now
  shows the model the part of each chunk that matches the question, not
  just its first 300 characters.
- Offline demo mode understands plain questions in any UI language (not
  only English keywords) and percentages in all of them ("15% от 4900",
  "15 % von 149", "999 的 15%", "149 का 15%", ...).
- Auto-context: before each turn the knowledge base is searched and the
  passages close to the best match are handed to the model with the
  instruction to answer only from them -- small local models (qwen2.5:3b)
  otherwise often answered without ever calling the search tool. The
  lookup shows up as a normal search step, so sources and grounding are
  always shown (`COGNIVORE_AUTO_CONTEXT`,
  `COGNIVORE_AUTO_CONTEXT_THRESHOLD`, `COGNIVORE_AUTO_CONTEXT_RELATIVE_SCORE`).
- Questions the knowledge base can't answer are handled end to end: the
  model is told the knowledge base was searched and has nothing on it (so
  it neither guesses nor sends the user to search the internet), the
  search tool reports "nothing found" instead of the least-unrelated
  passages, the question lands on the gap radar, and the UI shows no
  made-up sources or grounding for it. Arithmetic and small talk are not
  treated as knowledge questions (`cognivore.ml.intent`). By default
  (`COGNIVORE_STRICT_KNOWLEDGE_ANSWERS=true`) such a question is answered
  "the knowledge base has no information about this" directly, in the
  question's language, without calling the model -- a 3B model told
  exactly that still answered with its own meta-reasoning about half the
  time.
- Citation sentences ("This is from <source>") are shown muted instead of
  as part of the answer's claims.
- The answer language is named explicitly in the prompt; language
  detection now also tells Spanish, French, German and Italian apart.
- Upload size limits (`COGNIVORE_MAX_DOCUMENT_UPLOAD_MB`,
  `COGNIVORE_MAX_MEDIA_UPLOAD_MB`); oversized uploads get HTTP 413 instead
  of being buffered in memory.

### Changed

- The web UI is rewritten in all 9 languages, ordered Russian (default),
  English, then by global reach: Simplified Chinese, Spanish, Hindi,
  French, German, Japanese, Italian -- with proper plural forms
  (`Intl.PluralRules`) and locale number formatting. The browser language
  picks the initial one, falling back to Russian; the switcher is now a
  compact language menu.
- Default embedding model is now the multilingual
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (same
  384 dimensions) instead of the English-only `BAAI/bge-small-en-v1.5`.
- `POST /api/ingest/text` takes a JSON body (`{"source", "text"}`) instead
  of query parameters, so long documents no longer hit URL length limits.
- The Docker image persists the fastembed model cache in `/data`.

### Fixed

- `build_agent` treated an *empty* knowledge base as missing (an empty
  store is falsy) and gave the agent a separate store, so documents
  uploaded to a fresh install were never found in chat.
- Overlapping neighbour chunks no longer show up as duplicate sources, in
  the UI and in what the model reads.
- Switching the UI language after the first question left part of the
  interface in the previous language; runtime-generated texts (status
  lines, errors, upload log) are now re-translated too.
- A final answer written as a tool call ("Action: Final Answer" +
  "Action Input: ...", a common slip of small models) was dispatched as
  an unknown tool, and the model then relayed the error message to the
  user; it is now parsed as the final answer.
- A sentence that only attributes the answer to a source ("This is from
  the Skylark Cloud knowledge base.") was underlined as an unsupported
  claim; it is now recognised as a citation, and the model is asked not
  to name sources at all since the UI shows them.
- Knowledge-map topic labels no longer pile up or cover the question
  label; the map asks for at least three topics on larger knowledge bases.
- Hindi tokenization: Devanagari vowel signs are combining marks, not
  "word" characters, so words fell apart into bare consonants; the danda
  (।) is now also a sentence terminator.
- Sentence splitting silently dropped lines without terminal punctuation
  (headings, wrapped lines), which hurt both snippets and grounding.
- The offline hashing embedder ignores stopwords, so frequent function
  words no longer outrank topical ones.
- Chinese (and Japanese) retrieval: whitespace tokenization turned a whole
  CJK sentence into one token, so BM25 and the offline embedder scored
  every Chinese query 0 against every chunk. Tokenization is now
  CJK-aware (character bigrams) and strips punctuation in all languages.
- A saved knowledge base loaded with a different embedding model silently
  compared vectors from two different models; it is now re-embedded
  automatically (the embedder id is stored with the index).
- Ingestion and media uploads no longer block the event loop while
  embedding; the document store and the agent are safe to use from
  concurrent requests.
- Offline demo answers no longer echo every retrieved passage verbatim
  (source labels containing parentheses broke the passage parser).

## [0.1.0] - Unreleased

### Added

- Native C++ vector index (`FlatIndex` exact search, `NSWIndex` approximate
  graph search) via a hand-written pybind11 extension, with AVX2/FMA and
  OpenMP acceleration when available, and a pure-NumPy fallback when no C++
  toolchain is present at install time.
- A ReAct-style agent loop (`cognivore.agent`) with pluggable LLM backends:
  local GGUF inference via `llama-cpp-python`, or a dependency-free
  `FakeLLMBackend` for offline demos and tests.
- A hybrid (vector + BM25) RAG pipeline: recursive-splitter chunking,
  `fastembed`-based semantic embeddings with a hashing-trick fallback, and a
  persistent `DocumentStore`.
- Agent tools: safe (AST-based) calculator, knowledge-base search, audio
  transcription (`faster-whisper`), and video scene/OCR analysis (OpenCV).
- A FastAPI backend with SSE-streamed chat, file/audio/video ingestion
  endpoints, and a vanilla JS/HTML/CSS chat UI.
- `cognivore` CLI (`chat`, `ingest`, `serve`, `bench`) via Typer.
- Full test suite (pytest), lint/format (ruff), static typing (mypy, with a
  hand-written `.pyi` stub for the native extension), and a multi-OS/
  multi-Python CI matrix (GitHub Actions).
- Web UI localization: a language switcher (English, Russian, German,
  French, Italian, Spanish, Simplified Chinese, Japanese, Hindi) covering
  all interface copy, with the choice remembered per browser.
- Web UI theming (dark/light/aurora) and small live interactions: a
  working drag-and-drop for the knowledge-base/audio/video dropzones, an
  animated "thinking" indicator while the agent is working, a pulsing
  connection status dot, and expandable trace steps to see the full
  (untruncated) tool observation.
- `docker-compose.yml`: a one-command stack (Ollama + Cognivore, both
  containerized) that runs the same way on Windows, macOS, and Linux with
  nothing installed on the host but Docker. The `Dockerfile` itself now
  runs as a non-root user and ships a `HEALTHCHECK`. A new CI job builds
  the image and smoke-tests it (health endpoint, `HEALTHCHECK` status,
  non-root) on every push; tagged releases publish a multi-arch
  (amd64/arm64) image to GHCR via `docker-publish.yml`.

- Documentation: a rewritten `README.md` with a new `## Usage` section
  (CLI, Web UI, REST/SSE API, and library-usage examples, verified against
  the actual `cli.py`/`web/app.py`/`bootstrap.py` source), plus a full
  technical translation of it into all 8 other UI languages
  (`docs/i18n/README.<lang>.md` for ru, de, fr, it, es, zh, ja, hi) --
  code blocks, commands, paths, and benchmark numbers kept verbatim in
  every translation, only prose and table headers translated. `LICENSE`
  now names the author (Sergeev Anton, avsergeev1981@gmail.com).

- `COGNIVORE_SEED_DEMO_KB` / `cognivore seed-demo`: an empty knowledge base
  can now be seeded automatically (Docker Compose does this by default) or
  on demand with two bundled demo company handbooks (English + Russian --
  pricing, SLA, security, refund policy, support FAQ), packaged as
  installed package data so it works identically from source, a wheel, or
  the Docker image. Idempotent and never touches a knowledge base that
  already has real content in it.

### Changed

- README.md (and all 8 translations): documented the `docker compose down`
  / `up -d` stop-and-restart flow, that `restart: unless-stopped` brings
  both containers back on their own after a reboot if they weren't stopped
  manually first, and that `cognivore-data`/`ollama-data` survive `down`
  (only an explicit `down -v` removes them).
- README.md (and all 8 translations): documented that `analyze_video`'s
  OCR targets the content its own tool description names -- screencasts,
  lecture recordings, slide-based videos -- and is a much rougher ride on
  a raw terminal/IDE recording (small monospace font, compression
  artifacts right at scene cuts, glyphs like box-drawing characters the
  OCR model was never trained on). Tested that upscaling/thresholding the
  frame doesn't reliably help once compression has already discarded the
  detail, so the README points at recording with a larger font/higher
  resolution instead of promising a post-processing fix that doesn't
  actually work.

### Fixed

- On-screen text OCR in the video tool (`analyze_video`) silently produced
  no text at all: `pytesseract` was never declared as a dependency (in
  the `video` extra or anywhere else), so `_ocr_frame`'s `import
  pytesseract` always raised `ImportError`, which was caught and returned
  as `""` -- indistinguishable from "this frame just has no text on it."
  Scene detection and timestamps were unaffected (pure OpenCV), which is
  what made the gap easy to miss. Fixed by adding `pytesseract` to the
  `video` extra, installing the native `tesseract-ocr` binary it wraps in
  the Docker image's runtime stage, documenting the same binary
  requirement for native (non-Docker) installs, and adding real,
  non-mocked test coverage (`tests/test_video.py`) that exercises OCR
  end-to-end against a synthetic video with burned-in text.
- On-screen OCR was hardcoded to English-only, so Cyrillic text (a Russian
  terminal or UI in a screen recording, say) came back mis-recognized as
  look-alike Latin letters -- e.g. "Контейнеры запущены" as "KoHTewHepbi
  3anylueHbl" -- confirmed live, and easy to mistake for a bad-quality scan
  rather than a language mismatch, since Tesseract doesn't error out on a
  wrong-language request, it just quietly reads the wrong alphabet. Fixed
  by adding a new `ocr_languages` setting (`COGNIVORE_OCR_LANGUAGES`,
  default `"eng+rus"`, threaded through `VideoAnalyzeTool` ->
  `extract_keyframes` -> `_ocr_frame`), installing `tesseract-ocr-rus` in
  the Docker image alongside `tesseract-ocr-eng`, documenting the
  install step (and the silent-mismatch pitfall) for native installs, and
  adding a real regression test pair in `tests/test_video.py` that OCRs
  actual Cyrillic text and confirms it comes back correctly with the new
  default but garbled when forced to `eng`-only.
- Docker image: the `faster-whisper` speech-recognition model has no
  dedicated cache volume (unlike Ollama's `ollama-data`), so its
  Hugging Face model cache landed in the container's throwaway home
  directory -- re-downloaded on every `docker compose up --build`
  instead of once. Fixed by setting `HF_HOME=/data/hf-cache`, so it now
  shares the already-persisted `cognivore-data` volume. Also added
  `tests/test_audio.py` (previously zero coverage for the audio module):
  real, non-mocked tests for the speaker-turn heuristic, a locked-in
  check that a missing `faster-whisper` fails loudly (it already did --
  unlike the OCR case above), and an end-to-end transcription test
  against synthesized speech that self-skips without network access to
  Hugging Face or a local TTS engine.
- `cognivore ingest` (and the CLI generally) built a brand-new, empty
  `DocumentStore` on every invocation instead of loading whatever was
  already persisted at `store_dir` -- confirmed by writing a regression
  test for it -- so a *second* `ingest` run silently discarded everything
  a first one had saved. All store-loading (CLI and web server alike) now
  goes through one shared `load_or_build_document_store()` so this can't
  drift apart again.
- Docker image build: the runtime stage installed the `[all]` extra,
  which pulls in `llama-cpp-python` -- confirmed live, this has no
  prebuilt wheel for some platform/Python combinations and falls back to
  compiling from source, which fails outright in that stage since it has
  no C compiler by design. Switched to `[audio,video]`; the documented
  Docker paths (compose, or a single container + host Ollama) talk to
  Ollama over HTTP and never needed llama-cpp-python in the image at all.
- `docker-compose.yml`: the bundled `ollama` service published port
  11434 to the host -- confirmed live, this fails outright with "address
  already in use" for anyone (a very ordinary case, not an edge case)
  who already has a native Ollama install running, which is the same
  port. Nothing in the documented workflow (`docker compose exec ollama
  ollama pull ...`, or `cognivore` reaching it as `http://ollama:11434`
  over the compose network) actually needed that port published to the
  host at all, so the mapping is simply gone rather than moved.
