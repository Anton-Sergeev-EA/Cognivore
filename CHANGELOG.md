# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this
project uses [Semantic Versioning](https://semver.org/).

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

### Fixed

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
