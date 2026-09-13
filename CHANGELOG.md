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
