# Contributing to Cognivore

Thanks for considering a contribution. This is a personal portfolio project,
but it's built like production software and issues/PRs are genuinely
welcome.

## Setup

```bash
git clone https://github.com/Anton-Sergeev-EA/cognivore.git
cd cognivore
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,all]"
```

Building the native extension requires a C++17 compiler (GCC/Clang/MSVC).
If none is available, installation still succeeds and the pure-Python
fallback index is used automatically -- see `src/cognivore/index/__init__.py`.

Then wire up the pre-commit hooks (`.pre-commit-config.yaml` -- ruff lint
+ format, plus a few hygiene checks) so lint/format issues are caught at
commit time instead of in CI:

```bash
pre-commit install
```

## Before opening a PR

```bash
ruff check .
ruff format --check .
mypy -p cognivore
pytest --cov
```

All four must pass. CI runs the same checks across Linux/macOS/Windows and
Python 3.10-3.12.

## Project layout

- `native/` -- the C++ vector index core and its pybind11 bindings.
- `src/cognivore/index/` -- Python-side index selection (native vs. fallback).
- `src/cognivore/rag/` -- chunking, embeddings, the document store.
- `src/cognivore/agent/` -- the ReAct loop, memory, prompts.
- `src/cognivore/tools/` -- agent-callable tools (calculator, RAG search,
  audio, video).
- `src/cognivore/llm/` -- LLM backends (llama.cpp, and the dependency-free
  `FakeLLMBackend` used by the test suite and offline demos).
- `src/cognivore/web/` -- the FastAPI app and static chat UI.
- `benchmarks/` -- standalone scripts for the numbers quoted in README.md.

## Commit style

Short, imperative subject lines (`Add X`, `Fix Y`, `Refactor Z`); a body
paragraph if the "why" isn't obvious from the diff.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
