# Cognivore

**A local-first, torch-free multimodal agent framework.** LLM reasoning +
RAG, with a hand-written C++ vector index and a web chat UI -- everything
runs on a CPU-only laptop, nothing is required to leave your machine.

[![CI](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml/badge.svg)](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)

🌐 **Read this in another language:** [English](README.md) | [Русский](docs/i18n/README.ru.md) | [Deutsch](docs/i18n/README.de.md) | [Français](docs/i18n/README.fr.md) | [Italiano](docs/i18n/README.it.md) | [Español](docs/i18n/README.es.md) | [中文](docs/i18n/README.zh.md) | [日本語](docs/i18n/README.ja.md) | [हिन्दी](docs/i18n/README.hi.md)

Cognivore is a ReAct-style agent framework with retrieval-augmented
generation and multimodal tool support (audio transcription, video scene
analysis), built specifically to run entirely on a CPU-only machine with no
PyTorch dependency anywhere in the stack. Its retrieval index is a
hand-written C++ core (AVX2 SIMD, OpenMP-parallel exact search, and a
from-scratch approximate NSW graph), exposed to Python via pybind11, with a
pure-NumPy fallback so `pip install` never hard-fails for lack of a C++
compiler.

```
you> What does the ingested spec say about the retry policy, and what's 15% of that timeout in ms?

  [tool] search_knowledge_base({"query": "retry policy timeout"})
  [tool] calculator({"expression": "5000 * 0.15"})

cognivore> The spec (spec.md) sets a 5000ms timeout with exponential
backoff. 15% of that is 750ms.
```

## Why this exists

Most "AI agent" portfolio projects are a thin wrapper around an API call.
This one is built to show the opposite: a framework where the interesting
parts (a real ANN data structure, a provider-agnostic tool-calling
protocol, a hybrid retrieval pipeline, graceful degradation when optional
dependencies are missing) are implemented rather than imported.

## Features

- **ReAct agent loop** (Thought → Action → Observation) that works with
  *any* instruction-tuned local model, not just ones fine-tuned for a
  specific function-calling wire format -- see
  [docs/architecture.md](docs/architecture.md#why-a-react-loop-instead-of-native-function-calling).
- **Native C++ vector index** (`native/vector_index.cpp`): an exact
  `FlatIndex` (AVX2/FMA dot product, OpenMP-parallel scan) and an
  approximate `NSWIndex` (a from-scratch single-layer Navigable Small World
  graph -- insertion, neighbour pruning, greedy beam search), both bound to
  Python via a hand-written pybind11 extension with a `.pyi` type stub.
  Falls back to a pure-NumPy index automatically if no C++ compiler is
  present at install time.
- **Hybrid RAG**: recursive-splitter chunking, `fastembed` (ONNX, no
  PyTorch) semantic embeddings with a zero-download hashing-trick fallback,
  vector + BM25 hybrid retrieval.
- **Multimodal tools**: safe (AST-based, no `eval`) calculator, knowledge-
  base search, audio transcription + rough speaker turns
  (`faster-whisper`), and video scene-detection + OCR (OpenCV) -- all
  CPU-only, no PyTorch.
- **Pluggable LLM backend**: local GGUF inference via `llama-cpp-python`,
  or a deterministic, dependency-free `FakeLLMBackend` that exercises the
  exact same tool-calling code path with zero download -- what the test
  suite and CI run against.
- **Web UI**: FastAPI + SSE streaming + a vanilla JS/HTML/CSS chat
  interface (no build step, no framework) with file/audio/video
  drag-and-drop, live "thinking"/connection indicators, dark/light/aurora
  themes, and localization into English, Russian, German, French,
  Italian, Spanish, Simplified Chinese, Japanese, and Hindi.
- Full test suite, ruff lint+format, mypy (strict-ish, including a stub for
  the native extension), and a multi-OS/multi-Python CI matrix.

## Quickstart

```bash
git clone https://github.com/Anton-Sergeev-EA/cognivore.git
cd cognivore
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # builds the native extension if a C++17 compiler is present

cognivore chat                   # interactive chat, offline demo mode by default
cognivore ingest ./docs          # index a folder of markdown/text files
cognivore serve                  # web chat UI at http://127.0.0.1:8420
```

By default there's no LLM download and no network access at all: the agent
runs against `FakeLLMBackend`, a small deterministic backend that still
exercises the real tool-calling loop (try `What is 12 * 7?` or `search the
knowledge base for ...` after `ingest`ing something). To use a real local
LLM:

```bash
pip install -e ".[llm]"
# download e.g. https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
cp .env.example .env
echo 'COGNIVORE_LLM_MODEL_PATH=/path/to/model.gguf' >> .env
cognivore chat
```

Or, simpler and with no C++ toolchain involved at all, point it at
[Ollama](https://ollama.com) instead -- `COGNIVORE_LLM_PROVIDER=auto` (the
default) tries a locally running Ollama server before falling back to a
GGUF path or `FakeLLMBackend`:

```bash
ollama pull qwen2.5:3b   # any instruction-tuned model works
cognivore chat           # picks up the running Ollama server automatically
```

Audio/video tools need their own extras: `pip install -e ".[audio,video]"`
(or `.[all]` for everything, GGUF included). See `.env.example` for every
setting.

### Docker

The image is a multi-stage build (compiles the native C++ extension, then
discards the compiler for a slim runtime image), runs as a non-root user,
and ships a `HEALTHCHECK` -- built once with GitHub Actions' Docker layer
caching and verified end-to-end (server actually answers `/api/health`,
container reports `healthy`, runs as non-root) on every push, so it isn't
just "builds," it's checked.

**Easiest path -- no Python, no Ollama install, works the same on Windows,
macOS, and Linux:**

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:3b   # one-time, ~2GB
```

Then open <http://127.0.0.1:8420>. `docker-compose.yml` runs Ollama
*inside* its own container too, so there's nothing to install on the host
beyond Docker itself; Cognivore reaches it over the compose network by
service name (`http://ollama:11434`), which sidesteps the usual
host-networking differences between Windows/macOS/Linux entirely. Without
that one-time `ollama pull`, Cognivore still starts and runs fine -- it
just falls back to the offline `FakeLLMBackend` demo mode until a model is
available.

The compose stack also sets `COGNIVORE_SEED_DEMO_KB=true`, so a fresh
knowledge base is seeded automatically with two bundled demo company
handbooks (English + Russian -- pricing, SLA, security, refund policy,
support FAQ) instead of opening to an empty dropzone. It only ever seeds
an *empty* knowledge base: once you've ingested your own documents, this
is a permanent no-op. Set it to `false` in `docker-compose.yml` (or
`docker run -e COGNIVORE_SEED_DEMO_KB=false`) to start empty instead, or
run `cognivore seed-demo` yourself at any time to add the same demo docs
to an existing store on demand.

**Already have Ollama running on the host, or want a single container?**

```bash
docker build -t cognivore .
docker run -d -p 8420:8420 -v cognivore-data:/data \
  -e COGNIVORE_OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  cognivore
```

`host.docker.internal` is provided automatically on Docker Desktop
(Windows/macOS); the explicit `--add-host` above is what makes the same
command also work on plain Linux, where it otherwise doesn't resolve.

The image ships the audio/video tools (`faster-whisper`, OpenCV) but *not*
`llama-cpp-python` -- it talks to Ollama over plain HTTP for the LLM
instead of loading a GGUF file in-process, deliberately, since
llama-cpp-python has no prebuilt wheel for every platform and needs a
compiler the runtime stage doesn't carry. Want in-process GGUF inference
inside the container anyway? Add `build-essential` to the final stage and
switch its `pip install` back to the `[all]` extra.

**Prebuilt image (no build step at all):** tagged releases are published
multi-arch (amd64 + arm64 -- Apple Silicon and Raspberry Pi included) to
GHCR by [`.github/workflows/docker-publish.yml`](.github/workflows/docker-publish.yml):

```bash
docker pull ghcr.io/anton-sergeev-ea/cognivore:latest
```

## Usage

**CLI** (`cognivore --help` for the full list):

```bash
cognivore chat                              # interactive REPL
cognivore ingest ./docs                     # recursively indexes .txt/.md/.markdown/.rst
cognivore seed-demo                         # adds the bundled EN+RU demo company handbooks
cognivore serve --host 0.0.0.0 --port 8420  # web UI + REST/SSE API
cognivore bench                             # index build/search benchmarks (see below)
```

**Web UI** (`cognivore serve`, then open the printed URL): a chat interface
with live token streaming, a drag-and-drop zone for `.txt`/`.md` ingestion
and for audio/video files, a language switcher (9 languages), three themes
(dark/light/aurora), and a trace view of every tool call the agent made
for a given answer (click a step to see the full, untruncated tool
output).

**REST/SSE API**, once `cognivore serve` is running:

```bash
curl http://127.0.0.1:8420/api/health

curl -X POST http://127.0.0.1:8420/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 12 * 7?"}'

curl -N "http://127.0.0.1:8420/api/chat/stream?message=Summarize+the+ingested+docs"

curl -X POST http://127.0.0.1:8420/api/ingest/file -F "file=@./notes.md"
```

**As a library**, rather than through the CLI or the API (see
`examples/`):

```python
from cognivore.bootstrap import build_agent
from cognivore.config import get_settings

agent = build_agent(get_settings())
result = agent.run("What's 15% of 5000?")
print(result.answer)
```

## Benchmarks

Numbers from `python benchmarks/bench_index.py` on a 2-core CI-class
machine (`Dockerfile`'s image is portable; your actual laptop almost
certainly has more cores, which matters -- see below). Random,
uniformly-distributed 384-dimensional vectors, which is close to a *worst
case* for approximate search (no real cluster structure, so "nearest"
neighbours are only marginally closer than random ones); real embeddings
recall noticeably better at the same `ef`.

**Build time** (3,000 vectors) -- this is where the native extension's
advantage is unambiguous:

| Index | Build time | vs. NumPy fallback |
|---|---|---|
| `FlatIndexPy` (NumPy, `.add()` in a loop) | 0.615s | 1x |
| `FlatIndex` (C++) | 0.027s | **~23x faster** |

**Recall vs. speed is a tunable knob, not a single number** (`NSWIndex`, n=3,000):

| `ef` | ms/query | Recall@10 |
|---|---|---|
| 50  | 0.24 | 0.64 |
| 150 (default) | 0.46 | 0.93 |
| 400 | 0.78 | 1.00 |

**Search latency vs. collection size** -- an exact linear scan (even
AVX2+OpenMP-accelerated) is *fine* until the collection is large enough
that scanning it becomes the bottleneck; that crossover is where an
approximate graph index is supposed to start winning:

| n | `FlatIndex` ms/query | `NSWIndex` ms/query (ef=150) | Speedup |
|---|---|---|---|
| 1,000 | 0.036 | 0.240 | 0.2x (brute force wins -- collection too small) |
| 10,000 | 0.340 | 0.798 | 0.4x |
| 50,000 | 1.561 | 1.291 | 1.2x |

Read that table for what it actually says, not what would make a better
story: at these sizes, on 2 cores, a SIMD+parallel brute-force scan is
*competitive with or faster than* the from-scratch approximate index. This
is real, well-documented ANN behavior -- a hand-rolled single-layer NSW
graph has meaningfully more per-step overhead (heap operations, random
memory access across the graph, visited-set bookkeeping) than a
cache-friendly linear scan enjoys, and multi-layer HNSW (Faiss/hnswlib's
approach, not yet implemented here -- see the roadmap) exists specifically
to widen that gap at scale. The honest takeaway: this project's ANN index
demonstrates the *data structure and algorithm* correctly (see
`tests/test_index.py`'s recall regression test), and the crossover point
where it pays for itself moves with core count, `ef`, and how clustered
your actual embeddings are -- it is not a universal "always faster" claim,
and this README isn't going to pretend it is.

## Testing

```bash
pip install -e ".[dev]"
pytest --cov                 # 71 tests: calculator safety, chunking,
                              # native-vs-Python index parity, NSW recall,
                              # agent loop, RAG store, FastAPI endpoints
ruff check . && ruff format --check .
mypy -p cognivore
```

Every check above is what CI runs (`.github/workflows/ci.yml`), across
Ubuntu/macOS/Windows and Python 3.10-3.12; native-extension-only tests skip
themselves (rather than fail) on a matrix leg where the C++ toolchain isn't
available, matching the same fallback behavior the package itself has.

## Project layout

See [docs/architecture.md](docs/architecture.md) for a diagram and the
reasoning behind the two biggest design decisions (a text-based ReAct loop
instead of provider-specific function calling, and a hand-written index
instead of Faiss/hnswlib).

```
native/            C++ vector index core + pybind11 bindings
src/cognivore/
  index/            native-vs-fallback index selection
  rag/              chunking, embeddings, hybrid document store
  agent/            ReAct loop, memory, prompt/parsing
  tools/            calculator, RAG search, audio, video
  llm/              llama.cpp backend + FakeLLMBackend
  media/            faster-whisper / OpenCV wrappers
  web/              FastAPI app + static chat UI
  cli.py            cognivore chat|ingest|serve|bench
benchmarks/         standalone scripts for the numbers above
examples/           minimal library-usage scripts
tests/              pytest suite (71 tests)
```

## License

MIT -- see [LICENSE](LICENSE).

## Contact

Built by **Sergeev Anton** ([GitHub](https://github.com/Anton-Sergeev-EA), [avsergeev1981@gmail.com](mailto:avsergeev1981@gmail.com)).
Contributions welcome -- see [CONTRIBUTING.md](CONTRIBUTING.md).
