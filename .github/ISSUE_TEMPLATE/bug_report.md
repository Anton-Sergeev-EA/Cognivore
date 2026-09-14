---
name: Bug report
about: Something doesn't work the way this project's own docs say it should
title: ""
labels: bug
assignees: ""
---

**What happened**
A clear, concrete description. Include the actual output/error, not just
"it doesn't work" -- a stack trace or the exact CLI/API response is what
turns this into something fixable.

**What you expected**
What the README/docs led you to expect instead.

**How to reproduce**
The exact commands (or API calls) that trigger it, in order:

```bash
cognivore ...
```

**Environment**
- Cognivore version / commit: `git rev-parse HEAD` or `pip show cognivore`
- OS: (Linux/macOS/Windows, native or Docker)
- Python version: `python --version`
- LLM backend: (`FakeLLMBackend` / Ollama / llama-cpp-python) -- run
  `curl http://127.0.0.1:8420/api/health` if the server is up, it's in
  `llm_backend`
- Was the native C++ index built, or the NumPy fallback? (`native_index`
  in the same `/api/health` response, or `is_native` from
  `cognivore.index`)

**Logs**
Relevant output with `logging.basicConfig(level=logging.DEBUG)` if the
default INFO level isn't enough to see what's going on.
