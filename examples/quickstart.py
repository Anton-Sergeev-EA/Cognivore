#!/usr/bin/env python3
"""Minimal end-to-end example: ingest a document, then ask the agent about
it. Uses FakeLLMBackend so this runs with zero setup -- swap in
`COGNIVORE_LLM_MODEL_PATH` (see .env.example) for a real local LLM.

Run from the repo root: `python examples/quickstart.py`
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from cognivore.bootstrap import build_agent, build_document_store
from cognivore.config import Settings

DOCUMENT = """
Cognivore is a local-first agent framework. Its retrieval index has two
backends: an exact FlatIndex and an approximate NSWIndex, both implemented
in C++ and exposed to Python via pybind11. When no C++ compiler is
available at install time, it falls back to a pure NumPy implementation.
"""


def main() -> None:
    settings = Settings(prefer_semantic_embedder=False)  # no network needed for this demo
    store = build_document_store(settings)
    store.add_text(DOCUMENT, source="about_cognivore.md")

    agent = build_agent(settings, store=store, include_media=False)

    for question in [
        "search the knowledge base for what Cognivore's retrieval index backends are",
        "What is 17 * 23?",
    ]:
        print(f"\n> {question}")
        result = agent.run(question)
        for step in result.trace:
            if step.action:
                print(f"  [tool] {step.action}({step.action_input})")
        print(f"  {result.answer}")


if __name__ == "__main__":
    main()
