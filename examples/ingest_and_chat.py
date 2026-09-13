#!/usr/bin/env python3
"""A slightly larger example: ingest every markdown file in a directory,
save the resulting store to disk, reload it, and chat against it -- the
same flow the `cognivore ingest` / `cognivore chat` CLI commands wrap.

Run from the repo root: `python examples/ingest_and_chat.py <directory>`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

from cognivore.agent.core import Agent
from cognivore.bootstrap import build_document_store
from cognivore.config import Settings
from cognivore.llm.fake_backend import FakeLLMBackend
from cognivore.rag.store import DocumentStore
from cognivore.tools.base import ToolRegistry
from cognivore.tools.rag_search import RagSearchTool


def ingest_directory(directory: Path, settings: Settings) -> DocumentStore:
    store = build_document_store(settings)
    for path in sorted(directory.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        ids = store.add_text(text, source=str(path))
        print(f"+ {path} -> {len(ids)} chunks")
    return store


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <directory-of-markdown-files>")
        raise SystemExit(1)

    directory = Path(sys.argv[1])
    settings = Settings(prefer_semantic_embedder=False)
    store = ingest_directory(directory, settings)

    save_path = Path(".cognivore/example_store")
    store.save(save_path)
    print(f"\nSaved {len(store)} chunks to {save_path}")

    reloaded = DocumentStore.load(save_path, embedder=store.embedder)
    agent = Agent(llm=FakeLLMBackend(), tools=ToolRegistry([RagSearchTool(reloaded)]))

    print("\nChat with the reloaded store (Ctrl-D to exit):")
    while True:
        try:
            question = input("> ")
        except EOFError:
            break
        print(agent.run(question).answer)


if __name__ == "__main__":
    main()
