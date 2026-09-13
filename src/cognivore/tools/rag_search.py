"""Exposes a :class:`~cognivore.rag.store.DocumentStore` to the agent as a
callable tool, so the agent decides *when* retrieval is worth doing rather
than every query being unconditionally stuffed with context."""

from __future__ import annotations

from typing import Any, ClassVar

from cognivore.rag.store import DocumentStore
from cognivore.tools.base import Tool


class RagSearchTool(Tool):
    name = "search_knowledge_base"
    description = (
        "Searches the ingested documents (hybrid vector + keyword search) and returns the "
        "most relevant passages with their source. Use this before answering questions about "
        "content the user has uploaded."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    def run(self, query: str = "", top_k: int = 5, **_: object) -> str:
        if len(self.store) == 0:
            return "The knowledge base is empty. No documents have been ingested yet."
        hits = self.store.search(query, top_k=top_k)
        if not hits:
            return "No relevant passages found."
        lines = []
        for i, hit in enumerate(hits, start=1):
            snippet = hit.text[:500]
            lines.append(f"[{i}] (source: {hit.source}, score: {hit.score:.3f}) {snippet}")
        return "\n\n".join(lines)
