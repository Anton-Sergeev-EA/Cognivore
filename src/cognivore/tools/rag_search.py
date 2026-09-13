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
        "Searches the knowledge base (hybrid vector + keyword search) and returns the most "
        "relevant passages with their source. ALWAYS try this FIRST for any question that "
        "could plausibly be answered by specific, factual content someone has provided -- "
        "policies, prices, specs, procedures, names, dates, anything a document might state "
        "precisely. Prefer searching over answering from general knowledge, asking the user "
        "for more details, or saying more information is needed: the knowledge base may "
        "already contain the exact answer even if the question doesn't mention uploading or "
        "a specific document by name. Only skip it for questions that are clearly unrelated "
        "to any document (small talk, pure arithmetic, etc.). Write the query using the same "
        "language and, where possible, the same key words as the user's own question -- do "
        "not translate it. The default embedder matches text lexically, so a query translated "
        "into a different language than the documents will fail to find them even when they "
        "answer the question."
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

    def run(self, query: str = "", top_k: int = 5, _user_input: str = "", **_: object) -> str:
        if len(self.store) == 0:
            return "The knowledge base is empty. No documents have been ingested yet."
        # Also try the user's own, unmodified wording of the question --
        # see DocumentStore.search_multi for why: the model's own `query`
        # is sometimes translated/rewritten in a way that no longer
        # lexically matches the ingested documents.
        hits = self.store.search_multi([query, _user_input], top_k=top_k)
        if not hits:
            return "No relevant passages found."
        lines = []
        for i, hit in enumerate(hits, start=1):
            snippet = hit.text[:500]
            lines.append(f"[{i}] (source: {hit.source}, score: {hit.score:.3f}) {snippet}")
        return "\n\n".join(lines)
