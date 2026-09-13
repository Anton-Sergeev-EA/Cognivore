"""Exposes a :class:`~cognivore.rag.store.DocumentStore` to the agent as a
callable tool, so the agent decides *when* retrieval is worth doing rather
than every query being unconditionally stuffed with context."""

from __future__ import annotations

from typing import Any, ClassVar

from cognivore.rag.store import DocumentStore
from cognivore.tools.base import Tool


class RagSearchTool(Tool):
    name = "search_knowledge_base"
    # Kept deliberately short: every word here is repeated in the system
    # prompt on EVERY model call, and on CPU-only inference a longer
    # prompt means longer prefill on every single turn of the ReAct loop.
    # The behavioral nudges that earned their keep from live testing
    # (search proactively; don't translate the query) stay; the more
    # verbose, more "explain why" version tried first was reverted after
    # visibly slowing down every response.
    description = (
        "Searches the knowledge base and returns the most relevant passages with their "
        "source. Prefer searching over guessing or asking for more details -- it may already "
        "have the exact answer even to a plain factual question. Use the user's own language "
        "and wording; do not translate the query, since matching is literal-text-based."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 3},
        },
        "required": ["query"],
    }

    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    def run(self, query: str = "", top_k: int = 3, _user_input: str = "", **_: object) -> str:
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
            # Shorter snippets keep the Observation text (which the model
            # re-reads on its very next, slowest-so-far call) from adding
            # unnecessary prefill time on CPU-only inference.
            snippet = hit.text[:300]
            lines.append(f"[{i}] ({hit.source}) {snippet}")
        return "\n\n".join(lines)
