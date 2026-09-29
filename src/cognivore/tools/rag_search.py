"""Exposes a :class:`~cognivore.rag.store.DocumentStore` to the agent as a
callable tool, so the agent decides *when* retrieval is worth doing rather
than every query being unconditionally stuffed with context."""

from __future__ import annotations

from typing import Any, ClassVar

from cognivore.rag.snippets import distinct_snippets
from cognivore.rag.store import DocumentStore, RetrievedChunk
from cognivore.tools.base import Tool

NOTHING_FOUND = "No relevant passages found."


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

    def __init__(self, store: DocumentStore, min_confidence: float = 0.0) -> None:
        self.store = store
        # Below this retrieval confidence the tool reports that nothing was
        # found instead of returning the least-unrelated chunks: handing a
        # model the "best" of several irrelevant passages invites it to
        # answer from them (see cognivore.ml.insight.retrieval_confidence).
        self.min_confidence = min_confidence

    def run(self, query: str = "", top_k: int = 3, _user_input: str = "", **_: object) -> str:
        if len(self.store) == 0:
            # Imported here: cognivore.agent imports this module, so a
            # top-level import would be circular.
            from cognivore.agent.replies import empty_knowledge_base

            # In the user's language -- in the offline demo this text is
            # shown to them as the answer verbatim.
            return empty_knowledge_base(_user_input or query)
        # Also try the user's own, unmodified wording of the question --
        # see DocumentStore.search_multi for why: the model's own `query`
        # is sometimes translated/rewritten in a way that no longer
        # lexically matches the ingested documents.
        # Over-fetch a little: format_hits drops passages that repeat one
        # another (overlapping neighbour chunks).
        hits = self.store.search_multi([query, _user_input], top_k=top_k + 2)
        if not hits:
            return NOTHING_FOUND
        if self.min_confidence > 0:
            from cognivore.ml.insight import retrieval_confidence

            confidence = max(
                retrieval_confidence(q, hits) for q in (query, _user_input) if q.strip()
            )
            if confidence < self.min_confidence:
                return NOTHING_FOUND
        return format_hits(hits, f"{query} {_user_input}", top_k=top_k)


def format_hits(hits: list[RetrievedChunk], query: str, top_k: int | None = None) -> str:
    """Formats retrieved chunks the way the model sees them:
    ``[n] (source) snippet`` blocks. Snippets are kept short, because the
    model re-reads them on every following call and prefill on CPU-only
    inference is slow; they show the part of each chunk that matches the
    question (see cognivore.rag.snippets)."""
    return "\n\n".join(
        f"[{i}] ({hit.source}) {snippet}"
        for i, (hit, snippet) in enumerate(
            distinct_snippets(hits, query, limit=300, top_k=top_k), start=1
        )
    )
