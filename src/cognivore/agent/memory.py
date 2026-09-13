"""Two kinds of memory:

* :class:`ConversationBuffer` -- the last N raw turns, always included
  verbatim in the prompt (short-term memory).
* :class:`VectorMemory` -- every turn ever seen, embedded and stored in a
  vector index, recalled by similarity (long-term memory). This reuses the
  exact same C++-backed index as the RAG subsystem: from the index's point
  of view, "a past conversation turn" and "a document chunk" are the same
  thing -- a piece of text with an embedding.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from cognivore.index import FlatIndex, NSWIndex, is_native
from cognivore.llm.base import ChatMessage
from cognivore.rag.embeddings import EmbeddingModel


@dataclass
class ConversationBuffer:
    max_turns: int = 12

    def __post_init__(self) -> None:
        self._messages: deque[ChatMessage] = deque(maxlen=self.max_turns * 2)

    def add(self, role: str, content: str) -> None:
        self._messages.append(ChatMessage(role=role, content=content))

    def as_list(self) -> list[ChatMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


class VectorMemory:
    def __init__(self, embedder: EmbeddingModel, use_approximate_index: bool = True) -> None:
        self.embedder = embedder
        self._use_ann = use_approximate_index and is_native and NSWIndex is not None
        self._index = NSWIndex(embedder.dim, 16, 200) if self._use_ann else FlatIndex(embedder.dim)
        self._texts: dict[int, str] = {}
        self._next_id = 0

    def remember(self, text: str) -> None:
        vector = self.embedder.embed([text])[0]
        cid = self._next_id
        self._next_id += 1
        self._index.add(cid, vector)
        self._texts[cid] = text

    def recall(self, query: str, top_k: int = 3) -> list[str]:
        if not self._texts:
            return []
        vector = self.embedder.embed([query])[0]
        hits = (
            self._index.search(vector, top_k, 150)
            if self._use_ann
            else self._index.search(vector, top_k)
        )
        return [self._texts[h.id] for h in hits]

    def __len__(self) -> int:
        return len(self._texts)
