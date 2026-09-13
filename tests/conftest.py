from __future__ import annotations

import pytest

from cognivore.rag.embeddings import HashingEmbedder
from cognivore.rag.store import DocumentStore


@pytest.fixture
def hashing_embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=64)


@pytest.fixture
def document_store(hashing_embedder: HashingEmbedder) -> DocumentStore:
    return DocumentStore(embedder=hashing_embedder, use_approximate_index=True)
