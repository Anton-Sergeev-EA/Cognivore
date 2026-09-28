from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from cognivore.rag.embeddings import HashingEmbedder
from cognivore.rag.store import DocumentStore


@pytest.fixture
def hashing_embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=64)


@pytest.fixture
def document_store(hashing_embedder: HashingEmbedder) -> DocumentStore:
    return DocumentStore(embedder=hashing_embedder, use_approximate_index=True)


@pytest.fixture(scope="session", autouse=True)
def _isolated_from_the_developer_machine(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Makes the suite hermetic -- same result on a CI runner and on a
    developer laptop:

    * runs from an empty directory, so pydantic-settings never reads the
      developer's ``.env`` (API keys, a real model path, ...);
    * drops every ``COGNIVORE_*`` variable from the environment;
    * defaults the LLM to the offline ``FakeLLMBackend`` and the embedder to
      the offline hashing one. With the default ``llm_provider="auto"`` a
      locally running Ollama was picked up, and the web tests sent dozens
      of questions to a real CPU model -- the suite went from seconds to
      tens of minutes.

    Session-scoped so it is in place before module-scoped fixtures build
    their apps. Tests that exercise provider selection pass it explicitly,
    which takes precedence over the environment.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path_factory.mktemp("cwd"))
        for key in list(os.environ):
            if key.startswith("COGNIVORE_"):
                mp.delenv(key)
        mp.setenv("COGNIVORE_LLM_PROVIDER", "fake")
        mp.setenv("COGNIVORE_PREFER_SEMANTIC_EMBEDDER", "false")
        yield
