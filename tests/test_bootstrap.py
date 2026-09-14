from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from cognivore.bootstrap import (
    build_document_store,
    build_llm_backend,
    load_or_build_document_store,
    seed_demo_knowledge_base,
)
from cognivore.config import Settings
from cognivore.llm.fake_backend import FakeLLMBackend


def test_provider_fake_always_returns_fake_backend() -> None:
    settings = Settings(llm_provider="fake")
    assert isinstance(build_llm_backend(settings), FakeLLMBackend)


def test_provider_ollama_falls_back_to_fake_when_unreachable() -> None:
    settings = Settings(llm_provider="ollama", ollama_host="http://127.0.0.1:1")
    assert isinstance(build_llm_backend(settings), FakeLLMBackend)


def test_provider_llama_cpp_without_model_path_falls_back_to_fake() -> None:
    settings = Settings(llm_provider="llama_cpp", llm_model_path=None)
    assert isinstance(build_llm_backend(settings), FakeLLMBackend)


def test_auto_falls_back_to_fake_when_nothing_available() -> None:
    settings = Settings(llm_provider="auto", ollama_host="http://127.0.0.1:1", llm_model_path=None)
    assert isinstance(build_llm_backend(settings), FakeLLMBackend)


def test_auto_prefers_ollama_when_reachable() -> None:
    settings = Settings(llm_provider="auto")
    with (
        patch("cognivore.llm.ollama_backend.is_ollama_running", return_value=True),
        patch("cognivore.llm.ollama_backend.list_ollama_models", return_value=["qwen2.5:3b"]),
    ):
        backend = build_llm_backend(settings)
    assert type(backend).__name__ == "OllamaBackend"
    assert backend.model == "qwen2.5:3b"


def test_seed_demo_knowledge_base_adds_both_language_docs() -> None:
    settings = Settings(prefer_semantic_embedder=False)
    store = build_document_store(settings)
    added = seed_demo_knowledge_base(store, settings)
    assert added > 0
    assert len(store) == added
    sources = store.sources()
    assert any("EN" in s for s in sources)
    assert any("RU" in s for s in sources)


def test_seed_demo_knowledge_base_is_idempotent() -> None:
    settings = Settings(prefer_semantic_embedder=False)
    store = build_document_store(settings)
    first = seed_demo_knowledge_base(store, settings)
    second = seed_demo_knowledge_base(store, settings)
    assert first > 0
    assert second == 0
    assert len(store) == first


def test_load_or_build_document_store_round_trips(tmp_path: Path) -> None:
    settings = Settings(prefer_semantic_embedder=False)
    store_dir = tmp_path / "store"

    store = load_or_build_document_store(settings, store_dir)
    assert len(store) == 0
    store.add_text("hello world", source="test.md")
    store.save(store_dir)

    restored = load_or_build_document_store(settings, store_dir)
    assert len(restored) == 1
    assert restored.sources() == {"test.md"}


def test_ingest_command_accumulates_across_runs(tmp_path: Path) -> None:
    """Regression test: `cognivore ingest` used to call `build_document_store`
    directly, which built a fresh (empty) store every invocation -- a
    second `ingest` run would silently discard everything from the first
    one on save. `load_or_build_document_store` fixes that; this asserts
    two independent "ingest calls" against the same store_dir accumulate
    rather than overwrite."""
    settings = Settings(prefer_semantic_embedder=False)
    store_dir = tmp_path / "store"

    first_store = load_or_build_document_store(settings, store_dir)
    first_store.add_text("first document", source="first.md")
    first_store.save(store_dir)

    second_store = load_or_build_document_store(settings, store_dir)
    second_store.add_text("second document", source="second.md")
    second_store.save(store_dir)

    final_store = load_or_build_document_store(settings, store_dir)
    assert final_store.sources() == {"first.md", "second.md"}
