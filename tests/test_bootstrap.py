from __future__ import annotations

from unittest.mock import patch

from cognivore.bootstrap import build_llm_backend
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
