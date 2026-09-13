from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import patch

from cognivore.llm.base import ChatMessage, GenerationConfig
from cognivore.llm.ollama_backend import (
    OllamaBackend,
    OllamaUnavailableError,
    is_ollama_running,
    list_ollama_models,
)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._buf = BytesIO(payload)

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self):
        return iter(self._buf.readlines())


def test_is_ollama_running_true_when_reachable() -> None:
    with patch("urllib.request.urlopen", return_value=_FakeResponse(b"{}")):
        assert is_ollama_running("http://127.0.0.1:11434") is True


def test_is_ollama_running_false_when_connection_refused() -> None:
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        assert is_ollama_running("http://127.0.0.1:11434") is False


def test_list_ollama_models_parses_tags_response() -> None:
    payload = json.dumps({"models": [{"name": "qwen2.5:3b"}, {"name": "llama3.2:3b"}]}).encode()
    with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
        assert list_ollama_models("http://127.0.0.1:11434") == ["qwen2.5:3b", "llama3.2:3b"]


def test_list_ollama_models_returns_empty_on_error() -> None:
    with patch("urllib.request.urlopen", side_effect=OSError("no route")):
        assert list_ollama_models("http://127.0.0.1:11434") == []


def test_ollama_backend_picks_first_available_model_by_default() -> None:
    payload = json.dumps({"models": [{"name": "qwen2.5:3b"}]}).encode()
    with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
        backend = OllamaBackend(host="http://127.0.0.1:11434")
    assert backend.model == "qwen2.5:3b"


def test_ollama_backend_raises_when_no_models_pulled() -> None:
    with patch("urllib.request.urlopen", return_value=_FakeResponse(b'{"models": []}')):
        try:
            OllamaBackend(host="http://127.0.0.1:11434")
        except OllamaUnavailableError:
            pass
        else:
            raise AssertionError("expected OllamaUnavailableError")


def test_ollama_backend_generate_extracts_message_content() -> None:
    response_payload = json.dumps({"message": {"role": "assistant", "content": "42"}}).encode()
    with patch("urllib.request.urlopen", return_value=_FakeResponse(response_payload)):
        backend = OllamaBackend(host="http://127.0.0.1:11434", model="qwen2.5:3b")
        answer = backend.generate(
            [ChatMessage(role="user", content="What is 6*7?")], GenerationConfig()
        )
    assert answer == "42"


def test_ollama_backend_stream_yields_content_chunks() -> None:
    lines = [
        json.dumps({"message": {"content": "The "}, "done": False}).encode() + b"\n",
        json.dumps({"message": {"content": "answer"}, "done": False}).encode() + b"\n",
        json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n",
    ]
    with patch("urllib.request.urlopen", return_value=_FakeResponse(b"".join(lines))):
        backend = OllamaBackend(host="http://127.0.0.1:11434", model="qwen2.5:3b")
        chunks = list(backend.stream([ChatMessage(role="user", content="hi")], GenerationConfig()))
    assert chunks == ["The ", "answer"]
