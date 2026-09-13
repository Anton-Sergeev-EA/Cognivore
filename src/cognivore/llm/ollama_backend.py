"""LLM backend that talks to a locally running Ollama (https://ollama.com)
server over plain HTTP -- no extra Python dependency (stdlib
``urllib.request`` is enough for JSON-over-HTTP), and no model download
managed by this project: Ollama owns pulling/storing models, Cognivore just
asks it to generate.

This is deliberately the first thing ``build_llm_backend`` tries in "auto"
mode (see ``cognivore.bootstrap``): a huge number of people who already run
a local LLM at all run it through Ollama specifically, so detecting and
using it automatically turns "download and configure a GGUF file" into
"already works" for that whole audience.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator

from cognivore.llm.base import ChatMessage, GenerationConfig


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama server can't be reached or has no models."""


def is_ollama_running(host: str, timeout: float = 0.5) -> bool:
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def list_ollama_models(host: str, timeout: float = 2.0) -> list[str]:
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        return []


class OllamaBackend:
    def __init__(
        self,
        host: str,
        model: str | None = None,
        request_timeout: float = 600.0,
        num_thread: int | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.request_timeout = request_timeout
        self.model = model or self._pick_default_model()
        # Ollama's own auto-detected thread count can be conservative (e.g.
        # picking 2 threads on a 12-core laptop); explicitly passing the
        # CPU count gives CPU-only inference a real speed boost for free.
        # None means "don't send it, let Ollama decide" -- kept overridable
        # rather than hardcoded so a shared/constrained machine can still
        # cap it via COGNIVORE_LLM_N_THREADS.
        self.num_thread = num_thread

    def _pick_default_model(self) -> str:
        models = list_ollama_models(self.host)
        if not models:
            raise OllamaUnavailableError(
                f"Ollama at {self.host} has no models pulled. Run e.g. `ollama pull qwen2.5:3b` "
                "first, or set COGNIVORE_OLLAMA_MODEL explicitly."
            )
        return models[0]

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise OllamaUnavailableError(f"Ollama request to {self.host} failed: {exc}") from exc

    def _to_ollama_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _options(self, config: GenerationConfig) -> dict[str, object]:
        options: dict[str, object] = {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        }
        if self.num_thread is not None:
            options["num_thread"] = self.num_thread
        return options

    def generate(self, messages: list[ChatMessage], config: GenerationConfig) -> str:
        result = self._post(
            {
                "model": self.model,
                "messages": self._to_ollama_messages(messages),
                "stream": False,
                "options": self._options(config),
            }
        )
        return result.get("message", {}).get("content", "")

    def stream(self, messages: list[ChatMessage], config: GenerationConfig) -> Iterator[str]:
        body = json.dumps(
            {
                "model": self.model,
                "messages": self._to_ollama_messages(messages),
                "stream": True,
                "options": self._options(config),
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                for line in resp:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise OllamaUnavailableError(f"Ollama request to {self.host} failed: {exc}") from exc
