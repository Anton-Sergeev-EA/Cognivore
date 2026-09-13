"""The interface every LLM backend implements.

Keeping this as a tiny ``Protocol`` (rather than an abstract base class) is
what lets :mod:`cognivore.agent.core` stay backend-agnostic and lets tests
substitute :class:`~cognivore.llm.fake_backend.FakeLLMBackend` with zero
mocking framework involved.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.2
    stop: list[str] = field(default_factory=list)


@runtime_checkable
class LLMBackend(Protocol):
    """A minimal, provider-agnostic chat-completion interface.

    Implementations: :class:`cognivore.llm.llama_cpp_backend.LlamaCppBackend`
    (local GGUF models via llama.cpp) and
    :class:`cognivore.llm.fake_backend.FakeLLMBackend` (deterministic,
    dependency-free, used in tests and offline demos).
    """

    def generate(self, messages: list[ChatMessage], config: GenerationConfig) -> str:
        """Returns the full completion text for the given chat history."""
        ...

    def stream(self, messages: list[ChatMessage], config: GenerationConfig) -> Iterator[str]:
        """Yields the completion incrementally, token-chunk by token-chunk."""
        ...
