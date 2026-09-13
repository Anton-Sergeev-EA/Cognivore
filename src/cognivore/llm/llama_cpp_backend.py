"""Local LLM inference via ``llama-cpp-python`` (a Python binding for
llama.cpp). Runs quantized GGUF models entirely on CPU -- no PyTorch, no
GPU required, works fine on a laptop. The model file itself is not bundled
(GGUF checkpoints run from hundreds of MB to several GB); point
``COGNIVORE_LLM_MODEL_PATH`` at one you've downloaded, e.g. from
https://huggingface.co/models?library=gguf.
"""

from __future__ import annotations

from collections.abc import Iterator

from cognivore.llm.base import ChatMessage, GenerationConfig


class LlamaCppBackend:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int | None = None,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it with `pip install cognivore[llm]`."
            ) from exc
        self._model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,  # None lets llama.cpp auto-detect available cores
            verbose=False,
        )

    def _to_openai_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def generate(self, messages: list[ChatMessage], config: GenerationConfig) -> str:
        result = self._model.create_chat_completion(
            messages=self._to_openai_messages(messages),
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stop=config.stop or None,
        )
        return result["choices"][0]["message"]["content"] or ""

    def stream(self, messages: list[ChatMessage], config: GenerationConfig) -> Iterator[str]:
        stream = self._model.create_chat_completion(
            messages=self._to_openai_messages(messages),
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stop=config.stop or None,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta
