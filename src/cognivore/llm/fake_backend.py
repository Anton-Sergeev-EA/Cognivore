"""A deterministic, dependency-free LLM backend.

This is not a joke stand-in: it is what lets the entire agent loop, the API
endpoints, and the test suite run with zero model download and zero
network access, and it is what the CLI/web server fall back to when no
``COGNIVORE_LLM_MODEL_PATH`` is configured, so a fresh checkout is usable
in about ten seconds. It follows the exact ReAct textual protocol from
:mod:`cognivore.agent.prompts`, using a couple of small heuristics to
decide when a tool call is warranted, so demos exercise the real
tool-calling code path rather than a special-cased "fake mode".
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from cognivore.llm.base import ChatMessage, GenerationConfig

_MATH_CANDIDATE_RE = re.compile(r"[\d().\s+\-*/]{3,}")


def _extract_math_expression(text: str) -> str | None:
    """Finds the longest run of digits/operators/parentheses in `text` that
    contains at least one arithmetic operator -- greedy but good enough for
    routing a demo-mode tool call, e.g. picks the whole "(12 + 8) * 3" out
    of "What is (12 + 8) * 3?" rather than stopping at the first operand.
    """
    best: str | None = None
    for match in _MATH_CANDIDATE_RE.finditer(text):
        candidate = match.group(0).strip()
        if not any(op in candidate for op in "+-*/"):
            continue
        if best is None or len(candidate) > len(best):
            best = candidate
    return best


class FakeLLMBackend:
    def __init__(self, canned_answer: str | None = None) -> None:
        self.canned_answer = canned_answer

    def generate(self, messages: list[ChatMessage], config: GenerationConfig) -> str:
        last_user = _last_user_content(messages)

        if last_user.startswith("Observation:"):
            observation = last_user[len("Observation:") :].strip()
            # Agent._observation_message appends a blank line plus a
            # usage nudge after the actual observation text (a real model
            # reads both as one message; this fake backend still needs to
            # echo back only the observation itself to stay a faithful
            # stand-in for "the model used what it was given").
            observation = observation.split("\n\n", 1)[0]
            return f"Thought: I now have the observation I need.\nFinal Answer: {observation}"

        expr = _extract_math_expression(last_user)
        if expr:
            payload = json.dumps({"expression": expr})
            return f"Thought: This requires arithmetic, I'll use the calculator.\nAction: calculator\nAction Input: {payload}"

        if any(
            kw in last_user.lower() for kw in ("search", "knowledge base", "document", "find in")
        ):
            payload = json.dumps({"query": last_user})
            return (
                "Thought: This may be answered by the ingested documents; I'll search them.\n"
                f"Action: search_knowledge_base\nAction Input: {payload}"
            )

        answer = self.canned_answer or f"(offline demo mode) You said: {last_user}"
        return f"Thought: No tool is needed here.\nFinal Answer: {answer}"

    def stream(self, messages: list[ChatMessage], config: GenerationConfig) -> Iterator[str]:
        text = self.generate(messages, config)
        chunk_size = 24
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]


def _last_user_content(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""
