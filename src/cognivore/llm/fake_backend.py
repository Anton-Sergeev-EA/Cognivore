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

from cognivore.agent.replies import not_in_knowledge_base
from cognivore.llm.base import ChatMessage, GenerationConfig
from cognivore.ml.intent import looks_like_question
from cognivore.ml.lang import detect_language
from cognivore.tools.rag_search import NOTHING_FOUND

_MATH_CANDIDATE_RE = re.compile(r"[\d().\s+\-*/]{3,}")
# The search tool formats hits as "[1] (source) passage\n\n[2] ...".
# Lazy source match: labels themselves may contain parentheses, e.g.
# "(NordCloud demo knowledge base (RU))".
_TOP_PASSAGE_RE = re.compile(r"^\[1\] \((?P<source>.*?)\) (?P<text>.*?)(?:\n\n\[2\] |\Z)", re.S)
# Markdown heading markers can sit mid-line once chunking has joined lines.
_MARKDOWN_RE = re.compile(r"(?:^|(?<=\s))#{1,6}\s+|\*\*|__", re.M)

# Explicit "look it up" requests, in the three UI languages.
_SEARCH_KEYWORDS = (
    "search",
    "knowledge base",
    "document",
    "find in",
    "найди",
    "поиск",
    "база знаний",
    "базе знаний",
    "документ",
    "搜索",
    "查找",
    "知识库",
    "文档",
)
_OFFLINE_ECHO = {
    "ru": "(офлайн-демо) Вы написали: {text}",
    "zh": "（离线演示模式）您输入的是：{text}",
    "ja": "（オフラインデモ）入力内容：{text}",
    "hi": "(ऑफ़लाइन डेमो) आपने लिखा: {text}",
    "en": "(offline demo mode) You said: {text}",
}


_NUM = r"(\d+(?:[.,]\d+)?)"
# "15% of 149" and its forms in the UI languages: от/из (ru), de (es/fr),
# du (fr), von (de), di (it).
_PERCENT_OF_RE = re.compile(_NUM + r"\s*%\s*(?:of|от|из|de|du|von|di)\s+" + _NUM, re.I)
# Base-first word order: "999 的 15%" (zh), "149 の 15%" (ja), "149 का 15%" (hi).
_PERCENT_ZH_RE = re.compile(_NUM + r"\s*(?:的|の|का)\s*" + _NUM + r"\s*%")


def _extract_percent_expression(text: str) -> str | None:
    """Turns "15% of 149" (and its forms in the other UI languages) into
    ``149 * 15 / 100`` so the offline demo can route it to the calculator."""
    match = _PERCENT_OF_RE.search(text)
    if match:
        pct, base = match.groups()
    else:
        match = _PERCENT_ZH_RE.search(text)
        if not match:
            return None
        base, pct = match.groups()
    return f"{base.replace(',', '.')} * {pct.replace(',', '.')} / 100"


def _offline_answer_from_observation(observation: str, question: str = "") -> str:
    """With the search tool's formatted hits, answer with the best passage
    itself (the UI shows its source separately); with nothing found, say so
    in the question's language; anything else -- e.g. a calculator result --
    is echoed as-is."""
    if observation.strip() == NOTHING_FOUND:
        return not_in_knowledge_base(question)
    match = _TOP_PASSAGE_RE.match(observation)
    if match is None:
        return observation
    return _MARKDOWN_RE.sub("", match.group("text")).strip()


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
            observation = observation.rsplit("\n\nIf the passage above answers", 1)[0]
            answer = _offline_answer_from_observation(observation, _original_question(messages))
            return f"Thought: I now have the observation I need.\nFinal Answer: {answer}"

        expr = _extract_percent_expression(last_user) or _extract_math_expression(last_user)
        if expr:
            payload = json.dumps({"expression": expr})
            return f"Thought: This requires arithmetic, I'll use the calculator.\nAction: calculator\nAction Input: {payload}"

        # Only route to the search tool when the agent actually offers it
        # (it is listed in the system prompt); otherwise a question would
        # just produce an "unknown tool" error.
        has_search = any(
            m.role == "system" and "search_knowledge_base" in m.content for m in messages
        )
        wants_search = any(kw in last_user.lower() for kw in _SEARCH_KEYWORDS)
        if has_search and (wants_search or looks_like_question(last_user)):
            payload = json.dumps({"query": last_user}, ensure_ascii=False)
            return (
                "Thought: This may be answered by the ingested documents; I'll search them.\n"
                f"Action: search_knowledge_base\nAction Input: {payload}"
            )

        template = _OFFLINE_ECHO.get(detect_language(last_user), _OFFLINE_ECHO["en"])
        answer = self.canned_answer or template.format(text=last_user)
        return f"Thought: No tool is needed here.\nFinal Answer: {answer}"

    def stream(self, messages: list[ChatMessage], config: GenerationConfig) -> Iterator[str]:
        text = self.generate(messages, config)
        chunk_size = 24
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]


def _original_question(messages: list[ChatMessage]) -> str:
    """The user's own message of this turn (not a tool observation)."""
    for message in reversed(messages):
        if message.role == "user" and not message.content.startswith("Observation:"):
            return message.content
    return ""


def _last_user_content(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""
