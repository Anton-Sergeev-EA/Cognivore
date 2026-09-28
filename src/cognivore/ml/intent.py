"""Cheap intent heuristics: is a message a question about facts that would
live in the knowledge base (as opposed to arithmetic or small talk)?

Used to decide whether an empty knowledge-base lookup is worth telling the
model about and recording as a knowledge gap, and by the offline demo
backend to route questions to the search tool.
"""

from __future__ import annotations

import re

from cognivore.rag.tokenize import content_tokens

QUESTION_STARTS = (
    # English
    "what", "how", "which", "when", "where", "why", "who", "is ", "are ", "do ",
    "does ", "can ", "could ", "should ",
    # Russian
    "что", "как", "какой", "какая", "какие", "каков", "сколько", "когда", "где",
    "почему", "зачем", "кто", "есть ли", "можно ли",
    # Spanish, French, German, Italian
    "qué", "que ", "cuál", "cuánto", "cómo", "dónde", "cuándo", "quel", "quelle",
    "combien", "comment", "où", "quand", "was ", "wie ", "welche", "wann", "wo ",
    "gibt es", "qual", "quanto", "come ", "dove", "quando",
)  # fmt: skip
# Question markers that can appear anywhere (Chinese, Japanese, Hindi).
QUESTION_MARKERS = (
    "吗", "什么", "多少", "怎么", "如何", "哪", "是否", "能否", "ですか", "ますか",
    "क्या", "कैसे", "कितना", "कितनी", "कौन",
)  # fmt: skip

# "15% от 4900", "12 * 7", "149 का 15%": numbers joined by an operator.
_ARITHMETIC_RE = re.compile(r"\d\s*(?:[-+*/×÷^]|%)\s*\S|\d+(?:[.,]\d+)?\s*%")


def looks_like_question(text: str) -> bool:
    stripped = text.strip().lower()
    if stripped.endswith(("?", "？")):
        return True
    if stripped.startswith(QUESTION_STARTS):
        return True
    return any(marker in stripped for marker in QUESTION_MARKERS)


def looks_like_knowledge_question(text: str) -> bool:
    """A question with at least two content words and no arithmetic in it:
    "Do you have a mobile app for iPhone?" yes; "What is 15% of 149?",
    "How are you?" and "Hi" no."""
    if not looks_like_question(text) or _ARITHMETIC_RE.search(text):
        return False
    return len(set(content_tokens(text))) >= 2
