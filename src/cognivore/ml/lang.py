"""Language identification for the nine UI languages.

Deliberately not a statistical model. Non-Latin scripts are told apart by
counting characters per Unicode block, which is exact, instant and has no
failure mode on short inputs (where n-gram classifiers are weakest). The
Latin-script languages are told apart by very frequent function words and
by characters only one of them uses (ñ, ß, ç ...).
"""

from __future__ import annotations

import re

LANGUAGE_NAMES = {
    "ru": "Russian",
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "hi": "Hindi",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "it": "Italian",
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# High-frequency words that are distinctive for one Latin-script language.
_MARKERS: dict[str, frozenset[str]] = {
    "en": frozenset(
        [
            "the",
            "is",
            "are",
            "what",
            "how",
            "which",
            "who",
            "why",
            "when",
            "where",
            "do",
            "does",
            "can",
            "you",
            "your",
            "of",
            "and",
            "to",
            "with",
            "for",
            "this",
            "that",
            "there",
        ]
    ),
    "es": frozenset(
        [
            "el",
            "la",
            "los",
            "las",
            "es",
            "qué",
            "cuál",
            "cuánto",
            "cuánta",
            "cómo",
            "dónde",
            "cuándo",
            "por",
            "para",
            "con",
            "una",
            "del",
            "hay",
            "tiene",
            "usted",
            "ustedes",
            "su",
            "sus",
            "y",
        ]
    ),
    "fr": frozenset(
        [
            "le",
            "la",
            "les",
            "est",
            "quel",
            "quelle",
            "quels",
            "quelles",
            "combien",
            "comment",
            "où",
            "quand",
            "pour",
            "avec",
            "une",
            "des",
            "du",
            "et",
            "vous",
            "nous",
            "au",
            "aux",
        ]
    ),
    "de": frozenset(
        [
            "der",
            "die",
            "das",
            "ist",
            "sind",
            "wie",
            "was",
            "welche",
            "welcher",
            "welches",
            "wann",
            "wo",
            "gibt",
            "es",
            "und",
            "mit",
            "für",
            "ein",
            "eine",
            "einen",
            "nicht",
            "ich",
            "sie",
        ]
    ),
    "it": frozenset(
        [
            "il",
            "lo",
            "la",
            "gli",
            "è",
            "sono",
            "qual",
            "quale",
            "quali",
            "quanto",
            "come",
            "dove",
            "quando",
            "per",
            "con",
            "una",
            "del",
            "della",
            "e",
            "che",
            "non",
            "ci",
        ]
    ),
}
_UNIQUE_CHARS = {"es": "ñ¿¡", "de": "ßäöü", "fr": "çœêëîïôûùâ", "it": "ìò"}


def _latin_language(text: str) -> str | None:
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)
    scores = {lang: sum(w in markers for w in words) for lang, markers in _MARKERS.items()}
    for lang, chars in _UNIQUE_CHARS.items():
        scores[lang] += 2 * sum(lowered.count(c) for c in chars)
    best = max(scores, key=lambda lang: scores[lang])
    if scores[best] == 0:
        return None
    runner_up = max(score for lang, score in scores.items() if lang != best)
    return best if scores[best] > runner_up else None


def detect_language_confident(text: str) -> str | None:
    """The language of ``text`` as one of the UI language codes, or
    ``None`` when there isn't enough signal (e.g. "12 * 7")."""
    cjk = kana = cyrillic = latin = devanagari = 0
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            devanagari += 1
        elif 0x3040 <= code <= 0x30FF:
            kana += 1
        elif 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
            cjk += 1
        elif 0x0400 <= code <= 0x04FF:
            cyrillic += 1
        elif ch.isalpha() and code < 0x0250:
            latin += 1
    if kana:
        return "ja"
    # One CJK character carries roughly a word's worth of meaning, so it is
    # weighted against Latin/Cyrillic letters (≈ 5 per word) accordingly.
    if cjk and cjk * 5 >= max(cyrillic, latin):
        return "zh"
    if devanagari and devanagari > latin:
        return "hi"
    if cyrillic and cyrillic > latin:
        return "ru"
    if latin:
        return _latin_language(text)
    return None


def detect_language(text: str) -> str:
    """Like :func:`detect_language_confident`, defaulting to ``"en"``."""
    return detect_language_confident(text) or "en"
