"""Script-based language identification (Chinese, Japanese, Hindi, Russian, English).

Deliberately not a statistical model: for telling these scripts
apart, counting characters by Unicode script is exact, instant and
has no failure mode on short inputs (where n-gram classifiers are weakest).
"""

from __future__ import annotations


def detect_language(text: str) -> str:
    """Returns ``"zh"``, ``"ja"``, ``"hi"``, ``"ru"`` or ``"en"`` (the
    default, also for other Latin-script languages and for text with no
    letters at all)."""
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
        elif ch.isascii() and ch.isalpha():
            latin += 1
    if kana:
        return "ja"
    # One CJK character carries roughly a word's worth of meaning, so it is
    # weighted against Latin/Cyrillic letters (≈ 5 per word) accordingly.
    if cjk * 5 >= max(cyrillic, latin) and cjk:
        return "zh"
    if devanagari > latin:
        return "hi"
    if cyrillic > latin:
        return "ru"
    return "en"
