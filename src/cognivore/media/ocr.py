"""On-screen text recognition for video keyframes (Tesseract).

Three things make raw Tesseract output unfit to put in a knowledge base, and
this module handles all three:

* **Language packs.** Tesseract needs a trained-data pack per language, and
  asking for one that isn't installed doesn't degrade -- the whole call
  fails. :class:`OcrEngine` only ever asks for installed packs and reports
  the missing ones instead of silently returning nothing.
* **Choosing languages.** Recognizing with every pack at once works for any
  script but is 3-5x slower and a little less accurate than recognizing
  with the packs of the right script. The engine reads the first frame that
  has text with all packs, looks at which script came back, and uses only
  that script's packs from then on (Tesseract's own script detection,
  ``--psm 0``, proved unreliable: it took Chinese slides for Korean).
* **Artifacts.** Slide bullets come back as stray symbols or letters
  ("¢ Mettre...", "e Usage...", "» Пауза..."), and a Cyrillic word made only
  of letters that also exist in Latin can come back in Latin ("Ha" for
  "на"). :func:`clean_ocr_text` fixes both.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Tesseract packs for the nine UI languages, grouped by script. ``auto``
# (the default ``ocr_languages``) means "every installed pack from here".
SCRIPT_PACKS: dict[str, tuple[str, ...]] = {
    "latin": ("eng", "deu", "fra", "spa", "ita"),
    "cyrillic": ("rus",),
    "han": ("chi_sim",),
    "japanese": ("jpn",),
    "devanagari": ("hin",),
}
KNOWN_PACKS: tuple[str, ...] = tuple(p for packs in SCRIPT_PACKS.values() for p in packs)

# Mean word confidence below which a reading is double-checked with every
# pack (measured on slides: the right packs score 77-93, wrong ones 13-68).
_CONFIDENT = 75.0
_CJK_GAP = re.compile(
    r"(?<=[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]) (?=[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\d])|(?<=\d) (?=[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff])"
)


def script_of(text: str) -> str | None:
    """The dominant script of ``text`` (a key of :data:`SCRIPT_PACKS`), or
    ``None`` when it has no letters. Kana makes it Japanese: Japanese text
    mixes kana with the same Han characters Chinese uses. Only letters
    count -- a stray "・" (a kana-block bullet) doesn't make a slide Japanese."""
    counts = dict.fromkeys(SCRIPT_PACKS, 0)
    for ch in text:
        if not ch.isalpha():
            continue
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF:
            counts["japanese"] += 1
        elif 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
            counts["han"] += 1
        elif 0x0400 <= code <= 0x04FF:
            counts["cyrillic"] += 1
        elif 0x0900 <= code <= 0x097F:
            counts["devanagari"] += 1
        elif code < 0x0250:
            counts["latin"] += 1
    if counts["japanese"] >= 2:
        return "japanese"
    counts["japanese"] = 0
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] else None


# -- Cleanup -------------------------------------------------------------

# A bullet glyph Tesseract kept (or misread as another symbol), then space.
_BULLET = re.compile(r"^(?:[•·・。、»«¢©®°*○●■□▪►>–—\-~+<|.,‚„]|[\[(|]\w?)\s+")
# The same, glued to CJK text ("。订阅..." / "<サブ...").
_BULLET_CJK = re.compile(r"^[。、・<>•·»«*]+(?=[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff])")
# An opening quote with no closing one is a misread bullet, not a quotation.
_STRAY_QUOTE = re.compile(r"^[“\"„‚]\s*(?=\S)")
# A round bullet read as a Latin letter or zero: "e Usage", "0 सदस्यता", "0 14 दिनों".
_BULLET_LETTER = re.compile(r"^[eocO0]\s+(?=[A-ZÀ-ÖØ-Þ\d]|[^\x00-\x7f])")

# Latin letters that look exactly like Cyrillic ones.
_HOMOGLYPHS = str.maketrans("AaBCcEeHKMOoPpTXxyk", "АаВСсЕеНКМОоРрТХхук")
_HOMOGLYPH_WORD = re.compile(r"\b[AaBCcEeHKMOoPpTXxyk]+\b")


def _fix_homoglyphs(line: str) -> str:
    """In a line that is mostly Cyrillic, a word spelled only with Latin
    letters that also exist in Cyrillic is a misread Cyrillic word: "Ha 18
    процентов" -> "На 18 процентов". Genuine Latin words ("NordCloud",
    "TLS", "AES") contain letters with no Cyrillic twin and are kept."""
    cyrillic = sum(1 for ch in line if "\u0400" <= ch <= "\u04ff")
    latin = sum(1 for ch in line if ch.isascii() and ch.isalpha())
    if cyrillic <= latin:
        return line

    # All-caps words are left alone: those are usually real Latin acronyms
    # ("HP", "OK") rather than misreads, which come out mixed-case.
    def restore(match: re.Match[str]) -> str:
        word = match.group(0)
        if word.isupper():
            return word
        word = word.translate(_HOMOGLYPHS)
        # "Ha" was "на" mid-sentence; Tesseract's capital H is the misread.
        if match.start() > 0 and word[1:].islower():
            word = word.lower()
        return word

    return _HOMOGLYPH_WORD.sub(restore, line)


def clean_ocr_text(text: str) -> str:
    """Tidy raw Tesseract output: drop bullet artifacts and empty lines,
    restore misread Cyrillic words, collapse runs of spaces."""
    lines = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        line = _BULLET.sub("", line)
        line = _BULLET_CJK.sub("", line)
        line = _BULLET_LETTER.sub("", line)
        if _STRAY_QUOTE.match(line) and not any(q in line[1:] for q in '”"“'):
            line = _STRAY_QUOTE.sub("", line)
        line = _fix_homoglyphs(line)
        # A line with no letters or digits left is noise (a lone "|", "—").
        if any(ch.isalnum() for ch in line):
            lines.append(line)
    return "\n".join(lines)


# -- Engine --------------------------------------------------------------


@dataclass
class OcrEngine:
    """Tesseract with language selection. ``languages`` is ``"auto"`` or a
    ``+``-joined list of packs (``"eng+rus"``)."""

    languages: str = "auto"
    available: bool = False
    installed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    _packs: list[str] = field(default_factory=list)
    _chosen: str | None = None
    _chosen_script: str | None = None
    _pytesseract: Any = None

    def __post_init__(self) -> None:
        try:
            import pytesseract

            self.installed = sorted(set(pytesseract.get_languages(config="")) - {"osd"})
        except Exception:  # ImportError, or the tesseract binary missing
            logger.info("On-screen text recognition unavailable (Tesseract not found).")
            return
        self._pytesseract = pytesseract
        if self.languages.strip().lower() in ("", "auto"):
            wanted = [p for p in KNOWN_PACKS if p in self.installed] or self.installed
        else:
            wanted = [p for p in self.languages.split("+") if p]
            self.missing = [p for p in wanted if p not in self.installed]
            if self.missing:
                logger.warning(
                    "Tesseract language pack(s) not installed, skipped: %s",
                    ", ".join(self.missing),
                )
        self._packs = [p for p in wanted if p in self.installed]
        self.available = bool(self._packs)

    @property
    def languages_used(self) -> str:
        return self._chosen or "+".join(self._packs)

    def _script_packs(self, script: str | None) -> str | None:
        if script is None:
            return None
        packs = [p for p in SCRIPT_PACKS[script] if p in self._packs]
        # English rides along with every non-Latin script: slides mix in
        # Latin product names, units and acronyms.
        if script != "latin" and "eng" in self._packs and "eng" not in packs:
            packs.append("eng")
        return "+".join(packs) or None

    def _recognize(self, frame: Any, lang: str) -> tuple[str, float]:
        """Cleaned text and Tesseract's mean word confidence (0-100)."""
        data = self._pytesseract.image_to_data(
            frame, lang=lang, output_type=self._pytesseract.Output.DICT
        )
        lines: dict[tuple[int, int, int], list[str]] = {}
        confidences = []
        for i, word in enumerate(data["text"]):
            confidence = float(data["conf"][i])
            if confidence < 0 or not word.strip():
                continue
            confidences.append(confidence)
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
        text = "\n".join(" ".join(words) for words in lines.values())
        # Tesseract splits Chinese/Japanese into "words"; the text has no spaces.
        text = _CJK_GAP.sub("", text)
        mean = sum(confidences) / len(confidences) if confidences else 0.0
        return clean_ocr_text(text), mean

    def read(self, frame: Any) -> str:
        """The cleaned text on ``frame`` (a BGR image), or ``""``.

        Reads with the language set chosen so far; when that reading is
        unconvincing (low confidence, or text in another script -- a video
        that switches language), reads the frame with every pack, picks the
        set for the script that came back and reads it again with that.
        """
        if not self.available:
            return ""
        every = "+".join(self._packs)
        try:
            if self._chosen is not None:
                text, confidence = self._recognize(frame, self._chosen)
                expected = self._chosen_script
                if not text or (confidence >= _CONFIDENT and script_of(text) in (expected, None)):
                    return text
            text, confidence = self._recognize(frame, every)
            script = script_of(text)
            narrower = self._script_packs(script)
            if narrower and narrower != every:
                narrow_text, narrow_confidence = self._recognize(frame, narrower)
                if narrow_confidence >= confidence or not text:
                    text = narrow_text
                self._chosen, self._chosen_script = narrower, script
            return text
        except Exception:
            logger.warning("OCR failed on a frame.", exc_info=True)
            return ""
