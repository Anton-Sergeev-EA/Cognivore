"""OCR post-processing (cognivore.media.ocr): pure functions, no Tesseract
needed. The inputs are real Tesseract outputs seen on slide videos."""

from __future__ import annotations

import pytest

from cognivore.media.ocr import OcrEngine, clean_ocr_text, script_of


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Bullet glyphs Tesseract keeps or misreads, in several scripts.
        ("» Пауза подписки до 3 месяцев", "Пауза подписки до 3 месяцев"),
        ("¢ Mettre l'abonnement en pause", "Mettre l'abonnement en pause"),
        ("e Usage-based billing", "Usage-based billing"),
        ("« Abonnement bis zu 3 Monate", "Abonnement bis zu 3 Monate"),
        ("+ Pausa de la suscripción", "Pausa de la suscripción"),
        ("‚ Нажмите Экспорт данных", "Нажмите Экспорт данных"),
        (". Не более 10 ГБ трафика", "Не более 10 ГБ трафика"),
        ("。 首次付款后14天内可全额退款", "首次付款后14天内可全额退款"),
        ("・サブスクリプション", "サブスクリプション"),
        ("“首次付款", "首次付款"),
        ("[1 सदस्यता को 3 महीने", "सदस्यता को 3 महीने"),
        ("0 14 दिनों के भीतर", "14 दिनों के भीतर"),
        # Cyrillic words misread as look-alike Latin letters.
        ("Выручка выросла Ha 18 процентов", "Выручка выросла на 18 процентов"),
        ("Ha складе 40 позиций", "На складе 40 позиций"),
    ],
)
def test_clean_ocr_text_removes_artifacts(raw: str, expected: str) -> None:
    assert clean_ocr_text(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "3 месяца бесплатно",  # a leading number is content, not a bullet
        "$9 per month",
        "(a) first option",
        "HP и OK в одной строке",  # all-caps Latin acronyms stay Latin
        "Шифрование AES-256 и TLS 1.3",
        "Квартальный отчёт NordCloud",
        "“Цитата” — автор",  # a real quotation keeps its quotes
    ],
)
def test_clean_ocr_text_keeps_real_content(text: str) -> None:
    assert clean_ocr_text(text) == text


def test_clean_ocr_text_drops_noise_lines_and_spacing() -> None:
    assert clean_ocr_text("Title\n\n  |  \n—\nFirst   point ") == "Title\nFirst point"


@pytest.mark.parametrize(
    ("text", "script"),
    [
        ("Политика возврата", "cyrillic"),
        ("Refund policy", "latin"),
        ("Rückerstattung für Kunden", "latin"),
        ("退款政策", "han"),
        ("返金ポリシー", "japanese"),
        ("रिफ़ंड नीति", "devanagari"),
        # A kana-block bullet alone doesn't make English text Japanese.
        ("・ Usage-based billing", "latin"),
        ("12 * 7", None),
    ],
)
def test_script_of(text: str, script: str | None) -> None:
    assert script_of(text) == script


def test_ocr_engine_reports_missing_language_packs() -> None:
    engine = OcrEngine("eng+not_a_real_pack")
    if not engine.installed:
        pytest.skip("tesseract not installed")
    assert engine.missing == ["not_a_real_pack"]
    assert "not_a_real_pack" not in engine.languages_used


def test_ocr_engine_without_tesseract_is_unavailable_not_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pytesseract":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    engine = OcrEngine()
    assert not engine.available
    assert engine.read(object()) == ""
