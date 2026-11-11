from pathlib import Path

from cognivore.rag.embeddings import HashingEmbedder
from cognivore.review import compare_requirements

_STATIC = Path(__file__).resolve().parents[1] / "src/cognivore/web/static"


def test_demo_recovers_annotated_changes() -> None:
    before = (_STATIC / "demo-before.md").read_text(encoding="utf-8")
    after = (_STATIC / "demo-after.md").read_text(encoding="utf-8")
    findings = compare_requirements("v1", before, "v2", after, HashingEmbedder())
    expected = {
        ("changed", 4, 13): ["number or unit"],
        ("changed", 8, 9): ["number or unit"],
        ("removed", 9, None): [],
        ("changed", 12, 5): ["obligation", "condition or exception"],
        ("added", None, 6): [],
    }
    actual = {
        (
            item.kind,
            item.before.line if item.before else None,
            item.after.line if item.after else None,
        ): item.signals
        for item in findings
    }
    assert actual == expected


def test_equally_plausible_matches_are_marked_ambiguous() -> None:
    findings = compare_requirements(
        "v1",
        "The gateway must retain event logs for 30 days.",
        "v2",
        "The gateway must retain event logs for 60 days.\n"
        "The gateway must retain event logs for 90 days.",
        HashingEmbedder(),
    )
    assert findings[0].kind == "ambiguous"


def test_number_obligation_and_condition_changes_in_german_and_russian() -> None:
    examples = [
        (
            "NET-1: Das Gateway darf bei Störung 30 Sekunden warten, wenn der Puffer voll ist.",
            "NET-1: Das Gateway muss bei Störung 10 Minuten warten, nur wenn der Puffer voll ist.",
        ),
        (
            "NET-1: Шлюз может ждать 30 секунд, если буфер заполнен.",
            "NET-1: Шлюз должен ждать 10 минут, только если буфер заполнен.",
        ),
    ]
    for before, after in examples:
        findings = compare_requirements("v1", before, "v2", after, HashingEmbedder())
        assert len(findings) == 1
        assert findings[0].kind == "changed"
        assert findings[0].signals == ["number or unit", "obligation", "condition or exception"]
