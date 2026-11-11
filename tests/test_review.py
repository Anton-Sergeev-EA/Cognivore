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
    assert len(findings) == 1
    assert findings[0].kind == "ambiguous"
    assert findings[0].after is not None
    assert {findings[0].after.line, *(item.line for item in findings[0].alternatives)} == {1, 2}


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


def test_renumbered_requirement_is_flagged_without_asserting_a_match() -> None:
    findings = compare_requirements(
        "v1",
        "NET-1: The gateway must limit outbound connections to 10.",
        "v2",
        "NET-3: The gateway must limit outbound connections to 20.",
        HashingEmbedder(),
    )
    assert [item.kind for item in findings] == ["ambiguous"]


class SectionEmbedder:
    id = "section-test"
    dim = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "Retain logs.": [1.0, 0.0],
            "Archive records.": [0.6, 0.8],
            "Calibrate sensors.": [0.0, 1.0],
        }
        return [[1.0, 0.0] if "\n" in text else vectors[text] for text in texts]


def test_section_context_requires_similarity_in_the_requirement_itself() -> None:
    before = "# Operational history\nRetain logs."
    after = "# Auditing\nArchive records."
    model = SectionEmbedder()
    baseline = compare_requirements("v1", before, "v2", after, model, use_section_context=False)
    assert [item.kind for item in baseline] == ["ambiguous"]

    contextual = compare_requirements("v1", before, "v2", after, model)
    assert [item.kind for item in contextual] == ["changed"]
    assert contextual[0].before is not None
    assert contextual[0].before.section == "Operational history"
    assert contextual[0].before.line == 2

    unrelated = compare_requirements(
        "v1", before, "v2", "# Operational history\nCalibrate sensors.", model
    )
    assert [item.kind for item in unrelated] == ["removed", "added"]
