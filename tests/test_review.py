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
