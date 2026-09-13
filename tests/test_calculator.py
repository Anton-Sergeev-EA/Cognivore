from __future__ import annotations

import pytest

from cognivore.tools.calculator import CalculatorError, CalculatorTool, safe_eval


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 2", 4.0),
        ("(3 + 4) * 2", 14.0),
        ("2 ** 10", 1024.0),
        ("-5 + 3", -2.0),
        ("7 // 2", 3.0),
        ("7 % 2", 1.0),
        ("10 / 4", 2.5),
    ],
)
def test_safe_eval(expr: str, expected: float) -> None:
    assert safe_eval(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd')",
        "1 if True else 2",
        "[1, 2, 3]",
        "a + 1",
    ],
)
def test_safe_eval_rejects_unsafe_or_unsupported(expr: str) -> None:
    with pytest.raises(CalculatorError):
        safe_eval(expr)


def test_safe_eval_division_by_zero_raises() -> None:
    with pytest.raises(ZeroDivisionError):
        safe_eval("1 / 0")


def test_calculator_tool_returns_error_string_not_exception() -> None:
    tool = CalculatorTool()
    assert tool.run(expression="1 / 0").startswith("Error:")
    assert tool.run(expression="not an expression!!").startswith("Error:")


def test_calculator_tool_formats_integers_without_decimal() -> None:
    tool = CalculatorTool()
    assert tool.run(expression="2 + 2") == "4"
    assert tool.run(expression="10 / 4") == "2.5"
