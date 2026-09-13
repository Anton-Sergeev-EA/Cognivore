"""A safe arithmetic calculator tool.

Deliberately does *not* use ``eval``: it parses the expression into a
Python AST and only walks a small allow-listed set of node types
(numbers, +, -, *, /, //, %, **, unary +/-, and parentheses). This is a
common trap in agent-framework demos -- "just eval() it" is a code
execution vulnerability the moment an LLM (or a prompt-injected document)
can influence the expression -- so doing it properly here is itself part of
the portfolio story.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, ClassVar

from cognivore.tools.base import Tool

_ALLOWED_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorError(ValueError):
    pass


def safe_eval(expr: str) -> float:
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"invalid expression: {exc}") from exc
    return _eval_node(node.body)


def _eval_node(node: ast.expr) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise CalculatorError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return float(_ALLOWED_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return float(_ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand)))
    raise CalculatorError(f"unsupported expression element: {ast.dump(node)}")


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluates a basic arithmetic expression, e.g. '(3 + 4) * 2 / 7'."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    }

    def run(self, expression: str = "", **_: object) -> str:
        try:
            result = safe_eval(expression)
        except (CalculatorError, ZeroDivisionError, RecursionError) as exc:
            return f"Error: {exc}"
        if result.is_integer():
            return str(int(result))
        return str(result)
