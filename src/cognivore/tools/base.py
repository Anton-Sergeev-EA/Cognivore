"""The tool interface and a simple registry.

A tool accepts a JSON object and returns a text observation. The registry
checks required fields and supported primitive types before invoking the tool.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

_JSON_TYPES = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


class Tool(ABC):
    name: str
    description: str
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Executes the tool and returns a plain-text observation."""

    def as_prompt_spec(self) -> str:
        return f"- {self.name}: {self.description}\n  parameters: {self.parameters}"


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in (tools or [])}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def invoke(self, name: str, arguments: dict[str, Any], *, user_input: str = "") -> str:
        tool = self.get(name)
        if tool is None:
            available = ", ".join(self.names())
            return f"Error: unknown tool '{name}'. Available tools: {available}"
        try:
            # Pass the original question separately for tools such as RAG search.
            # Model-provided arguments cannot replace this context.
            call_kwargs = {**arguments, "_user_input": user_input}
            for field in tool.parameters.get("required", []):
                if field not in arguments:
                    return f"Error running tool '{name}': missing required argument '{field}'."
            for field, schema in tool.parameters.get("properties", {}).items():
                expected = schema.get("type")
                accepted = _JSON_TYPES.get(expected) if isinstance(expected, str) else None
                if field in arguments and accepted and type(arguments[field]) not in accepted:
                    return f"Error running tool '{name}': argument '{field}' must be {expected}."
            return tool.run(**call_kwargs)
        except Exception as exc:
            return f"Error running tool '{name}': {exc}"

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def prompt_block(self) -> str:
        if not self._tools:
            return "(no tools available)"
        return "\n".join(t.as_prompt_spec() for t in self._tools.values())
