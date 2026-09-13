"""The tool interface and a simple registry.

A ``Tool`` is anything the agent can call by name with a single JSON-object
argument and get back a string observation. Tools declare a JSON-schema-ish
``parameters`` dict purely for documentation/prompting purposes -- we don't
do full JSON Schema validation here to keep things dependency-light, but the
shape follows the JSON Schema conventions models are trained on so it slots
into function-calling APIs unchanged if a backend supports those natively.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


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
