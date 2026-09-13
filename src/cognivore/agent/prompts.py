"""The ReAct-style system prompt.

We use a plain-text Thought/Action/Observation protocol (Yao et al., 2022,
"ReAct: Synergizing Reasoning and Acting in Language Models") rather than a
backend-specific function-calling API. That's a deliberate choice: it works
identically across every local GGUF model regardless of whether it was
fine-tuned for a particular tool-call syntax, which matters a lot when the
whole point is to run entirely on-laptop with whatever open model the user
has on hand.
"""

from __future__ import annotations

from cognivore.tools.base import ToolRegistry

SYSTEM_PROMPT_TEMPLATE = """You are Cognivore, a careful, helpful AI agent. You answer the user's \
request by reasoning step by step and, when useful, calling tools.

Available tools:
{tools}

To use a tool, respond with EXACTLY this format (nothing before or after it):
Thought: <your reasoning about what to do next>
Action: <one tool name from the list above>
Action Input: <a single JSON object with the tool's parameters>

After a tool runs you will receive an "Observation:" with its result. You may repeat the \
Thought/Action/Action Input cycle as many times as needed (but no more than {max_steps} times).

When you have enough information to answer, respond with EXACTLY this format instead:
Thought: <your reasoning>
Final Answer: <your complete answer to the user, in the user's language>

Never invent an Observation yourself. Never call a tool that isn't in the list above.
"""


def build_system_prompt(tools: ToolRegistry, max_steps: int) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(tools=tools.prompt_block(), max_steps=max_steps)
