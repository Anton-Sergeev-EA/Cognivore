"""The agent loop: a ReAct-style Thought/Action/Observation cycle over a
pluggable :class:`~cognivore.llm.base.LLMBackend`, dispatching to tools from
a :class:`~cognivore.tools.base.ToolRegistry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cognivore.agent.memory import ConversationBuffer, VectorMemory
from cognivore.agent.parsing import AgentStep, parse_step
from cognivore.agent.prompts import build_system_prompt
from cognivore.llm.base import ChatMessage, GenerationConfig, LLMBackend
from cognivore.tools.base import ToolRegistry


@dataclass
class TraceEntry:
    thought: str
    action: str | None
    action_input: dict
    observation: str | None


@dataclass
class AgentResult:
    answer: str
    trace: list[TraceEntry] = field(default_factory=list)
    hit_step_limit: bool = False


class Agent:
    def __init__(
        self,
        llm: LLMBackend,
        tools: ToolRegistry,
        max_steps: int = 6,
        generation_config: GenerationConfig | None = None,
        conversation: ConversationBuffer | None = None,
        vector_memory: VectorMemory | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.generation_config = generation_config or GenerationConfig()
        self.conversation = conversation or ConversationBuffer()
        self.vector_memory = vector_memory
        self.system_prompt = build_system_prompt(tools, max_steps)

    def run(self, user_input: str) -> AgentResult:
        recalled = self.vector_memory.recall(user_input) if self.vector_memory else []
        messages: list[ChatMessage] = [ChatMessage(role="system", content=self.system_prompt)]
        if recalled:
            memory_block = "\n".join(f"- {m}" for m in recalled)
            messages.append(
                ChatMessage(role="system", content=f"Relevant memory from earlier:\n{memory_block}")
            )
        messages.extend(self.conversation.as_list())
        messages.append(ChatMessage(role="user", content=user_input))

        trace: list[TraceEntry] = []
        for _step_num in range(self.max_steps):
            completion = self.llm.generate(messages, self.generation_config)
            step: AgentStep = parse_step(completion)

            if step.is_final:
                answer = step.final_answer or ""
                trace.append(TraceEntry(step.thought, None, {}, None))
                self._remember_turn(user_input, answer)
                return AgentResult(answer=answer, trace=trace)

            observation = self._dispatch(step)
            trace.append(TraceEntry(step.thought, step.action, step.action_input, observation))
            messages.append(ChatMessage(role="assistant", content=completion))
            messages.append(ChatMessage(role="user", content=f"Observation: {observation}"))

        # Step budget exhausted: ask once more for a best-effort final answer.
        messages.append(
            ChatMessage(
                role="user",
                content="You've used all available steps. Give your best Final Answer now.",
            )
        )
        completion = self.llm.generate(messages, self.generation_config)
        final_step = parse_step(completion)
        answer = final_step.final_answer or completion.strip()
        self._remember_turn(user_input, answer)
        return AgentResult(answer=answer, trace=trace, hit_step_limit=True)

    def _dispatch(self, step: AgentStep) -> str:
        if step.action is None:
            return "Error: no action specified."
        tool = self.tools.get(step.action)
        if tool is None:
            available = ", ".join(self.tools.names())
            return f"Error: unknown tool '{step.action}'. Available tools: {available}"
        try:
            return tool.run(**step.action_input)
        except Exception as exc:
            return f"Error running tool '{step.action}': {exc}"

    def _remember_turn(self, user_input: str, answer: str) -> None:
        self.conversation.add("user", user_input)
        self.conversation.add("assistant", answer)
        if self.vector_memory is not None:
            self.vector_memory.remember(f"User asked: {user_input}\nAgent answered: {answer}")
