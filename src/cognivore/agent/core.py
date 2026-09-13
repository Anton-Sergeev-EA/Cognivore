"""The agent loop: a ReAct-style Thought/Action/Observation cycle over a
pluggable :class:`~cognivore.llm.base.LLMBackend`, dispatching to tools from
a :class:`~cognivore.tools.base.ToolRegistry`.
"""

from __future__ import annotations

import re
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field

from cognivore.agent.memory import ConversationBuffer, VectorMemory
from cognivore.agent.parsing import AgentStep, parse_step
from cognivore.agent.prompts import build_system_prompt
from cognivore.llm.base import ChatMessage, GenerationConfig, LLMBackend
from cognivore.tools.base import ToolRegistry

# Cheap presence/position checks used only to decide, *while tokens are
# still streaming in*, whether the model has started its "Final Answer:"
# section (in which case raw tokens from here on ARE the answer and can be
# forwarded to the caller live) or an "Action:" call (in which case we keep
# buffering silently -- streaming half-typed ReAct/JSON syntax to a chat UI
# would look broken, not impressive). ``parsing.parse_step`` re-parses the
# complete text afterwards regardless, so a wrong guess here only affects
# how much appears to stream live, never correctness.
_FINAL_MARKER_RE = re.compile(r"Final Answer:\s*", re.I)
_ACTION_MARKER_RE = re.compile(r"\bAction:\s*", re.I)


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

    def run_stream(self, user_input: str) -> Iterator[tuple[str, TraceEntry | str | AgentResult]]:
        """Like :meth:`run`, but yields events as the turn progresses instead
        of blocking until it's entirely finished.

        A real local model can easily take tens of seconds for a full
        ReAct turn; ``run`` (and the old streaming endpoint, which just
        chunked up ``run``'s already-complete answer for a typewriter
        effect) leaves the UI silent for all of that. This instead streams
        the model's *final* answer token-by-token as it's generated, so
        the person watches an answer materialize instead of staring at a
        blank chat bubble.

        Yields ``("trace", TraceEntry)`` for each completed intermediate
        Thought/Action/Observation step, ``("answer_delta", str)`` for
        live chunks of the final answer as they stream in, and finally
        ``("final", AgentResult)`` once the whole turn is done.
        """
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
            completion = yield from self._stream_step(messages)
            step: AgentStep = parse_step(completion)

            if step.is_final:
                answer = step.final_answer or ""
                trace.append(TraceEntry(step.thought, None, {}, None))
                self._remember_turn(user_input, answer)
                yield ("final", AgentResult(answer=answer, trace=trace))
                return

            observation = self._dispatch(step)
            entry = TraceEntry(step.thought, step.action, step.action_input, observation)
            trace.append(entry)
            yield ("trace", entry)
            messages.append(ChatMessage(role="assistant", content=completion))
            messages.append(ChatMessage(role="user", content=f"Observation: {observation}"))

        messages.append(
            ChatMessage(
                role="user",
                content="You've used all available steps. Give your best Final Answer now.",
            )
        )
        completion = yield from self._stream_step(messages)
        final_step = parse_step(completion)
        answer = final_step.final_answer or completion.strip()
        self._remember_turn(user_input, answer)
        yield ("final", AgentResult(answer=answer, trace=trace, hit_step_limit=True))

    def _stream_step(self, messages: list[ChatMessage]) -> Generator[tuple[str, str], None, str]:
        """Streams one model completion, forwarding raw text live once (and
        only once) it looks like a "Final Answer:" section has started.
        Always returns the full, unmodified completion text so the caller
        can run the real parser on it, regardless of what was streamed.
        """
        buffer = ""
        streaming_answer = False
        for chunk in self.llm.stream(messages, self.generation_config):
            buffer += chunk
            if streaming_answer:
                yield ("answer_delta", chunk)
                continue
            final_match = _FINAL_MARKER_RE.search(buffer)
            action_match = _ACTION_MARKER_RE.search(buffer)
            if final_match and (not action_match or final_match.start() < action_match.start()):
                streaming_answer = True
                already_streamed = buffer[final_match.end() :]
                if already_streamed:
                    yield ("answer_delta", already_streamed)
        return buffer

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
