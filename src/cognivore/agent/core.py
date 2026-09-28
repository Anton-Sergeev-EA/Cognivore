"""The agent loop: a ReAct-style Thought/Action/Observation cycle over a
pluggable :class:`~cognivore.llm.base.LLMBackend`, dispatching to tools from
a :class:`~cognivore.tools.base.ToolRegistry`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field

from cognivore.agent.memory import ConversationBuffer, VectorMemory
from cognivore.agent.parsing import AgentStep, parse_step
from cognivore.agent.prompts import build_system_prompt
from cognivore.agent.replies import not_in_knowledge_base
from cognivore.llm.base import ChatMessage, GenerationConfig, LLMBackend
from cognivore.ml.lang import LANGUAGE_NAMES, detect_language_confident
from cognivore.tools.base import ToolRegistry
from cognivore.tools.rag_search import NOTHING_FOUND

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

SEARCH_TOOL = "search_knowledge_base"

# For a user message: formatted knowledge-base passages; "" when the
# knowledge base was searched and has nothing on it; None when a lookup
# doesn't apply (see bootstrap.build_context_provider).
ContextProvider = Callable[[str], str | None]


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
        context_provider: ContextProvider | None = None,
        strict_knowledge_answers: bool = False,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.generation_config = generation_config or GenerationConfig()
        self.conversation = conversation or ConversationBuffer()
        self.vector_memory = vector_memory
        self.context_provider = context_provider
        # When the knowledge base was searched for a question and has nothing
        # on it, answer "not in the knowledge base" directly instead of
        # asking the model. Live testing: told exactly that, a 3B model
        # still answered with its own meta-reasoning ("I must tell the user
        # that...") or a garbled tool call about half the time.
        self.strict_knowledge_answers = strict_knowledge_answers
        self.system_prompt = build_system_prompt(tools, max_steps)

    def _prepare(self, user_input: str) -> tuple[list[ChatMessage], list[TraceEntry], str | None]:
        """Builds the prompt for a turn, plus any trace steps that happened
        before the model was called.

        Two things are done here rather than left to the model, because
        live testing showed small local models (3B) skipping them:

        * **Retrieval.** Given a ``context_provider``, relevant knowledge-base
          passages are put in front of the model up front, instead of
          hoping it decides to call ``search_knowledge_base`` -- a 3B model
          frequently answered from its own guesses without ever searching.
          The lookup is recorded as a normal search step, so the trace and
          the explainability layer see it like any other tool call.
        * **Answer language.** The model is told the language by name;
          a generic "same language as the user" rule wasn't enough to stop
          a Qwen model from drifting into Chinese mid-sentence.
        """
        recalled = self.vector_memory.recall(user_input) if self.vector_memory else []
        messages: list[ChatMessage] = [ChatMessage(role="system", content=self.system_prompt)]
        if recalled:
            memory_block = "\n".join(f"- {m}" for m in recalled)
            messages.append(
                ChatMessage(role="system", content=f"Relevant memory from earlier:\n{memory_block}")
            )

        pre_trace: list[TraceEntry] = []
        direct_answer: str | None = None
        passages = self.context_provider(user_input) if self.context_provider else None
        if passages == "" and self.strict_knowledge_answers:
            pre_trace.append(
                TraceEntry(
                    thought="Searched the knowledge base before answering.",
                    action=SEARCH_TOOL,
                    action_input={"query": user_input},
                    observation=NOTHING_FOUND,
                )
            )
            direct_answer = not_in_knowledge_base(user_input)
        elif passages == "":
            # Searched, and the knowledge base has nothing on this. Saying so
            # explicitly: left to itself, a small model either guessed or
            # told the user to go search the internet.
            messages.append(
                ChatMessage(
                    role="system",
                    content=(
                        "The knowledge base was searched for the user's next message and "
                        "contains nothing about it. Tell the user plainly that the "
                        "knowledge base has no information on this. Do not guess, do not "
                        "invent an answer, and do not tell them to search elsewhere."
                    ),
                )
            )
            pre_trace.append(
                TraceEntry(
                    thought="Searched the knowledge base before answering.",
                    action=SEARCH_TOOL,
                    action_input={"query": user_input},
                    observation=NOTHING_FOUND,
                )
            )
        elif passages:
            messages.append(
                ChatMessage(
                    role="system",
                    content=(
                        "Passages retrieved from the knowledge base for the user's next "
                        f"message:\n\n{passages}\n\n"
                        "Base your Final Answer on these passages and keep every number, "
                        "condition and limit they state. Use only the passages that are "
                        "about what the user asked; ignore the others, and never combine "
                        "facts from different topics into one statement. If the passages "
                        "don't answer the question, say that the knowledge base doesn't "
                        "cover it. Never add details that are not in the passages. Don't "
                        "name or cite the passages or their sources in the answer: the "
                        "interface already shows them next to it."
                    ),
                )
            )
            pre_trace.append(
                TraceEntry(
                    thought="Searched the knowledge base before answering.",
                    action=SEARCH_TOOL,
                    action_input={"query": user_input},
                    observation=passages,
                )
            )

        language = detect_language_confident(user_input)
        if language is not None:
            name = LANGUAGE_NAMES[language]
            messages.append(
                ChatMessage(
                    role="system",
                    content=f"The user writes in {name}. Write the Final Answer entirely in "
                    f"{name}, without switching to any other language.",
                )
            )

        messages.extend(self.conversation.as_list())
        messages.append(ChatMessage(role="user", content=user_input))
        return messages, pre_trace, direct_answer

    def run(self, user_input: str) -> AgentResult:
        messages, trace, direct_answer = self._prepare(user_input)
        if direct_answer is not None:
            trace.append(TraceEntry("Answered without the model.", None, {}, None))
            self._remember_turn(user_input, direct_answer)
            return AgentResult(answer=direct_answer, trace=trace)
        for _step_num in range(self.max_steps):
            completion = self.llm.generate(messages, self.generation_config)
            step: AgentStep = parse_step(completion)

            if step.is_final:
                answer = step.final_answer or ""
                trace.append(TraceEntry(step.thought, None, {}, None))
                self._remember_turn(user_input, answer)
                return AgentResult(answer=answer, trace=trace)

            observation = self._dispatch(step, user_input=user_input)
            trace.append(TraceEntry(step.thought, step.action, step.action_input, observation))
            messages.append(ChatMessage(role="assistant", content=completion))
            messages.append(self._observation_message(observation))

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
        messages, trace, direct_answer = self._prepare(user_input)
        for entry in trace:
            yield ("trace", entry)
        if direct_answer is not None:
            trace.append(TraceEntry("Answered without the model.", None, {}, None))
            self._remember_turn(user_input, direct_answer)
            yield ("answer_delta", direct_answer)
            yield ("final", AgentResult(answer=direct_answer, trace=trace))
            return
        for _step_num in range(self.max_steps):
            completion = yield from self._stream_step(messages)
            step: AgentStep = parse_step(completion)

            if step.is_final:
                answer = step.final_answer or ""
                trace.append(TraceEntry(step.thought, None, {}, None))
                self._remember_turn(user_input, answer)
                yield ("final", AgentResult(answer=answer, trace=trace))
                return

            observation = self._dispatch(step, user_input=user_input)
            entry = TraceEntry(step.thought, step.action, step.action_input, observation)
            trace.append(entry)
            yield ("trace", entry)
            messages.append(ChatMessage(role="assistant", content=completion))
            messages.append(self._observation_message(observation))

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

    def _observation_message(self, observation: str) -> ChatMessage:
        # Live-tested finding: a small model can retrieve the exact right
        # passage via search_knowledge_base and then still answer as if
        # it found nothing ("that's not specified, contact support"),
        # apparently not registering the Observation as something to
        # actually use. Naming that expectation explicitly, right where
        # the model reads the passage, is a direct attempt to fix that --
        # not yet re-verified live at the time of writing this comment.
        return ChatMessage(
            role="user",
            content=(
                f"Observation: {observation}\n\n"
                "If the passage above answers the question, use it directly in your Final "
                "Answer. Don't say the information isn't available or ask the user for more "
                "details when it's right there in the Observation."
            ),
        )

    def _dispatch(self, step: AgentStep, user_input: str = "") -> str:
        if step.action is None:
            return "Error: no action specified."
        tool = self.tools.get(step.action)
        if tool is None:
            available = ", ".join(self.tools.names())
            return f"Error: unknown tool '{step.action}'. Available tools: {available}"
        try:
            # Every tool accepts and ignores unknown kwargs, so this is safe
            # to pass unconditionally. It exists so a tool like
            # RagSearchTool can fall back to the user's own wording of the
            # question when the model rewrites (e.g. translates) its own
            # "query" argument into something that no longer lexically
            # matches the ingested documents -- a real, observed failure
            # mode with small local models, not a hypothetical one.
            call_kwargs = {**step.action_input, "_user_input": user_input}
            return tool.run(**call_kwargs)
        except Exception as exc:
            return f"Error running tool '{step.action}': {exc}"

    def _remember_turn(self, user_input: str, answer: str) -> None:
        self.conversation.add("user", user_input)
        self.conversation.add("assistant", answer)
        if self.vector_memory is not None:
            self.vector_memory.remember(f"User asked: {user_input}\nAgent answered: {answer}")
