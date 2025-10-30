"""ReAct agent workflow implementation."""

import logging

from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import ToolOutput, ToolSelection
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from deepresearch.utils import get_llm


def _extract_answer_only(text: str) -> str:
    """Return only the content after an 'Answer:' label if present.

    Otherwise, strip common ReAct prefixes like 'Thought:' and 'Action:' lines at the top.
    """
    if not text:
        return text
    lower = text.lower()
    # Prefer explicit Answer: label
    if "answer:" in lower:
        idx = lower.find("answer:")
        return text[idx + len("answer:") :].strip()
    # Otherwise, drop leading Thought:/Action:/Observation: lines
    lines = list(text.splitlines())
    cleaned = []
    skipping = True
    for ln in lines:
        ln_stripped = ln.strip()
        if skipping and (ln_stripped.lower().startswith("thought:") or ln_stripped.lower().startswith("action:") or ln_stripped.lower().startswith("observation:")):
            # skip these header lines
            continue
        skipping = False
        cleaned.append(ln)
    out = "\n".join(cleaned).strip()
    return out if out else text

logger = logging.getLogger(__name__)


class PrepEvent(Event):
    """Event to prepare chat history."""

    pass


class InputEvent(Event):
    """Event containing LLM input."""

    input: list[ChatMessage]


class StreamEvent(Event):
    """Event for streaming LLM response."""

    delta: str


class ReasoningEvent(Event):
    """Event for streaming reasoning steps."""

    reasoning: str


class ToolCallEvent(Event):
    """Event containing tool calls to execute."""

    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    """Event containing tool output."""

    output: ToolOutput


class ReActAgent(Workflow):
    """ReAct agent workflow that can use tools to answer questions."""

    def __init__(
        self,
        *args: object,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        extra_context: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.llm = llm or get_llm()
        self.formatter = ReActChatFormatter.from_defaults(context=extra_context or "")
        self.output_parser = ReActOutputParser()

    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        """Handle new user message and prepare context."""
        # Clear sources
        ctx.data["sources"] = []
        # Reset loop/parse guards
        ctx.data["iterations"] = 0
        ctx.data["parse_errors"] = 0

        # Init memory if needed
        memory = ctx.data.get("memory", None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        # Get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        # Clear current reasoning
        ctx.data["current_reasoning"] = []

        # Set memory
        ctx.data["memory"] = memory

        return PrepEvent()

    @step
    async def prepare_chat_history(self, ctx: Context, ev: PrepEvent) -> InputEvent:
        """Prepare chat history and format ReAct prompt."""
        # Get chat history
        memory = ctx.data.get("memory")
        chat_history = memory.get()
        current_reasoning = ctx.data.get("current_reasoning", [])

        # Format the prompt with react instructions
        llm_input = self.formatter.format(
            self.tools, chat_history, current_reasoning=current_reasoning
        )
        return InputEvent(input=llm_input)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent | PrepEvent:
        """Handle LLM input and parse response."""
        # Iteration guard to prevent infinite loops
        MAX_ITERATIONS = 12
        iterations = int(ctx.data.get("iterations", 0)) + 1
        ctx.data["iterations"] = iterations
        if iterations > MAX_ITERATIONS:
            # Fallback: compose a brief answer from available sources
            sources = ctx.data.get("sources", [])
            fallback = "I'm concluding to avoid a loop."
            if sources:
                try:
                    latest = sources[-1].content if hasattr(sources[-1], "content") else str(sources[-1])
                    fallback = latest
                except Exception as _e:
                    logger.exception("Failed selecting fallback source: %s", _e)
            return StopEvent(result={
                "response": fallback,
                "sources": sources,
                "reasoning": ctx.data.get("current_reasoning", []),
            })
        # If a previous tool produced a final response, short-circuit here
        forced = ctx.data.get("force_final_response")
        if forced:
            ctx.data.pop("force_final_response", None)
            return StopEvent(
                result={
                    "response": forced,
                    "sources": ctx.data.get("sources", []),
                    "reasoning": ctx.data.get("current_reasoning", []),
                }
            )

        chat_history = ev.input
        current_reasoning = ctx.data.get("current_reasoning", [])
        memory = ctx.data.get("memory")

        # Use non-streaming mode to avoid JSON parsing issues
        response = await self.llm.achat(chat_history)

        if not response:
            return PrepEvent()

        # Stream the raw response for debugging
        ctx.write_event_to_stream(
            StreamEvent(delta=f"\n🤔 Raw LLM Response:\n{response.message.content}\n")
        )

        try:
            reasoning_step = self.output_parser.parse(response.message.content)
            current_reasoning.append(reasoning_step)

            # Stream reasoning step details
            if isinstance(reasoning_step, ActionReasoningStep):
                ctx.write_event_to_stream(
                    ReasoningEvent(
                        reasoning=f"🔧 Action: {reasoning_step.action}\n"
                        f"📝 Input: {reasoning_step.action_input}\n"
                        f"💭 Thought: {reasoning_step.thought}\n"
                    )
                )
            elif isinstance(reasoning_step, ObservationReasoningStep):
                ctx.write_event_to_stream(
                    ReasoningEvent(
                        reasoning=f"👁️ Observation: {reasoning_step.observation}\n"
                    )
                )
            else:
                ctx.write_event_to_stream(
                    ReasoningEvent(
                        reasoning=f"📋 Response: {reasoning_step.response}\n"
                    )
                )

            if reasoning_step.is_done:
                memory.put(
                    ChatMessage(role="assistant", content=_extract_answer_only(reasoning_step.response))
                )
                ctx.data["memory"] = memory
                ctx.data["current_reasoning"] = current_reasoning

                sources = ctx.data.get("sources", [])

                return StopEvent(
                    result={
                        "response": _extract_answer_only(reasoning_step.response),
                        "sources": sources,
                        "reasoning": current_reasoning,
                    }
                )
            elif isinstance(reasoning_step, ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="fake",
                            tool_name=tool_name,
                            tool_kwargs=tool_args,
                        )
                    ]
                )
        except Exception as e:
            # Try to handle common parsing errors gracefully
            content = response.message.content
            # Parse error guard
            ctx.data["parse_errors"] = int(ctx.data.get("parse_errors", 0)) + 1
            MAX_PARSE_ERRORS = 3
            if ctx.data["parse_errors"] >= MAX_PARSE_ERRORS:
                # Fallback: return best available content/sources
                sources = ctx.data.get("sources", [])
                fallback = _extract_answer_only(content) if content else "Unable to parse response."
                return StopEvent(result={
                    "response": fallback,
                    "sources": sources,
                    "reasoning": ctx.data.get("current_reasoning", []),
                })

            # Check if this looks like a final answer with incorrect formatting
            is_final_answer = False
            answer_text = None

            # Check for explicit Answer: format
            if "Answer:" in content:
                try:
                    answer_start = content.find("Answer:")
                    if answer_start != -1:
                        answer_text = content[answer_start + 7 :].strip()
                        if answer_text:
                            is_final_answer = True
                except Exception as _e:
                    logger.exception("Failed extracting explicit Answer: %s", _e)

            # Check for indicators that LLM wants to provide final answer without tools
            if not is_final_answer:
                # Look for phrases that indicate a final answer is coming
                final_answer_indicators = [
                    "I can answer without using any more tools",
                    "I'll use the user's language to answer",
                    "I will now provide the final answer",
                    "I can now provide a comprehensive answer",
                    "I have gathered enough information to answer",
                    "Based on the information collected",
                ]

                for indicator in final_answer_indicators:
                    if indicator.lower() in content.lower():
                        # Try to extract the answer part after the indicator
                        indicator_pos = content.lower().find(indicator.lower())
                        if indicator_pos >= 0:
                            # Extract everything after the indicator as potential answer
                            potential_answer = content[
                                indicator_pos + len(indicator) :
                            ].strip()
                            # Remove "Thought:" or "Action:" prefixes if present
                            for prefix in ["Thought:", "Action:", "Observation:"]:
                                if potential_answer.startswith(prefix):
                                    potential_answer = potential_answer[
                                        len(prefix) :
                                    ].strip()

                            MIN_FINAL_ANSWER_LEN = 20
                            if len(potential_answer) > MIN_FINAL_ANSWER_LEN:  # Reasonable answer length
                                answer_text = potential_answer
                                is_final_answer = True
                                break

                # If no indicator found but content is substantial and doesn't contain Action:
                # and we've done some reasoning steps, treat as final answer
                MIN_REASONING_STEPS = 2
                if not is_final_answer and len(current_reasoning) > MIN_REASONING_STEPS:
                    MIN_CONTENT_FOR_FINAL = 100
                    if "Action:" not in content and len(content) > MIN_CONTENT_FOR_FINAL:
                        # Check if this looks like a thoughtful conclusion
                        if any(
                            word in content.lower()
                            for word in [
                                "therefore",
                                "in conclusion",
                                "in summary",
                                "based on",
                                "the answer is",
                            ]
                        ):
                            answer_text = content
                            is_final_answer = True

            # If we detected a final answer, return it
            if is_final_answer and answer_text:
                try:
                    from llama_index.core.agent.react.types import (
                        ResponseReasoningStep,
                    )

                    final_step = ResponseReasoningStep(response=answer_text)
                    current_reasoning.append(final_step)

                    # Stream the final response
                    ctx.write_event_to_stream(
                        ReasoningEvent(reasoning=f"📋 Response: {answer_text}\n")
                    )

                    # Store in memory and return stop event
                    memory.put(ChatMessage(role="assistant", content=answer_text))
                    ctx.data["memory"] = memory
                    ctx.data["current_reasoning"] = current_reasoning

                    sources = ctx.data.get("sources", [])
                    return StopEvent(
                        result={
                            "response": answer_text,
                            "sources": sources,
                            "reasoning": current_reasoning,
                        }
                    )
                except Exception as parse_error:
                    # Fall through to original error handling if answer extraction fails
                    _ = parse_error  # Suppress unused variable warning

            # Original error handling for other cases
            # Only log the error if we haven't successfully extracted an answer
            error_msg = f"There was an error in parsing my reasoning: {e}"
            current_reasoning.append(ObservationReasoningStep(observation=error_msg))
            ctx.write_event_to_stream(
                ReasoningEvent(reasoning=f"❌ Error: {error_msg}\n")
            )
            ctx.data["current_reasoning"] = current_reasoning

        # If no tool calls or final response, iterate again
        return PrepEvent()

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> PrepEvent:
        """Handle tool calls and execute them."""
        import inspect

        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        current_reasoning = ctx.data.get("current_reasoning", [])
        sources = ctx.data.get("sources", [])

        # Call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                error_msg = f"Tool {tool_call.tool_name} does not exist"
                current_reasoning.append(
                    ObservationReasoningStep(observation=error_msg)
                )
                ctx.write_event_to_stream(ReasoningEvent(reasoning=f"❌ {error_msg}\n"))
                continue

            try:
                ctx.write_event_to_stream(
                    ReasoningEvent(
                        reasoning=f"🔧 Executing tool: {tool_call.tool_name}\n"
                        f"📝 With arguments: {tool_call.tool_kwargs}\n"
                    )
                )

                # Handle both sync and async tools
                tool_output = tool(**tool_call.tool_kwargs)
                
                # Check if tool is async and await if needed
                if inspect.iscoroutine(tool_output):
                    tool_output = await tool_output

                # If output is already a ToolOutput, use it; otherwise wrap it
                if not isinstance(tool_output, ToolOutput):
                    tool_output = ToolOutput(
                        content=str(tool_output), tool_name=tool_call.tool_name
                    )

                # Keep full tool outputs to preserve completeness
                sources.append(tool_output)

                # Stream tool output
                ctx.write_event_to_stream(
                    ReasoningEvent(reasoning=f"✅ Tool result: {tool_output.content}\n")
                )

                current_reasoning.append(
                    ObservationReasoningStep(observation=tool_output.content)
                )

                # Conditional stop: if sub-agent returned a substantive answer, end
                if tool_call.tool_name == "run_sub_agent_with_logging":
                    content = tool_output.content or ""
                    cleaned = content.strip()
                    # Strip common prefix like: "Sub-agent 'name' response: ..."
                    import re as _re
                    cleaned = _re.sub(r"^Sub-agent\s+'[^']+'\s+response:\s*", "", cleaned, flags=_re.IGNORECASE)
                    # consider empty/placeholder responses as incomplete
                    is_no_result = cleaned.strip() == "NO_RESULT"
                    MIN_SUBSTANTIVE_LEN = 50
                    is_substantive = (len(cleaned.strip()) > MIN_SUBSTANTIVE_LEN) and ("No response" not in cleaned)
                    if is_substantive and not is_no_result:
                        # set flag for next step to return StopEvent with cleaned content
                        ctx.data["force_final_response"] = cleaned
            except Exception as e:
                error_msg = f"Error calling tool {tool.metadata.get_name()}: {e}"
                # Suppress cancel scope errors as they're usually cleanup issues
                if "cancel scope" not in str(e).lower():
                    current_reasoning.append(
                        ObservationReasoningStep(observation=error_msg)
                    )
                    ctx.write_event_to_stream(ReasoningEvent(reasoning=f"❌ {error_msg}\n"))
                else:
                    # For cancel scope errors, try to get a result if possible
                    logger.warning("Cancel scope error when calling tool %s (ignoring): %s", tool_call.tool_name, e)
                    current_reasoning.append(
                        ObservationReasoningStep(observation=f"Tool {tool_call.tool_name} completed but encountered cleanup error")
                    )

        # Save new state in context
        ctx.data["sources"] = sources
        ctx.data["current_reasoning"] = current_reasoning

        # Prep the next iteration
        return PrepEvent()


# Test the workflow
if __name__ == "__main__":
    from llama_index.core.tools import FunctionTool

    def add(x: int, y: int) -> int:
        """Useful function to add two numbers."""
        return x + y

    def multiply(x: int, y: int) -> int:
        """Useful function to multiply two numbers."""
        return x * y

    tools = [
        FunctionTool.from_defaults(add),
        FunctionTool.from_defaults(multiply),
    ]

    agent = ReActAgent(tools=tools)
    print("ReAct agent workflow created successfully!")
