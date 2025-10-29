"""Chainlit app for ReAct agent research workflow."""

import os
import sys

# Set environment variables to avoid telemetry
os.environ["CHAINLIT_NO_TELEMETRY"] = "1"

try:
    import chainlit as cl
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

from llama_index.core.tools import FunctionTool

from deepresearch.tools.react_agent_tool import (
    SUB_AGENT_REGISTRY,
    create_react_agent,
    create_specialized_react_agent,
    list_sub_agents,
    run_sub_agent,
)
from deepresearch.workflows.react_agent_workflow import ReActAgent

# from deepresearch.workflows.react_agent_workflow_minimal import ReActAgent


async def _execute_sub_agent_with_logging(agent_name: str, question: str) -> None:
    """Execute a sub-agent and stream its execution to Chainlit.

    This is called from the main event handler when a sub-agent tool call is detected.
    """
    try:
        if agent_name not in SUB_AGENT_REGISTRY:
            await cl.Message(
                content=f"❌ Sub-agent '{agent_name}' not found.",
                author="Sub-Agent Executor",
            ).send()
            return

        agent = SUB_AGENT_REGISTRY[agent_name]

        # Send header message for sub-agent execution
        await cl.Message(
            content=f"## 🔄 Executing Sub-Agent: **{agent_name}**\n\n**Question:** {question}",
            author="Sub-Agent Executor",
        ).send()

        # Run the sub-agent and stream its events
        handler = agent.run(input=question)

        # Stream sub-agent events as persistent messages
        async for event in handler.stream_events():
            if hasattr(event, "reasoning") and event.reasoning:
                reasoning_text = event.reasoning.strip()
                if reasoning_text:
                    # Send each reasoning step as a separate message with sub-agent context
                    await cl.Message(
                        content=f"**[{agent_name}]** {reasoning_text}",
                        author="Sub-Agent Executor",
                    ).send()

        # Get final result
        result = await handler
        response = result.get("response", "No response generated")
        sources = result.get("sources", [])

        # Send final response with tool execution summary
        final_content = (
            f"## ✅ Sub-Agent '{agent_name}' Complete\n\n**Response:** {response}"
        )

        if sources:
            final_content += f"\n\n**Tools Executed:** {len(sources)}\n"
            for source in sources:
                tool_name = getattr(source, "tool_name", "Unknown")
                final_content += f"  - `{tool_name}`\n"

        await cl.Message(content=final_content, author="Sub-Agent Executor").send()

    except Exception as e:
        await cl.Message(
            content=f"❌ **Error running sub-agent '{agent_name}':** {str(e)}",
            author="Sub-Agent Executor",
        ).send()


async def run_sub_agent_with_logging(agent_name: str, question: str) -> str:
    """Run a sub-agent - logging will be handled by the main event stream handler.

    This function just executes the sub-agent normally. The Chainlit handler
    will intercept tool calls and provide logging.
    """
    # Just use the regular run_sub_agent - logging will be handled externally
    return await run_sub_agent(agent_name, question)


# Create orchestration tools (focus on delegation, not direct research)
def get_orchestration_tools():
    """Get the orchestration tools for the ReAct agent."""
    return [
        FunctionTool.from_defaults(create_react_agent),
        FunctionTool.from_defaults(create_specialized_react_agent),
        FunctionTool.from_defaults(
            run_sub_agent_with_logging
        ),  # Use Chainlit-aware wrapper
        FunctionTool.from_defaults(list_sub_agents),
    ]


# Extra context for the agent
EXTRA_CONTEXT = """
You are a research orchestrator that delegates research tasks to specialized agents. Your role is to:

1. Analyze research questions to determine what expertise is needed
2. Create specialized ReAct agents for specific domains
3. Delegate research tasks to sub-agents rather than doing research directly
4. Coordinate multiple sub-agents for complex multi-domain questions
5. List and manage available sub-agents

RESEARCH STRATEGY - Think strategically about multi-step research:
- After getting a sub-agent response, analyze if additional research is needed
- Consider complementary domains (e.g., genomics → materials science for delivery methods)
- Look for gaps in the research that require different expertise
- Only provide a final answer when you have comprehensive coverage of the topic

EXAMPLES OF MULTI-STEP RESEARCH:
- Question about climate change 
    - Create climate specialist 
    - Get climate data 
    - Create another specialist to get more data 
    - Repeat until you have comprehensive coverage of the topic
    - Synthesize comprehensive answer

You should NOT do research directly. Instead, create appropriate sub-agents and delegate the research to them.
For each research question, determine the domain expertise needed and create/use specialized agents accordingly.

COMPREHENSIVE ANSWER STRATEGY:
- After you have comprehensive coverage of the topic, create a specialized agent to synthesize a comprehensive answer
- The answer should be a detailed, coherent response that addresses the question in a thorough manner
- Sources should be cited at the bottom of the comprehensive answer

IMPORTANT - When providing your final answer after receiving sub-agent responses:
- Use this exact format when you're ready to provide the final answer:
  Thought: I have gathered enough information from the sub-agents to provide a comprehensive answer.
  Answer: [Your comprehensive final answer here]
- Do NOT continue using tools once you have received sufficient information from sub-agents
- Make sure to include "Answer:" followed by your response when completing the task
"""


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize the chat session."""
    await cl.Message(
        content="🤖 **ReAct Research Agent**\n\n"
        "I'm an intelligent research agent that can answer complex questions by reasoning about them step-by-step.\n\n"
        "I can delegate tasks to specialized sub-agents and coordinate multi-step research across different domains.\n\n"
        "What would you like to know?"
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming messages."""
    user_input = message.content.strip()

    if not user_input:
        await cl.Message(content="Please provide a question to research.").send()
        return

    try:
        # Create orchestration tools
        orchestration_tools = get_orchestration_tools()

        # Create agent
        agent = ReActAgent(
            tools=orchestration_tools, timeout=120, extra_context=EXTRA_CONTEXT
        )

        # Send agent initialized message
        await cl.Message(
            content=f"🔍 **Question:** {user_input}\n\n✅ ReAct agent initialized\n🧠 Starting reasoning process...",
            author="ReAct Agent",
        ).send()

        # Run workflow
        handler = agent.run(input=user_input)

        # Stream events and create persistent messages for each reasoning step
        pending_sub_agent_execution = None

        async for event in handler.stream_events():
            if hasattr(event, "delta") and event.delta:
                # Raw LLM response - we can skip or show in a summary message
                pass
            elif hasattr(event, "reasoning") and event.reasoning:
                reasoning_text = event.reasoning.strip()
                if not reasoning_text:
                    continue

                # Check if this is a tool execution event for run_sub_agent_with_logging
                if "🔧 Executing tool: run_sub_agent_with_logging" in reasoning_text:
                    # Try to extract arguments from this message or the next one
                    pending_sub_agent_execution = reasoning_text

                # Check if this has tool arguments (could be in same or next message)
                if pending_sub_agent_execution:
                    full_text = pending_sub_agent_execution
                    if (
                        "📝 With arguments:" in reasoning_text
                        and pending_sub_agent_execution != reasoning_text
                    ):
                        full_text += "\n" + reasoning_text
                    elif (
                        "📝 With arguments:" not in full_text
                        and "📝 With arguments:" in reasoning_text
                    ):
                        full_text = reasoning_text

                    if "📝 With arguments:" in full_text:
                        # Parse the arguments to extract agent_name and question
                        try:
                            import ast

                            # Extract the arguments string (could be in dict format)
                            args_start = full_text.find("{")
                            args_end = full_text.rfind("}") + 1
                            if args_start >= 0 and args_end > args_start:
                                args_str = full_text[args_start:args_end]
                                args_dict = ast.literal_eval(args_str)
                                agent_name = args_dict.get("agent_name")
                                question = args_dict.get("question")

                                if agent_name and question:
                                    # Run sub-agent with logging (don't display the tool execution message)
                                    await _execute_sub_agent_with_logging(
                                        agent_name, question
                                    )
                                    pending_sub_agent_execution = None
                                    continue
                        except Exception:
                            # If parsing fails, just continue normally
                            pass
                        pending_sub_agent_execution = None

                # Skip displaying the tool execution message for run_sub_agent_with_logging
                # since we handle it specially above
                if (
                    "🔧 Executing tool: run_sub_agent_with_logging" in reasoning_text
                    or (
                        pending_sub_agent_execution
                        and "📝 With arguments:" in reasoning_text
                    )
                ):
                    continue

                # Send each reasoning step as a separate persistent message
                # This ensures all intermediate steps are preserved in chat history
                await cl.Message(content=reasoning_text, author="ReAct Agent").send()

        # Get final result
        result = await handler
        final_response = result.get("response", "No response received")
        reasoning_steps = result.get("reasoning", [])
        sources = result.get("sources", [])

        # Send final response as a separate persistent message
        # Show reasoning process first, then the final answer
        final_content = ""

        if reasoning_steps:
            final_content += "## 🧠 Reasoning Process\n\n"

            # Filter to only show meaningful reasoning steps (those with thought or action)
            meaningful_steps = []
            for step in reasoning_steps:
                # Include steps that have thoughts (ActionReasoningStep) or are responses (ResponseReasoningStep)
                if (hasattr(step, "thought") and step.thought) or (
                    hasattr(step, "response")
                    and step.response
                    and hasattr(step, "is_done")
                    and step.is_done
                ):
                    meaningful_steps.append(step)

            # Number only the meaningful steps sequentially
            for i, step in enumerate(meaningful_steps, 1):
                if hasattr(step, "thought") and step.thought:
                    final_content += f"\n**Step {i}:** {step.thought}\n"
                if hasattr(step, "action") and step.action:
                    final_content += f"**Action:** {step.action}\n"
                    if hasattr(step, "action_input"):
                        final_content += f"**Input:** {step.action_input}\n"
                if hasattr(step, "observation") and step.observation:
                    # Show full observation if short, truncate if long
                    obs = str(step.observation)
                    if len(obs) > 300:
                        obs = obs[:300] + "..."
                    final_content += f"**Observation:** {obs}\n"
                if hasattr(step, "response") and step.response:
                    final_content += f"**Response:** {step.response}\n"

            final_content += "\n\n---\n\n"

        final_content += f"## ✅ Final Answer\n\n{final_response}"

        await cl.Message(content=final_content, author="ReAct Agent").send()

    except Exception as e:
        await cl.Message(
            content=f"❌ **Error during research:** {str(e)}\n\n"
            "Please try again with a different question."
        ).send()


if __name__ == "__main__":
    # This allows the script to be run directly
    pass
