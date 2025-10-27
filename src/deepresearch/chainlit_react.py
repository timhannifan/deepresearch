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
    create_react_agent,
    create_specialized_react_agent,
    list_sub_agents,
    run_sub_agent,
)
from deepresearch.workflows.react_agent_workflow import ReActAgent


# Create orchestration tools (focus on delegation, not direct research)
def get_orchestration_tools():
    """Get the orchestration tools for the ReAct agent."""
    return [
        FunctionTool.from_defaults(create_react_agent),
        FunctionTool.from_defaults(create_specialized_react_agent),
        FunctionTool.from_defaults(run_sub_agent),
        FunctionTool.from_defaults(list_sub_agents),
    ]


# Extra context for the agent
EXTRA_CONTEXT = """
You are a research orchestrator that delegates research tasks to specialized agents. Your role is to:

1. Analyze research questions to determine what expertise is needed
2. Create specialized ReAct agents for specific domains (climate, genomics, materials, etc.)
3. Delegate research tasks to sub-agents rather than doing research directly
4. Coordinate multiple sub-agents for complex multi-domain questions
5. List and manage available sub-agents

RESEARCH STRATEGY - Think strategically about multi-step research:
- After getting a sub-agent response, analyze if additional research is needed
- Consider complementary domains (e.g., genomics → materials science for delivery methods)
- Look for gaps in the research that require different expertise
- Only provide a final answer when you have comprehensive coverage of the topic

EXAMPLES OF MULTI-STEP RESEARCH:
- Question about CRISPR → Create genomics specialist → Get CRISPR info → Create materials specialist → Ask about delivery methods → Synthesize comprehensive answer
- Question about climate change → Create climate specialist → Get climate data → Create economics specialist → Ask about economic impacts → Synthesize comprehensive answer

You should NOT do research directly. Instead, create appropriate sub-agents and delegate the research to them.
For each research question, determine the domain expertise needed and create/use specialized agents accordingly.
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

    # Show initial message
    msg = cl.Message(
        content=f"🔍 **Thinking about:** {user_input}\n\n"
        "Initializing ReAct agent...",
        author="ReAct Agent"
    )
    await msg.send()

    try:
        # Create orchestration tools
        orchestration_tools = get_orchestration_tools()

        # Create agent
        agent = ReActAgent(
            tools=orchestration_tools, 
            timeout=120, 
            extra_context=EXTRA_CONTEXT
        )

        # Update with agent created
        msg.content = f"🔍 **Question:** {user_input}\n\n"
        msg.content += "✅ ReAct agent initialized\n"
        msg.content += "🧠 Starting reasoning process...\n"
        msg.content += "\n---\n"
        await msg.update()

        # Run workflow
        handler = agent.run(input=user_input)

        # Stream events and display progress
        full_response = ""
        async for event in handler.stream_events():
            if hasattr(event, "delta") and event.delta:
                full_response += event.delta
                # Update message with incremental content
                msg.content = f"🔍 **Question:** {user_input}\n\n"
                msg.content += "✅ ReAct agent initialized\n"
                msg.content += "🧠 Starting reasoning process...\n"
                msg.content += "\n---\n"
                msg.content += full_response
                await msg.update()
            elif hasattr(event, "reasoning") and event.reasoning:
                full_response += event.reasoning
                # Update message with reasoning
                msg.content = f"🔍 **Question:** {user_input}\n\n"
                msg.content += "✅ ReAct agent initialized\n"
                msg.content += "🧠 Starting reasoning process...\n"
                msg.content += "\n---\n"
                msg.content += full_response
                await msg.update()

        # Get final result
        result = await handler
        final_response = result.get("response", "No response received")
        
        # Update with final response
        msg.content = f"🔍 **Question:** {user_input}\n\n"
        msg.content += "✅ Research completed\n"
        msg.content += "\n---\n"
        msg.content += f"**Answer:**\n\n{final_response}"
        await msg.update()

    except Exception as e:
        await cl.Message(
            content=f"❌ **Error during research:** {str(e)}\n\n"
            "Please try again with a different question."
        ).send()


if __name__ == "__main__":
    # This allows the script to be run directly
    pass
