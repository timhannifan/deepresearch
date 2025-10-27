#!/usr/bin/env python3
"""Direct Chainlit app that uses the workflow directly."""

import asyncio
import os
import sys

# Set environment variables to avoid telemetry
os.environ["CHAINLIT_NO_TELEMETRY"] = "1"

try:
    import chainlit as cl
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

from deepresearch.workflows.research_workflow import DeepResearchWorkflow


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize the chat session."""
    await cl.Message(
        content="🔬 **Deep Research Assistant**\n\n"
        "I can help you conduct comprehensive research on any topic using AI agents and scientific literature analysis.\n\n"
        "What would you like to research today?"
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming messages."""
    try:
        # Get the user's research topic
        research_topic = message.content.strip()
        
        if not research_topic:
            await cl.Message(content="Please provide a research topic to investigate.").send()
            return

        # Show initial message
        msg = cl.Message(
            content=f"🔍 **Starting deep research on:** {research_topic}\n\n"
            "Initializing research workflow...",
            author="Deep Research Assistant"
        )
        await msg.send()

        # Create and run the workflow
        workflow = DeepResearchWorkflow()
        
        # Update message with progress
        msg.content = f"🔍 **Researching:** {research_topic}\n\n"
        msg.content += "✅ Research workflow initialized\n"
        msg.content += "🔬 Generating research questions...\n"
        await msg.update()

        # Run the workflow
        result = await workflow.run(topic=research_topic)
        
        # Update with completion
        msg.content = f"🔍 **Research completed for:** {research_topic}\n\n"
        msg.content += "✅ Research workflow completed\n"
        msg.content += "📊 Generating final report...\n"
        await msg.update()

        # Send the final result
        await cl.Message(
            content=f"# Research Report\n\n{result}",
            author="Deep Research Assistant"
        ).send()

    except Exception as e:
        await cl.Message(
            content=f"❌ **Error during research:** {str(e)}\n\n"
            "Please try again with a different research topic."
        ).send()


if __name__ == "__main__":
    # This allows the script to be run directly
    pass
