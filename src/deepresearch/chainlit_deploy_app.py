#!/usr/bin/env python3
"""Simple Chainlit app that connects to llama-deploy."""

import asyncio
import json
import os
import sys

import httpx

# Set environment variables to avoid telemetry
os.environ["CHAINLIT_NO_TELEMETRY"] = "1"

try:
    import chainlit as cl
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configuration
API_BASE_URL = "http://host.docker.internal:4501"
DEPLOYMENT_NAME = "DeepResearchDeployment"


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize the chat session."""
    try:
        # Create session using the correct API endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/deployments/{DEPLOYMENT_NAME}/sessions/create",
                json={"session_name": "Deep Research Session"},
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                session_data = response.json()
                session_id = session_data.get("session_id")
                cl.user_session.set("session_id", session_id)
                cl.user_session.set("history", [])
                await cl.Message(
                    content="🔬 **Deep Research Assistant**\n\n"
                    "I can help you conduct comprehensive research on any topic using AI agents and scientific literature analysis.\n\n"
                    "What would you like to research today?"
                ).send()
            else:
                await cl.Message(
                    f"❌ Error creating session: {response.status_code} - {response.text}"
                ).send()
    except Exception as e:
        await cl.Message(f"❌ Error initializing session: {e}").send()


@cl.on_message
async def on_chat_message(message: cl.Message) -> None:
    """Handle chat messages and run research workflow."""
    try:
        session_id = cl.user_session.get("session_id")
        history = cl.user_session.get("history")

        if not session_id:
            await cl.Message("❌ No session available. Please refresh the page.").send()
            return

        # Show initial loading message
        msg = cl.Message(
            content="🚀 **Starting Deep Research Workflow**\n\nInitializing research session...",
            author="Deep Research Assistant",
        )
        await msg.send()

        # Run the workflow using the correct API endpoint
        async with httpx.AsyncClient() as client:
            # Create a task
            task_response = await client.post(
                f"{API_BASE_URL}/deployments/{DEPLOYMENT_NAME}/tasks/create",
                json={
                    "session_id": session_id,
                    "service_id": "deep_research_workflow",
                    "input": json.dumps(
                        {"user_input": message.content, "chat_history": history}
                    ),
                },
                headers={"Content-Type": "application/json"},
            )

            if task_response.status_code == 200:
                task_data = task_response.json()
                task_id = task_data.get("task_id")

                # Update with task creation success
                msg.content = "✅ **Task Created Successfully**\n\n🔍 **Deep Research Process Started**\n\n**Step 1:** Setting up research topic and initializing workflow..."
                await msg.update()
                await asyncio.sleep(3)

                # Show step 2 immediately
                msg.content = "🔍 **Step 2:** Generating comprehensive research questions...\n\n⏳ This may take 30-60 seconds..."
                await msg.update()
                await asyncio.sleep(3)

                # Wait for the task to complete and get results
                # Add retry logic for long-running tasks
                max_retries = 30  # 5 minutes with 10-second intervals
                retry_count = 0
                result_response = None

                while retry_count < max_retries:
                    result_response = await client.get(
                        f"{API_BASE_URL}/deployments/{DEPLOYMENT_NAME}/tasks/{task_id}/results?session_id={session_id}",
                        headers={"Content-Type": "application/json"},
                    )

                    if result_response.status_code == 200:
                        result_data = result_response.json()
                        result = result_data.get("result")
                        if result:  # Task completed
                            break

                    # Update progress message with simple steps
                    step = retry_count + 1
                    if step <= 5:
                        progress_msg = f"Generating research question... ({step}/30)"
                    elif step <= 15:
                        progress_msg = f"Researching topic... ({step}/30)"
                    elif step <= 25:
                        progress_msg = f"Analyzing findings... ({step}/30)"
                    else:
                        progress_msg = f"Creating report... ({step}/30)"

                    # Update the main message with progress
                    msg.content = f"**Deep Research in Progress**\n\n{progress_msg}\n\nPlease wait while we research your topic..."
                    await msg.update()

                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(10)  # Wait 10 seconds before retry

                if result_response and result_response.status_code == 200:
                    result_data = result_response.json()
                    result = result_data.get("result", "No result returned")

                    # Update the message with the clean result
                    msg.content = result
                    await msg.update()

                    # Update history
                    cl.user_session.set(
                        "history", [{"role": "assistant", "content": result}]
                    )
                else:
                    msg.content = f"❌ Error getting results: {result_response.status_code if result_response else 'No response'} - {result_response.text if result_response else 'Task may still be running'}"
                    await msg.update()
            else:
                msg.content = f"❌ Error creating task: {task_response.status_code} - {task_response.text}"
                await msg.update()

    except Exception as e:
        await cl.Message(f"❌ Error: {e}").send()


if __name__ == "__main__":
    print("Starting Chainlit app...")
