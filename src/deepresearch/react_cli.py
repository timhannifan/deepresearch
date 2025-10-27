"""Research-enabled ReAct agent workflow."""

import asyncio
import logging
import sys

from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool

from deepresearch.tools.react_agent_tool import (
    create_react_agent,
    create_specialized_react_agent,
    list_sub_agents,
    run_sub_agent,
)
from deepresearch.tools.vector_search import query_papers_with_llm
from deepresearch.tools.web_search import search_web
from deepresearch.workflows.react_agent_workflow import ReActAgent

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def search_scientific_papers(query: str, top_k: int = 3) -> str:
    """Search scientific papers for relevant information.

    Args:
        query: Research query to search for
        top_k: Number of top results to return

    Returns:
        Formatted string with relevant paper excerpts
    """
    try:
        return await query_papers_with_llm(query, top_k=top_k)
    except Exception as e:
        return f"Error searching papers: {str(e)}"


async def search_web_research(query: str, max_results: int = 3) -> str:
    """Search the web for relevant information.

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        Formatted string with search results
    """
    try:
        return await search_web(query, max_results=max_results)
    except Exception as e:
        return f"Error searching web: {str(e)}"


async def main() -> None:
    """Run research-enabled ReAct agent workflow."""
    if len(sys.argv) < 2:  # noqa: PLR2004
        logger.error("Usage: uv run python -m deepresearch.react_cli 'your question'")
        sys.exit(1)

    user_input = sys.argv[1]
    logger.info("Starting research-enabled ReAct agent with input: %s", user_input)

    # Create orchestration tools (focus on delegation, not direct research)
    orchestration_tools = [
        FunctionTool.from_defaults(create_react_agent),
        FunctionTool.from_defaults(create_specialized_react_agent),
        FunctionTool.from_defaults(run_sub_agent),
        FunctionTool.from_defaults(list_sub_agents),
    ]

    # Create agent with orchestration context
    extra_context = """
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

    agent = ReActAgent(
        tools=orchestration_tools, timeout=120, extra_context=extra_context
    )

    # Run workflow
    handler = agent.run(input=user_input)

    # Stream events and display progress
    async for event in handler.stream_events():
        if hasattr(event, "delta") and event.delta:
            print(event.delta, end="", flush=True)
        elif hasattr(event, "reasoning") and event.reasoning:
            print(event.reasoning, end="", flush=True)
        else:
            logger.info("Event: %s", event)

    # Get final result
    result = await handler
    logger.info("\nResearch ReAct agent completed!")
    logger.info("Final result: %s", result.get("response", "No response"))


if __name__ == "__main__":
    asyncio.run(main())
