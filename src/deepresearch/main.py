"""CLI entry point for deep research agent."""

import asyncio
import logging
import sys

from dotenv import load_dotenv

from deepresearch.workflows.research_workflow import DeepResearchWorkflow

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run deep research workflow."""
    if len(sys.argv) < 2:  # noqa: PLR2004
        logger.error("Usage: uv run python -m deepresearch.main 'research topic'")
        sys.exit(1)

    research_topic = sys.argv[1]
    logger.info("Starting deep research workflow for topic: %s", research_topic)

    # Create workflow
    workflow = DeepResearchWorkflow()

    # Run workflow with topic as input
    result = await workflow.run(topic=research_topic)
    
    logger.info("Deep research workflow completed!")
    logger.info("Final result: %s", result)


if __name__ == "__main__":
    asyncio.run(main())
