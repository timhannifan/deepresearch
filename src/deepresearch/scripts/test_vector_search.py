"""Test and demonstrate vector search functionality."""

import argparse
import asyncio
import logging
import sys

from deepresearch.tools.vector_search import search_papers

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def demo_queries() -> None:
    """Run demonstration queries to show vector search capabilities."""
    logger.info("=== Vector Search Demonstration ===")
    logger.info("This script demonstrates Qdrant + LlamaIndex vector search.")
    logger.info("Make sure you have loaded documents first with:")
    logger.info("  uv run python -m deepresearch.scripts.load_vector_db")

    demo_queries = [
        "What is climate modeling?",
        "What are the benefits of HPC?",
    ]

    for i, query in enumerate(demo_queries, 1):
        logger.info(f"Demo Query {i}: {query}")
        logger.info("-" * 60)

        try:
            results = await search_papers(query, top_k=3)
            logger.info(results)
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.info("Make sure Qdrant is running and documents are loaded.")

        logger.info("=" * 60)

    logger.info("=== Demo Complete! ===")
    logger.info("You can also run individual searches:")
    logger.info(
        "uv run python -m deepresearch.scripts.test_vector_search 'your query here'"
    )


async def main() -> None:
    """Test vector search with command line interface."""
    parser = argparse.ArgumentParser(
        description="Test and demonstrate vector search functionality"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Query to search for (if not provided, runs demo queries)",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of top results (default: 3)"
    )
    parser.add_argument("--demo", action="store_true", help="Run demonstration queries")

    args = parser.parse_args()

    # Run demo if no query provided or --demo flag used
    if not args.query or args.demo:
        await demo_queries()
        return

    # Run single query
    logger.info("Searching for: %s", args.query)
    try:
        results = await search_papers(args.query, top_k=args.top_k)
        logger.info(results)
    except Exception as e:
        logger.error("Search failed: %s", e)
        logger.info("Make sure Qdrant is running and documents are loaded.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
