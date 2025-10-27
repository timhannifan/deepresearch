"""Query documents from Qdrant vector database."""

import argparse
import logging

from deepresearch.utils import get_query_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def query_documents(query: str, top_k: int = 3) -> str:
    """Query documents from the existing Qdrant collection."""
    # Get query engine with Cerebras LLM
    query_engine = get_query_engine(similarity_top_k=top_k)

    # Query
    response = query_engine.query(query)
    logger.info(f"Query: {query}")
    logger.info(f"Answer: {response}")

    return response


def main() -> None:
    """Query documents with command line interface."""
    parser = argparse.ArgumentParser(description="Query documents from Qdrant database")
    parser.add_argument("query", help="Query to search for")
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of top results (default: 3)"
    )

    args = parser.parse_args()

    query_documents(args.query, args.top_k)


if __name__ == "__main__":
    main()
