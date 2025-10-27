"""Load sample scientific papers into Qdrant vector database."""

import argparse
import logging
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from deepresearch.utils import (
    config,
    get_embed_model,
    get_qdrant_client,
    init_storage_context,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_collection_exists() -> bool:
    """Check if the collection already exists in Qdrant."""
    client = get_qdrant_client()
    return client.collection_exists(config.COLLECTION_NAME)


def main() -> None:
    """Load papers from data/sample_papers into Qdrant using shared utilities."""
    parser = argparse.ArgumentParser(
        description="Load scientific papers into Qdrant vector database"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing collection if it exists (default: False)",
    )

    args = parser.parse_args()

    # Check if collection exists
    if check_collection_exists():
        if args.overwrite:
            logger.info(
                "Collection '%s' exists. Overwriting...", config.COLLECTION_NAME
            )
        else:
            logger.info(
                "Collection '%s' already exists. Use --overwrite to replace it.",
                config.COLLECTION_NAME,
            )
            logger.info("Exiting without changes.")
            return

    # Setup paths
    papers_dir = Path(config.SAMPLE_PAPERS_DIR)

    if not papers_dir.exists():
        logger.info("Creating %s directory...", papers_dir)
        papers_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("Please add scientific papers (PDF or TXT) to %s", papers_dir)
        return

    # Check for papers
    paper_files = list(papers_dir.glob("*"))
    if not paper_files:
        logger.error("No files found in %s", papers_dir)
        logger.info("Please add scientific papers (PDF or TXT) to this directory")
        return

    logger.info("Found %d files in %s", len(paper_files), papers_dir)

    # Load documents
    logger.info("Loading documents...")
    documents = SimpleDirectoryReader(
        input_dir=str(papers_dir),
        recursive=True,
    ).load_data()

    logger.info("Loaded %d document chunks", len(documents))

    # Setup storage context
    logger.info("Connecting to Qdrant...")
    storage_context = init_storage_context(
        config.COLLECTION_NAME, create_collection_if_missing=True
    )

    # Get embedding model
    embed_model = get_embed_model()

    # Create index and load documents
    logger.info("Creating vector index and storing embeddings...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    logger.info("✓ Successfully loaded %d chunks into Qdrant", len(documents))
    logger.info("  Collection name: %s", config.COLLECTION_NAME)


if __name__ == "__main__":
    main()
