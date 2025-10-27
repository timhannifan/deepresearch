"""Shared utilities for Qdrant and embedding operations."""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.cerebras import Cerebras
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

if TYPE_CHECKING:
    from llama_index.core.query_engine import BaseQueryEngine

logger = logging.getLogger(__name__)


class QdrantConfig:
    """Configuration settings for Qdrant operations."""

    # Qdrant settings
    QDRANT_PATH = "data/qdrant_db"
    COLLECTION_NAME = "scientific_papers"

    # Embedding settings
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIM = 1536  # OpenAI embedding dimension

    # LLM settings
    CEREBRAS_MODEL = "llama-3.3-70b"

    # Sample data
    SAMPLE_PAPERS_DIR = "data/sample_papers"


config = QdrantConfig()


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client - use service in Docker, local file otherwise."""
    # Check if we're in Docker (service available) or running locally
    if Path("/.dockerenv").exists():
        # Running in Docker - use Qdrant service
        return QdrantClient(host="qdrant", port=6333)
    else:
        # Running locally - use local file storage
        return QdrantClient(path=config.QDRANT_PATH)


def create_collection(collection_name: str) -> None:
    """Create a Qdrant collection if it doesn't exist."""
    client = get_qdrant_client()

    if not client.collection_exists(collection_name):
        logger.info(f"Creating collection {collection_name}...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=config.EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )


def init_storage_context(
    collection_name: str, create_collection_if_missing: bool = True
) -> StorageContext:
    """Initialize storage context with Qdrant vector store."""
    # Get appropriate Qdrant client
    client = get_qdrant_client()

    if create_collection_if_missing:
        create_collection(collection_name)

    # Create vector store
    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        client=client,
    )

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return storage_context


def get_embed_model() -> OpenAIEmbedding:
    """Get OpenAI embedding model."""
    return OpenAIEmbedding(
        model=config.EMBEDDING_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_vector_index() -> VectorStoreIndex:
    """Get vector store index."""
    client = get_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
    )

    embed_model = get_embed_model()

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )


def get_llm() -> Cerebras:
    """Get Cerebras LLM configured directly."""
    return Cerebras(model=config.CEREBRAS_MODEL, api_key=os.environ["CEREBRAS_API_KEY"])


def get_query_engine(similarity_top_k: int = 3) -> "BaseQueryEngine":
    """Get query engine with Cerebras LLM."""
    index = get_vector_index()
    llm = get_llm()

    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        llm=llm,
    )
