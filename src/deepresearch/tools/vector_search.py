"""Vector search tool using Qdrant for scientific literature retrieval."""

from dotenv import load_dotenv

from deepresearch.utils import get_query_engine, get_vector_index

# Load environment variables from .env file
load_dotenv()


async def search_papers(query: str, top_k: int = 5) -> str:
    """Search scientific papers for relevant information.

    Args:
        query: Research query to search for
        top_k: Number of top results to return

    Returns:
        Formatted string with relevant paper excerpts
    """
    index = get_vector_index()
    retriever = index.as_retriever(similarity_top_k=top_k)

    nodes = retriever.retrieve(query)

    if not nodes:
        return "No relevant papers found in the database."

    results = []
    for i, node in enumerate(nodes, 1):
        results.append(f"Result {i}:")
        results.append(f"Score: {node.score:.3f}")
        results.append(f"Content: {node.text}")
        results.append("")

    return "\n".join(results)


async def query_papers_with_llm(query: str, top_k: int = 5) -> str:
    """Query scientific papers using LLM for natural language responses.

    Args:
        query: Research query to search for
        top_k: Number of top results to use as context

    Returns:
        Natural language response from LLM using retrieved context
    """
    query_engine = get_query_engine(similarity_top_k=top_k)
    response = query_engine.query(query)
    return str(response)
