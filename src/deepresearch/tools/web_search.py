"""Web search tool using Tavily API."""

import os

from tavily import TavilyClient


async def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for relevant information using Tavily API.

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        Formatted string with search results
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return (
            "Tavily API key not found. Please set TAVILY_API_KEY environment variable."
        )

    client = TavilyClient(api_key=api_key)

    try:
        response = client.search(query=query, max_results=max_results)

        if not response.get("results"):
            return "No relevant web results found."

        results = []
        for i, result in enumerate(response["results"], 1):
            results.append(f"Result {i}:")
            results.append(f"Title: {result.get('title', 'N/A')}")
            results.append(f"URL: {result.get('url', 'N/A')}")
            results.append(f"Content: {result.get('content', 'N/A')}")
            results.append("")

        return "\n".join(results)

    except Exception as e:
        return f"Error searching web: {str(e)}"
