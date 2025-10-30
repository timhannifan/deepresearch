"""Utilities for loading and using MCP tools in ReAct workflows.

NOTE: llama-index-tools-mcp conflicts with chainlit due to httpx version
requirements. This module will gracefully handle missing imports.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool

logger = logging.getLogger(__name__)

# Detect if we're running in Docker
def _get_default_mcp_url() -> str:
    """Get the default MCP server URL, adjusting for Docker if needed."""
    # Check if MCP_SERVER_URL is explicitly set
    if "MCP_SERVER_URL" in os.environ:
        return os.environ["MCP_SERVER_URL"]
    
    # Detect if we're in Docker (container environment)
    if Path("/.dockerenv").exists() or os.environ.get("DOCKER") == "1":
        # If MCP_SERVER_URL is set in docker-compose, use that
        # Otherwise, try host.docker.internal (works on Mac/Windows Docker Desktop)
        # or use service name if MCP server is also in Docker
        return os.environ.get("MCP_SERVER_URL", "http://host.docker.internal:10000/mcp")
    
    # Default for local development
    return "http://127.0.0.1:10000/mcp"


DEFAULT_MCP_URL = _get_default_mcp_url()


async def load_mcp_tools(mcp_url: str | None = None) -> list["BaseTool"]:
    """Load tools from an MCP server.

    Args:
        mcp_url: URL of the MCP server (default: http://127.0.0.1:10000/mcp)

    Returns:
        List of FunctionTools from the MCP server
    """
    try:
        from llama_index.tools.mcp import aget_tools_from_mcp_url

        url = mcp_url or DEFAULT_MCP_URL
        logger.info("Loading MCP tools from %s", url)
        tools = await aget_tools_from_mcp_url(url)
        logger.info("Loaded %d MCP tools", len(tools))
        return tools
    except ImportError as e:
        logger.warning("llama-index-tools-mcp not available: %s", e)
        return []
    except Exception as e:
        logger.error("Error loading MCP tools from %s: %s", mcp_url or DEFAULT_MCP_URL, e)
        return []


def load_mcp_tools_sync(mcp_url: str | None = None) -> list["BaseTool"]:
    """Load tools from an MCP server (synchronous version).

    Args:
        mcp_url: URL of the MCP server (default: http://127.0.0.1:10000/mcp)

    Returns:
        List of FunctionTools from the MCP server
    """
    try:
        from llama_index.tools.mcp import get_tools_from_mcp_url

        url = mcp_url or DEFAULT_MCP_URL
        logger.info("Loading MCP tools from %s", url)
        tools = get_tools_from_mcp_url(url)
        logger.info("Loaded %d MCP tools", len(tools))
        return tools
    except ImportError as e:
        logger.warning("llama-index-tools-mcp not available: %s", e)
        return []
    except Exception as e:
        logger.error("Error loading MCP tools from %s: %s", mcp_url or DEFAULT_MCP_URL, e)
        return []

