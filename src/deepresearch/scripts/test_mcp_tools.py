"""Test script to verify MCP tools are loading correctly."""

import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Set environment variables
os.environ["CHAINLIT_NO_TELEMETRY"] = "1"
# MCP_SERVER_URL will be auto-detected based on Docker environment


async def test_mcp_tools() -> bool | None:
    """Test loading MCP tools."""
    logger.info("Testing MCP tools loading...")
    from deepresearch.tools.mcp_tools import DEFAULT_MCP_URL
    logger.info("Using MCP_SERVER_URL: %s", os.environ.get("MCP_SERVER_URL", DEFAULT_MCP_URL))

    try:
        from deepresearch.tools.mcp_tools import load_mcp_tools

        logger.info("✓ Successfully imported load_mcp_tools")
        tools = await load_mcp_tools()

        if tools:
            logger.info(f"✓ Successfully loaded {len(tools)} MCP tools:")
            for i, tool in enumerate(tools, 1):
                tool_name = tool.metadata.get_name()
                tool_desc = tool.metadata.description or "No description"
                logger.info(f"  {i}. {tool_name}: {tool_desc}")
            return True
        else:
            logger.warning("⚠ No MCP tools loaded (empty list)")
            return False
    except ImportError as e:
        logger.error(f"✗ Failed to import MCP tools module: {e}")
        logger.info("Make sure llama-index-tools-mcp is installed")
        return False
    except Exception as e:
        logger.error(f"✗ Error loading MCP tools: {e}")
        logger.info("Make sure the MCP server is running at %s", os.environ.get("MCP_SERVER_URL"))
        return False


async def test_create_specialized_agent() -> bool | None:
    """Test creating a specialized agent with MCP tools."""
    logger.info("\nTesting specialized agent creation with MCP tools...")

    try:
        from deepresearch.tools.react_agent_tool import create_specialized_react_agent

        logger.info("✓ Successfully imported create_specialized_react_agent")

        # Create a test agent
        result = await create_specialized_react_agent("test_domain")
        logger.info(f"Result: {result}")

        # Check if MCP tools are mentioned in the result
        if "MCP" in result or "mcp" in result.lower():
            logger.info("✓ MCP tools appear to be loaded")
        else:
            logger.info("⚠ MCP tools not mentioned in agent creation response")

        return True
    except Exception as e:
        logger.error(f"✗ Error creating specialized agent: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main() -> None:
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("MCP Tools Test")
    logger.info("=" * 60)

    # First test: Can we load MCP tools directly?
    mcp_loaded = await test_mcp_tools()

    if not mcp_loaded:
        logger.error("\n⚠ MCP tools not loaded. Cannot proceed with agent test.")
        logger.info("\nTroubleshooting:")
        logger.info("1. Make sure the MCP server is running:")
        logger.info("   make mcp-server-http")
        logger.info("2. Check MCP_SERVER_URL environment variable")
        logger.info("3. Verify llama-index-tools-mcp is installed")
        sys.exit(1)

    # Second test: Can we create an agent with MCP tools?
    await test_create_specialized_agent()

    logger.info("\n" + "=" * 60)
    logger.info("Test complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

