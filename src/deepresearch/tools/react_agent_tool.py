"""Tool for creating ReAct agents with different configurations."""

import logging

from llama_index.core.tools import FunctionTool

from deepresearch.utils import get_llm
from deepresearch.workflows.react_agent_workflow import ReActAgent

logger = logging.getLogger(__name__)

# Global registry to store created sub-agents
SUB_AGENT_REGISTRY: dict[str, ReActAgent] = {}


async def run_sub_agent(
    agent_name: str, question: str, capture_execution: bool = False
) -> str:
    """Run a sub-agent that was previously created.

    Args:
        agent_name: Name of the sub-agent to run
        question: Question to ask the sub-agent
        capture_execution: If True, include execution details (tools used, reasoning)

    Returns:
        Response from the sub-agent, optionally including execution details
    """
    try:
        # Try direct match, then common variations
        target_name = agent_name
        agent = SUB_AGENT_REGISTRY.get(target_name)
        if not agent:
            # Try with _specialist suffix
            alt_name = f"{agent_name}_specialist"
            agent = SUB_AGENT_REGISTRY.get(alt_name)
            if agent:
                target_name = alt_name

        # If still missing, try to auto-create a specialized agent with this name
        if not agent:
            try:
                _ = await create_specialized_react_agent(agent_name)
                # Newly created agents use the _specialist suffix
                target_name = f"{agent_name}_specialist"
                agent = SUB_AGENT_REGISTRY.get(target_name)
            except Exception:
                agent = None

        if not agent:
            return (
                f"Error: Sub-agent '{agent_name}' not found and could not be auto-created. "
                f"Available agents: {list(SUB_AGENT_REGISTRY.keys())}"
            )

        # Use resolved agent
        agent = agent

        # Run the sub-agent
        handler = agent.run(input=question)

        # If capture_execution is True, we'll include details in the response
        # Note: We can't stream events here and then await handler - we need to choose one
        # For Chainlit integration, we'll create a wrapper that handles streaming separately

        # Collect the response
        result = await handler
        response = result.get("response", "No response generated")
        reasoning_steps = result.get("reasoning", [])
        sources = result.get("sources", [])

        # Format response with execution details if requested
        if capture_execution and (reasoning_steps or sources):
            details_text = f"\n\n--- Sub-agent '{agent_name}' Execution Log ---\n"

            if reasoning_steps:
                details_text += "\n**Reasoning Steps:**\n"
                for i, step in enumerate(reasoning_steps, 1):
                    if hasattr(step, "thought") and step.thought:
                        details_text += f"\n{i}. Thought: {step.thought}\n"
                    if hasattr(step, "action") and step.action:
                        details_text += f"   Action: {step.action}\n"
                        if hasattr(step, "action_input"):
                            details_text += f"   Input: {step.action_input}\n"
                    if hasattr(step, "observation") and step.observation:
                        # Truncate long observations
                        obs = str(step.observation)
                        OBS_PREVIEW_LIMIT = 300
                        if len(obs) > OBS_PREVIEW_LIMIT:
                            obs = obs[:OBS_PREVIEW_LIMIT] + "..."
                        details_text += f"   Observation: {obs}\n"

            if sources:
                details_text += f"\n**Tools Executed:** {len(sources)}\n"
                for source in sources:
                    tool_name = getattr(source, "tool_name", "Unknown")
                    details_text += f"  - {tool_name}\n"

            return f"Sub-agent '{agent_name}' response: {response}{details_text}"

        return f"Sub-agent '{agent_name}' response: {response}"

    except Exception as e:
        return f"Error running sub-agent '{agent_name}': {str(e)}"


async def list_sub_agents() -> str:
    """List all available sub-agents.

    Returns:
        List of available sub-agents
    """
    if not SUB_AGENT_REGISTRY:
        return "No sub-agents have been created yet."

    agent_list = []
    for name, agent in SUB_AGENT_REGISTRY.items():
        tool_names = [tool.metadata.get_name() for tool in agent.tools]
        agent_list.append(f"• {name}: {tool_names}")

    return "Available sub-agents:\n" + "\n".join(agent_list)


async def create_react_agent(
    tools: list | None = None,
    llm: object | None = None,
    timeout: int = 300,
    extra_context: str | None = None,
    agent_name: str | None = None,
) -> str:
    """Create a ReAct agent with specified configuration.

    Args:
        tools: List of tools to give the agent access to
        llm: Language model to use (defaults to Cerebras LLM)
        timeout: Timeout in seconds for the agent
        extra_context: Additional context/instructions for the agent
        agent_name: Optional name for the agent (for identification)

    Returns:
        String confirmation of agent creation with configuration details
    """
    try:
        # Use default LLM if none provided
        if llm is None:
            llm = get_llm()

        # Use default tools if none provided
        if tools is None:
            tools = []

        # Create the ReAct agent
        agent = ReActAgent(
            tools=tools, llm=llm, timeout=timeout, extra_context=extra_context or ""
        )

        # Store in registry
        final_name = agent_name or f"agent_{len(SUB_AGENT_REGISTRY) + 1}"
        SUB_AGENT_REGISTRY[final_name] = agent

        # Generate confirmation message
        tool_names = [tool.metadata.get_name() for tool in tools] if tools else []
        agent_info = f"ReAct Agent '{final_name}' created and stored successfully"

        config_details = [
            f"• Tools: {tool_names if tool_names else 'None'}",
            f"• Timeout: {timeout} seconds",
            f"• LLM: {llm.__class__.__name__}",
            f"• Extra Context: {'Yes' if extra_context else 'No'}",
            "• Use 'run_sub_agent' to execute this agent",
        ]

        return f"{agent_info}:\n" + "\n".join(config_details)

    except Exception as e:
        return f"Error creating ReAct agent: {str(e)}"


async def create_specialized_react_agent(
    specialization: str, tools: list | None = None, timeout: int = 300
) -> str:
    """Create a ReAct agent specialized for a specific domain.

    Args:
        specialization: The domain/field the agent should specialize in
        tools: Additional tools to include
        timeout: Timeout in seconds for the agent

    Returns:
        String confirmation of specialized agent creation
    """
    try:
        # Default tools (ensure they are FunctionTool objects)
        default_tools = []
        if tools:
            for tool in tools:
                if isinstance(tool, FunctionTool):
                    default_tools.append(tool)
                else:
                    # If it's a string, try to find the tool by name
                    # For now, skip non-FunctionTool objects
                    pass

        base_context = f"""You are an assistant that specializes in {specialization}.
        Your goal is to provide a thorough and detailed answer to the question.
        STRICT TOOL POLICY:
        - FIRST use `search_arxiv` to retrieve relevant arXiv IDs.
        - THEN use `extract_info` for each ID to get details (title, authors, summary, pdf_url, published).
        - DO NOT use other tools unless explicitly provided and relevant.
        - If you cannot find relevant items with these tools, respond EXACTLY with: NO_RESULT

        Answer composition:
        - Summarize what you found based on extract_info outputs.
        - Cite sources with PDF URLs when available.
        - If nothing relevant was found, return NO_RESULT.
        """

        # Add research tools (disabled for this specialist to avoid confusion)
        # from deepresearch.tools.vector_search import query_papers_with_llm
        from deepresearch.tools.web_search import search_web

        # Disabled: vector DB paper search for this specialist
        # async def search_papers(query: str, top_k: int = 3) -> str:
        #     """Search scientific papers for relevant information."""
        #     try:
        #         return await query_papers_with_llm(query, top_k=top_k)
        #     except Exception as e:
        #         return f"Error searching papers: {str(e)}"

        async def search_web_research(query: str, max_results: int = 3) -> str:
            """Search the web for relevant information."""
            try:
                return await search_web(query, max_results=max_results)
            except Exception as e:
                return f"Error searching web: {str(e)}"

        research_tools = [
            # FunctionTool.from_defaults(search_papers),
            # FunctionTool.from_defaults(search_web_research),
        ]

        # Load MCP tools if available (may not work with chainlit due to dependency conflicts)
        mcp_tools = []
        try:
            from deepresearch.tools.mcp_tools import load_mcp_tools

            logger.info("Attempting to load MCP tools from server...")
            loaded_mcp_tools = await load_mcp_tools()
            # Keep only the arXiv pipeline tools in desired order
            name_to_tool = {t.metadata.get_name(): t for t in loaded_mcp_tools}
            mcp_tools = [name_to_tool[n] for n in ["search_arxiv", "extract_info"] if n in name_to_tool]
            if mcp_tools:
                logger.info("✓ Loaded MCP tools: %s", [t.metadata.get_name() for t in mcp_tools])
            else:
                logger.warning("⚠ MCP tools loading returned empty list")
        except ImportError as e:
            logger.warning("Could not import MCP tools module (llama-index-tools-mcp may not be installed): %s", e)
        except Exception as e:
            logger.warning("Could not load MCP tools from server (make sure MCP server is running): %s", e)

        all_tools = default_tools + research_tools + mcp_tools

        # Create the agent
        agent = ReActAgent(tools=all_tools, timeout=timeout, extra_context=base_context)

        # Store in registry with specialization name
        agent_name = f"{specialization}_specialist"
        SUB_AGENT_REGISTRY[agent_name] = agent

        tool_names = [tool.metadata.get_name() for tool in all_tools]
        return f"Specialized ReAct Agent '{agent_name}' created and stored successfully with tools: {tool_names}. Use 'run_sub_agent' to execute this agent."

    except Exception as e:
        return f"Error creating specialized ReAct agent: {str(e)}"
