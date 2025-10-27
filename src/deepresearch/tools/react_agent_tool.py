"""Tool for creating ReAct agents with different configurations."""

from llama_index.core.tools import FunctionTool

from deepresearch.utils import get_llm
from deepresearch.workflows.react_agent_workflow import ReActAgent

# Global registry to store created sub-agents
SUB_AGENT_REGISTRY: dict[str, ReActAgent] = {}


async def run_sub_agent(agent_name: str, question: str) -> str:
    """Run a sub-agent that was previously created.

    Args:
        agent_name: Name of the sub-agent to run
        question: Question to ask the sub-agent

    Returns:
        Response from the sub-agent
    """
    try:
        if agent_name not in SUB_AGENT_REGISTRY:
            return f"Error: Sub-agent '{agent_name}' not found. Available agents: {list(SUB_AGENT_REGISTRY.keys())}"

        agent = SUB_AGENT_REGISTRY[agent_name]

        # Run the sub-agent
        handler = agent.run(input=question)

        # Collect the response
        result = await handler
        response = result.get("response", "No response generated")

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

        # Create specialized context based on domain
        specializations = {
            "climate": "You are a climate science research assistant specializing in climate modeling, environmental data analysis, and sustainability research.",
            "genomics": "You are a genomics research assistant specializing in DNA analysis, genome assembly, and bioinformatics.",
            "materials": "You are a materials science research assistant specializing in material discovery, property prediction, and nanotechnology.",
            "physics": "You are a physics research assistant specializing in theoretical physics, quantum mechanics, and computational physics.",
            "chemistry": "You are a chemistry research assistant specializing in molecular analysis, chemical reactions, and computational chemistry.",
            "ai": "You are an AI research assistant specializing in machine learning, neural networks, and artificial intelligence research.",
        }

        base_context = specializations.get(
            specialization.lower(),
            f"You are a research assistant specializing in {specialization}.",
        )

        # Add research tools
        from deepresearch.tools.vector_search import query_papers_with_llm
        from deepresearch.tools.web_search import search_web

        async def search_papers(query: str, top_k: int = 3) -> str:
            """Search scientific papers for relevant information."""
            try:
                return await query_papers_with_llm(query, top_k=top_k)
            except Exception as e:
                return f"Error searching papers: {str(e)}"

        async def search_web_research(query: str, max_results: int = 3) -> str:
            """Search the web for relevant information."""
            try:
                return await search_web(query, max_results=max_results)
            except Exception as e:
                return f"Error searching web: {str(e)}"

        research_tools = [
            FunctionTool.from_defaults(search_papers),
            FunctionTool.from_defaults(search_web_research),
        ]

        all_tools = default_tools + research_tools

        # Create the agent
        agent = ReActAgent(tools=all_tools, timeout=timeout, extra_context=base_context)

        # Store in registry with specialization name
        agent_name = f"{specialization}_specialist"
        SUB_AGENT_REGISTRY[agent_name] = agent

        tool_names = [tool.metadata.get_name() for tool in all_tools]
        return f"Specialized ReAct Agent '{agent_name}' created and stored successfully with tools: {tool_names}. Use 'run_sub_agent' to execute this agent."

    except Exception as e:
        return f"Error creating specialized ReAct agent: {str(e)}"
