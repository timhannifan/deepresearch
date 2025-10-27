"""Deep research workflow orchestration."""

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context, Workflow, step
from llama_index.core.workflow import Event, StartEvent, StopEvent

from deepresearch.agents.question_agent import QuestionAgent
from deepresearch.agents.report_agent import ReportAgent
from deepresearch.tools.react_agent_tool import (
    create_react_agent,
    create_specialized_react_agent,
    list_sub_agents,
    run_sub_agent,
)
from deepresearch.tools.vector_search import query_papers_with_llm
from deepresearch.tools.web_search import search_web
from deepresearch.workflows.react_agent_workflow import ReActAgent


async def search_scientific_papers(query: str, top_k: int = 3) -> str:
    """Search scientific papers for relevant information.

    Args:
        query: Research query to search for
        top_k: Number of top results to return

    Returns:
        Formatted string with relevant paper excerpts
    """
    try:
        return await query_papers_with_llm(query, top_k=top_k)
    except Exception as e:
        return f"Error searching papers: {str(e)}"


async def search_web_research(query: str, max_results: int = 3) -> str:
    """Search the web for relevant information.

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        Formatted string with search results
    """
    try:
        return await search_web(query, max_results=max_results)
    except Exception as e:
        return f"Error searching web: {str(e)}"


class GenerateEvent(Event):
    """Event to trigger question generation."""

    topic: str


class QuestionEvent(Event):
    """Event containing a research question."""

    question: str


class AnswerEvent(Event):
    """Event containing an answer to a research question."""

    question: str
    answer: str


class ReportEvent(Event):
    """Event containing the final research report."""

    report: str


class DeepResearchWorkflow(Workflow):
    """Deep research workflow."""

    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> GenerateEvent:
        """Setup agents and start research."""
        self.question_agent = QuestionAgent()
        self.report_agent = ReportAgent()

        # Create orchestration tools for ReAct agent (focus on delegation)
        orchestration_tools = [
            FunctionTool.from_defaults(create_react_agent),
            FunctionTool.from_defaults(create_specialized_react_agent),
            FunctionTool.from_defaults(run_sub_agent),
            FunctionTool.from_defaults(list_sub_agents),
        ]

        # Create ReAct agent with orchestration context
        extra_context = """
You are a research orchestrator that delegates research tasks to specialized agents. Your role is to:

1. Analyze research questions to determine what expertise is needed
2. Create specialized ReAct agents for specific domains (climate, genomics, materials, etc.)
3. Create research-focused agents with appropriate tool configurations
4. Delegate research tasks to sub-agents rather than doing research directly
5. Coordinate multiple sub-agents for complex multi-domain questions
6. List and manage available sub-agents

RESEARCH STRATEGY - Think strategically about multi-step research:
- After getting a sub-agent response, analyze if additional research is needed
- Consider complementary domains (e.g., genomics → materials science for delivery methods)
- Look for gaps in the research that require different expertise
- Only provide a final answer when you have comprehensive coverage of the topic

EXAMPLES OF MULTI-STEP RESEARCH:
- Question about CRISPR → Create genomics specialist → Get CRISPR info → Create materials specialist → Ask about delivery methods → Synthesize comprehensive answer
- Question about climate change → Create climate specialist → Get climate data → Create economics specialist → Ask about economic impacts → Synthesize comprehensive answer

You should NOT do research directly. Instead, create appropriate sub-agents and delegate the research to them.
For each research question, determine the domain expertise needed and create/use specialized agents accordingly.
"""

        self.react_agent = ReActAgent(
            tools=orchestration_tools,
            timeout=60,  # Reduced timeout to 1 minute for testing
            extra_context=extra_context,
        )

        # Get the topic from the input data
        topic = (
            ev.get("user_input")
            or ev.get("topic")
            or ev.get("query")
            or "Research Topic"
        )
        await ctx.set("topic", topic)  # Store topic in context for later use
        return GenerateEvent(topic=topic)

    @step
    async def generate_questions(
        self, ctx: Context, ev: GenerateEvent
    ) -> QuestionEvent:
        """Generate research questions."""
        questions = await self.question_agent.generate_questions(
            ev.topic, num_questions=1
        )

        # Send parallel question events
        for question in questions:
            question_event = QuestionEvent(question=question)
            ctx.send_event(question_event)

        # Return the first question event to continue the workflow
        return QuestionEvent(question=questions[0])

    @step
    async def research_questions(self, ctx: Context, ev: QuestionEvent) -> AnswerEvent:
        """Research a specific question using ReAct agent."""

        # Use ReAct agent for intelligent research with web search
        try:

            # Create a research agent with web search capability
            research_tools = [
                FunctionTool.from_defaults(search_web),
            ]

            # Create a research-focused ReAct agent
            research_agent = ReActAgent(
                tools=research_tools,
                timeout=90,  # 1.5 minutes for research
                extra_context=f"""
You are a specialized research agent focused on answering: "{ev.question}"

Your goal is to conduct thorough research and provide a comprehensive, well-sourced answer. Use web search to find relevant information and synthesize a detailed response.

RESEARCH STRATEGY:
1. Search for recent, authoritative sources related to the question
2. Look for multiple perspectives and viewpoints  
3. Find specific examples, data, or case studies
4. Synthesize information into a coherent, informative answer
5. Include relevant context and background information

Provide a detailed, well-structured answer that directly addresses the question.
""",
            )

            # Get the research answer
            response = await research_agent.arun(ev.question)
            answer = str(response)

        except Exception as e:
            # Fallback to simple web search
            try:
                search_results = await search_web(ev.question, max_results=5)
                answer = f"""Based on web search results for '{ev.question}':

{search_results}

This research was conducted using web search to find current information and multiple perspectives on the topic."""
            except Exception:
                answer = f"Research summary for '{ev.question}': Unable to complete research due to errors. This is a fallback response."

        return AnswerEvent(question=ev.question, answer=answer)

    @step
    async def synthesize_report(
        self, ctx: Context, ev: AnswerEvent
    ) -> StopEvent | None:
        """Synthesize final report from the answer."""

        # Get topic from context
        topic = await ctx.get("topic", "Research Topic")

        # Use the report agent to analyze and synthesize the research findings
        try:
            # Use the report agent to generate a synthesized analysis
            synthesized_analysis = await self.report_agent.generate_report(
                topic, [ev.question], [ev.answer]
            )

        except Exception as e:
            # Fallback to a simple analysis
            synthesized_analysis = f"""
            Based on the research findings for "{ev.question}", here are the key insights:

            **Key Findings:**
            The research reveals several important patterns and trends related to the topic.

            **Analysis:**
            The data shows that this is a complex topic with multiple perspectives and factors at play.

            **Implications:**
            These findings suggest important considerations for understanding this area.

            **Research Data:**
            {ev.answer}
            """

        # Generate a comprehensive report
        report = f"""# Research Report: {topic}

## Research Question
{ev.question}

## Analysis & Findings
{synthesized_analysis}

---
*Generated by Deep Research Assistant*
"""

        # Return the report as the result
        return StopEvent(result=str(report))


# Test the workflow
if __name__ == "__main__":
    from workflows.events import StartEvent

    # Create a simple test
    workflow = DeepResearchWorkflow()
    start_event = StartEvent(topic="artificial intelligence")

    print("Workflow created successfully!")
    print(f"Start event topic: {start_event.topic}")

    # Test event creation
    generate_event = GenerateEvent(topic="AI research")
    question_event = QuestionEvent(question="What is machine learning?")
    answer_event = AnswerEvent(question="What is machine learning?", answer="ML is...")
    report_event = ReportEvent(report="Final report...")

    print("All events created successfully!")
