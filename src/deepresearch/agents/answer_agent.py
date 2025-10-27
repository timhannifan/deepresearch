"""Answer agent that researches specific questions."""

from deepresearch.tools.vector_search import query_papers_with_llm
from deepresearch.tools.web_search import search_web
from deepresearch.utils import get_llm


class AnswerAgent:
    """Agent that researches and answers specific questions."""

    def __init__(self) -> None:
        self.llm = get_llm()

    async def research_question(self, question: str) -> str:
        """Research a specific question using both vector search and web search.

        Args:
            question: Research question to answer

        Returns:
            Comprehensive answer based on research
        """
        # Search scientific papers
        paper_results = await query_papers_with_llm(question, top_k=3)

        # Search web for recent information
        web_results = await search_web(question, max_results=3)

        # Synthesize findings
        prompt = f"""Based on the following research findings, provide a comprehensive answer to: "{question}"

SCIENTIFIC LITERATURE FINDINGS:
{paper_results}

WEB RESEARCH FINDINGS:
{web_results}

Please provide a well-structured answer that:
1. Directly addresses the question
2. Synthesizes information from both sources
3. Highlights key findings and insights
4. Notes any limitations or gaps in the research
"""

        response = self.llm.complete(prompt)
        return response.text
