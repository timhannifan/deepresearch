"""Report synthesis agent."""

from deepresearch.utils import get_llm


class ReportAgent:
    """Agent that synthesizes research findings into a comprehensive report."""

    def __init__(self) -> None:
        self.llm = get_llm()

    async def generate_report(
        self, topic: str, questions: list[str], answers: list[str]
    ) -> str:
        """Generate a comprehensive research report.

        Args:
            topic: Research topic
            questions: List of research questions
            answers: List of corresponding answers

        Returns:
            Comprehensive research report
        """
        # Prepare Q&A sections
        qa_sections = []
        for i, (question, answer) in enumerate(zip(questions, answers), 1):
            qa_sections.append(f"## {i}. {question}\n\n{answer}\n")

        qa_content = "\n".join(qa_sections)

        prompt = f"""Create a comprehensive research report on "{topic}" based on the following research findings:

{qa_content}

Please structure the report as follows:

# Research Report: {topic}

## Executive Summary
Provide a brief overview of the key findings and insights.

## Research Questions and Findings
Include the Q&A sections above, but enhance them with better formatting and flow.

## Key Insights and Trends
Synthesize the most important findings across all research questions.

## Conclusions and Future Directions
Summarize the main conclusions and suggest areas for future research.

## References and Sources
Note the sources used (scientific literature and web research).

Make the report professional, well-structured, and informative.
"""

        response = self.llm.complete(prompt)
        return response.text
