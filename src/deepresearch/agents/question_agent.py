"""Question generation agent."""

from deepresearch.utils import get_llm


class QuestionAgent:
    """Agent that generates research questions for a given topic."""

    def __init__(self) -> None:
        self.llm = get_llm()

    async def generate_questions(self, topic: str, num_questions: int = 3) -> list[str]:
        """Generate research questions for a given topic.

        Args:
            topic: Research topic
            num_questions: Number of questions to generate

        Returns:
            List of research questions
        """
        prompt = f"""Generate {num_questions} specific, researchable questions about "{topic}".
Each question should be focused and answerable through scientific literature or web research.
Format each question on a new line, starting with a number.

Example format:
1. What are the current challenges in [specific aspect]?
2. How does [specific method] compare to [alternative method]?
3. What recent advances have been made in [specific area]?
"""

        response = self.llm.complete(prompt)
        questions = []

        for line in response.text.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering and clean up
                question = line.split(".", 1)[-1].strip()
                if question:
                    questions.append(question)

        return questions[:num_questions]
