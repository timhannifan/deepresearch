"""Deep research agent for scientific literature synthesis."""

from deepresearch.workflows.research_workflow import DeepResearchWorkflow

# Export workflow instance for llama-deploy
deep_research_w = DeepResearchWorkflow()

__all__ = ["DeepResearchWorkflow", "deep_research_w"]
