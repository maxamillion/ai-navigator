"""Model Catalog Agent implementation."""

import structlog

from ai_navigator.a2a.base_agent import BaseAgent
from ai_navigator.a2a.message import Message
from ai_navigator.a2a.skills import SkillInput, SkillResult
from ai_navigator.a2a.task import Task
from ai_navigator.agents.model_catalog.skills import register_skills
from ai_navigator.config import AgentSettings
from ai_navigator.mcp.openshift_ai_tools import OpenShiftAITools
from ai_navigator.mcp.trustyai_tools import TrustyAITools

logger = structlog.get_logger(__name__)


class ModelCatalogAgent(BaseAgent):
    """
    Model Catalog Agent for querying model registry and TrustyAI.

    Skills:
    - query_models: Search for models in the registry
    - get_model_details: Get detailed model information
    - get_benchmarks: Get model benchmarks
    - get_trustyai_scores: Get TrustyAI evaluation scores
    - recommend_for_workload: Recommend models for a workload type
    """

    def __init__(self, settings: AgentSettings | None = None) -> None:
        """
        Initialize the Model Catalog Agent.

        Args:
            settings: Agent configuration
        """
        if settings is None:
            settings = AgentSettings(
                name="model-catalog",
                description="Model Catalog Agent for OpenShift AI",
            )

        super().__init__(
            name="Model Catalog Agent",
            description=(
                "Queries model registry for available models, retrieves benchmarks, "
                "and provides TrustyAI evaluation scores for model selection."
            ),
            settings=settings,
        )

        # Initialize MCP tools
        self.openshift_ai = OpenShiftAITools()
        self.trustyai = TrustyAITools()

        # Register skills
        register_skills(self)

    async def on_startup(self) -> None:
        """Initialize resources on startup."""
        logger.info("model_catalog_agent_starting")

    async def on_shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("model_catalog_agent_stopping")

    async def process_message(self, task: Task, message: Message) -> SkillResult:
        """
        Process messages that don't target a specific skill.

        Analyzes the message to determine the appropriate skill.
        """
        text = message.get_text().lower()

        # Route to appropriate skill based on message content
        if "recommend" in text or "suggest" in text or "best model" in text:
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("recommend_for_workload", skill_input)
        elif "list" in text or "search" in text or "find" in text or "query" in text:
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("query_models", skill_input)
        elif "detail" in text or "info" in text:
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("get_model_details", skill_input)
        elif "benchmark" in text or "performance" in text:
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("get_benchmarks", skill_input)
        elif "trust" in text or "safety" in text or "evaluation" in text:
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("get_trustyai_scores", skill_input)
        else:
            # Default to querying models
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("query_models", skill_input)


# Create FastAPI app for running as standalone service
def create_app():
    """Create the FastAPI application."""
    agent = ModelCatalogAgent()
    return agent.app


app = create_app()
