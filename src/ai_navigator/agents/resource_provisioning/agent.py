"""Resource Provisioning Agent implementation."""

import structlog

from ai_navigator.a2a.base_agent import BaseAgent
from ai_navigator.a2a.message import Message
from ai_navigator.a2a.skills import SkillInput, SkillResult
from ai_navigator.a2a.task import Task
from ai_navigator.agents.resource_provisioning.skills import register_skills
from ai_navigator.config import AgentSettings
from ai_navigator.mcp.openshift_ai_tools import OpenShiftAITools
from ai_navigator.mcp.trustyai_tools import TrustyAITools

logger = structlog.get_logger(__name__)


class ResourceProvisioningAgent(BaseAgent):
    """
    Resource Provisioning Agent for deployment configuration.

    Skills:
    - generate_deployment_config: Generate KServe InferenceService config
    - estimate_cost: Estimate deployment costs
    - validate_slo_compliance: Validate config against SLOs
    - apply_deployment: Apply deployment to cluster
    - generate_guardrails: Generate TrustyAI guardrails config
    """

    def __init__(self, settings: AgentSettings | None = None) -> None:
        """
        Initialize the Resource Provisioning Agent.

        Args:
            settings: Agent configuration
        """
        if settings is None:
            settings = AgentSettings(
                name="resource-provisioning",
                description="Resource Provisioning Agent for OpenShift AI",
            )

        super().__init__(
            name="Resource Provisioning Agent",
            description=(
                "Generates deployment configurations for KServe and TrustyAI, "
                "estimates costs, and validates SLO compliance."
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
        logger.info("resource_provisioning_agent_starting")

    async def on_shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("resource_provisioning_agent_stopping")

    async def process_message(self, task: Task, message: Message) -> SkillResult:
        """
        Process messages that don't target a specific skill.

        Analyzes the message to determine the appropriate skill.
        """
        text = message.get_text().lower()

        # Route to appropriate skill based on message content
        if "deploy" in text and ("apply" in text or "create" in text or "install" in text):
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("apply_deployment", skill_input)
        elif "config" in text or "generate" in text or "create" in text:
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("generate_deployment_config", skill_input)
        elif "cost" in text or "price" in text or "estimate" in text:
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("estimate_cost", skill_input)
        elif "slo" in text or "validate" in text or "compliance" in text:
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("validate_slo_compliance", skill_input)
        elif "guardrail" in text or "safety" in text or "protect" in text:
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("generate_guardrails", skill_input)
        else:
            # Default to generating deployment config
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("generate_deployment_config", skill_input)


# Create FastAPI app for running as standalone service
def create_app():
    """Create the FastAPI application."""
    agent = ResourceProvisioningAgent()
    return agent.app


app = create_app()
