"""Deployment Monitor Agent implementation."""

import structlog

from ai_navigator.a2a.base_agent import BaseAgent
from ai_navigator.a2a.message import Message
from ai_navigator.a2a.skills import SkillInput, SkillResult
from ai_navigator.a2a.task import Task
from ai_navigator.agents.deployment_monitor.skills import register_skills
from ai_navigator.config import AgentSettings
from ai_navigator.mcp.observability_tools import ObservabilityTools
from ai_navigator.mcp.openshift_ai_tools import OpenShiftAITools

logger = structlog.get_logger(__name__)


class DeploymentMonitorAgent(BaseAgent):
    """
    Deployment Monitor Agent for observability and monitoring.

    Skills:
    - get_deployment_status: Get current deployment status
    - query_metrics: Query Prometheus metrics
    - check_slo_violations: Check for SLO violations
    - get_pod_logs: Get pod logs
    - get_health_summary: Get overall health summary
    """

    def __init__(self, settings: AgentSettings | None = None) -> None:
        """
        Initialize the Deployment Monitor Agent.

        Args:
            settings: Agent configuration
        """
        if settings is None:
            settings = AgentSettings(
                name="deployment-monitor",
                description="Deployment Monitor Agent for OpenShift AI",
            )

        super().__init__(
            name="Deployment Monitor Agent",
            description=(
                "Monitors deployments, queries metrics, checks SLO compliance, "
                "and provides health summaries for inference services."
            ),
            settings=settings,
        )

        # Initialize MCP tools
        self.openshift_ai = OpenShiftAITools()
        self.observability = ObservabilityTools()

        # Register skills
        register_skills(self)

    async def on_startup(self) -> None:
        """Initialize resources on startup."""
        logger.info("deployment_monitor_agent_starting")

    async def on_shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("deployment_monitor_agent_stopping")

    async def process_message(self, task: Task, message: Message) -> SkillResult:
        """
        Process messages that don't target a specific skill.

        Analyzes the message to determine the appropriate skill.
        """
        text = message.get_text().lower()

        # Route to appropriate skill based on message content
        if "status" in text or "state" in text:
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("get_deployment_status", skill_input)
        elif "metric" in text or "performance" in text or "latency" in text:
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("query_metrics", skill_input)
        elif "slo" in text or "violation" in text or "breach" in text:
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("check_slo_violations", skill_input)
        elif "log" in text or "error" in text:
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("get_pod_logs", skill_input)
        elif "health" in text or "summary" in text or "overview" in text:
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("get_health_summary", skill_input)
        else:
            # Default to health summary
            skill_input = SkillInput(task=task, message=message, params={})
            return await self.skills.dispatch("get_health_summary", skill_input)


# Create FastAPI app for running as standalone service
def create_app():
    """Create the FastAPI application."""
    agent = DeploymentMonitorAgent()
    return agent.app


app = create_app()
