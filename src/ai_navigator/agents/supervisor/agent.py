"""Supervisor Agent implementation."""

import structlog

from ai_navigator.a2a.base_agent import BaseAgent
from ai_navigator.a2a.message import Message
from ai_navigator.a2a.skills import SkillInput, SkillResult
from ai_navigator.a2a.task import Task
from ai_navigator.agents.supervisor.aggregator import ResultAggregator
from ai_navigator.agents.supervisor.decomposer import IntentDecomposer
from ai_navigator.agents.supervisor.delegator import SubAgentDelegator
from ai_navigator.agents.supervisor.skills import register_skills
from ai_navigator.config import AgentSettings, KubernetesSettings, get_settings
from ai_navigator.llm.client import LLMClient

logger = structlog.get_logger(__name__)


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent for orchestrating AI Navigator sub-agents.

    The Supervisor Agent:
    - Receives user requests via A2A protocol
    - Decomposes requests into sub-tasks using LLM
    - Delegates sub-tasks to specialized sub-agents
    - Aggregates results into coherent responses

    Skills:
    - recommend_model: Get model recommendations
    - deploy_model: Deploy a model to the cluster
    - check_status: Check deployment status
    - list_models: List available models
    - get_metrics: Query performance metrics
    - check_slos: Check SLO compliance
    - estimate_cost: Estimate deployment costs
    """

    def __init__(
        self,
        settings: AgentSettings | None = None,
        kubernetes_settings: KubernetesSettings | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        """
        Initialize the Supervisor Agent.

        Args:
            settings: Agent configuration
            kubernetes_settings: Kubernetes configuration for agent discovery
            llm_client: LLM client for decomposition and aggregation
        """
        if settings is None:
            settings = AgentSettings(
                name="supervisor",
                description="AI Navigator Supervisor Agent",
            )

        super().__init__(
            name="AI Navigator Supervisor",
            description=(
                "Orchestrates AI model deployment on OpenShift AI. "
                "Decomposes user requests and delegates to specialized sub-agents "
                "for model catalog, resource provisioning, and deployment monitoring."
            ),
            settings=settings,
        )

        # Get global settings if not provided
        if kubernetes_settings is None:
            kubernetes_settings = get_settings().kubernetes

        if llm_client is None:
            llm_client = LLMClient()

        # Initialize components
        self.llm = llm_client
        self.decomposer = IntentDecomposer(llm_client)
        self.delegator = SubAgentDelegator(kubernetes_settings)
        self.aggregator = ResultAggregator(llm_client)

        # Register default static endpoints for development
        self._register_default_endpoints()

        # Register skills
        register_skills(self)

    def _register_default_endpoints(self) -> None:
        """Register default static endpoints for development."""
        # These will be overridden by Kubernetes discovery in production
        default_endpoints = {
            "model_catalog": "http://localhost:8001",
            "resource_provisioning": "http://localhost:8002",
            "deployment_monitor": "http://localhost:8003",
        }

        for name, url in default_endpoints.items():
            self.delegator.register_static_endpoint(name, url)

    async def on_startup(self) -> None:
        """Initialize resources on startup."""
        logger.info("supervisor_agent_starting")

        # Discover available sub-agents
        try:
            agents = await self.delegator.discover_agents()
            logger.info("discovered_agents", agents=list(agents.keys()))
        except Exception as e:
            logger.warning("agent_discovery_failed", error=str(e))

    async def on_shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("supervisor_agent_stopping")

    async def process_message(self, task: Task, message: Message) -> SkillResult:
        """
        Process a message by decomposing and delegating.

        This is the main entry point for handling user requests
        when no specific skill is targeted.
        """
        user_text = message.get_text()
        logger.info("processing_request", text=user_text[:100])

        try:
            # Decompose the request
            decomposition = await self.decomposer.decompose(user_text)
            logger.info(
                "request_decomposed",
                intent=decomposition.intent,
                subtasks=len(decomposition.subtasks),
            )

            # If no subtasks, try to match a skill directly
            if not decomposition.subtasks:
                return await self._route_to_skill(task, message)

            # Delegate to sub-agents
            results = await self.delegator.delegate(decomposition, user_text)

            # Aggregate results
            response = await self.aggregator.aggregate(user_text, results)

            if response.success:
                return SkillResult.ok(
                    message=response.summary,
                    data={"details": response.details},
                )
            elif response.failed_agents:
                # Partial success
                return SkillResult.ok(
                    message=response.summary,
                    data={
                        "details": response.details,
                        "failed_agents": response.failed_agents,
                    },
                )
            else:
                return SkillResult.error("No results received from sub-agents")

        except Exception as e:
            logger.exception("request_processing_failed", error=str(e))
            return SkillResult.error(f"Failed to process request: {e}")

    async def _route_to_skill(self, task: Task, message: Message) -> SkillResult:
        """Route to a skill based on message content."""
        text = message.get_text().lower()

        # Determine which skill to use
        if any(w in text for w in ["recommend", "suggest", "best"]):
            skill_id = "recommend_model"
        elif any(w in text for w in ["deploy", "create", "install"]):
            skill_id = "deploy_model"
        elif any(w in text for w in ["status", "health", "running"]):
            skill_id = "check_status"
        elif any(w in text for w in ["list", "show", "available"]):
            skill_id = "list_models"
        elif any(w in text for w in ["metric", "latency", "performance"]):
            skill_id = "get_metrics"
        elif any(w in text for w in ["slo", "violation", "compliance"]):
            skill_id = "check_slos"
        elif any(w in text for w in ["cost", "price", "estimate"]):
            skill_id = "estimate_cost"
        else:
            skill_id = "list_models"  # Default

        skill_input = SkillInput(task=task, message=message, params={})
        return await self.skills.dispatch(skill_id, skill_input)


# Create FastAPI app for running as standalone service
def create_app():
    """Create the FastAPI application."""
    agent = SupervisorAgent()
    return agent.app


app = create_app()
