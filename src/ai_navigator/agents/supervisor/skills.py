"""Skills for the Supervisor Agent."""

from typing import TYPE_CHECKING

import structlog

from ai_navigator.a2a.skills import SkillInput, SkillResult

if TYPE_CHECKING:
    from ai_navigator.agents.supervisor.agent import SupervisorAgent

logger = structlog.get_logger(__name__)


def register_skills(agent: "SupervisorAgent") -> None:
    """Register all skills for the Supervisor Agent."""

    @agent.skills.register(
        id="recommend_model",
        name="Recommend Model",
        description="Get model recommendations for a specific workload",
        tags=["recommendation", "models", "planning"],
        examples=[
            "Recommend a model for our chatbot",
            "What model should I use for text classification?",
            "Suggest the best model for code generation",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "workload_type": {
                    "type": "string",
                    "description": "Type of workload (chat, classification, code, etc.)",
                },
                "constraints": {
                    "type": "object",
                    "description": "Resource constraints",
                },
            },
        },
    )
    async def recommend_model(input: SkillInput) -> SkillResult:
        """Get model recommendations."""
        user_message = input.message.get_text()

        # Decompose and delegate
        decomposition = await agent.decomposer.decompose(user_message)

        # Override to use recommendation workflow
        if decomposition.intent != "recommend_model":
            decomposition = await agent.decomposer._heuristic_decompose(
                f"recommend model for {user_message}"
            )

        results = await agent.delegator.delegate(decomposition, user_message)

        # Aggregate results
        response = await agent.aggregator.aggregate(user_message, results)

        if response.success:
            return SkillResult.ok(message=response.summary, data={"details": response.details})
        else:
            return SkillResult.error(
                f"Failed to get recommendations: {', '.join(response.failed_agents)}"
            )

    @agent.skills.register(
        id="deploy_model",
        name="Deploy Model",
        description="Deploy a model to OpenShift AI",
        tags=["deployment", "models", "kserve"],
        examples=[
            "Deploy granite-4.0-h-tiny",
            "Create an inference service for Llama 3",
            "Set up the model with guardrails",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Model to deploy"},
                "namespace": {"type": "string", "description": "Target namespace"},
                "dry_run": {"type": "boolean", "description": "Dry run mode"},
            },
        },
    )
    async def deploy_model(input: SkillInput) -> SkillResult:
        """Deploy a model."""
        user_message = input.message.get_text()

        # Decompose for deployment workflow
        decomposition = await agent.decomposer.decompose(user_message)

        if decomposition.intent != "deploy_model":
            decomposition = await agent.decomposer._heuristic_decompose(
                f"deploy {user_message}"
            )

        results = await agent.delegator.delegate(decomposition, user_message)
        response = await agent.aggregator.aggregate(user_message, results)

        if response.success:
            return SkillResult.ok(message=response.summary, data={"details": response.details})
        else:
            return SkillResult.error(
                f"Deployment failed: {', '.join(response.failed_agents)}"
            )

    @agent.skills.register(
        id="check_status",
        name="Check Deployment Status",
        description="Check the status and health of a deployment",
        tags=["monitoring", "status", "health"],
        examples=[
            "Check the status of granite deployment",
            "Is my model running?",
            "Get health summary",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Service name"},
                "namespace": {"type": "string", "description": "Namespace"},
            },
        },
    )
    async def check_status(input: SkillInput) -> SkillResult:
        """Check deployment status."""
        user_message = input.message.get_text()

        # Direct delegation to deployment monitor
        result = await agent.delegator.delegate_direct(
            agent_name="deployment_monitor",
            skill_id="get_health_summary",
            message=user_message,
        )

        if result.success:
            return SkillResult.ok(message=result.message, data=result.data)
        else:
            return SkillResult.error(f"Status check failed: {result.error}")

    @agent.skills.register(
        id="list_models",
        name="List Models",
        description="List available models in the registry",
        tags=["models", "catalog", "search"],
        examples=[
            "List all available models",
            "Show me LLM models",
            "What models are in the registry?",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "filter": {"type": "string", "description": "Filter criteria"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
    )
    async def list_models(input: SkillInput) -> SkillResult:
        """List available models."""
        user_message = input.message.get_text()

        result = await agent.delegator.delegate_direct(
            agent_name="model_catalog",
            skill_id="query_models",
            message=user_message,
        )

        if result.success:
            return SkillResult.ok(message=result.message, data=result.data)
        else:
            return SkillResult.error(f"Model query failed: {result.error}")

    @agent.skills.register(
        id="get_metrics",
        name="Get Metrics",
        description="Query performance metrics for a deployment",
        tags=["metrics", "monitoring", "performance"],
        examples=[
            "Show latency metrics",
            "What's the GPU utilization?",
            "Get performance data",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Service name"},
                "metric_type": {"type": "string", "description": "Type of metric"},
            },
        },
    )
    async def get_metrics(input: SkillInput) -> SkillResult:
        """Get deployment metrics."""
        user_message = input.message.get_text()

        result = await agent.delegator.delegate_direct(
            agent_name="deployment_monitor",
            skill_id="query_metrics",
            message=user_message,
        )

        if result.success:
            return SkillResult.ok(message=result.message, data=result.data)
        else:
            return SkillResult.error(f"Metrics query failed: {result.error}")

    @agent.skills.register(
        id="check_slos",
        name="Check SLO Compliance",
        description="Check SLO compliance and violations",
        tags=["slo", "compliance", "monitoring"],
        examples=[
            "Are we meeting our SLOs?",
            "Check for violations",
            "SLO status report",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Service name"},
            },
        },
    )
    async def check_slos(input: SkillInput) -> SkillResult:
        """Check SLO compliance."""
        user_message = input.message.get_text()

        result = await agent.delegator.delegate_direct(
            agent_name="deployment_monitor",
            skill_id="check_slo_violations",
            message=user_message,
        )

        if result.success:
            return SkillResult.ok(message=result.message, data=result.data)
        else:
            return SkillResult.error(f"SLO check failed: {result.error}")

    @agent.skills.register(
        id="estimate_cost",
        name="Estimate Cost",
        description="Estimate deployment costs",
        tags=["cost", "planning", "estimation"],
        examples=[
            "How much will this cost?",
            "Estimate deployment costs",
            "What's the monthly cost?",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "replicas": {"type": "integer"},
                "gpu_count": {"type": "integer"},
            },
        },
    )
    async def estimate_cost(input: SkillInput) -> SkillResult:
        """Estimate deployment costs."""
        user_message = input.message.get_text()

        result = await agent.delegator.delegate_direct(
            agent_name="resource_provisioning",
            skill_id="estimate_cost",
            message=user_message,
        )

        if result.success:
            return SkillResult.ok(message=result.message, data=result.data)
        else:
            return SkillResult.error(f"Cost estimation failed: {result.error}")
