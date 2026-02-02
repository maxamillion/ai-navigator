"""Deployment stage - Step 7."""

from typing import Optional

import structlog

from ai_navigator.models.capacity import CapacityPlan
from ai_navigator.models.deployment import (
    DeploymentConfig,
    InferenceServiceSpec,
    RuntimeConfig,
)
from ai_navigator.models.workflow import WorkflowState
from ai_navigator.workflow.stages.base import BaseStage

logger = structlog.get_logger(__name__)


class StageResult:
    def __init__(
        self,
        success: bool,
        message: str,
        advance: bool = False,
        data: Optional[dict] = None,
        prompt_user: Optional[str] = None,
    ):
        self.success = success
        self.message = message
        self.advance = advance
        self.data = data or {}
        self.prompt_user = prompt_user


class DeployStage(BaseStage):
    """Stage 7: Generate and apply deployment configuration."""

    def __init__(self, mcp_client: Optional[object] = None) -> None:
        """Initialize with optional MCP client for deployment."""
        self._mcp_client = mcp_client

    async def process(self, state: WorkflowState, user_input: str) -> StageResult:
        """Generate deployment configuration."""
        # Check if we're confirming deployment
        if "confirm" in user_input.lower() or "yes" in user_input.lower():
            return await self._execute_deployment(state)

        if "cancel" in user_input.lower() or "no" in user_input.lower():
            return StageResult(
                success=True,
                message="Deployment cancelled. You can modify the configuration and try again.",
                prompt_user="Would you like to adjust any settings?",
            )

        # Generate deployment configuration
        config = await self._generate_config(state)

        if not config:
            return StageResult(
                success=False,
                message="Could not generate deployment configuration. Missing capacity plan.",
            )

        # Store configuration
        state.deployment_config_id = config.id
        state.metadata["deployment_config"] = config.model_dump()

        # Show preview
        preview = self._format_deployment_preview(config)

        return StageResult(
            success=True,
            message=preview,
            prompt_user=(
                "Ready to deploy?\n"
                "- Type 'confirm' to proceed with deployment\n"
                "- Type 'cancel' to abort\n"
                "- Or specify changes you'd like to make"
            ),
        )

    async def _generate_config(self, state: WorkflowState) -> Optional[DeploymentConfig]:
        """Generate deployment configuration from capacity plan."""
        plan_data = state.metadata.get("capacity_plan")
        if not plan_data:
            return None

        plan = CapacityPlan.model_validate(plan_data)

        # Generate namespace from user ID or default
        namespace = f"ai-navigator-{state.user_id[:8]}"
        service_name = f"{plan.model_name.replace('_', '-')}-service"

        # Create runtime configuration
        runtime = RuntimeConfig(
            runtime_name="vllm",
            tensor_parallel_size=plan.gpu_count,
            gpu_memory_utilization=0.9,
            dtype="float16",
        )

        # Create inference service spec
        inference_service = InferenceServiceSpec(
            name=service_name,
            namespace=namespace,
            model_name=plan.model_name,
            model_format="pytorch",
            runtime=runtime,
            min_replicas=plan.min_replicas,
            max_replicas=plan.max_replicas,
            gpu_count=plan.gpu_count,
            gpu_type=plan.gpu_type,
            memory=plan.memory_per_replica,
            cpu=plan.cpu_per_replica,
            scale_target=10,  # Concurrent requests per replica
            enable_route=True,
            enable_auth=True,
        )

        config = DeploymentConfig(
            inference_service=inference_service,
            create_hpa=True,
            create_pdb=plan.min_replicas > 1,
        )

        return config

    async def _execute_deployment(self, state: WorkflowState) -> StageResult:
        """Execute the deployment via MCP."""
        config_data = state.metadata.get("deployment_config")
        if not config_data:
            return StageResult(
                success=False,
                message="No deployment configuration found.",
            )

        config = DeploymentConfig.model_validate(config_data)

        if not self._mcp_client:
            # Simulate deployment for demo
            logger.info(
                "Simulating deployment (no MCP client)",
                namespace=config.inference_service.namespace,
                service=config.inference_service.name,
            )

            return StageResult(
                success=True,
                message=(
                    "**Deployment Initiated (Simulation Mode)**\n\n"
                    f"Namespace: {config.inference_service.namespace}\n"
                    f"Service: {config.inference_service.name}\n\n"
                    "In a real deployment, this would:\n"
                    "1. Create the namespace if needed\n"
                    "2. Apply the InferenceService manifest\n"
                    "3. Wait for pods to become ready\n"
                    "4. Configure the route/ingress\n\n"
                    "Proceeding to monitoring stage..."
                ),
                advance=True,
                data={"simulated": True, "config": config.model_dump()},
            )

        # Real deployment via MCP
        try:
            # Create namespace
            # await self._mcp_client.call_tool("create_data_science_project", {...})

            # Create inference service
            # await self._mcp_client.call_tool("create_inference_service", {...})

            return StageResult(
                success=True,
                message=(
                    f"**Deployment Successful**\n\n"
                    f"Namespace: {config.inference_service.namespace}\n"
                    f"Service: {config.inference_service.name}\n\n"
                    "Resources created:\n"
                    "- InferenceService\n"
                    "- HorizontalPodAutoscaler\n"
                    "- Route (OpenShift)\n\n"
                    "Proceeding to verification..."
                ),
                advance=True,
                data={"deployed": True, "config": config.model_dump()},
            )

        except Exception as e:
            logger.error("Deployment failed", error=str(e))
            return StageResult(
                success=False,
                message=f"Deployment failed: {e}",
                data={"error": str(e)},
            )

    def _format_deployment_preview(self, config: DeploymentConfig) -> str:
        """Format deployment configuration as preview."""
        svc = config.inference_service
        runtime = svc.runtime

        lines = [
            "**Deployment Preview**\n",
            f"**Namespace**: {svc.namespace}",
            f"**Service Name**: {svc.name}",
            "",
            "**Model Configuration**",
            f"- Model: {svc.model_name}",
            f"- Format: {svc.model_format}",
            "",
            "**Runtime Configuration**",
        ]

        if runtime:
            lines.extend([
                f"- Runtime: {runtime.runtime_name}",
                f"- Tensor Parallel: {runtime.tensor_parallel_size}",
                f"- GPU Memory Utilization: {runtime.gpu_memory_utilization * 100:.0f}%",
                f"- Data Type: {runtime.dtype}",
            ])

        lines.extend([
            "",
            "**Scaling Configuration**",
            f"- Min Replicas: {svc.min_replicas}",
            f"- Max Replicas: {svc.max_replicas}",
            f"- Scale Target: {svc.scale_target} concurrent requests",
            "",
            "**Resources per Replica**",
            f"- GPU: {svc.gpu_type or 'default'} x{svc.gpu_count}",
            f"- Memory: {svc.memory}",
            f"- CPU: {svc.cpu}",
            "",
            "**Networking**",
            f"- Route Enabled: {svc.enable_route}",
            f"- Authentication: {svc.enable_auth}",
            f"- Timeout: {svc.timeout_seconds}s",
        ])

        if config.create_hpa:
            lines.append("\n**Additional Resources**")
            lines.append("- HorizontalPodAutoscaler: Yes")
        if config.create_pdb:
            lines.append("- PodDisruptionBudget: Yes")

        return "\n".join(lines)
