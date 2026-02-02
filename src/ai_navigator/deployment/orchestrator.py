"""Deployment orchestration via MCP."""

from dataclasses import dataclass
from typing import Optional

import structlog

from ai_navigator.models.deployment import (
    DeploymentConfig,
    DeploymentResult,
    DeploymentStatus,
)
from ai_navigator.mcp.client import MCPClient, MCPToolError
from ai_navigator.mcp.orchestrator import MCPOrchestrator, ToolCall
from ai_navigator.deployment.generator import YAMLGenerator
from ai_navigator.deployment.validators import DeploymentValidator, ValidationResult

logger = structlog.get_logger(__name__)


@dataclass
class DeploymentStep:
    """Single step in deployment process."""

    name: str
    description: str
    completed: bool = False
    error: Optional[str] = None


class DeploymentOrchestrator:
    """Orchestrates the complete deployment process."""

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        generator: Optional[YAMLGenerator] = None,
        validator: Optional[DeploymentValidator] = None,
    ) -> None:
        """Initialize orchestrator."""
        self._mcp_client = mcp_client
        self._generator = generator or YAMLGenerator()
        self._validator = validator or DeploymentValidator(mcp_client)
        self._orchestrator = MCPOrchestrator(mcp_client) if mcp_client else None

    async def deploy(
        self,
        config: DeploymentConfig,
        dry_run: bool = False,
    ) -> DeploymentResult:
        """Execute full deployment workflow."""
        steps: list[DeploymentStep] = []

        # Step 1: Validate configuration
        steps.append(DeploymentStep(name="validate", description="Validating configuration"))
        validation = await self._validator.validate(config)
        steps[-1].completed = True

        if not validation.passed:
            steps[-1].error = "; ".join(validation.errors)
            return DeploymentResult(
                deployment_id=config.id,
                status=DeploymentStatus.FAILED,
                message="Validation failed",
                error="; ".join(validation.errors),
                error_details={"errors": validation.errors, "warnings": validation.warnings},
            )

        # Step 2: Generate manifests
        steps.append(DeploymentStep(name="generate", description="Generating manifests"))
        try:
            manifests = self._generator.generate_all(config)
            steps[-1].completed = True
        except Exception as e:
            steps[-1].error = str(e)
            return DeploymentResult(
                deployment_id=config.id,
                status=DeploymentStatus.FAILED,
                message="Manifest generation failed",
                error=str(e),
            )

        if dry_run:
            return DeploymentResult(
                deployment_id=config.id,
                status=DeploymentStatus.PENDING,
                message="Dry run completed successfully",
                inference_service_name=config.inference_service.name,
                namespace=config.inference_service.namespace,
            )

        # Step 3: Ensure namespace exists
        steps.append(DeploymentStep(name="namespace", description="Creating namespace"))
        namespace_result = await self._ensure_namespace(config)
        steps[-1].completed = namespace_result is None
        if namespace_result:
            steps[-1].error = namespace_result
            return DeploymentResult(
                deployment_id=config.id,
                status=DeploymentStatus.FAILED,
                message="Failed to create namespace",
                error=namespace_result,
            )

        # Step 4: Apply manifests
        steps.append(DeploymentStep(name="apply", description="Applying manifests"))
        apply_result = await self._apply_manifests(config, manifests)
        if apply_result.status == DeploymentStatus.FAILED:
            steps[-1].error = apply_result.error
            return apply_result
        steps[-1].completed = True

        # Step 5: Wait for ready
        steps.append(DeploymentStep(name="wait", description="Waiting for deployment"))
        ready_result = await self._wait_for_ready(config)
        steps[-1].completed = ready_result.status == DeploymentStatus.RUNNING

        return ready_result

    async def rollback(self, config: DeploymentConfig) -> DeploymentResult:
        """Rollback a deployment."""
        if not self._mcp_client:
            return DeploymentResult(
                deployment_id=config.id,
                status=DeploymentStatus.DELETED,
                message="Rollback simulated (no MCP client)",
            )

        try:
            await self._mcp_client.call_tool(
                "delete_inference_service",
                {
                    "namespace": config.inference_service.namespace,
                    "name": config.inference_service.name,
                },
            )

            return DeploymentResult(
                deployment_id=config.id,
                status=DeploymentStatus.DELETED,
                message="Deployment rolled back successfully",
                inference_service_name=config.inference_service.name,
                namespace=config.inference_service.namespace,
            )

        except MCPToolError as e:
            return DeploymentResult(
                deployment_id=config.id,
                status=DeploymentStatus.FAILED,
                message="Rollback failed",
                error=str(e),
            )

    async def get_status(self, config: DeploymentConfig) -> DeploymentResult:
        """Get current deployment status."""
        if not self._mcp_client:
            return DeploymentResult(
                deployment_id=config.id,
                status=DeploymentStatus.RUNNING,
                message="Status check simulated (no MCP client)",
                inference_service_name=config.inference_service.name,
                namespace=config.inference_service.namespace,
                endpoint_url=f"https://{config.inference_service.name}-{config.inference_service.namespace}.apps.example.com",
                ready_replicas=config.inference_service.min_replicas,
                total_replicas=config.inference_service.min_replicas,
            )

        try:
            result = await self._mcp_client.call_tool(
                "get_inference_service",
                {
                    "namespace": config.inference_service.namespace,
                    "name": config.inference_service.name,
                },
            )

            status = self._parse_status(result)
            return DeploymentResult(
                deployment_id=config.id,
                status=status,
                inference_service_name=config.inference_service.name,
                namespace=config.inference_service.namespace,
                endpoint_url=result.get("url"),
                ready_replicas=result.get("readyReplicas", 0),
                total_replicas=result.get("replicas", 0),
                conditions=result.get("conditions", []),
            )

        except MCPToolError as e:
            return DeploymentResult(
                deployment_id=config.id,
                status=DeploymentStatus.FAILED,
                message="Status check failed",
                error=str(e),
            )

    async def _ensure_namespace(self, config: DeploymentConfig) -> Optional[str]:
        """Ensure namespace exists, return error message if failed."""
        if not self._mcp_client:
            logger.info(
                "Simulating namespace creation",
                namespace=config.inference_service.namespace,
            )
            return None

        try:
            await self._mcp_client.call_tool(
                "create_data_science_project",
                {
                    "name": config.inference_service.namespace,
                    "display_name": f"AI Navigator: {config.inference_service.name}",
                },
            )
            return None
        except MCPToolError as e:
            if "already exists" in str(e).lower():
                return None  # Namespace already exists, that's fine
            return str(e)

    async def _apply_manifests(
        self,
        config: DeploymentConfig,
        manifests: dict[str, str],
    ) -> DeploymentResult:
        """Apply generated manifests to cluster."""
        if not self._mcp_client:
            logger.info(
                "Simulating manifest application",
                manifests=list(manifests.keys()),
            )
            return DeploymentResult(
                deployment_id=config.id,
                status=DeploymentStatus.CREATING,
                message="Manifests applied (simulated)",
                inference_service_name=config.inference_service.name,
                namespace=config.inference_service.namespace,
            )

        try:
            spec = config.inference_service

            # Create inference service via MCP
            result = await self._mcp_client.call_tool(
                "create_inference_service",
                {
                    "namespace": spec.namespace,
                    "name": spec.name,
                    "model_name": spec.model_name,
                    "min_replicas": spec.min_replicas,
                    "max_replicas": spec.max_replicas,
                    "gpu_count": spec.gpu_count,
                    "memory": spec.memory,
                    "cpu": spec.cpu,
                },
            )

            return DeploymentResult(
                deployment_id=config.id,
                status=DeploymentStatus.CREATING,
                message="InferenceService created",
                inference_service_name=spec.name,
                namespace=spec.namespace,
            )

        except MCPToolError as e:
            return DeploymentResult(
                deployment_id=config.id,
                status=DeploymentStatus.FAILED,
                message="Failed to apply manifests",
                error=str(e),
            )

    async def _wait_for_ready(
        self,
        config: DeploymentConfig,
        timeout_seconds: int = 300,
    ) -> DeploymentResult:
        """Wait for deployment to become ready."""
        if not self._mcp_client:
            # Simulate ready state
            spec = config.inference_service
            return DeploymentResult(
                deployment_id=config.id,
                status=DeploymentStatus.RUNNING,
                message="Deployment ready (simulated)",
                inference_service_name=spec.name,
                namespace=spec.namespace,
                endpoint_url=f"https://{spec.name}-{spec.namespace}.apps.cluster.example.com",
                ready_replicas=spec.min_replicas,
                total_replicas=spec.min_replicas,
            )

        import asyncio

        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout_seconds:
            status = await self.get_status(config)

            if status.status == DeploymentStatus.RUNNING:
                return status

            if status.status == DeploymentStatus.FAILED:
                return status

            await asyncio.sleep(10)

        return DeploymentResult(
            deployment_id=config.id,
            status=DeploymentStatus.FAILED,
            message="Deployment timed out",
            error=f"Deployment did not become ready within {timeout_seconds} seconds",
        )

    def _parse_status(self, result: dict) -> DeploymentStatus:
        """Parse deployment status from MCP result."""
        conditions = result.get("conditions", [])

        for condition in conditions:
            if condition.get("type") == "Ready":
                if condition.get("status") == "True":
                    return DeploymentStatus.RUNNING
                elif condition.get("reason") == "RevisionFailed":
                    return DeploymentStatus.FAILED

        if result.get("readyReplicas", 0) > 0:
            return DeploymentStatus.RUNNING

        return DeploymentStatus.CREATING
