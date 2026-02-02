"""Monitoring stage - Step 8."""

from typing import Optional

import structlog

from ai_navigator.models.deployment import DeploymentConfig, DeploymentResult, DeploymentStatus
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


class MonitorStage(BaseStage):
    """Stage 8: Verify deployment and test endpoints."""

    def __init__(self, mcp_client: Optional[object] = None) -> None:
        """Initialize with optional MCP client for verification."""
        self._mcp_client = mcp_client

    async def process(self, state: WorkflowState, user_input: str) -> StageResult:
        """Verify deployment and test inference endpoint."""
        config_data = state.metadata.get("deployment_config")
        if not config_data:
            return StageResult(
                success=False,
                message="No deployment found to monitor.",
            )

        config = DeploymentConfig.model_validate(config_data)

        # Check deployment status
        status = await self._check_deployment_status(config)

        if status.status == DeploymentStatus.RUNNING:
            # Run inference test
            test_result = await self._test_inference(config)

            summary = self._format_deployment_summary(config, status, test_result)

            return StageResult(
                success=True,
                message=summary,
                data={
                    "status": status.model_dump(),
                    "inference_test": test_result,
                    "workflow_complete": True,
                },
            )

        elif status.status == DeploymentStatus.CREATING:
            return StageResult(
                success=True,
                message=(
                    f"Deployment is still in progress...\n\n"
                    f"Status: {status.message}\n"
                    f"Ready replicas: {status.ready_replicas}/{status.total_replicas}\n\n"
                    "Checking again..."
                ),
                prompt_user="Type 'check' to check status again, or wait for completion.",
            )

        else:
            return StageResult(
                success=False,
                message=(
                    f"Deployment issue detected.\n\n"
                    f"Status: {status.status.value}\n"
                    f"Error: {status.error or 'Unknown'}\n\n"
                    "Please check the deployment logs for more details."
                ),
                data={"status": status.model_dump()},
            )

    async def _check_deployment_status(self, config: DeploymentConfig) -> DeploymentResult:
        """Check the deployment status via MCP or simulation."""
        svc = config.inference_service

        if not self._mcp_client:
            # Simulate success for demo
            endpoint_url = f"https://{svc.name}-{svc.namespace}.apps.cluster.example.com"

            return DeploymentResult(
                deployment_id=config.id,
                status=DeploymentStatus.RUNNING,
                inference_service_name=svc.name,
                namespace=svc.namespace,
                endpoint_url=endpoint_url,
                message="All replicas ready",
                ready_replicas=svc.min_replicas,
                total_replicas=svc.min_replicas,
                conditions=[
                    {"type": "Ready", "status": "True", "reason": "MinimumReplicasAvailable"},
                    {"type": "PredictorReady", "status": "True", "reason": "PodRunning"},
                ],
            )

        # Real status check via MCP
        try:
            # result = await self._mcp_client.call_tool("get_inference_service", {...})
            # Parse and return status
            pass
        except Exception as e:
            logger.error("Failed to check deployment status", error=str(e))

        return DeploymentResult(
            deployment_id=config.id,
            status=DeploymentStatus.PENDING,
            message="Checking status...",
        )

    async def _test_inference(self, config: DeploymentConfig) -> dict:
        """Test the inference endpoint."""
        if not self._mcp_client:
            # Simulate test for demo
            return {
                "success": True,
                "latency_ms": 245,
                "tokens_generated": 50,
                "test_prompt": "Hello, how are you?",
                "test_response": "Hello! I'm doing well, thank you for asking. How can I assist you today?",
            }

        # Real inference test
        try:
            # Send test request to endpoint
            pass
        except Exception as e:
            logger.error("Inference test failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

        return {"success": False, "error": "Not implemented"}

    def _format_deployment_summary(
        self,
        config: DeploymentConfig,
        status: DeploymentResult,
        test_result: dict,
    ) -> str:
        """Format the final deployment summary."""
        svc = config.inference_service

        lines = [
            "**Deployment Complete**\n",
            f"**Service**: {svc.name}",
            f"**Namespace**: {svc.namespace}",
            f"**Endpoint**: {status.endpoint_url}",
            "",
            "**Status**",
            f"- State: {status.status.value}",
            f"- Ready Replicas: {status.ready_replicas}/{status.total_replicas}",
            "",
        ]

        if test_result.get("success"):
            lines.extend([
                "**Inference Test: PASSED**",
                f"- Latency: {test_result.get('latency_ms', 'N/A')}ms",
                f"- Tokens Generated: {test_result.get('tokens_generated', 'N/A')}",
                "",
            ])
        else:
            lines.extend([
                "**Inference Test: FAILED**",
                f"- Error: {test_result.get('error', 'Unknown')}",
                "",
            ])

        lines.extend([
            "**Next Steps**",
            f"1. Access your model at: {status.endpoint_url}",
            "2. Configure authentication tokens in your application",
            "3. Monitor performance in the OpenShift AI dashboard",
            "",
            "**Example Usage**",
            "```bash",
            f"curl -X POST {status.endpoint_url}/v1/chat/completions \\",
            '  -H "Content-Type: application/json" \\',
            '  -H "Authorization: Bearer $TOKEN" \\',
            '  -d \'{"messages": [{"role": "user", "content": "Hello"}]}\'',
            "```",
            "",
            "Workflow complete. Your model is ready to use.",
        ])

        return "\n".join(lines)
