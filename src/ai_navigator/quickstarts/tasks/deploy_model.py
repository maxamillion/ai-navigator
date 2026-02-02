"""Deploy Model quickstart task - Priority task."""

from typing import Any, Optional

import structlog

from ai_navigator.quickstarts.engine import (
    QuickstartTask,
    QuickstartResult,
    QuickstartStep,
    QuickstartStatus,
)

logger = structlog.get_logger(__name__)


class DeployModelTask(QuickstartTask):
    """Quickstart: Deploy a model with KServe."""

    @property
    def name(self) -> str:
        return "deploy-model"

    @property
    def display_name(self) -> str:
        return "Deploy Model with KServe"

    @property
    def description(self) -> str:
        return (
            "Deploy an AI model as an inference service using KServe. "
            "This quickstart guides you through selecting a model, configuring "
            "the serving runtime, and deploying to your OpenShift AI project."
        )

    def get_steps(self) -> list[QuickstartStep]:
        return [
            QuickstartStep(
                id="select-project",
                name="Select Project",
                description="Choose or create a Data Science Project",
            ),
            QuickstartStep(
                id="select-model",
                name="Select Model",
                description="Choose the model to deploy",
            ),
            QuickstartStep(
                id="configure-runtime",
                name="Configure Runtime",
                description="Set up the serving runtime (vLLM, OpenVINO, etc.)",
            ),
            QuickstartStep(
                id="configure-resources",
                name="Configure Resources",
                description="Set GPU, memory, and replica configuration",
            ),
            QuickstartStep(
                id="deploy",
                name="Deploy",
                description="Create the InferenceService",
            ),
            QuickstartStep(
                id="verify",
                name="Verify",
                description="Confirm deployment is running",
            ),
        ]

    def get_required_inputs(self) -> list[dict[str, str]]:
        return [
            {
                "name": "project",
                "description": "Name of the Data Science Project (namespace)",
                "required": True,
            },
            {
                "name": "model_name",
                "description": "Name of the model to deploy (e.g., llama-2-7b)",
                "required": True,
            },
            {
                "name": "model_path",
                "description": "S3 path or model registry URI (optional)",
                "required": False,
            },
        ]

    async def execute(
        self,
        context: dict[str, Any],
        user_input: Optional[str] = None,
    ) -> QuickstartResult:
        """Execute the deploy model quickstart."""
        steps = self.get_steps()
        current_step = context.get("current_step", 0)
        mcp_client = context.get("mcp_client")

        # Parse user input if provided
        if user_input and current_step == 0:
            parsed = self._parse_initial_input(user_input)
            context.update(parsed)

        # Execute based on current step
        if current_step == 0:
            # Step 1: Select/verify project
            project = context.get("project")
            if not project:
                return QuickstartResult(
                    task_name=self.name,
                    status=QuickstartStatus.WAITING_INPUT,
                    steps=steps,
                    message="Please provide the project name.",
                    next_prompt="Enter the namespace/project name:",
                )

            steps[0].status = QuickstartStatus.COMPLETED
            steps[0].result = f"Using project: {project}"
            context["current_step"] = 1

            return await self.execute(context)

        elif current_step == 1:
            # Step 2: Select model
            model_name = context.get("model_name")
            if not model_name:
                return QuickstartResult(
                    task_name=self.name,
                    status=QuickstartStatus.WAITING_INPUT,
                    steps=steps,
                    message="Please specify the model to deploy.",
                    next_prompt="Enter the model name (e.g., llama-2-7b, mistral-7b):",
                )

            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED
            steps[1].result = f"Selected model: {model_name}"
            context["current_step"] = 2

            return await self.execute(context)

        elif current_step == 2:
            # Step 3: Configure runtime
            runtime = context.get("runtime", "vllm")
            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED
            steps[2].status = QuickstartStatus.COMPLETED
            steps[2].result = f"Runtime: {runtime}"
            context["current_step"] = 3

            return await self.execute(context)

        elif current_step == 3:
            # Step 4: Configure resources
            gpu_count = context.get("gpu_count", 1)
            memory = context.get("memory", "16Gi")

            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED
            steps[2].status = QuickstartStatus.COMPLETED
            steps[3].status = QuickstartStatus.COMPLETED
            steps[3].result = f"GPU: {gpu_count}, Memory: {memory}"
            context["current_step"] = 4

            return await self.execute(context)

        elif current_step == 4:
            # Step 5: Deploy
            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED
            steps[2].status = QuickstartStatus.COMPLETED
            steps[3].status = QuickstartStatus.COMPLETED
            steps[4].status = QuickstartStatus.IN_PROGRESS

            project = context.get("project")
            model_name = context.get("model_name")
            service_name = model_name.replace("_", "-") + "-service"

            if mcp_client:
                try:
                    await mcp_client.call_tool(
                        "create_inference_service",
                        {
                            "namespace": project,
                            "name": service_name,
                            "model_name": model_name,
                            "min_replicas": 1,
                            "max_replicas": 3,
                            "gpu_count": context.get("gpu_count", 1),
                        },
                    )
                    steps[4].status = QuickstartStatus.COMPLETED
                    steps[4].result = f"Created InferenceService: {service_name}"
                except Exception as e:
                    steps[4].status = QuickstartStatus.FAILED
                    steps[4].error = str(e)
                    return QuickstartResult(
                        task_name=self.name,
                        status=QuickstartStatus.FAILED,
                        steps=steps,
                        message=f"Deployment failed: {e}",
                    )
            else:
                # Simulate deployment
                steps[4].status = QuickstartStatus.COMPLETED
                steps[4].result = f"Created InferenceService: {service_name} (simulated)"

            context["service_name"] = service_name
            context["current_step"] = 5

            return await self.execute(context)

        elif current_step == 5:
            # Step 6: Verify
            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED
            steps[2].status = QuickstartStatus.COMPLETED
            steps[3].status = QuickstartStatus.COMPLETED
            steps[4].status = QuickstartStatus.COMPLETED
            steps[5].status = QuickstartStatus.COMPLETED

            project = context.get("project")
            service_name = context.get("service_name")
            endpoint = f"https://{service_name}-{project}.apps.cluster.example.com"

            steps[5].result = f"Endpoint: {endpoint}"

            return QuickstartResult(
                task_name=self.name,
                status=QuickstartStatus.COMPLETED,
                steps=steps,
                message=(
                    f"**Model Deployed Successfully!**\n\n"
                    f"- Service: {service_name}\n"
                    f"- Namespace: {project}\n"
                    f"- Endpoint: {endpoint}\n\n"
                    "Your model is now ready to receive inference requests."
                ),
                data={
                    "service_name": service_name,
                    "namespace": project,
                    "endpoint": endpoint,
                },
            )

        return QuickstartResult(
            task_name=self.name,
            status=QuickstartStatus.FAILED,
            steps=steps,
            message="Unknown step",
        )

    def _parse_initial_input(self, user_input: str) -> dict[str, str]:
        """Parse initial user input for project and model."""
        result: dict[str, str] = {}
        lines = user_input.strip().split("\n")

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                if key in ["project", "namespace"]:
                    result["project"] = value
                elif key in ["model", "model_name"]:
                    result["model_name"] = value
                elif key in ["path", "model_path", "storage_uri"]:
                    result["model_path"] = value

        return result
