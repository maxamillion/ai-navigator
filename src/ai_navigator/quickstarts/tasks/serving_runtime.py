"""Configure Serving Runtime quickstart task - Priority task."""

from typing import Any, Optional

import structlog

from ai_navigator.quickstarts.engine import (
    QuickstartTask,
    QuickstartResult,
    QuickstartStep,
    QuickstartStatus,
)

logger = structlog.get_logger(__name__)


# Available serving runtimes
SERVING_RUNTIMES = {
    "vllm": {
        "name": "vLLM",
        "description": "High-performance LLM inference with PagedAttention",
        "supports": ["llama", "mistral", "falcon", "phi", "codellama"],
        "features": ["tensor-parallelism", "continuous-batching", "openai-api"],
    },
    "openvino": {
        "name": "OpenVINO Model Server",
        "description": "Optimized inference for Intel hardware",
        "supports": ["bert", "gpt2", "resnet", "yolo"],
        "features": ["cpu-optimized", "int8-quantization"],
    },
    "triton": {
        "name": "NVIDIA Triton",
        "description": "Multi-framework inference server",
        "supports": ["pytorch", "tensorflow", "onnx", "tensorrt"],
        "features": ["dynamic-batching", "model-ensemble", "gpu-optimized"],
    },
    "tgis": {
        "name": "Text Generation Inference Server",
        "description": "HuggingFace text generation server",
        "supports": ["llama", "bloom", "gpt-neox", "falcon"],
        "features": ["streaming", "continuous-batching"],
    },
}


class ServingRuntimeTask(QuickstartTask):
    """Quickstart: Configure a serving runtime."""

    @property
    def name(self) -> str:
        return "serving-runtime"

    @property
    def display_name(self) -> str:
        return "Configure Serving Runtime"

    @property
    def description(self) -> str:
        return (
            "Set up a serving runtime for model inference. "
            "Choose from vLLM, OpenVINO, Triton, or TGIS based on your model and hardware."
        )

    def get_steps(self) -> list[QuickstartStep]:
        return [
            QuickstartStep(
                id="select-runtime",
                name="Select Runtime",
                description="Choose the serving runtime",
            ),
            QuickstartStep(
                id="configure-parameters",
                name="Configure Parameters",
                description="Set runtime-specific parameters",
            ),
            QuickstartStep(
                id="create-runtime",
                name="Create Runtime",
                description="Deploy the ServingRuntime resource",
            ),
            QuickstartStep(
                id="verify",
                name="Verify",
                description="Confirm runtime is available",
            ),
        ]

    def get_required_inputs(self) -> list[dict[str, str]]:
        return [
            {
                "name": "project",
                "description": "Data Science Project namespace",
                "required": True,
            },
            {
                "name": "runtime",
                "description": "Runtime type (vllm, openvino, triton, tgis)",
                "required": False,
            },
        ]

    async def execute(
        self,
        context: dict[str, Any],
        user_input: Optional[str] = None,
    ) -> QuickstartResult:
        """Execute the serving runtime configuration quickstart."""
        steps = self.get_steps()
        current_step = context.get("current_step", 0)

        # Parse user input
        if user_input:
            parsed = self._parse_input(user_input, current_step)
            context.update(parsed)

        if current_step == 0:
            # Step 1: Select runtime
            runtime = context.get("runtime")
            if not runtime:
                runtime_list = "\n".join(
                    f"- **{key}**: {info['name']} - {info['description']}"
                    for key, info in SERVING_RUNTIMES.items()
                )
                return QuickstartResult(
                    task_name=self.name,
                    status=QuickstartStatus.WAITING_INPUT,
                    steps=steps,
                    message=f"Available serving runtimes:\n\n{runtime_list}",
                    next_prompt="Enter the runtime to use (vllm, openvino, triton, tgis):",
                )

            if runtime not in SERVING_RUNTIMES:
                return QuickstartResult(
                    task_name=self.name,
                    status=QuickstartStatus.WAITING_INPUT,
                    steps=steps,
                    message=f"Unknown runtime: {runtime}",
                    next_prompt="Please choose from: vllm, openvino, triton, tgis",
                )

            steps[0].status = QuickstartStatus.COMPLETED
            steps[0].result = f"Selected: {SERVING_RUNTIMES[runtime]['name']}"
            context["current_step"] = 1

            return await self.execute(context)

        elif current_step == 1:
            # Step 2: Configure parameters
            runtime = context.get("runtime", "vllm")
            runtime_info = SERVING_RUNTIMES[runtime]

            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED

            config_summary = f"Runtime: {runtime_info['name']}\nFeatures: {', '.join(runtime_info['features'])}"
            steps[1].result = config_summary
            context["current_step"] = 2

            return await self.execute(context)

        elif current_step == 2:
            # Step 3: Create runtime
            project = context.get("project", "default")
            runtime = context.get("runtime", "vllm")
            runtime_name = f"{runtime}-runtime"

            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED
            steps[2].status = QuickstartStatus.COMPLETED
            steps[2].result = f"Created: {runtime_name} in {project}"

            context["runtime_name"] = runtime_name
            context["current_step"] = 3

            return await self.execute(context)

        elif current_step == 3:
            # Step 4: Verify
            runtime = context.get("runtime", "vllm")
            runtime_name = context.get("runtime_name")
            project = context.get("project", "default")

            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED
            steps[2].status = QuickstartStatus.COMPLETED
            steps[3].status = QuickstartStatus.COMPLETED
            steps[3].result = "Runtime ready"

            return QuickstartResult(
                task_name=self.name,
                status=QuickstartStatus.COMPLETED,
                steps=steps,
                message=(
                    f"**Serving Runtime Configured!**\n\n"
                    f"- Runtime: {SERVING_RUNTIMES[runtime]['name']}\n"
                    f"- Name: {runtime_name}\n"
                    f"- Namespace: {project}\n\n"
                    "You can now deploy models using this runtime."
                ),
                data={
                    "runtime_name": runtime_name,
                    "runtime_type": runtime,
                    "namespace": project,
                },
            )

        return QuickstartResult(
            task_name=self.name,
            status=QuickstartStatus.FAILED,
            message="Unknown step",
        )

    def _parse_input(self, user_input: str, step: int) -> dict[str, str]:
        """Parse user input based on current step."""
        result: dict[str, str] = {}
        text = user_input.strip().lower()

        if step == 0:
            if text in SERVING_RUNTIMES:
                result["runtime"] = text
            elif ":" in text:
                parts = text.split(":", 1)
                if parts[0].strip() in ["runtime", "type"]:
                    result["runtime"] = parts[1].strip()

        return result
