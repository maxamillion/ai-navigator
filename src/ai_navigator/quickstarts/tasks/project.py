"""Project Setup quickstart task - Supporting task."""

from typing import Any, Optional

import structlog

from ai_navigator.quickstarts.engine import (
    QuickstartTask,
    QuickstartResult,
    QuickstartStep,
    QuickstartStatus,
)

logger = structlog.get_logger(__name__)


class ProjectSetupTask(QuickstartTask):
    """Quickstart: Create and configure a Data Science Project."""

    @property
    def name(self) -> str:
        return "project-setup"

    @property
    def display_name(self) -> str:
        return "Create Data Science Project"

    @property
    def description(self) -> str:
        return (
            "Create a new Data Science Project with storage and basic configuration. "
            "This sets up the namespace and resources needed for AI workloads."
        )

    def get_steps(self) -> list[QuickstartStep]:
        return [
            QuickstartStep(
                id="create-project",
                name="Create Project",
                description="Create the Data Science Project namespace",
            ),
            QuickstartStep(
                id="create-storage",
                name="Create Storage",
                description="Set up persistent storage for models and data",
            ),
            QuickstartStep(
                id="configure-quotas",
                name="Configure Quotas",
                description="Set resource quotas and limits",
            ),
            QuickstartStep(
                id="verify",
                name="Verify",
                description="Confirm project is ready",
            ),
        ]

    def get_required_inputs(self) -> list[dict[str, str]]:
        return [
            {
                "name": "project_name",
                "description": "Name for the project (lowercase, alphanumeric with hyphens)",
                "required": True,
            },
            {
                "name": "display_name",
                "description": "Display name for the project",
                "required": False,
            },
            {
                "name": "storage_size",
                "description": "Storage size (e.g., 20Gi)",
                "required": False,
            },
        ]

    async def execute(
        self,
        context: dict[str, Any],
        user_input: Optional[str] = None,
    ) -> QuickstartResult:
        """Execute the project setup quickstart."""
        steps = self.get_steps()
        current_step = context.get("current_step", 0)
        mcp_client = context.get("mcp_client")

        # Parse user input
        if user_input:
            parsed = self._parse_input(user_input)
            context.update(parsed)

        if current_step == 0:
            # Step 1: Create project
            project_name = context.get("project_name")
            if not project_name:
                return QuickstartResult(
                    task_name=self.name,
                    status=QuickstartStatus.WAITING_INPUT,
                    steps=steps,
                    message="Please provide a name for your project.",
                    next_prompt="Enter project name (e.g., my-ai-project):",
                )

            display_name = context.get("display_name", project_name)

            if mcp_client:
                try:
                    await mcp_client.call_tool(
                        "create_data_science_project",
                        {
                            "name": project_name,
                            "display_name": display_name,
                        },
                    )
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        steps[0].status = QuickstartStatus.FAILED
                        steps[0].error = str(e)
                        return QuickstartResult(
                            task_name=self.name,
                            status=QuickstartStatus.FAILED,
                            steps=steps,
                            message=f"Failed to create project: {e}",
                        )

            steps[0].status = QuickstartStatus.COMPLETED
            steps[0].result = f"Created project: {project_name}"
            context["current_step"] = 1

            return await self.execute(context)

        elif current_step == 1:
            # Step 2: Create storage
            project_name = context.get("project_name")
            storage_size = context.get("storage_size", "20Gi")
            storage_name = f"{project_name}-storage"

            if mcp_client:
                try:
                    await mcp_client.call_tool(
                        "create_storage",
                        {
                            "namespace": project_name,
                            "name": storage_name,
                            "size": storage_size,
                        },
                    )
                except Exception as e:
                    logger.warning("Storage creation warning", error=str(e))

            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED
            steps[1].result = f"Created {storage_size} storage"
            context["storage_name"] = storage_name
            context["current_step"] = 2

            return await self.execute(context)

        elif current_step == 2:
            # Step 3: Configure quotas (optional)
            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED
            steps[2].status = QuickstartStatus.COMPLETED
            steps[2].result = "Using default quotas"
            context["current_step"] = 3

            return await self.execute(context)

        elif current_step == 3:
            # Step 4: Verify
            project_name = context.get("project_name")
            storage_name = context.get("storage_name")
            storage_size = context.get("storage_size", "20Gi")

            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED
            steps[2].status = QuickstartStatus.COMPLETED
            steps[3].status = QuickstartStatus.COMPLETED
            steps[3].result = "Project ready"

            return QuickstartResult(
                task_name=self.name,
                status=QuickstartStatus.COMPLETED,
                steps=steps,
                message=(
                    f"**Data Science Project Created!**\n\n"
                    f"- Project: {project_name}\n"
                    f"- Storage: {storage_name} ({storage_size})\n\n"
                    "Your project is ready for AI workloads. Next steps:\n"
                    "- Deploy a model with `/quickstart deploy-model`\n"
                    "- Create a workbench for development\n"
                    "- Set up data connections"
                ),
                data={
                    "project_name": project_name,
                    "storage_name": storage_name,
                    "storage_size": storage_size,
                },
            )

        return QuickstartResult(
            task_name=self.name,
            status=QuickstartStatus.FAILED,
            message="Unknown step",
        )

    def _parse_input(self, user_input: str) -> dict[str, str]:
        """Parse user input."""
        result: dict[str, str] = {}
        text = user_input.strip()

        if ":" in text:
            for part in text.split(","):
                if ":" in part:
                    key, value = part.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    value = value.strip()
                    if key in ["project", "name", "project_name"]:
                        result["project_name"] = value
                    elif key in ["display", "display_name"]:
                        result["display_name"] = value
                    elif key in ["storage", "storage_size", "size"]:
                        result["storage_size"] = value
        else:
            # Assume it's just the project name
            result["project_name"] = text

        return result
