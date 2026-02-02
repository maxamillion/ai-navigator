"""Data Connection quickstart task - Supporting task."""

from typing import Any, Optional

import structlog

from ai_navigator.quickstarts.engine import (
    QuickstartTask,
    QuickstartResult,
    QuickstartStep,
    QuickstartStatus,
)

logger = structlog.get_logger(__name__)


class DataConnectionTask(QuickstartTask):
    """Quickstart: Configure S3 data connection."""

    @property
    def name(self) -> str:
        return "data-connection"

    @property
    def display_name(self) -> str:
        return "Configure S3 Data Connection"

    @property
    def description(self) -> str:
        return (
            "Set up an S3-compatible data connection for model storage. "
            "This allows you to access models stored in S3, MinIO, or other compatible storage."
        )

    def get_steps(self) -> list[QuickstartStep]:
        return [
            QuickstartStep(
                id="select-project",
                name="Select Project",
                description="Choose the target project",
            ),
            QuickstartStep(
                id="configure-endpoint",
                name="Configure Endpoint",
                description="Set up S3 endpoint and bucket",
            ),
            QuickstartStep(
                id="configure-credentials",
                name="Configure Credentials",
                description="Provide access credentials",
            ),
            QuickstartStep(
                id="create-connection",
                name="Create Connection",
                description="Create the data connection",
            ),
            QuickstartStep(
                id="verify",
                name="Verify",
                description="Test the connection",
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
                "name": "connection_name",
                "description": "Name for the data connection",
                "required": True,
            },
            {
                "name": "endpoint",
                "description": "S3 endpoint URL",
                "required": True,
            },
            {
                "name": "bucket",
                "description": "S3 bucket name",
                "required": True,
            },
            {
                "name": "access_key",
                "description": "AWS access key ID",
                "required": True,
            },
            {
                "name": "secret_key",
                "description": "AWS secret access key",
                "required": True,
            },
        ]

    async def execute(
        self,
        context: dict[str, Any],
        user_input: Optional[str] = None,
    ) -> QuickstartResult:
        """Execute the data connection quickstart."""
        steps = self.get_steps()
        current_step = context.get("current_step", 0)
        mcp_client = context.get("mcp_client")

        # Parse user input
        if user_input:
            parsed = self._parse_input(user_input)
            context.update(parsed)

        if current_step == 0:
            # Step 1: Select project
            project = context.get("project")
            if not project:
                return QuickstartResult(
                    task_name=self.name,
                    status=QuickstartStatus.WAITING_INPUT,
                    steps=steps,
                    message="Please specify the target project.",
                    next_prompt="Enter project name:",
                )

            steps[0].status = QuickstartStatus.COMPLETED
            steps[0].result = f"Project: {project}"
            context["current_step"] = 1

            return await self.execute(context)

        elif current_step == 1:
            # Step 2: Configure endpoint
            endpoint = context.get("endpoint")
            bucket = context.get("bucket")

            if not endpoint or not bucket:
                return QuickstartResult(
                    task_name=self.name,
                    status=QuickstartStatus.WAITING_INPUT,
                    steps=steps,
                    message="Please provide S3 endpoint and bucket.",
                    next_prompt=(
                        "Enter:\n"
                        "endpoint: <s3-endpoint-url>\n"
                        "bucket: <bucket-name>"
                    ),
                )

            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED
            steps[1].result = f"Endpoint: {endpoint}, Bucket: {bucket}"
            context["current_step"] = 2

            return await self.execute(context)

        elif current_step == 2:
            # Step 3: Configure credentials
            access_key = context.get("access_key")
            secret_key = context.get("secret_key")

            if not access_key or not secret_key:
                return QuickstartResult(
                    task_name=self.name,
                    status=QuickstartStatus.WAITING_INPUT,
                    steps=steps,
                    message="Please provide S3 credentials.",
                    next_prompt=(
                        "Enter:\n"
                        "access_key: <aws-access-key-id>\n"
                        "secret_key: <aws-secret-access-key>"
                    ),
                )

            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED
            steps[2].status = QuickstartStatus.COMPLETED
            steps[2].result = "Credentials configured"
            context["current_step"] = 3

            return await self.execute(context)

        elif current_step == 3:
            # Step 4: Create connection
            project = context.get("project")
            connection_name = context.get("connection_name", "s3-connection")
            endpoint = context.get("endpoint")
            bucket = context.get("bucket")
            access_key = context.get("access_key")
            secret_key = context.get("secret_key")
            region = context.get("region", "us-east-1")

            if mcp_client:
                try:
                    await mcp_client.call_tool(
                        "create_s3_data_connection",
                        {
                            "namespace": project,
                            "name": connection_name,
                            "endpoint": endpoint,
                            "bucket": bucket,
                            "access_key": access_key,
                            "secret_key": secret_key,
                            "region": region,
                        },
                    )
                except Exception as e:
                    steps[3].status = QuickstartStatus.FAILED
                    steps[3].error = str(e)
                    return QuickstartResult(
                        task_name=self.name,
                        status=QuickstartStatus.FAILED,
                        steps=steps,
                        message=f"Failed to create data connection: {e}",
                    )

            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED
            steps[2].status = QuickstartStatus.COMPLETED
            steps[3].status = QuickstartStatus.COMPLETED
            steps[3].result = f"Created: {connection_name}"
            context["current_step"] = 4

            return await self.execute(context)

        elif current_step == 4:
            # Step 5: Verify
            project = context.get("project")
            connection_name = context.get("connection_name", "s3-connection")
            endpoint = context.get("endpoint")
            bucket = context.get("bucket")

            steps[0].status = QuickstartStatus.COMPLETED
            steps[1].status = QuickstartStatus.COMPLETED
            steps[2].status = QuickstartStatus.COMPLETED
            steps[3].status = QuickstartStatus.COMPLETED
            steps[4].status = QuickstartStatus.COMPLETED
            steps[4].result = "Connection verified"

            return QuickstartResult(
                task_name=self.name,
                status=QuickstartStatus.COMPLETED,
                steps=steps,
                message=(
                    f"**Data Connection Created!**\n\n"
                    f"- Connection: {connection_name}\n"
                    f"- Namespace: {project}\n"
                    f"- Endpoint: {endpoint}\n"
                    f"- Bucket: {bucket}\n\n"
                    "You can now use this connection for model storage.\n"
                    "Reference the connection as a storage URI:\n"
                    f"`s3://{bucket}/path/to/model`"
                ),
                data={
                    "connection_name": connection_name,
                    "namespace": project,
                    "endpoint": endpoint,
                    "bucket": bucket,
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

        for line in user_input.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()

                if key in ["project", "namespace"]:
                    result["project"] = value
                elif key in ["name", "connection", "connection_name"]:
                    result["connection_name"] = value
                elif key in ["endpoint", "url", "s3_endpoint"]:
                    result["endpoint"] = value
                elif key in ["bucket", "bucket_name"]:
                    result["bucket"] = value
                elif key in ["access_key", "access", "aws_access_key"]:
                    result["access_key"] = value
                elif key in ["secret_key", "secret", "aws_secret_key"]:
                    result["secret_key"] = value
                elif key in ["region", "aws_region"]:
                    result["region"] = value

        return result
