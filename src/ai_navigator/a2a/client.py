"""A2A Client for agent-to-agent communication."""

import structlog
from httpx import AsyncClient, HTTPStatusError, RequestError
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ai_navigator.a2a.agent_card import AgentCard
from ai_navigator.a2a.message import Message
from ai_navigator.a2a.task import Task, TaskState

logger = structlog.get_logger(__name__)


class A2AClientError(Exception):
    """Error during A2A communication."""

    pass


class SendMessageRequest(BaseModel):
    """Request to send a message to an agent."""

    message: Message = Field(..., description="Message to send")
    task_id: str | None = Field(default=None, description="Existing task ID")
    session_id: str | None = Field(default=None, description="Session ID")
    skill_id: str | None = Field(default=None, description="Target skill ID")


class A2AClient:
    """Client for A2A protocol communication between agents."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize A2A client.

        Args:
            base_url: Base URL of the target agent
            timeout: Request timeout in seconds
            headers: Additional headers to include
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = headers or {}
        self._client: AsyncClient | None = None

    async def __aenter__(self) -> "A2AClient":
        """Enter async context."""
        self._client = AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._headers,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context."""
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> AsyncClient:
        """Get the HTTP client."""
        if self._client is None:
            self._client = AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self._headers,
            )
        return self._client

    @retry(
        retry=retry_if_exception_type((RequestError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def get_agent_card(self) -> AgentCard:
        """
        Fetch the agent card from /.well-known/agent.json.

        Returns:
            AgentCard: The agent's capabilities and skills

        Raises:
            A2AClientError: If the request fails
        """
        try:
            response = await self.client.get("/.well-known/agent.json")
            response.raise_for_status()
            data = response.json()
            return AgentCard(**data)
        except HTTPStatusError as e:
            raise A2AClientError(f"Failed to fetch agent card: {e}") from e
        except RequestError as e:
            raise A2AClientError(f"Connection error: {e}") from e

    @retry(
        retry=retry_if_exception_type((RequestError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def send_message(
        self,
        message: Message,
        task_id: str | None = None,
        session_id: str | None = None,
        skill_id: str | None = None,
    ) -> Task:
        """
        Send a message to the agent.

        Args:
            message: The message to send
            task_id: Optional existing task ID to continue
            session_id: Optional session ID for task grouping
            skill_id: Optional target skill ID

        Returns:
            Task: The task created or updated by the message

        Raises:
            A2AClientError: If the request fails
        """
        request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": message.model_dump(),
            },
        }

        if task_id:
            request["params"]["taskId"] = task_id
        if session_id:
            request["params"]["sessionId"] = session_id
        if skill_id:
            request["params"]["skillId"] = skill_id

        try:
            response = await self.client.post("/", json=request)
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise A2AClientError(f"Agent error: {result['error']}")

            task_data = result.get("result", {})
            return Task(
                id=task_data.get("id", ""),
                session_id=task_data.get("sessionId"),
                metadata=task_data,
            )
        except HTTPStatusError as e:
            raise A2AClientError(f"Request failed: {e}") from e
        except RequestError as e:
            raise A2AClientError(f"Connection error: {e}") from e

    async def send_text(
        self,
        text: str,
        task_id: str | None = None,
        session_id: str | None = None,
        skill_id: str | None = None,
    ) -> Task:
        """
        Send a text message to the agent.

        Args:
            text: Text content to send
            task_id: Optional existing task ID
            session_id: Optional session ID
            skill_id: Optional target skill ID

        Returns:
            Task: The resulting task
        """
        return await self.send_message(
            message=Message.user(text),
            task_id=task_id,
            session_id=session_id,
            skill_id=skill_id,
        )

    @retry(
        retry=retry_if_exception_type((RequestError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def get_task(self, task_id: str) -> Task:
        """
        Get the current state of a task.

        Args:
            task_id: The task ID to query

        Returns:
            Task: The current task state

        Raises:
            A2AClientError: If the request fails
        """
        request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tasks/get",
            "params": {"id": task_id},
        }

        try:
            response = await self.client.post("/", json=request)
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise A2AClientError(f"Agent error: {result['error']}")

            task_data = result.get("result", {})
            return Task(
                id=task_data.get("id", task_id),
                session_id=task_data.get("sessionId"),
                metadata=task_data,
            )
        except HTTPStatusError as e:
            raise A2AClientError(f"Request failed: {e}") from e
        except RequestError as e:
            raise A2AClientError(f"Connection error: {e}") from e

    @retry(
        retry=retry_if_exception_type((RequestError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def cancel_task(self, task_id: str) -> Task:
        """
        Cancel a running task.

        Args:
            task_id: The task ID to cancel

        Returns:
            Task: The canceled task

        Raises:
            A2AClientError: If the request fails
        """
        request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tasks/cancel",
            "params": {"id": task_id},
        }

        try:
            response = await self.client.post("/", json=request)
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise A2AClientError(f"Agent error: {result['error']}")

            task_data = result.get("result", {})
            return Task(
                id=task_data.get("id", task_id),
                session_id=task_data.get("sessionId"),
                metadata=task_data,
            )
        except HTTPStatusError as e:
            raise A2AClientError(f"Request failed: {e}") from e
        except RequestError as e:
            raise A2AClientError(f"Connection error: {e}") from e

    async def health_check(self) -> bool:
        """
        Check if the agent is healthy.

        Returns:
            bool: True if agent is healthy
        """
        try:
            response = await self.client.get("/healthz")
            return response.status_code == 200
        except RequestError:
            return False

    async def wait_for_completion(
        self,
        task_id: str,
        poll_interval: float = 1.0,
        max_polls: int = 300,
    ) -> Task:
        """
        Wait for a task to reach a terminal state.

        Args:
            task_id: The task ID to wait for
            poll_interval: Seconds between polls
            max_polls: Maximum number of polls

        Returns:
            Task: The completed task

        Raises:
            A2AClientError: If max polls exceeded or task fails
        """
        import asyncio

        for _ in range(max_polls):
            task = await self.get_task(task_id)

            if task.status.state == TaskState.COMPLETED:
                return task
            elif task.status.state == TaskState.FAILED:
                error_msg = "Task failed"
                if task.status.message:
                    error_msg = task.status.message.get_text()
                raise A2AClientError(error_msg)
            elif task.status.state == TaskState.CANCELED:
                raise A2AClientError("Task was canceled")
            elif task.status.state == TaskState.INPUT_REQUIRED:
                # Return for caller to handle
                return task

            await asyncio.sleep(poll_interval)

        raise A2AClientError(f"Task did not complete within {max_polls} polls")
