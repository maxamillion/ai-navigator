"""Quickstart engine for running guided tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class QuickstartStatus(str, Enum):
    """Quickstart execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QuickstartStep:
    """A single step in a quickstart task."""

    id: str
    name: str
    description: str
    status: QuickstartStatus = QuickstartStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class QuickstartResult:
    """Result of quickstart execution."""

    task_name: str
    status: QuickstartStatus
    steps: list[QuickstartStep] = field(default_factory=list)
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    next_prompt: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.status in {QuickstartStatus.COMPLETED, QuickstartStatus.FAILED}


class QuickstartTask(ABC):
    """Abstract base class for quickstart tasks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Task name for identification."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable task name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Task description."""
        ...

    @abstractmethod
    def get_steps(self) -> list[QuickstartStep]:
        """Define the steps for this task."""
        ...

    @abstractmethod
    async def execute(
        self,
        context: dict[str, Any],
        user_input: Optional[str] = None,
    ) -> QuickstartResult:
        """Execute the quickstart task."""
        ...

    def get_required_inputs(self) -> list[dict[str, str]]:
        """Get list of required inputs for this task."""
        return []


class QuickstartEngine:
    """Engine for running quickstart tasks."""

    def __init__(self, mcp_client: Optional[object] = None) -> None:
        """Initialize engine with optional MCP client."""
        self._mcp_client = mcp_client
        self._tasks: dict[str, QuickstartTask] = {}
        self._active_sessions: dict[str, dict[str, Any]] = {}

    def register_task(self, task: QuickstartTask) -> None:
        """Register a quickstart task."""
        self._tasks[task.name] = task
        logger.debug("Registered quickstart task", task=task.name)

    def list_tasks(self) -> list[dict[str, str]]:
        """List all available quickstart tasks."""
        return [
            {
                "name": task.name,
                "display_name": task.display_name,
                "description": task.description,
            }
            for task in self._tasks.values()
        ]

    def get_task(self, task_name: str) -> Optional[QuickstartTask]:
        """Get a task by name."""
        return self._tasks.get(task_name)

    async def start_task(
        self,
        task_name: str,
        session_id: str,
        initial_context: Optional[dict[str, Any]] = None,
    ) -> QuickstartResult:
        """Start a new quickstart task session."""
        task = self._tasks.get(task_name)
        if not task:
            return QuickstartResult(
                task_name=task_name,
                status=QuickstartStatus.FAILED,
                message=f"Unknown task: {task_name}",
            )

        # Initialize session
        context = initial_context or {}
        context["session_id"] = session_id
        context["task_name"] = task_name
        context["mcp_client"] = self._mcp_client
        context["current_step"] = 0

        self._active_sessions[session_id] = context

        # Get task introduction
        steps = task.get_steps()
        required_inputs = task.get_required_inputs()

        if required_inputs:
            # Need to collect inputs first
            prompt_lines = [
                f"**{task.display_name}**",
                "",
                task.description,
                "",
                "Please provide the following information:",
            ]
            for input_spec in required_inputs:
                prompt_lines.append(f"- **{input_spec['name']}**: {input_spec['description']}")

            return QuickstartResult(
                task_name=task_name,
                status=QuickstartStatus.WAITING_INPUT,
                steps=steps,
                message="\n".join(prompt_lines),
                next_prompt="Provide the required values, or type 'cancel' to abort.",
            )

        # Execute directly if no inputs needed
        return await task.execute(context)

    async def continue_task(
        self,
        session_id: str,
        user_input: str,
    ) -> QuickstartResult:
        """Continue an active quickstart task."""
        context = self._active_sessions.get(session_id)
        if not context:
            return QuickstartResult(
                task_name="unknown",
                status=QuickstartStatus.FAILED,
                message="No active session found",
            )

        task_name = context.get("task_name", "")
        task = self._tasks.get(task_name)

        if not task:
            return QuickstartResult(
                task_name=task_name,
                status=QuickstartStatus.FAILED,
                message="Task not found",
            )

        # Check for cancel
        if user_input.lower().strip() == "cancel":
            del self._active_sessions[session_id]
            return QuickstartResult(
                task_name=task_name,
                status=QuickstartStatus.CANCELLED,
                message="Task cancelled",
            )

        # Execute with user input
        result = await task.execute(context, user_input)

        # Clean up if complete
        if result.is_complete:
            self._active_sessions.pop(session_id, None)

        return result

    async def cancel_task(self, session_id: str) -> bool:
        """Cancel an active task session."""
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            return True
        return False

    def get_session_status(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get status of an active session."""
        context = self._active_sessions.get(session_id)
        if not context:
            return None

        return {
            "session_id": session_id,
            "task_name": context.get("task_name"),
            "current_step": context.get("current_step", 0),
        }
