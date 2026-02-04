"""A2A Task lifecycle management."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ai_navigator.a2a.message import Artifact, Message


class TaskState(str, Enum):
    """A2A Task states."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TaskStatus(BaseModel):
    """Status information for a task."""

    state: TaskState = Field(..., description="Current task state")
    message: Message | None = Field(default=None, description="Status message from agent")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Status update time")

    @classmethod
    def working(cls, message: str | None = None) -> "TaskStatus":
        """Create a working status."""
        return cls(
            state=TaskState.WORKING,
            message=Message.agent(message) if message else None,
        )

    @classmethod
    def completed(cls, message: str | None = None) -> "TaskStatus":
        """Create a completed status."""
        return cls(
            state=TaskState.COMPLETED,
            message=Message.agent(message) if message else None,
        )

    @classmethod
    def failed(cls, error: str) -> "TaskStatus":
        """Create a failed status."""
        return cls(
            state=TaskState.FAILED,
            message=Message.agent(error),
        )

    @classmethod
    def input_required(cls, prompt: str) -> "TaskStatus":
        """Create an input-required status."""
        return cls(
            state=TaskState.INPUT_REQUIRED,
            message=Message.agent(prompt),
        )


class Task(BaseModel):
    """A2A Task representing a unit of work."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique task ID")
    session_id: str | None = Field(
        default=None,
        description="Session ID for task continuity",
    )
    status: TaskStatus = Field(
        default_factory=lambda: TaskStatus(state=TaskState.SUBMITTED),
        description="Current task status",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="Conversation history",
    )
    artifacts: list[Artifact] = Field(
        default_factory=list,
        description="Artifacts produced during execution",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Task metadata",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Task creation time",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update time",
    )

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    def add_artifact(self, artifact: Artifact) -> None:
        """Add an artifact to the task."""
        self.artifacts.append(artifact)
        self.updated_at = datetime.utcnow()

    def set_working(self, message: str | None = None) -> None:
        """Transition task to working state."""
        self.status = TaskStatus.working(message)
        self.updated_at = datetime.utcnow()

    def set_completed(self, message: str | None = None) -> None:
        """Transition task to completed state."""
        self.status = TaskStatus.completed(message)
        self.updated_at = datetime.utcnow()

    def set_failed(self, error: str) -> None:
        """Transition task to failed state."""
        self.status = TaskStatus.failed(error)
        self.updated_at = datetime.utcnow()

    def set_input_required(self, prompt: str) -> None:
        """Transition task to input-required state."""
        self.status = TaskStatus.input_required(prompt)
        self.updated_at = datetime.utcnow()

    @property
    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status.state in {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELED,
        }

    def model_dump_jsonrpc(self) -> dict[str, Any]:
        """Dump task for JSON-RPC response."""
        return {
            "id": self.id,
            "sessionId": self.session_id,
            "status": {
                "state": self.status.state.value,
                "message": self.status.message.model_dump() if self.status.message else None,
                "timestamp": self.status.timestamp.isoformat(),
            },
            "artifacts": [a.model_dump() for a in self.artifacts],
        }


class TaskStore:
    """In-memory task storage."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def create(self, task: Task) -> Task:
        """Store a new task."""
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        """Retrieve a task by ID."""
        return self._tasks.get(task_id)

    def update(self, task: Task) -> Task:
        """Update an existing task."""
        self._tasks[task.id] = task
        return task

    def delete(self, task_id: str) -> bool:
        """Delete a task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    def list_by_session(self, session_id: str) -> list[Task]:
        """List all tasks for a session."""
        return [t for t in self._tasks.values() if t.session_id == session_id]
