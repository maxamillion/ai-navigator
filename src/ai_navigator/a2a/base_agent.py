"""Base Agent class with A2A protocol compliance."""

from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from ai_navigator.a2a.agent_card import AgentCard, AgentCardBuilder
from ai_navigator.a2a.message import Message, MessageRole, TextPart
from ai_navigator.a2a.skills import SkillInput, SkillRegistry, SkillResult
from ai_navigator.a2a.task import Task, TaskState, TaskStore
from ai_navigator.config import AgentSettings

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "a2a_requests_total",
    "Total A2A requests",
    ["method", "status"],
)
REQUEST_LATENCY = Histogram(
    "a2a_request_latency_seconds",
    "A2A request latency",
    ["method"],
)
TASK_COUNT = Counter(
    "a2a_tasks_total",
    "Total tasks created",
    ["state"],
)


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request."""

    jsonrpc: str = Field(default="2.0")
    id: str | int | None = Field(default=None)
    method: str = Field(...)
    params: dict[str, Any] = Field(default_factory=dict)


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error."""

    code: int
    message: str
    data: Any | None = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response."""

    jsonrpc: str = Field(default="2.0")
    id: str | int | None = Field(default=None)
    result: Any | None = None
    error: JSONRPCError | None = None


class BaseAgent:
    """
    Base class for A2A-compliant agents.

    Provides:
    - Agent card at /.well-known/agent.json
    - JSON-RPC message handling at /
    - Health endpoint at /healthz
    - Metrics endpoint at /metrics
    - Skill registry and dispatch
    """

    def __init__(
        self,
        name: str,
        description: str,
        settings: AgentSettings | None = None,
    ) -> None:
        """
        Initialize the base agent.

        Args:
            name: Agent name
            description: Agent description
            settings: Agent configuration
        """
        self.name = name
        self.description = description
        self.settings = settings or AgentSettings()
        self.skills = SkillRegistry()
        self.tasks = TaskStore()
        self._app: FastAPI | None = None

    def build_agent_card(self) -> AgentCard:
        """Build the agent card for this agent."""
        builder = (
            AgentCardBuilder(
                name=self.name,
                description=self.description,
                url=self.settings.endpoint,
            )
            .version(self.settings.version)
            .provider("Red Hat", "https://www.redhat.com")
            .capabilities(
                streaming=False,
                push_notifications=False,
                state_transition_history=True,
            )
        )

        # Add skills from registry
        for skill in self.skills.list():
            builder.skill(
                id=skill.id,
                name=skill.name,
                description=skill.description,
                tags=skill.tags,
                examples=skill.examples,
                input_schema=skill.input_schema,
                output_schema=skill.output_schema,
            )

        return builder.build()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """FastAPI lifespan context manager."""
        logger.info("agent_starting", name=self.name)
        await self.on_startup()
        yield
        logger.info("agent_stopping", name=self.name)
        await self.on_shutdown()

    async def on_startup(self) -> None:
        """Hook called on agent startup. Override in subclasses."""
        pass

    async def on_shutdown(self) -> None:
        """Hook called on agent shutdown. Override in subclasses."""
        pass

    def create_app(self) -> FastAPI:
        """Create the FastAPI application."""
        app = FastAPI(
            title=self.name,
            description=self.description,
            version=self.settings.version,
            lifespan=self.lifespan,
        )

        # Register routes
        @app.get("/.well-known/agent.json")
        async def agent_card() -> dict[str, Any]:
            """Return the agent card."""
            return self.build_agent_card().model_dump_json_ld()

        @app.get("/healthz")
        async def health() -> dict[str, str]:
            """Health check endpoint."""
            return {"status": "healthy"}

        @app.get("/readyz")
        async def ready() -> dict[str, str]:
            """Readiness check endpoint."""
            return {"status": "ready"}

        @app.get("/metrics")
        async def metrics() -> Response:
            """Prometheus metrics endpoint."""
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )

        @app.post("/")
        async def handle_jsonrpc(request: Request) -> JSONResponse:
            """Handle JSON-RPC requests."""
            try:
                body = await request.json()
                rpc_request = JSONRPCRequest(**body)

                with REQUEST_LATENCY.labels(method=rpc_request.method).time():
                    result = await self._dispatch_method(rpc_request)

                REQUEST_COUNT.labels(
                    method=rpc_request.method,
                    status="success",
                ).inc()

                return JSONResponse(
                    content=JSONRPCResponse(
                        id=rpc_request.id,
                        result=result,
                    ).model_dump()
                )
            except Exception as e:
                logger.exception("jsonrpc_error", error=str(e))
                REQUEST_COUNT.labels(
                    method=body.get("method", "unknown") if isinstance(body, dict) else "unknown",
                    status="error",
                ).inc()

                return JSONResponse(
                    content=JSONRPCResponse(
                        id=body.get("id") if isinstance(body, dict) else None,
                        error=JSONRPCError(
                            code=-32603,
                            message=str(e),
                        ),
                    ).model_dump(),
                    status_code=200,  # JSON-RPC errors return 200
                )

        self._app = app
        return app

    async def _dispatch_method(self, request: JSONRPCRequest) -> Any:
        """Dispatch a JSON-RPC method."""
        method = request.method
        params = request.params

        if method == "message/send":
            return await self._handle_message_send(params)
        elif method == "tasks/get":
            return await self._handle_task_get(params)
        elif method == "tasks/cancel":
            return await self._handle_task_cancel(params)
        else:
            raise ValueError(f"Unknown method: {method}")

    async def _handle_message_send(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle message/send method."""
        message_data = params.get("message", {})
        task_id = params.get("taskId")
        session_id = params.get("sessionId")
        skill_id = params.get("skillId")

        # Parse message
        parts = []
        for part_data in message_data.get("parts", []):
            if part_data.get("type") == "text":
                parts.append(TextPart(text=part_data.get("text", "")))

        message = Message(
            role=MessageRole(message_data.get("role", "user")),
            parts=parts,
            metadata=message_data.get("metadata", {}),
        )

        # Get or create task
        if task_id:
            task = self.tasks.get(task_id)
            if task is None:
                raise ValueError(f"Task not found: {task_id}")
        else:
            task = Task(session_id=session_id)
            self.tasks.create(task)
            TASK_COUNT.labels(state="created").inc()

        # Add message to task
        task.add_message(message)
        task.set_working("Processing request")

        # Dispatch to skill or default handler
        try:
            if skill_id:
                skill_input = SkillInput(
                    task=task,
                    message=message,
                    params=params.get("params", {}),
                )
                result = await self.skills.dispatch(skill_id, skill_input)
            else:
                result = await self.process_message(task, message)

            # Update task based on result
            if result.requires_input:
                task.set_input_required(result.input_prompt or "Additional input required")
            elif result.success:
                if result.message:
                    task.add_message(Message.agent(result.message))
                task.set_completed(result.message)
                TASK_COUNT.labels(state="completed").inc()
            else:
                task.set_failed(result.message or "Unknown error")
                TASK_COUNT.labels(state="failed").inc()

            self.tasks.update(task)
            return task.model_dump_jsonrpc()

        except Exception as e:
            logger.exception("message_processing_error", error=str(e))
            task.set_failed(str(e))
            self.tasks.update(task)
            TASK_COUNT.labels(state="failed").inc()
            return task.model_dump_jsonrpc()

    async def _handle_task_get(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tasks/get method."""
        task_id = params.get("id")
        if not task_id:
            raise ValueError("Task ID required")

        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        return task.model_dump_jsonrpc()

    async def _handle_task_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tasks/cancel method."""
        task_id = params.get("id")
        if not task_id:
            raise ValueError("Task ID required")

        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        if not task.is_terminal:
            task.status.state = TaskState.CANCELED
            self.tasks.update(task)
            TASK_COUNT.labels(state="canceled").inc()

        return task.model_dump_jsonrpc()

    async def process_message(self, task: Task, message: Message) -> SkillResult:
        """
        Process a message without explicit skill routing.

        Override this in subclasses to implement default message handling.

        Args:
            task: The current task
            message: The incoming message

        Returns:
            SkillResult: The processing result
        """
        # Default implementation tries to match skills by analyzing the message
        text = message.get_text()

        # Try to find a matching skill
        matching_skills = self.skills.find_by_text(text)
        if matching_skills:
            skill = matching_skills[0]
            skill_input = SkillInput(task=task, message=message, params={})
            return await skill.execute(skill_input)

        # No matching skill found
        return SkillResult.error(
            f"No skill matched for message. Available skills: "
            f"{[s.name for s in self.skills.list()]}"
        )

    @property
    def app(self) -> FastAPI:
        """Get or create the FastAPI application."""
        if self._app is None:
            self._app = self.create_app()
        return self._app
