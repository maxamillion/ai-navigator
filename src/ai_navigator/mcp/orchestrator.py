"""Multi-tool orchestration for complex operations."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import structlog

from ai_navigator.mcp.client import MCPClient, MCPToolError
from ai_navigator.mcp.cache import MCPCache
from ai_navigator.mcp.recovery import RetryPolicy, with_retry

logger = structlog.get_logger(__name__)


@dataclass
class ToolCall:
    """Represents a single tool call in an orchestration."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    result_key: str = ""
    transform: Optional[Callable[[dict[str, Any]], Any]] = None

    def __post_init__(self) -> None:
        if not self.result_key:
            self.result_key = self.tool_name


@dataclass
class OrchestrationResult:
    """Result of an orchestration execution."""

    success: bool
    results: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)


class MCPOrchestrator:
    """Orchestrates multiple MCP tool calls with dependency management."""

    def __init__(
        self,
        client: MCPClient,
        cache: Optional[MCPCache] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        """Initialize orchestrator."""
        self._client = client
        self._cache = cache
        self._retry_policy = retry_policy or RetryPolicy()

    async def execute_parallel(
        self,
        calls: list[ToolCall],
        stop_on_error: bool = True,
    ) -> OrchestrationResult:
        """Execute independent tool calls in parallel."""
        results: dict[str, Any] = {}
        errors: dict[str, str] = {}

        tasks = []
        for call in calls:
            tasks.append(self._execute_single(call, results))

        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for call, result in zip(calls, completed):
            if isinstance(result, Exception):
                errors[call.result_key] = str(result)
                if stop_on_error:
                    return OrchestrationResult(
                        success=False,
                        results=results,
                        errors=errors,
                        execution_order=[c.result_key for c in calls],
                    )
            else:
                results[call.result_key] = result

        return OrchestrationResult(
            success=len(errors) == 0,
            results=results,
            errors=errors,
            execution_order=[c.result_key for c in calls],
        )

    async def execute_sequential(
        self,
        calls: list[ToolCall],
        context: Optional[dict[str, Any]] = None,
        stop_on_error: bool = True,
    ) -> OrchestrationResult:
        """Execute tool calls sequentially, passing results between calls."""
        results: dict[str, Any] = context.copy() if context else {}
        errors: dict[str, str] = {}
        execution_order: list[str] = []

        for call in calls:
            # Resolve any argument references to previous results
            resolved_args = self._resolve_arguments(call.arguments, results)

            try:
                result = await self._execute_with_retry(
                    call.tool_name, resolved_args
                )
                if call.transform:
                    result = call.transform(result)
                results[call.result_key] = result
                execution_order.append(call.result_key)
                logger.debug(
                    "Sequential call completed",
                    tool=call.tool_name,
                    result_key=call.result_key,
                )
            except Exception as e:
                errors[call.result_key] = str(e)
                execution_order.append(call.result_key)
                logger.error(
                    "Sequential call failed",
                    tool=call.tool_name,
                    error=str(e),
                )
                if stop_on_error:
                    break

        return OrchestrationResult(
            success=len(errors) == 0,
            results=results,
            errors=errors,
            execution_order=execution_order,
        )

    async def execute_dag(
        self,
        calls: list[ToolCall],
        context: Optional[dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """Execute tool calls respecting dependencies (DAG execution)."""
        results: dict[str, Any] = context.copy() if context else {}
        errors: dict[str, str] = {}
        execution_order: list[str] = []
        pending = {call.result_key: call for call in calls}
        completed: set[str] = set(results.keys())

        while pending:
            # Find calls whose dependencies are satisfied
            ready = [
                call
                for call in pending.values()
                if all(dep in completed for dep in call.depends_on)
            ]

            if not ready:
                # Circular dependency or missing dependency
                remaining = list(pending.keys())
                for key in remaining:
                    errors[key] = "Unresolved dependencies"
                break

            # Execute ready calls in parallel
            tasks = []
            for call in ready:
                resolved_args = self._resolve_arguments(call.arguments, results)
                tasks.append(
                    (call, self._execute_with_retry(call.tool_name, resolved_args))
                )

            for call, coro in tasks:
                try:
                    result = await coro
                    if call.transform:
                        result = call.transform(result)
                    results[call.result_key] = result
                    completed.add(call.result_key)
                    execution_order.append(call.result_key)
                    del pending[call.result_key]
                except Exception as e:
                    errors[call.result_key] = str(e)
                    execution_order.append(call.result_key)
                    del pending[call.result_key]
                    logger.error(
                        "DAG call failed",
                        tool=call.tool_name,
                        error=str(e),
                    )

        return OrchestrationResult(
            success=len(errors) == 0,
            results=results,
            errors=errors,
            execution_order=execution_order,
        )

    async def _execute_single(
        self,
        call: ToolCall,
        context: dict[str, Any],
    ) -> Any:
        """Execute a single tool call."""
        resolved_args = self._resolve_arguments(call.arguments, context)
        result = await self._execute_with_retry(call.tool_name, resolved_args)
        if call.transform:
            result = call.transform(result)
        return result

    async def _execute_with_retry(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute tool with retry logic."""
        # Check cache first
        if self._cache:
            cached = await self._cache.get(tool_name, arguments)
            if cached is not None:
                return cached

        # Execute with retry
        result = await with_retry(
            lambda: self._client.call_tool(tool_name, arguments),
            self._retry_policy,
        )

        # Cache result if cacheable
        if self._cache:
            await self._cache.set(tool_name, arguments, result)

        return result

    def _resolve_arguments(
        self,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve argument references to context values."""
        resolved = {}
        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to context value: $result_key.field
                ref = value[1:]
                parts = ref.split(".", 1)
                result_key = parts[0]
                if result_key in context:
                    if len(parts) > 1:
                        # Nested access
                        resolved[key] = self._get_nested(
                            context[result_key], parts[1]
                        )
                    else:
                        resolved[key] = context[result_key]
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved

    def _get_nested(self, obj: Any, path: str) -> Any:
        """Get nested value from object using dot notation."""
        for part in path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(part)
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj


# Pre-defined orchestration patterns
async def setup_project_orchestration(
    orchestrator: MCPOrchestrator,
    namespace: str,
    display_name: str,
    storage_size: str = "20Gi",
) -> OrchestrationResult:
    """Set up a complete Data Science Project with storage."""
    calls = [
        ToolCall(
            tool_name="create_data_science_project",
            arguments={
                "name": namespace,
                "display_name": display_name,
            },
            result_key="project",
        ),
        ToolCall(
            tool_name="create_storage",
            arguments={
                "namespace": namespace,
                "name": f"{namespace}-storage",
                "size": storage_size,
            },
            depends_on=["project"],
            result_key="storage",
        ),
    ]
    return await orchestrator.execute_dag(calls)


async def get_project_overview_orchestration(
    orchestrator: MCPOrchestrator,
    namespace: str,
) -> OrchestrationResult:
    """Get comprehensive project overview in parallel."""
    calls = [
        ToolCall(
            tool_name="get_project_status",
            arguments={"namespace": namespace},
            result_key="status",
        ),
        ToolCall(
            tool_name="list_workbenches",
            arguments={"namespace": namespace},
            result_key="workbenches",
        ),
        ToolCall(
            tool_name="list_inference_services",
            arguments={"namespace": namespace},
            result_key="models",
        ),
    ]
    return await orchestrator.execute_parallel(calls)
