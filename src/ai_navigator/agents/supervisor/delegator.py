"""Sub-agent delegation for the Supervisor Agent."""

import asyncio
from typing import Any

import structlog
from pydantic import BaseModel, Field

from ai_navigator.a2a.client import A2AClient, A2AClientError
from ai_navigator.a2a.message import Message
from ai_navigator.agents.supervisor.decomposer import DecompositionResult, SubTask
from ai_navigator.config import KubernetesSettings

logger = structlog.get_logger(__name__)


class AgentEndpoint(BaseModel):
    """Information about a sub-agent endpoint."""

    name: str = Field(..., description="Agent name")
    url: str = Field(..., description="Agent URL")
    healthy: bool = Field(default=True, description="Agent health status")


class DelegationResult(BaseModel):
    """Result of delegating a sub-task."""

    task_id: str = Field(..., description="Sub-task ID")
    agent: str = Field(..., description="Agent that handled the task")
    skill: str = Field(..., description="Skill that was invoked")
    success: bool = Field(..., description="Whether delegation succeeded")
    message: str | None = Field(default=None, description="Result message")
    data: dict[str, Any] | None = Field(default=None, description="Result data")
    error: str | None = Field(default=None, description="Error message if failed")


class SubAgentDelegator:
    """
    Handles delegation of tasks to sub-agents.

    Supports:
    - Agent discovery via Kubernetes API or static configuration
    - Parallel execution of independent tasks
    - Dependency-aware task ordering
    - Error handling and retries
    """

    def __init__(
        self,
        kubernetes_settings: KubernetesSettings | None = None,
        static_endpoints: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the delegator.

        Args:
            kubernetes_settings: Kubernetes config for dynamic discovery
            static_endpoints: Static agent endpoints for development
        """
        self.k8s_settings = kubernetes_settings or KubernetesSettings()
        self._static_endpoints = static_endpoints or {}
        self._agent_cache: dict[str, AgentEndpoint] = {}

    def register_static_endpoint(self, name: str, url: str) -> None:
        """Register a static agent endpoint."""
        self._static_endpoints[name] = url
        self._agent_cache[name] = AgentEndpoint(name=name, url=url)

    async def discover_agents(self) -> dict[str, AgentEndpoint]:
        """
        Discover available sub-agents.

        Returns:
            Dictionary of agent name to endpoint info
        """
        # Start with static endpoints
        agents = {
            name: AgentEndpoint(name=name, url=url)
            for name, url in self._static_endpoints.items()
        }

        # Try Kubernetes discovery if configured
        if self.k8s_settings.in_cluster:
            try:
                k8s_agents = await self._discover_kubernetes_agents()
                agents.update(k8s_agents)
            except Exception as e:
                logger.warning("kubernetes_discovery_failed", error=str(e))

        self._agent_cache = agents
        return agents

    async def _discover_kubernetes_agents(self) -> dict[str, AgentEndpoint]:
        """Discover agents via Kubernetes API."""
        # In production, use kubernetes client to list services/pods
        # with the appropriate labels
        agents = {}

        # Default service discovery based on naming convention
        namespace = self.k8s_settings.namespace
        agent_names = ["model-catalog", "resource-provisioning", "deployment-monitor"]

        for name in agent_names:
            url = f"http://{name}.{namespace}.svc.cluster.local:8000"
            agents[name.replace("-", "_")] = AgentEndpoint(name=name, url=url)

        return agents

    async def get_agent_endpoint(self, agent_name: str) -> AgentEndpoint | None:
        """
        Get endpoint for a specific agent.

        Args:
            agent_name: Agent name (e.g., "model_catalog")

        Returns:
            AgentEndpoint or None if not found
        """
        # Normalize name
        normalized = agent_name.replace("-", "_")

        # Check cache first
        if normalized in self._agent_cache:
            return self._agent_cache[normalized]

        # Try discovery
        await self.discover_agents()
        return self._agent_cache.get(normalized)

    async def delegate(
        self,
        decomposition: DecompositionResult,
        original_message: str,
    ) -> list[DelegationResult]:
        """
        Execute a decomposed task by delegating to sub-agents.

        Args:
            decomposition: The decomposed task with sub-tasks
            original_message: The original user message

        Returns:
            List of delegation results
        """
        results: list[DelegationResult] = []
        completed: dict[str, DelegationResult] = {}

        # Execute parallel groups in order
        for group in decomposition.parallel_groups:
            # Filter to tasks in this group that have dependencies met
            ready_tasks = []
            for task_id in group:
                task = next((t for t in decomposition.subtasks if t.id == task_id), None)
                if task is None:
                    continue

                # Check if dependencies are met
                deps_met = all(
                    dep_id in completed and completed[dep_id].success
                    for dep_id in task.depends_on
                )

                if deps_met:
                    ready_tasks.append(task)

            if not ready_tasks:
                continue

            # Execute tasks in parallel
            group_results = await asyncio.gather(
                *[
                    self._delegate_task(task, original_message, completed)
                    for task in ready_tasks
                ],
                return_exceptions=True,
            )

            # Process results
            for task, result in zip(ready_tasks, group_results):
                if isinstance(result, Exception):
                    delegation_result = DelegationResult(
                        task_id=task.id,
                        agent=task.agent,
                        skill=task.skill,
                        success=False,
                        error=str(result),
                    )
                else:
                    delegation_result = result

                results.append(delegation_result)
                completed[task.id] = delegation_result

        return results

    async def _delegate_task(
        self,
        task: SubTask,
        original_message: str,
        completed: dict[str, DelegationResult],
    ) -> DelegationResult:
        """
        Delegate a single task to a sub-agent.

        Args:
            task: The sub-task to delegate
            original_message: Original user message for context
            completed: Already completed tasks

        Returns:
            DelegationResult
        """
        logger.info(
            "delegating_task",
            task_id=task.id,
            agent=task.agent,
            skill=task.skill,
        )

        # Get agent endpoint
        endpoint = await self.get_agent_endpoint(task.agent)
        if endpoint is None:
            return DelegationResult(
                task_id=task.id,
                agent=task.agent,
                skill=task.skill,
                success=False,
                error=f"Agent not found: {task.agent}",
            )

        # Prepare message with context from dependencies
        context_parts = [f"Original request: {original_message}"]
        for dep_id in task.depends_on:
            if dep_id in completed and completed[dep_id].success:
                dep_result = completed[dep_id]
                context_parts.append(f"Result from {dep_result.agent}: {dep_result.message}")

        message_text = "\n\n".join(context_parts)

        try:
            async with A2AClient(endpoint.url) as client:
                result_task = await client.send_message(
                    message=Message.user(message_text),
                    skill_id=task.skill,
                )

                # Wait for completion
                completed_task = await client.wait_for_completion(
                    result_task.id,
                    poll_interval=1.0,
                    max_polls=120,
                )

                # Extract result
                result_message = None
                if completed_task.status.message:
                    result_message = completed_task.status.message.get_text()

                return DelegationResult(
                    task_id=task.id,
                    agent=task.agent,
                    skill=task.skill,
                    success=True,
                    message=result_message,
                    data=completed_task.metadata,
                )

        except A2AClientError as e:
            logger.error("delegation_failed", task_id=task.id, error=str(e))
            return DelegationResult(
                task_id=task.id,
                agent=task.agent,
                skill=task.skill,
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.exception("delegation_error", task_id=task.id, error=str(e))
            return DelegationResult(
                task_id=task.id,
                agent=task.agent,
                skill=task.skill,
                success=False,
                error=f"Unexpected error: {e}",
            )

    async def delegate_direct(
        self,
        agent_name: str,
        skill_id: str,
        message: str,
        params: dict[str, Any] | None = None,
    ) -> DelegationResult:
        """
        Directly delegate to a specific agent skill.

        Args:
            agent_name: Target agent
            skill_id: Skill to invoke
            message: Message to send
            params: Optional skill parameters

        Returns:
            DelegationResult
        """
        endpoint = await self.get_agent_endpoint(agent_name)
        if endpoint is None:
            return DelegationResult(
                task_id="direct",
                agent=agent_name,
                skill=skill_id,
                success=False,
                error=f"Agent not found: {agent_name}",
            )

        try:
            async with A2AClient(endpoint.url) as client:
                result_task = await client.send_message(
                    message=Message.user(message),
                    skill_id=skill_id,
                )

                completed_task = await client.wait_for_completion(
                    result_task.id,
                    poll_interval=1.0,
                    max_polls=60,
                )

                result_message = None
                if completed_task.status.message:
                    result_message = completed_task.status.message.get_text()

                return DelegationResult(
                    task_id="direct",
                    agent=agent_name,
                    skill=skill_id,
                    success=True,
                    message=result_message,
                    data=completed_task.metadata,
                )

        except Exception as e:
            return DelegationResult(
                task_id="direct",
                agent=agent_name,
                skill=skill_id,
                success=False,
                error=str(e),
            )
