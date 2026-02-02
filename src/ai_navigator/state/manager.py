"""State manager abstraction for workflow state persistence."""

from abc import ABC, abstractmethod
from typing import Optional

from ai_navigator.models.workflow import WorkflowState
from ai_navigator.models.capacity import CapacityPlan
from ai_navigator.models.deployment import DeploymentConfig


class StateStore(ABC):
    """Abstract base class for state storage backends."""

    @abstractmethod
    async def save_workflow(self, state: WorkflowState) -> None:
        """Save or update workflow state."""
        ...

    @abstractmethod
    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowState]:
        """Retrieve workflow state by ID."""
        ...

    @abstractmethod
    async def get_workflow_by_user(self, user_id: str) -> Optional[WorkflowState]:
        """Get the active workflow for a user."""
        ...

    @abstractmethod
    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete workflow state."""
        ...

    @abstractmethod
    async def save_capacity_plan(self, plan: CapacityPlan) -> None:
        """Save capacity plan."""
        ...

    @abstractmethod
    async def get_capacity_plan(self, plan_id: str) -> Optional[CapacityPlan]:
        """Retrieve capacity plan by ID."""
        ...

    @abstractmethod
    async def save_deployment_config(self, config: DeploymentConfig) -> None:
        """Save deployment configuration."""
        ...

    @abstractmethod
    async def get_deployment_config(self, config_id: str) -> Optional[DeploymentConfig]:
        """Retrieve deployment configuration by ID."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections."""
        ...


class StateManager:
    """Manager for workflow state operations."""

    def __init__(self, store: StateStore) -> None:
        """Initialize with a state store backend."""
        self._store = store

    async def create_workflow(self, user_id: str) -> WorkflowState:
        """Create a new workflow for a user."""
        state = WorkflowState(user_id=user_id)
        await self._store.save_workflow(state)
        return state

    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get workflow by ID."""
        return await self._store.get_workflow(workflow_id)

    async def get_active_workflow(self, user_id: str) -> Optional[WorkflowState]:
        """Get the active workflow for a user."""
        return await self._store.get_workflow_by_user(user_id)

    async def save_workflow(self, state: WorkflowState) -> None:
        """Save workflow state."""
        await self._store.save_workflow(state)

    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        return await self._store.delete_workflow(workflow_id)

    async def save_capacity_plan(self, plan: CapacityPlan) -> None:
        """Save a capacity plan."""
        await self._store.save_capacity_plan(plan)

    async def get_capacity_plan(self, plan_id: str) -> Optional[CapacityPlan]:
        """Get capacity plan by ID."""
        return await self._store.get_capacity_plan(plan_id)

    async def save_deployment_config(self, config: DeploymentConfig) -> None:
        """Save deployment configuration."""
        await self._store.save_deployment_config(config)

    async def get_deployment_config(self, config_id: str) -> Optional[DeploymentConfig]:
        """Get deployment configuration by ID."""
        return await self._store.get_deployment_config(config_id)

    async def close(self) -> None:
        """Close the state store."""
        await self._store.close()
