"""In-memory state store for development and testing."""

from typing import Optional

from ai_navigator.models.capacity import CapacityPlan
from ai_navigator.models.deployment import DeploymentConfig
from ai_navigator.models.workflow import WorkflowState
from ai_navigator.state.manager import StateStore


class InMemoryStateStore(StateStore):
    """In-memory implementation of state store for development."""

    def __init__(self) -> None:
        """Initialize empty stores."""
        self._workflows: dict[str, WorkflowState] = {}
        self._user_workflows: dict[str, str] = {}  # user_id -> workflow_id
        self._capacity_plans: dict[str, CapacityPlan] = {}
        self._deployment_configs: dict[str, DeploymentConfig] = {}

    async def save_workflow(self, state: WorkflowState) -> None:
        """Save or update workflow state."""
        self._workflows[state.id] = state
        self._user_workflows[state.user_id] = state.id

    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowState]:
        """Retrieve workflow state by ID."""
        return self._workflows.get(workflow_id)

    async def get_workflow_by_user(self, user_id: str) -> Optional[WorkflowState]:
        """Get the active workflow for a user."""
        workflow_id = self._user_workflows.get(user_id)
        if workflow_id:
            return self._workflows.get(workflow_id)
        return None

    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete workflow state."""
        if workflow_id in self._workflows:
            state = self._workflows.pop(workflow_id)
            if state.user_id in self._user_workflows:
                if self._user_workflows[state.user_id] == workflow_id:
                    del self._user_workflows[state.user_id]
            return True
        return False

    async def save_capacity_plan(self, plan: CapacityPlan) -> None:
        """Save capacity plan."""
        self._capacity_plans[plan.id] = plan

    async def get_capacity_plan(self, plan_id: str) -> Optional[CapacityPlan]:
        """Retrieve capacity plan by ID."""
        return self._capacity_plans.get(plan_id)

    async def save_deployment_config(self, config: DeploymentConfig) -> None:
        """Save deployment configuration."""
        self._deployment_configs[config.id] = config

    async def get_deployment_config(self, config_id: str) -> Optional[DeploymentConfig]:
        """Retrieve deployment configuration by ID."""
        return self._deployment_configs.get(config_id)

    async def close(self) -> None:
        """No-op for in-memory store."""
        pass

    def clear(self) -> None:
        """Clear all stored state (for testing)."""
        self._workflows.clear()
        self._user_workflows.clear()
        self._capacity_plans.clear()
        self._deployment_configs.clear()
