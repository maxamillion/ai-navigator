"""Tests for state management."""

import pytest
from ai_navigator.state.memory import InMemoryStateStore
from ai_navigator.state.manager import StateManager
from ai_navigator.models.workflow import WorkflowState
from ai_navigator.models.capacity import CapacityPlan
from ai_navigator.models.deployment import DeploymentConfig, InferenceServiceSpec


class TestInMemoryStateStore:
    """Test in-memory state store."""

    @pytest.fixture
    def store(self):
        """Create fresh store for each test."""
        return InMemoryStateStore()

    @pytest.mark.asyncio
    async def test_save_and_get_workflow(self, store):
        """Test saving and retrieving workflow state."""
        state = WorkflowState(user_id="user-1")

        await store.save_workflow(state)
        retrieved = await store.get_workflow(state.id)

        assert retrieved is not None
        assert retrieved.id == state.id
        assert retrieved.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_get_workflow_by_user(self, store):
        """Test getting workflow by user ID."""
        state = WorkflowState(user_id="user-1")
        await store.save_workflow(state)

        retrieved = await store.get_workflow_by_user("user-1")

        assert retrieved is not None
        assert retrieved.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_workflow(self, store):
        """Test getting nonexistent workflow returns None."""
        retrieved = await store.get_workflow("nonexistent-id")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_workflow(self, store):
        """Test deleting workflow."""
        state = WorkflowState(user_id="user-1")
        await store.save_workflow(state)

        result = await store.delete_workflow(state.id)
        assert result is True

        retrieved = await store.get_workflow(state.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_workflow(self, store):
        """Test deleting nonexistent workflow returns False."""
        result = await store.delete_workflow("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_save_and_get_capacity_plan(self, store):
        """Test saving and retrieving capacity plan."""
        plan = CapacityPlan(
            model_name="test-model",
            min_replicas=1,
            max_replicas=3,
            target_replicas=2,
            gpu_type="A100-40GB",
            gpu_count=1,
            gpu_memory_gb=40,
            memory_per_replica="32Gi",
            cpu_per_replica="8",
            estimated_throughput_tps=100,
            estimated_rps=10,
            estimated_p95_latency_ms=250,
            estimated_p99_latency_ms=400,
            meets_slo=True,
        )

        await store.save_capacity_plan(plan)
        retrieved = await store.get_capacity_plan(plan.id)

        assert retrieved is not None
        assert retrieved.id == plan.id
        assert retrieved.model_name == "test-model"

    @pytest.mark.asyncio
    async def test_save_and_get_deployment_config(self, store):
        """Test saving and retrieving deployment config."""
        spec = InferenceServiceSpec(
            name="test-service",
            namespace="test-ns",
            model_name="test-model",
        )
        config = DeploymentConfig(inference_service=spec)

        await store.save_deployment_config(config)
        retrieved = await store.get_deployment_config(config.id)

        assert retrieved is not None
        assert retrieved.id == config.id
        assert retrieved.inference_service.name == "test-service"

    def test_clear_store(self, store):
        """Test clearing all stored state."""
        store.clear()
        # Should not raise any errors
        assert store._workflows == {}
        assert store._capacity_plans == {}


class TestStateManager:
    """Test state manager."""

    @pytest.fixture
    def manager(self):
        """Create state manager with in-memory store."""
        store = InMemoryStateStore()
        return StateManager(store)

    @pytest.mark.asyncio
    async def test_create_workflow(self, manager):
        """Test creating a new workflow."""
        state = await manager.create_workflow("user-1")

        assert state is not None
        assert state.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_get_active_workflow(self, manager):
        """Test getting active workflow for user."""
        created = await manager.create_workflow("user-1")
        retrieved = await manager.get_active_workflow("user-1")

        assert retrieved is not None
        assert retrieved.id == created.id

    @pytest.mark.asyncio
    async def test_save_workflow_updates(self, manager):
        """Test saving workflow updates."""
        state = await manager.create_workflow("user-1")
        state.intent = "Deploy a model"

        await manager.save_workflow(state)
        retrieved = await manager.get_workflow(state.id)

        assert retrieved.intent == "Deploy a model"
