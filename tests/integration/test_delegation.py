"""Integration tests for agent delegation."""

import pytest
from unittest.mock import AsyncMock, patch

from ai_navigator.agents.supervisor.decomposer import IntentDecomposer, SubTask, DecompositionResult
from ai_navigator.agents.supervisor.delegator import SubAgentDelegator, DelegationResult
from ai_navigator.agents.supervisor.aggregator import ResultAggregator
from ai_navigator.llm.client import LLMClient


class TestDelegationFlow:
    """Tests for end-to-end delegation flow."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        mock = AsyncMock(spec=LLMClient)
        mock.complete.return_value = "Synthesized response from all agents."
        return mock

    @pytest.fixture
    def decomposer(self, mock_llm):
        """Create decomposer with mock LLM."""
        return IntentDecomposer(mock_llm)

    @pytest.fixture
    def delegator(self):
        """Create delegator with static endpoints."""
        delegator = SubAgentDelegator()
        delegator.register_static_endpoint("model_catalog", "http://localhost:8001")
        delegator.register_static_endpoint("resource_provisioning", "http://localhost:8002")
        delegator.register_static_endpoint("deployment_monitor", "http://localhost:8003")
        return delegator

    @pytest.fixture
    def aggregator(self, mock_llm):
        """Create aggregator with mock LLM."""
        return ResultAggregator(mock_llm)

    def test_decomposition_creates_valid_subtasks(self, decomposer):
        """Test that decomposition creates valid subtasks."""
        result = decomposer._heuristic_decompose("Recommend a model for text classification")

        assert result.intent is not None
        assert len(result.subtasks) > 0

        for task in result.subtasks:
            assert task.id is not None
            assert task.agent is not None
            assert task.skill is not None

    def test_delegator_has_registered_endpoints(self, delegator):
        """Test that delegator has registered endpoints."""
        assert "model_catalog" in delegator._static_endpoints
        assert "resource_provisioning" in delegator._static_endpoints
        assert "deployment_monitor" in delegator._static_endpoints

    @pytest.mark.asyncio
    async def test_delegator_get_endpoint(self, delegator):
        """Test getting agent endpoint."""
        endpoint = await delegator.get_agent_endpoint("model_catalog")

        assert endpoint is not None
        assert endpoint.name == "model_catalog"
        assert endpoint.url == "http://localhost:8001"

    @pytest.mark.asyncio
    async def test_delegator_get_unknown_endpoint(self, delegator):
        """Test getting unknown agent endpoint."""
        endpoint = await delegator.get_agent_endpoint("unknown_agent")
        assert endpoint is None

    def test_aggregator_template_synthesis_success(self, aggregator):
        """Test template-based synthesis with successful results."""
        results = [
            DelegationResult(
                task_id="1",
                agent="model_catalog",
                skill="query_models",
                success=True,
                message="Found 3 models matching criteria.",
            ),
            DelegationResult(
                task_id="2",
                agent="resource_provisioning",
                skill="estimate_cost",
                success=True,
                message="Estimated monthly cost: $500",
            ),
        ]

        response = aggregator.aggregate_sync("What models are available?", results)

        assert response.success is True
        assert len(response.failed_agents) == 0
        assert "Found 3 models" in response.summary
        assert "Estimated monthly cost" in response.summary

    def test_aggregator_template_synthesis_partial_failure(self, aggregator):
        """Test template synthesis with partial failures."""
        results = [
            DelegationResult(
                task_id="1",
                agent="model_catalog",
                skill="query_models",
                success=True,
                message="Found 3 models.",
            ),
            DelegationResult(
                task_id="2",
                agent="resource_provisioning",
                skill="estimate_cost",
                success=False,
                error="Connection timeout",
            ),
        ]

        response = aggregator.aggregate_sync("Get models and cost", results)

        assert response.success is False
        assert "resource_provisioning" in response.failed_agents
        assert "Issues Encountered" in response.summary

    def test_aggregator_template_synthesis_all_failed(self, aggregator):
        """Test template synthesis when all tasks fail."""
        results = [
            DelegationResult(
                task_id="1",
                agent="model_catalog",
                skill="query_models",
                success=False,
                error="Service unavailable",
            ),
        ]

        response = aggregator.aggregate_sync("List models", results)

        assert response.success is False
        assert "Failed" in response.summary

    @pytest.mark.asyncio
    async def test_full_recommendation_flow(self, decomposer, delegator, aggregator):
        """Test the full recommendation flow (mocked delegation)."""
        # Step 1: Decompose
        decomposition = decomposer._heuristic_decompose(
            "Recommend a model for our chatbot"
        )

        assert decomposition.intent == "recommend_model"
        assert len(decomposition.subtasks) >= 1

        # Step 2: Mock delegation results (simulating sub-agent responses)
        mock_results = [
            DelegationResult(
                task_id=task.id,
                agent=task.agent,
                skill=task.skill,
                success=True,
                message=f"Result from {task.agent}:{task.skill}",
            )
            for task in decomposition.subtasks
        ]

        # Step 3: Aggregate
        response = await aggregator.aggregate(
            "Recommend a model for our chatbot",
            mock_results,
        )

        assert response.success is True


class TestDelegationResult:
    """Tests for DelegationResult model."""

    def test_successful_result(self):
        """Test creating a successful delegation result."""
        result = DelegationResult(
            task_id="1",
            agent="model_catalog",
            skill="query_models",
            success=True,
            message="Found 5 models",
            data={"count": 5},
        )

        assert result.success is True
        assert result.error is None
        assert result.data["count"] == 5

    def test_failed_result(self):
        """Test creating a failed delegation result."""
        result = DelegationResult(
            task_id="1",
            agent="model_catalog",
            skill="query_models",
            success=False,
            error="Connection refused",
        )

        assert result.success is False
        assert result.error == "Connection refused"
        assert result.message is None


class TestParallelExecution:
    """Tests for parallel task execution."""

    @pytest.fixture
    def decomposer(self):
        """Create decomposer."""
        mock_llm = AsyncMock(spec=LLMClient)
        return IntentDecomposer(mock_llm)

    def test_parallel_groups_for_deployment(self, decomposer):
        """Test that deployment creates proper parallel groups."""
        result = decomposer._heuristic_decompose(
            "Deploy granite-4.0-h-tiny"
        )

        # Should have multiple parallel groups for deployment workflow
        assert len(result.parallel_groups) >= 2

        # First group should be independent tasks
        first_group = result.parallel_groups[0]
        first_tasks = [t for t in result.subtasks if t.id in first_group]

        # First tasks should have no dependencies
        for task in first_tasks:
            assert len(task.depends_on) == 0

    def test_dependency_ordering(self, decomposer):
        """Test that dependencies are correctly ordered in groups."""
        result = decomposer._heuristic_decompose(
            "Deploy granite-4.0-h-tiny"
        )

        # Build a map of task ID to its position in parallel groups
        task_order = {}
        for i, group in enumerate(result.parallel_groups):
            for task_id in group:
                task_order[task_id] = i

        # Verify all dependencies appear before the dependent task
        for task in result.subtasks:
            for dep_id in task.depends_on:
                assert dep_id in task_order, f"Dependency {dep_id} not in any group"
                assert task_order[dep_id] < task_order[task.id], \
                    f"Dependency {dep_id} should come before {task.id}"
