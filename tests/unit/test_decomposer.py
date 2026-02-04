"""Unit tests for intent decomposition."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ai_navigator.agents.supervisor.decomposer import (
    IntentDecomposer,
    DecompositionResult,
    SubTask,
)
from ai_navigator.llm.client import LLMClient


class TestIntentDecomposer:
    """Tests for IntentDecomposer."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        mock = AsyncMock(spec=LLMClient)
        return mock

    @pytest.fixture
    def decomposer(self, mock_llm):
        """Create a decomposer with mock LLM."""
        return IntentDecomposer(mock_llm)

    def test_heuristic_recommend_intent(self, decomposer):
        """Test heuristic decomposition for recommendation."""
        result = decomposer._heuristic_decompose(
            "Recommend a model for our chatbot"
        )

        assert result.intent == "recommend_model"
        assert len(result.subtasks) > 0
        assert result.subtasks[0].agent == "model_catalog"
        assert result.subtasks[0].skill == "recommend_for_workload"

    def test_heuristic_deploy_intent(self, decomposer):
        """Test heuristic decomposition for deployment."""
        result = decomposer._heuristic_decompose(
            "Deploy granite-4.0-h-tiny to the cluster"
        )

        assert result.intent == "deploy_model"
        assert len(result.subtasks) >= 3  # get details, generate config, validate, guardrails

        # Check task dependencies are set correctly
        agents = [t.agent for t in result.subtasks]
        assert "model_catalog" in agents
        assert "resource_provisioning" in agents

    def test_heuristic_status_intent(self, decomposer):
        """Test heuristic decomposition for status check."""
        result = decomposer._heuristic_decompose(
            "Check the status of my deployment"
        )

        assert result.intent == "check_status"
        assert len(result.subtasks) == 1
        assert result.subtasks[0].agent == "deployment_monitor"
        assert result.subtasks[0].skill == "get_health_summary"

    def test_heuristic_list_intent(self, decomposer):
        """Test heuristic decomposition for listing models."""
        result = decomposer._heuristic_decompose(
            "Show me available models"
        )

        assert result.intent == "list_models"
        assert len(result.subtasks) == 1
        assert result.subtasks[0].agent == "model_catalog"
        assert result.subtasks[0].skill == "query_models"

    def test_heuristic_metrics_intent(self, decomposer):
        """Test heuristic decomposition for metrics query."""
        result = decomposer._heuristic_decompose(
            "Show me the latency metrics"
        )

        assert result.intent == "query_metrics"
        assert result.subtasks[0].agent == "deployment_monitor"
        assert result.subtasks[0].skill == "query_metrics"

    def test_heuristic_slo_intent(self, decomposer):
        """Test heuristic decomposition for SLO check."""
        result = decomposer._heuristic_decompose(
            "Are there any SLO violations?"
        )

        assert result.intent == "check_slo"
        assert result.subtasks[0].agent == "deployment_monitor"
        assert result.subtasks[0].skill == "check_slo_violations"

    def test_workload_type_extraction(self, decomposer):
        """Test workload type extraction."""
        assert decomposer._extract_workload_type("chatbot application") == "text-generation"
        assert decomposer._extract_workload_type("classification task") == "classification"
        assert decomposer._extract_workload_type("code generation") == "code-generation"
        assert decomposer._extract_workload_type("embedding model") == "embedding"
        assert decomposer._extract_workload_type("unknown task") == "text-generation"

    def test_model_name_extraction(self, decomposer):
        """Test model name extraction."""
        assert decomposer._extract_model_name("deploy granite model") == "granite-4.0-h-tiny"
        assert decomposer._extract_model_name("use llama-3") == "llama-3-8b"
        assert decomposer._extract_model_name("install mistral") == "mistral-7b"
        assert decomposer._extract_model_name("some model") == "granite-4.0-h-tiny"

    def test_parallel_groups(self, decomposer):
        """Test parallel execution groups."""
        result = decomposer._heuristic_decompose(
            "Deploy granite-4.0-h-tiny"
        )

        # Verify parallel groups exist
        assert len(result.parallel_groups) > 0

        # Verify all task IDs are accounted for
        all_task_ids = [t.id for t in result.subtasks]
        grouped_ids = [tid for group in result.parallel_groups for tid in group]

        for tid in all_task_ids:
            assert tid in grouped_ids

    @pytest.mark.asyncio
    async def test_llm_decomposition_fallback(self, decomposer, mock_llm):
        """Test fallback to heuristics when LLM fails."""
        mock_llm.complete.side_effect = Exception("LLM unavailable")

        result = await decomposer.decompose("Recommend a model")

        # Should fall back to heuristic
        assert result.intent == "recommend_model"
        assert len(result.subtasks) > 0

    @pytest.mark.asyncio
    async def test_llm_decomposition_success(self, decomposer, mock_llm):
        """Test successful LLM decomposition."""
        mock_llm.complete.return_value = '''
        {
            "intent": "deploy_model",
            "subtasks": [
                {
                    "id": "1",
                    "agent": "model_catalog",
                    "skill": "get_model_details",
                    "params": {"model_id": "granite-4-tiny"},
                    "depends_on": []
                }
            ],
            "parallel_groups": [["1"]]
        }
        '''

        result = await decomposer.decompose("Deploy granite model")

        assert result.intent == "deploy_model"
        assert len(result.subtasks) == 1
        assert result.subtasks[0].agent == "model_catalog"

    def test_subtask_dependencies(self, decomposer):
        """Test that task dependencies are correctly set."""
        result = decomposer._heuristic_decompose(
            "Deploy granite-4.0-h-tiny to the cluster"
        )

        # Find tasks that depend on other tasks
        tasks_with_deps = [t for t in result.subtasks if t.depends_on]

        # Deployment tasks should have dependencies
        assert len(tasks_with_deps) > 0

        # Verify dependencies reference valid task IDs
        all_task_ids = [t.id for t in result.subtasks]
        for task in tasks_with_deps:
            for dep_id in task.depends_on:
                assert dep_id in all_task_ids


class TestSubTask:
    """Tests for SubTask model."""

    def test_subtask_creation(self):
        """Test creating a subtask."""
        task = SubTask(
            id="1",
            agent="model_catalog",
            skill="query_models",
            params={"filter": "llm"},
            depends_on=["0"],
        )

        assert task.id == "1"
        assert task.agent == "model_catalog"
        assert task.skill == "query_models"
        assert task.params == {"filter": "llm"}
        assert task.depends_on == ["0"]

    def test_subtask_defaults(self):
        """Test subtask default values."""
        task = SubTask(
            id="1",
            agent="test",
            skill="test",
        )

        assert task.params == {}
        assert task.depends_on == []


class TestDecompositionResult:
    """Tests for DecompositionResult model."""

    def test_result_creation(self):
        """Test creating a decomposition result."""
        subtask = SubTask(id="1", agent="test", skill="test")
        result = DecompositionResult(
            intent="test_intent",
            subtasks=[subtask],
            parallel_groups=[["1"]],
        )

        assert result.intent == "test_intent"
        assert len(result.subtasks) == 1
        assert result.parallel_groups == [["1"]]

    def test_result_defaults(self):
        """Test result default values."""
        result = DecompositionResult(intent="test")

        assert result.subtasks == []
        assert result.parallel_groups == []
