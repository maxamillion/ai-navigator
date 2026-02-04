"""Unit tests for agent implementations."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai_navigator.a2a.message import Message
from ai_navigator.a2a.task import Task
from ai_navigator.a2a.skills import SkillInput, SkillResult
from ai_navigator.config import AgentSettings


class TestModelCatalogAgent:
    """Tests for ModelCatalogAgent."""

    @pytest.fixture
    def agent_settings(self):
        """Create test agent settings."""
        return AgentSettings(
            name="test-model-catalog",
            description="Test Model Catalog Agent",
            port=8001,
        )

    @pytest.fixture
    def mock_openshift_ai(self):
        """Create mock OpenShiftAI tools."""
        mock = AsyncMock()
        mock.query_model_registry.return_value = [
            MagicMock(
                id="granite-4-tiny",
                name="granite-4.0-h-tiny",
                description="Test model",
                parameters=7_000_000_000,
                tags=["llm"],
            ),
        ]
        mock.get_model_details.return_value = MagicMock(
            id="granite-4-tiny",
            name="granite-4.0-h-tiny",
            description="Test model",
            version="4.0.0",
            format="safetensors",
            parameters=7_000_000_000,
            size_bytes=14_000_000_000,
            tags=["llm"],
            metadata={},
        )
        return mock

    @pytest.fixture
    def mock_trustyai(self):
        """Create mock TrustyAI tools."""
        mock = AsyncMock()
        mock.get_model_evaluation.return_value = MagicMock(
            model_id="granite-4-tiny",
            model_name="granite-4.0-h-tiny",
            hap_score=0.95,
            bias_score=0.92,
            fairness_score=0.90,
            explainability_score=0.88,
            overall_trust_score=0.91,
            evaluation_date="2025-01-15",
            details={},
        )
        return mock

    @pytest.mark.asyncio
    async def test_agent_creation(self, agent_settings):
        """Test agent creation."""
        from ai_navigator.agents.model_catalog.agent import ModelCatalogAgent

        agent = ModelCatalogAgent(settings=agent_settings)
        assert agent.name == "Model Catalog Agent"
        assert agent.settings.name == "test-model-catalog"

    @pytest.mark.asyncio
    async def test_agent_has_skills(self, agent_settings):
        """Test that agent has registered skills."""
        from ai_navigator.agents.model_catalog.agent import ModelCatalogAgent

        agent = ModelCatalogAgent(settings=agent_settings)

        skills = agent.skills.list()
        skill_ids = [s.id for s in skills]

        assert "query_models" in skill_ids
        assert "get_model_details" in skill_ids
        assert "get_benchmarks" in skill_ids
        assert "get_trustyai_scores" in skill_ids
        assert "recommend_for_workload" in skill_ids

    @pytest.mark.asyncio
    async def test_agent_card_generation(self, agent_settings):
        """Test agent card generation."""
        from ai_navigator.agents.model_catalog.agent import ModelCatalogAgent

        agent = ModelCatalogAgent(settings=agent_settings)
        card = agent.build_agent_card()

        assert card.name == "Model Catalog Agent"
        assert len(card.skills) > 0
        assert card.provider.organization == "Red Hat"


class TestResourceProvisioningAgent:
    """Tests for ResourceProvisioningAgent."""

    @pytest.fixture
    def agent_settings(self):
        """Create test agent settings."""
        return AgentSettings(
            name="test-resource-provisioning",
            description="Test Resource Provisioning Agent",
            port=8002,
        )

    @pytest.mark.asyncio
    async def test_agent_creation(self, agent_settings):
        """Test agent creation."""
        from ai_navigator.agents.resource_provisioning.agent import ResourceProvisioningAgent

        agent = ResourceProvisioningAgent(settings=agent_settings)
        assert agent.name == "Resource Provisioning Agent"

    @pytest.mark.asyncio
    async def test_agent_has_skills(self, agent_settings):
        """Test that agent has registered skills."""
        from ai_navigator.agents.resource_provisioning.agent import ResourceProvisioningAgent

        agent = ResourceProvisioningAgent(settings=agent_settings)

        skills = agent.skills.list()
        skill_ids = [s.id for s in skills]

        assert "generate_deployment_config" in skill_ids
        assert "estimate_cost" in skill_ids
        assert "validate_slo_compliance" in skill_ids
        assert "apply_deployment" in skill_ids
        assert "generate_guardrails" in skill_ids


class TestDeploymentMonitorAgent:
    """Tests for DeploymentMonitorAgent."""

    @pytest.fixture
    def agent_settings(self):
        """Create test agent settings."""
        return AgentSettings(
            name="test-deployment-monitor",
            description="Test Deployment Monitor Agent",
            port=8003,
        )

    @pytest.mark.asyncio
    async def test_agent_creation(self, agent_settings):
        """Test agent creation."""
        from ai_navigator.agents.deployment_monitor.agent import DeploymentMonitorAgent

        agent = DeploymentMonitorAgent(settings=agent_settings)
        assert agent.name == "Deployment Monitor Agent"

    @pytest.mark.asyncio
    async def test_agent_has_skills(self, agent_settings):
        """Test that agent has registered skills."""
        from ai_navigator.agents.deployment_monitor.agent import DeploymentMonitorAgent

        agent = DeploymentMonitorAgent(settings=agent_settings)

        skills = agent.skills.list()
        skill_ids = [s.id for s in skills]

        assert "get_deployment_status" in skill_ids
        assert "query_metrics" in skill_ids
        assert "check_slo_violations" in skill_ids
        assert "get_pod_logs" in skill_ids
        assert "get_health_summary" in skill_ids


class TestSupervisorAgent:
    """Tests for SupervisorAgent."""

    @pytest.fixture
    def agent_settings(self):
        """Create test agent settings."""
        return AgentSettings(
            name="test-supervisor",
            description="Test Supervisor Agent",
            port=8000,
        )

    @pytest.mark.asyncio
    async def test_agent_creation(self, agent_settings):
        """Test agent creation."""
        from ai_navigator.agents.supervisor.agent import SupervisorAgent

        agent = SupervisorAgent(settings=agent_settings)
        assert agent.name == "AI Navigator Supervisor"

    @pytest.mark.asyncio
    async def test_agent_has_skills(self, agent_settings):
        """Test that agent has registered skills."""
        from ai_navigator.agents.supervisor.agent import SupervisorAgent

        agent = SupervisorAgent(settings=agent_settings)

        skills = agent.skills.list()
        skill_ids = [s.id for s in skills]

        assert "recommend_model" in skill_ids
        assert "deploy_model" in skill_ids
        assert "check_status" in skill_ids
        assert "list_models" in skill_ids
        assert "get_metrics" in skill_ids

    @pytest.mark.asyncio
    async def test_agent_has_components(self, agent_settings):
        """Test that supervisor has required components."""
        from ai_navigator.agents.supervisor.agent import SupervisorAgent

        agent = SupervisorAgent(settings=agent_settings)

        assert agent.decomposer is not None
        assert agent.delegator is not None
        assert agent.aggregator is not None
        assert agent.llm is not None
