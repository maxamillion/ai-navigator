"""Unit tests for MCP tool servers."""

import pytest
from datetime import datetime, timezone

from ai_navigator.mcp.openshift_ai_tools import OpenShiftAITools, ModelInfo, InferenceServiceSpec
from ai_navigator.mcp.trustyai_tools import TrustyAITools, ModelEvaluation, GuardrailConfig
from ai_navigator.mcp.observability_tools import (
    ObservabilityTools,
    MetricResult,
    MetricValue,
    SLOStatus,
)
from ai_navigator.config import MCPSettings


class TestOpenShiftAITools:
    """Tests for OpenShiftAI tools."""

    @pytest.fixture
    def tools(self):
        """Create OpenShiftAI tools with default settings."""
        return OpenShiftAITools()

    def test_mock_models_available(self, tools):
        """Test that mock models are available."""
        models = tools._get_mock_models()

        assert len(models) > 0
        model_names = [m.name for m in models]
        assert "granite-4.0-h-tiny" in model_names

    def test_mock_models_filter_by_name(self, tools):
        """Test filtering mock models by name."""
        models = tools._get_mock_models(name_filter="granite")

        assert len(models) >= 1
        for model in models:
            assert "granite" in model.name.lower()

    def test_mock_models_filter_by_tags(self, tools):
        """Test filtering mock models by tags."""
        models = tools._get_mock_models(tags=["llm"])

        assert len(models) >= 1
        for model in models:
            assert "llm" in model.tags

    def test_mock_models_limit(self, tools):
        """Test limiting mock model results."""
        models = tools._get_mock_models(limit=1)
        assert len(models) == 1

    @pytest.mark.asyncio
    async def test_query_model_registry_fallback(self, tools):
        """Test that query falls back to mock data."""
        # Without a real registry, should return mock data
        models = await tools.query_model_registry(name_filter="granite")

        assert len(models) >= 1
        assert any("granite" in m.name.lower() for m in models)

    @pytest.mark.asyncio
    async def test_create_inference_service(self, tools):
        """Test creating an InferenceService manifest."""
        spec = InferenceServiceSpec(
            name="test-service",
            namespace="test-ns",
            model_name="granite-4.0-h-tiny",
            runtime="vllm",
            gpu_count=1,
            min_replicas=1,
            max_replicas=3,
            memory="16Gi",
            storage_uri="s3://models/granite",
        )

        manifest = await tools.create_inference_service(spec)

        assert manifest["kind"] == "InferenceService"
        assert manifest["metadata"]["name"] == "test-service"
        assert manifest["metadata"]["namespace"] == "test-ns"
        assert "predictor" in manifest["spec"]

    @pytest.mark.asyncio
    async def test_get_inference_service_status(self, tools):
        """Test getting InferenceService status."""
        status = await tools.get_inference_service_status("test-service")

        assert "name" in status
        assert "ready" in status
        assert "replicas" in status


class TestTrustyAITools:
    """Tests for TrustyAI tools."""

    @pytest.fixture
    def tools(self):
        """Create TrustyAI tools with default settings."""
        return TrustyAITools()

    def test_mock_evaluation_exists(self, tools):
        """Test that mock evaluations exist."""
        eval_result = tools._get_mock_evaluation("granite-4-tiny")

        assert eval_result.model_id == "granite-4-tiny"
        assert eval_result.overall_trust_score > 0

    def test_mock_evaluation_scores(self, tools):
        """Test mock evaluation score structure."""
        eval_result = tools._get_mock_evaluation("granite-4-tiny")

        assert 0 <= eval_result.hap_score <= 1
        assert 0 <= eval_result.bias_score <= 1
        assert 0 <= eval_result.fairness_score <= 1
        assert 0 <= eval_result.overall_trust_score <= 1

    @pytest.mark.asyncio
    async def test_get_model_evaluation(self, tools):
        """Test getting model evaluation."""
        evaluation = await tools.get_model_evaluation("granite-4-tiny")

        assert evaluation.model_id == "granite-4-tiny"
        assert evaluation.hap_score > 0
        assert evaluation.overall_trust_score > 0

    @pytest.mark.asyncio
    async def test_generate_guardrails_config(self, tools):
        """Test generating guardrails configuration."""
        config = await tools.generate_guardrails_config(
            name="test-guardrails",
            enable_hap=True,
            enable_pii=True,
            enable_prompt_injection=True,
        )

        assert config.name == "test-guardrails"
        assert len(config.detectors) == 3
        detector_names = [d["name"] for d in config.detectors]
        assert "hap-detector" in detector_names
        assert "pii-detector" in detector_names
        assert "prompt-injection-detector" in detector_names

    @pytest.mark.asyncio
    async def test_generate_guardrails_partial(self, tools):
        """Test generating partial guardrails configuration."""
        config = await tools.generate_guardrails_config(
            name="hap-only",
            enable_hap=True,
            enable_pii=False,
            enable_prompt_injection=False,
        )

        assert len(config.detectors) == 1
        assert config.detectors[0]["name"] == "hap-detector"

    @pytest.mark.asyncio
    async def test_generate_guardrails_manifest(self, tools):
        """Test generating Kubernetes manifest for guardrails."""
        config = await tools.generate_guardrails_config(
            name="test-guardrails",
            enable_hap=True,
        )

        manifest = await tools.generate_guardrails_manifest(config)

        assert manifest["kind"] == "GuardrailsOrchestrator"
        assert manifest["metadata"]["name"] == "test-guardrails"
        assert "spec" in manifest


class TestObservabilityTools:
    """Tests for observability tools."""

    @pytest.fixture
    def tools(self):
        """Create observability tools with default settings."""
        return ObservabilityTools()

    def test_mock_metric_latency(self, tools):
        """Test mock latency metric."""
        result = tools._get_mock_metric("request_latency_seconds")

        assert result.metric_name == "request_latency_seconds"
        assert len(result.values) > 0

    def test_mock_metric_gpu(self, tools):
        """Test mock GPU metric."""
        result = tools._get_mock_metric("gpu_utilization")

        assert len(result.values) > 0
        assert 0 <= result.values[0].value <= 1

    @pytest.mark.asyncio
    async def test_get_inference_service_metrics(self, tools):
        """Test getting inference service metrics."""
        metrics = await tools.get_inference_service_metrics("test-service")

        assert "service" in metrics
        assert "latency_p95_ms" in metrics
        assert "throughput_rps" in metrics
        assert "gpu_utilization_pct" in metrics
        assert "memory_gb" in metrics

    @pytest.mark.asyncio
    async def test_check_slo_status(self, tools):
        """Test checking SLO status."""
        slos = await tools.check_slo_status("test-service")

        assert len(slos) > 0
        slo_names = [s.name for s in slos]
        assert "availability" in slo_names
        assert "latency_p95" in slo_names

    @pytest.mark.asyncio
    async def test_get_pod_status(self, tools):
        """Test getting pod status."""
        status = await tools.get_pod_status("test-service")

        assert status.namespace == "ai-navigator"
        assert status.phase in ["Running", "Pending", "Succeeded", "Failed"]
        assert isinstance(status.ready, bool)

    @pytest.mark.asyncio
    async def test_get_pod_logs(self, tools):
        """Test getting pod logs."""
        logs = await tools.get_pod_logs("test-service")

        assert isinstance(logs, list)
        assert len(logs) > 0


class TestModelInfo:
    """Tests for ModelInfo model."""

    def test_model_info_creation(self):
        """Test creating ModelInfo."""
        model = ModelInfo(
            id="test-model",
            name="Test Model",
            version="1.0.0",
            description="A test model",
            format="pytorch",
            size_bytes=1_000_000_000,
            parameters=1_000_000_000,
            tags=["test", "llm"],
            metadata={"publisher": "test"},
        )

        assert model.id == "test-model"
        assert model.name == "Test Model"
        assert model.parameters == 1_000_000_000

    def test_model_info_defaults(self):
        """Test ModelInfo default values."""
        model = ModelInfo(id="test", name="Test")

        assert model.version == "1.0.0"
        assert model.format == "pytorch"
        assert model.tags == []


class TestMetricResult:
    """Tests for MetricResult model."""

    def test_metric_result_creation(self):
        """Test creating MetricResult."""
        now = datetime.now(timezone.utc)
        values = [
            MetricValue(timestamp=now, value=0.5, labels={"quantile": "0.5"}),
            MetricValue(timestamp=now, value=0.9, labels={"quantile": "0.9"}),
        ]

        result = MetricResult(
            metric_name="test_metric",
            values=values,
            result_type="vector",
        )

        assert result.metric_name == "test_metric"
        assert len(result.values) == 2
        assert result.result_type == "vector"


class TestSLOStatus:
    """Tests for SLOStatus model."""

    def test_slo_compliant(self):
        """Test compliant SLO."""
        slo = SLOStatus(
            name="availability",
            target=0.999,
            current=0.9995,
            compliant=True,
            error_budget_remaining=50.0,
        )

        assert slo.compliant is True
        assert slo.current > slo.target

    def test_slo_violated(self):
        """Test violated SLO."""
        slo = SLOStatus(
            name="latency_p95",
            target=0.5,
            current=0.75,
            compliant=False,
            error_budget_remaining=0.0,
        )

        assert slo.compliant is False
        assert slo.current > slo.target
