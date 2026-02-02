"""Tests for data models."""

import pytest
from datetime import datetime

from ai_navigator.models.workflow import (
    WorkflowState,
    WorkflowStage,
    Message,
    MessageRole,
    TrafficProfile,
    TrafficPattern,
    SLORequirements,
    ModelRequirements,
)
from ai_navigator.models.capacity import (
    BenchmarkData,
    CapacityPlan,
    GPURecommendation,
)
from ai_navigator.models.deployment import (
    DeploymentConfig,
    InferenceServiceSpec,
    RuntimeConfig,
    DeploymentStatus,
    DeploymentResult,
)


class TestWorkflowModels:
    """Test workflow data models."""

    def test_workflow_state_creation(self):
        """Test creating a workflow state."""
        state = WorkflowState(user_id="test-user")

        assert state.id is not None
        assert state.user_id == "test-user"
        assert state.stage == WorkflowStage.INTENT
        assert state.conversation_history == []
        assert not state.is_complete()

    def test_workflow_state_add_message(self):
        """Test adding messages to conversation history."""
        state = WorkflowState(user_id="test-user")
        state.add_message(MessageRole.USER, "Hello")
        state.add_message(MessageRole.ASSISTANT, "Hi there!")

        assert len(state.conversation_history) == 2
        assert state.conversation_history[0].role == MessageRole.USER
        assert state.conversation_history[0].content == "Hello"
        assert state.conversation_history[1].role == MessageRole.ASSISTANT

    def test_workflow_state_advance_stage(self):
        """Test advancing workflow stages."""
        state = WorkflowState(user_id="test-user")

        assert state.stage == WorkflowStage.INTENT
        state.advance_stage()
        assert state.stage == WorkflowStage.TRAFFIC
        state.advance_stage()
        assert state.stage == WorkflowStage.SLO

    def test_workflow_state_complete(self):
        """Test workflow completion check."""
        state = WorkflowState(user_id="test-user")
        assert not state.is_complete()

        # Advance to final stage
        for _ in range(7):  # 8 stages total
            state.advance_stage()

        assert state.stage == WorkflowStage.MONITOR
        assert state.is_complete()

    def test_traffic_profile_creation(self):
        """Test traffic profile creation."""
        profile = TrafficProfile(
            pattern=TrafficPattern.BURST,
            requests_per_second=50,
            peak_rps=150,
            average_input_tokens=256,
            average_output_tokens=128,
        )

        assert profile.pattern == TrafficPattern.BURST
        assert profile.requests_per_second == 50
        assert profile.peak_rps == 150

    def test_slo_requirements_validation(self):
        """Test SLO requirements validation."""
        slo = SLORequirements(
            p50_latency_ms=500,
            p95_latency_ms=1000,
            p99_latency_ms=2000,
            availability_percent=99.9,
        )

        assert slo.p50_latency_ms == 500
        assert slo.p95_latency_ms == 1000
        assert slo.availability_percent == 99.9

    def test_model_requirements(self):
        """Test model requirements creation."""
        requirements = ModelRequirements(
            model_family="llama",
            max_parameters=13,
            capabilities=["chat", "code"],
        )

        assert requirements.model_family == "llama"
        assert requirements.max_parameters == 13
        assert "chat" in requirements.capabilities


class TestCapacityModels:
    """Test capacity planning models."""

    def test_benchmark_data_creation(self):
        """Test benchmark data creation."""
        benchmark = BenchmarkData(
            model_name="llama-2-7b",
            model_version="1.0",
            gpu_type="A100-40GB",
            gpu_count=1,
            p50_latency_ms=150,
            p95_latency_ms=250,
            p99_latency_ms=400,
            tokens_per_second=120,
            requests_per_second=15,
            gpu_memory_gb=14,
            gpu_utilization_percent=85,
        )

        assert benchmark.model_name == "llama-2-7b"
        assert benchmark.gpu_type == "A100-40GB"
        assert benchmark.tokens_per_second == 120

    def test_capacity_plan_creation(self):
        """Test capacity plan creation."""
        plan = CapacityPlan(
            model_name="llama-2-7b",
            min_replicas=2,
            max_replicas=5,
            target_replicas=3,
            gpu_type="A100-40GB",
            gpu_count=1,
            gpu_memory_gb=40,
            memory_per_replica="32Gi",
            cpu_per_replica="8",
            estimated_throughput_tps=360,
            estimated_rps=45,
            estimated_p95_latency_ms=250,
            estimated_p99_latency_ms=400,
            meets_slo=True,
        )

        assert plan.model_name == "llama-2-7b"
        assert plan.min_replicas == 2
        assert plan.max_replicas == 5
        assert plan.meets_slo

    def test_gpu_recommendation(self):
        """Test GPU recommendation creation."""
        rec = GPURecommendation(
            gpu_type="A100-80GB",
            gpu_count=2,
            estimated_cost_per_hour=8.0,
            meets_slo=True,
            headroom_percent=25,
            notes=["Supports larger models"],
        )

        assert rec.gpu_type == "A100-80GB"
        assert rec.gpu_count == 2
        assert rec.meets_slo


class TestDeploymentModels:
    """Test deployment configuration models."""

    def test_runtime_config_creation(self):
        """Test runtime config creation."""
        runtime = RuntimeConfig(
            runtime_name="vllm",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            dtype="float16",
        )

        assert runtime.runtime_name == "vllm"
        assert runtime.tensor_parallel_size == 2

    def test_inference_service_spec(self):
        """Test inference service spec creation."""
        spec = InferenceServiceSpec(
            name="my-model-service",
            namespace="my-project",
            model_name="llama-2-7b",
            min_replicas=2,
            max_replicas=5,
            gpu_count=1,
            memory="32Gi",
            cpu="8",
        )

        assert spec.name == "my-model-service"
        assert spec.namespace == "my-project"
        assert spec.min_replicas == 2

    def test_deployment_config(self):
        """Test deployment config creation."""
        spec = InferenceServiceSpec(
            name="test-service",
            namespace="test-ns",
            model_name="test-model",
        )
        config = DeploymentConfig(
            inference_service=spec,
            create_hpa=True,
            create_pdb=False,
        )

        assert config.inference_service.name == "test-service"
        assert config.create_hpa
        assert not config.create_pdb

    def test_deployment_result(self):
        """Test deployment result creation."""
        result = DeploymentResult(
            deployment_id="test-id",
            status=DeploymentStatus.RUNNING,
            endpoint_url="https://test.example.com",
            ready_replicas=2,
            total_replicas=2,
        )

        assert result.status == DeploymentStatus.RUNNING
        assert result.is_ready()

    def test_deployment_result_not_ready(self):
        """Test deployment result when not ready."""
        result = DeploymentResult(
            deployment_id="test-id",
            status=DeploymentStatus.CREATING,
            ready_replicas=0,
            total_replicas=2,
        )

        assert not result.is_ready()
