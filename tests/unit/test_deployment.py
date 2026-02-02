"""Tests for deployment automation."""

import pytest
import yaml

from ai_navigator.deployment.generator import YAMLGenerator
from ai_navigator.deployment.validators import DeploymentValidator, ValidationResult
from ai_navigator.models.deployment import (
    DeploymentConfig,
    InferenceServiceSpec,
    RuntimeConfig,
)


class TestYAMLGenerator:
    """Test YAML generator."""

    @pytest.fixture
    def generator(self):
        """Create YAML generator."""
        return YAMLGenerator()

    @pytest.fixture
    def basic_spec(self):
        """Create basic inference service spec."""
        return InferenceServiceSpec(
            name="test-model-service",
            namespace="test-project",
            model_name="llama-2-7b",
            min_replicas=2,
            max_replicas=5,
            gpu_count=1,
            gpu_type="A100-40GB",
            memory="32Gi",
            cpu="8",
        )

    @pytest.fixture
    def full_spec(self):
        """Create full inference service spec with runtime."""
        runtime = RuntimeConfig(
            runtime_name="vllm",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            dtype="float16",
        )
        return InferenceServiceSpec(
            name="llama-service",
            namespace="ai-project",
            model_name="llama-2-70b",
            model_format="pytorch",
            storage_uri="s3://models/llama-2-70b",
            runtime=runtime,
            min_replicas=1,
            max_replicas=3,
            gpu_count=4,
            gpu_type="A100-80GB",
            memory="256Gi",
            cpu="32",
            scale_target=10,
            enable_route=True,
            enable_auth=True,
        )

    def test_generate_inference_service(self, generator, basic_spec):
        """Test generating InferenceService manifest."""
        yaml_str = generator.generate_inference_service(basic_spec)
        manifest = yaml.safe_load(yaml_str)

        assert manifest["apiVersion"] == "serving.kserve.io/v1beta1"
        assert manifest["kind"] == "InferenceService"
        assert manifest["metadata"]["name"] == "test-model-service"
        assert manifest["metadata"]["namespace"] == "test-project"
        assert manifest["spec"]["predictor"]["minReplicas"] == 2
        assert manifest["spec"]["predictor"]["maxReplicas"] == 5

    def test_generate_inference_service_with_runtime(self, generator, full_spec):
        """Test generating InferenceService with runtime config."""
        yaml_str = generator.generate_inference_service(full_spec)
        manifest = yaml.safe_load(yaml_str)

        model_spec = manifest["spec"]["predictor"]["model"]
        assert model_spec["runtime"] == "vllm"
        assert "--tensor-parallel-size=2" in model_spec.get("args", [])
        assert model_spec["storageUri"] == "s3://models/llama-2-70b"

    def test_generate_hpa(self, generator, basic_spec):
        """Test generating HorizontalPodAutoscaler manifest."""
        yaml_str = generator.generate_hpa(basic_spec)
        manifest = yaml.safe_load(yaml_str)

        assert manifest["apiVersion"] == "autoscaling/v2"
        assert manifest["kind"] == "HorizontalPodAutoscaler"
        assert manifest["spec"]["minReplicas"] == 2
        assert manifest["spec"]["maxReplicas"] == 5

    def test_generate_pdb(self, generator, basic_spec):
        """Test generating PodDisruptionBudget manifest."""
        yaml_str = generator.generate_pdb(basic_spec)
        manifest = yaml.safe_load(yaml_str)

        assert manifest["apiVersion"] == "policy/v1"
        assert manifest["kind"] == "PodDisruptionBudget"
        assert manifest["spec"]["minAvailable"] == 1

    def test_generate_all_manifests(self, generator, basic_spec):
        """Test generating all manifests."""
        config = DeploymentConfig(
            inference_service=basic_spec,
            create_hpa=True,
            create_pdb=True,
        )

        manifests = generator.generate_all(config)

        assert "inferenceservice.yaml" in manifests
        assert "hpa.yaml" in manifests
        assert "pdb.yaml" in manifests

    def test_generate_serving_runtime(self, generator):
        """Test generating ServingRuntime manifest."""
        yaml_str = generator.generate_serving_runtime(
            name="vllm-runtime",
            namespace="ai-project",
            gpu_type="A100-80GB",
        )
        manifest = yaml.safe_load(yaml_str)

        assert manifest["kind"] == "ServingRuntime"
        assert manifest["metadata"]["name"] == "vllm-runtime"


class TestDeploymentValidator:
    """Test deployment validator."""

    @pytest.fixture
    def validator(self):
        """Create deployment validator."""
        return DeploymentValidator()

    @pytest.fixture
    def valid_config(self):
        """Create valid deployment config."""
        spec = InferenceServiceSpec(
            name="valid-service",
            namespace="valid-project",
            model_name="llama-2-7b",
            min_replicas=2,
            max_replicas=5,
            gpu_count=1,
            memory="32Gi",
            cpu="8",
        )
        return DeploymentConfig(inference_service=spec)

    @pytest.mark.asyncio
    async def test_validate_valid_config(self, validator, valid_config):
        """Test validating a valid configuration."""
        result = await validator.validate(valid_config)

        assert result.passed
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_missing_name(self, validator):
        """Test validation catches missing name."""
        spec = InferenceServiceSpec(
            name="",
            namespace="test",
            model_name="test-model",
        )
        config = DeploymentConfig(inference_service=spec)

        result = await validator.validate(config)

        assert not result.passed
        assert any("name" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_validate_invalid_replicas(self, validator):
        """Test validation catches invalid replica config."""
        spec = InferenceServiceSpec(
            name="test",
            namespace="test",
            model_name="test-model",
            min_replicas=5,
            max_replicas=2,  # Invalid: less than min
        )
        config = DeploymentConfig(inference_service=spec)

        result = await validator.validate(config)

        assert not result.passed
        assert any("replica" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_validate_invalid_memory_format(self, validator):
        """Test validation catches invalid memory format."""
        spec = InferenceServiceSpec(
            name="test",
            namespace="test",
            model_name="test-model",
            memory="invalid",
        )
        config = DeploymentConfig(inference_service=spec)

        result = await validator.validate(config)

        assert not result.passed

    @pytest.mark.asyncio
    async def test_validate_vllm_runtime(self, validator):
        """Test validation of vLLM runtime config."""
        runtime = RuntimeConfig(
            runtime_name="vllm",
            tensor_parallel_size=4,
            gpu_memory_utilization=0.95,
        )
        spec = InferenceServiceSpec(
            name="test",
            namespace="test",
            model_name="test-model",
            gpu_count=2,  # Less than tensor_parallel_size
            runtime=runtime,
        )
        config = DeploymentConfig(inference_service=spec)

        result = await validator.validate(config)

        assert not result.passed
        assert any("tensor_parallel" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_validate_warnings(self, validator):
        """Test that validator produces warnings."""
        spec = InferenceServiceSpec(
            name="test",
            namespace="test",
            model_name="test-model",
            max_replicas=200,  # Excessive
        )
        config = DeploymentConfig(inference_service=spec)

        result = await validator.validate(config)

        # Should pass but have warnings
        assert len(result.warnings) > 0
