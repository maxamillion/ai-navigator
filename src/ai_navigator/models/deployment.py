"""Deployment configuration models."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DeploymentStatus(str, Enum):
    """Deployment status."""

    PENDING = "pending"
    CREATING = "creating"
    RUNNING = "running"
    UPDATING = "updating"
    FAILED = "failed"
    DELETED = "deleted"


class RuntimeConfig(BaseModel):
    """Serving runtime configuration."""

    runtime_name: str = Field(description="Runtime name (e.g., vllm, openvino)")
    runtime_version: Optional[str] = Field(default=None)

    # vLLM-specific settings
    tensor_parallel_size: int = Field(default=1, ge=1)
    max_model_len: Optional[int] = Field(default=None, ge=1)
    gpu_memory_utilization: float = Field(default=0.9, ge=0, le=1)
    enforce_eager: bool = Field(default=False)
    dtype: str = Field(default="auto")

    # General settings
    env_vars: dict[str, str] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)


class InferenceServiceSpec(BaseModel):
    """KServe InferenceService specification."""

    name: str = Field(description="Service name")
    namespace: str = Field(description="Target namespace")

    # Model configuration
    model_name: str = Field(description="Model name/path")
    model_format: str = Field(default="pytorch", description="Model format")
    storage_uri: Optional[str] = Field(default=None, description="Model storage URI")

    # Runtime
    runtime: Optional[RuntimeConfig] = Field(default=None)

    # Resources
    min_replicas: int = Field(default=1, ge=0)
    max_replicas: int = Field(default=3, ge=1)
    gpu_count: int = Field(default=1, ge=1)
    gpu_type: Optional[str] = Field(default=None)
    memory: str = Field(default="16Gi")
    cpu: str = Field(default="4")

    # Autoscaling
    scale_target: int = Field(default=100, description="Concurrent requests target")
    scale_metric: str = Field(default="concurrency", description="Scaling metric")

    # Networking
    enable_route: bool = Field(default=True)
    enable_auth: bool = Field(default=True)

    # Timeouts
    timeout_seconds: int = Field(default=60)

    # Labels and annotations
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)


class DeploymentConfig(BaseModel):
    """Complete deployment configuration."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Core configuration
    inference_service: InferenceServiceSpec

    # Data connection
    data_connection_name: Optional[str] = Field(default=None)
    data_connection_secret: Optional[str] = Field(default=None)

    # Additional resources
    create_hpa: bool = Field(default=True, description="Create HorizontalPodAutoscaler")
    create_pdb: bool = Field(default=False, description="Create PodDisruptionBudget")

    # Generated manifests (stored after generation)
    manifests: dict[str, str] = Field(
        default_factory=dict, description="Generated YAML manifests"
    )

    # Validation results
    validation_passed: bool = Field(default=False)
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)


class DeploymentResult(BaseModel):
    """Result of a deployment operation."""

    deployment_id: str = Field(description="Deployment configuration ID")
    status: DeploymentStatus
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Deployed resources
    inference_service_name: Optional[str] = Field(default=None)
    namespace: Optional[str] = Field(default=None)
    endpoint_url: Optional[str] = Field(default=None)

    # Status details
    message: str = Field(default="")
    ready_replicas: int = Field(default=0)
    total_replicas: int = Field(default=0)

    # Conditions
    conditions: list[dict[str, Any]] = Field(default_factory=list)

    # Error details
    error: Optional[str] = Field(default=None)
    error_details: Optional[dict[str, Any]] = Field(default=None)

    # Test results
    inference_test_passed: Optional[bool] = Field(default=None)
    inference_test_latency_ms: Optional[float] = Field(default=None)

    def is_ready(self) -> bool:
        """Check if deployment is ready to serve traffic."""
        return (
            self.status == DeploymentStatus.RUNNING
            and self.ready_replicas > 0
            and self.endpoint_url is not None
        )
