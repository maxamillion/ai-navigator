"""Data models for Model Registry entities."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class CustomProperties(BaseModel):
    """Custom properties for model metadata."""

    properties: dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a property value."""
        return self.properties.get(key, default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float property value."""
        value = self.properties.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_int(self, key: str, default: int = 0) -> int:
        """Get an int property value."""
        value = self.properties.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default


class ModelArtifact(BaseModel):
    """Model artifact in the registry."""

    id: str
    name: str
    uri: str
    description: Optional[str] = None
    model_format_name: Optional[str] = None
    model_format_version: Optional[str] = None
    storage_key: Optional[str] = None
    storage_path: Optional[str] = None
    service_account_name: Optional[str] = None
    custom_properties: CustomProperties = Field(default_factory=CustomProperties)
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None


class ModelVersion(BaseModel):
    """Model version in the registry."""

    id: str
    name: str
    registered_model_id: str
    state: str = "LIVE"
    description: Optional[str] = None
    author: Optional[str] = None
    custom_properties: CustomProperties = Field(default_factory=CustomProperties)
    artifacts: list[ModelArtifact] = Field(default_factory=list)
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None


class RegisteredModel(BaseModel):
    """Registered model in the registry."""

    id: str
    name: str
    description: Optional[str] = None
    owner: Optional[str] = None
    state: str = "LIVE"
    custom_properties: CustomProperties = Field(default_factory=CustomProperties)
    versions: list[ModelVersion] = Field(default_factory=list)
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None

    def get_latest_version(self) -> Optional[ModelVersion]:
        """Get the most recent version."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.create_time or datetime.min)


class ValidationMetrics(BaseModel):
    """Validation/benchmark metrics from experiment runs."""

    model_name: str
    model_version: str
    run_id: Optional[str] = None

    # Latency metrics (milliseconds)
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    mean_latency_ms: Optional[float] = None

    # Throughput metrics
    tokens_per_second: Optional[float] = None
    requests_per_second: Optional[float] = None

    # Resource metrics
    gpu_memory_gb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    peak_memory_gb: Optional[float] = None

    # Test conditions
    gpu_type: Optional[str] = None
    gpu_count: int = 1
    input_tokens: int = 512
    output_tokens: int = 256
    batch_size: int = 1
    concurrency: int = 1
    tensor_parallel_size: int = 1

    # Quality metrics (optional)
    accuracy: Optional[float] = None
    perplexity: Optional[float] = None

    # Metadata
    benchmark_date: Optional[datetime] = None
    notes: Optional[str] = None


class MetricHistoryPoint(BaseModel):
    """Single point in metric history."""

    step: int
    timestamp: Optional[datetime] = None
    value: float


class MetricHistory(BaseModel):
    """Metric history from an experiment run."""

    metric_name: str
    run_id: str
    history: list[MetricHistoryPoint] = Field(default_factory=list)

    def get_last_value(self) -> Optional[float]:
        """Get the most recent metric value."""
        if not self.history:
            return None
        return self.history[-1].value

    def get_average(self) -> Optional[float]:
        """Get average metric value."""
        if not self.history:
            return None
        return sum(p.value for p in self.history) / len(self.history)
