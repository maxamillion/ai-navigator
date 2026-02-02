"""Capacity planning models."""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class BenchmarkData(BaseModel):
    """Model benchmark/validation data from Model Registry."""

    model_name: str = Field(description="Model identifier")
    model_version: str = Field(description="Model version")
    gpu_type: str = Field(description="GPU type used for benchmark")
    gpu_count: int = Field(default=1, ge=1, description="Number of GPUs")

    # Latency metrics (milliseconds)
    p50_latency_ms: float = Field(ge=0, description="50th percentile latency")
    p95_latency_ms: float = Field(ge=0, description="95th percentile latency")
    p99_latency_ms: float = Field(ge=0, description="99th percentile latency")

    # Throughput metrics
    tokens_per_second: float = Field(ge=0, description="Token generation throughput")
    requests_per_second: float = Field(ge=0, description="Request throughput")

    # Resource usage
    gpu_memory_gb: float = Field(ge=0, description="GPU memory usage")
    gpu_utilization_percent: float = Field(ge=0, le=100, description="GPU utilization")

    # Test conditions
    input_tokens: int = Field(default=512, ge=1, description="Input tokens in test")
    output_tokens: int = Field(default=256, ge=1, description="Output tokens in test")
    batch_size: int = Field(default=1, ge=1, description="Batch size used")
    concurrency: int = Field(default=1, ge=1, description="Concurrency level")

    # Metadata
    benchmark_date: Optional[datetime] = Field(default=None)
    source: str = Field(default="model_registry", description="Data source")


class GPURecommendation(BaseModel):
    """GPU recommendation for model serving."""

    gpu_type: str = Field(description="GPU type (e.g., A100-40GB)")
    gpu_count: int = Field(ge=1, description="Number of GPUs per replica")
    estimated_cost_per_hour: Optional[float] = Field(default=None, ge=0)
    meets_slo: bool = Field(description="Whether this config meets SLO requirements")
    headroom_percent: float = Field(
        default=0, description="Headroom above SLO requirements"
    )
    notes: list[str] = Field(default_factory=list, description="Additional notes")


class CapacityPlan(BaseModel):
    """Complete capacity plan for model deployment."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Model selection
    model_name: str = Field(description="Selected model")
    model_version: Optional[str] = Field(default=None)

    # Replica configuration
    min_replicas: int = Field(ge=0, description="Minimum replicas")
    max_replicas: int = Field(ge=1, description="Maximum replicas")
    target_replicas: int = Field(ge=1, description="Target replica count")

    # GPU configuration
    gpu_type: str = Field(description="GPU type")
    gpu_count: int = Field(ge=1, description="GPUs per replica")
    gpu_memory_gb: float = Field(ge=0, description="GPU memory per GPU")

    # Resource allocation
    memory_per_replica: str = Field(description="Memory request (e.g., '32Gi')")
    cpu_per_replica: str = Field(description="CPU request (e.g., '4')")

    # Expected performance
    estimated_throughput_tps: float = Field(ge=0, description="Expected tokens/sec")
    estimated_rps: float = Field(ge=0, description="Expected requests/sec")
    estimated_p95_latency_ms: float = Field(ge=0)
    estimated_p99_latency_ms: float = Field(ge=0)

    # SLO compliance
    meets_slo: bool = Field(description="Whether plan meets all SLO requirements")
    slo_violations: list[str] = Field(
        default_factory=list, description="List of SLO violations if any"
    )

    # Cost estimate
    estimated_monthly_cost: Optional[float] = Field(default=None, ge=0)

    # Alternative options
    alternatives: list[GPURecommendation] = Field(
        default_factory=list, description="Alternative configurations"
    )

    # Metadata
    benchmark_source: Optional[str] = Field(default=None)
    assumptions: list[str] = Field(default_factory=list)


class WhatIfScenario(BaseModel):
    """What-if analysis input scenario."""

    name: str = Field(description="Scenario name")
    description: Optional[str] = Field(default=None)

    # Traffic changes
    rps_multiplier: float = Field(default=1.0, ge=0, description="RPS multiplier")
    new_rps: Optional[float] = Field(default=None, ge=0, description="New absolute RPS")

    # SLO changes
    p95_latency_ms: Optional[int] = Field(default=None, ge=0)
    p99_latency_ms: Optional[int] = Field(default=None, ge=0)

    # Resource changes
    gpu_type: Optional[str] = Field(default=None)
    max_replicas: Optional[int] = Field(default=None, ge=1)


class WhatIfResult(BaseModel):
    """What-if analysis result."""

    scenario: WhatIfScenario
    original_plan: CapacityPlan
    modified_plan: CapacityPlan

    # Comparison
    replica_delta: int = Field(description="Change in replicas")
    cost_delta_percent: Optional[float] = Field(default=None)
    latency_delta_percent: float = Field(description="Change in p95 latency")
    throughput_delta_percent: float = Field(description="Change in throughput")

    # Feasibility
    is_feasible: bool = Field(description="Whether scenario is achievable")
    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
