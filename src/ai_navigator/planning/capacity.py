"""Core capacity planning algorithms."""

import math
from typing import Optional

import structlog

from ai_navigator.models.capacity import BenchmarkData, CapacityPlan, GPURecommendation
from ai_navigator.models.workflow import SLORequirements, TrafficProfile

logger = structlog.get_logger(__name__)


# GPU specifications and costs
GPU_CATALOG = {
    "A100-80GB": {
        "memory_gb": 80,
        "fp16_tflops": 312,
        "cost_per_hour": 4.00,
        "availability": "high",
    },
    "A100-40GB": {
        "memory_gb": 40,
        "fp16_tflops": 312,
        "cost_per_hour": 2.50,
        "availability": "high",
    },
    "A10": {
        "memory_gb": 24,
        "fp16_tflops": 125,
        "cost_per_hour": 1.00,
        "availability": "high",
    },
    "L4": {
        "memory_gb": 24,
        "fp16_tflops": 121,
        "cost_per_hour": 0.70,
        "availability": "medium",
    },
    "T4": {
        "memory_gb": 16,
        "fp16_tflops": 65,
        "cost_per_hour": 0.35,
        "availability": "high",
    },
    "H100-80GB": {
        "memory_gb": 80,
        "fp16_tflops": 989,
        "cost_per_hour": 8.00,
        "availability": "low",
    },
}

# Model size to memory requirements (approximate)
MODEL_MEMORY_ESTIMATES = {
    "7b": {"fp16_gb": 14, "int8_gb": 7, "int4_gb": 4},
    "13b": {"fp16_gb": 26, "int8_gb": 13, "int4_gb": 7},
    "34b": {"fp16_gb": 68, "int8_gb": 34, "int4_gb": 17},
    "70b": {"fp16_gb": 140, "int8_gb": 70, "int4_gb": 35},
}


class CapacityPlanner:
    """SLO-driven capacity planning engine."""

    def __init__(self) -> None:
        """Initialize capacity planner."""
        self._gpu_catalog = GPU_CATALOG

    def calculate_capacity_plan(
        self,
        benchmark: BenchmarkData,
        traffic_profile: TrafficProfile,
        slo_requirements: SLORequirements,
        availability_zones: int = 1,
    ) -> CapacityPlan:
        """Calculate a complete capacity plan from requirements."""
        # Calculate base replica count for RPS
        min_replicas = self._calculate_min_replicas(
            benchmark=benchmark,
            target_rps=traffic_profile.requests_per_second,
            slo=slo_requirements,
        )

        # Account for burst traffic
        peak_rps = traffic_profile.peak_rps or traffic_profile.requests_per_second * 2
        max_replicas = self._calculate_max_replicas(
            benchmark=benchmark,
            peak_rps=peak_rps,
        )

        # Ensure HA across zones
        if availability_zones > 1:
            min_replicas = max(min_replicas, availability_zones)

        # Target replicas for steady-state
        target_replicas = self._calculate_target_replicas(
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            traffic_pattern=traffic_profile.pattern,
        )

        # Calculate resource allocation
        memory_per_replica = self._calculate_memory_allocation(benchmark)
        cpu_per_replica = self._calculate_cpu_allocation(benchmark)

        # Estimate performance
        total_rps = target_replicas * benchmark.requests_per_second
        total_tps = target_replicas * benchmark.tokens_per_second

        # Check SLO compliance
        meets_slo, violations = self._check_slo_compliance(benchmark, slo_requirements)

        # Calculate cost
        gpu_spec = self._gpu_catalog.get(benchmark.gpu_type, self._gpu_catalog["A100-40GB"])
        hourly_cost = gpu_spec["cost_per_hour"] * benchmark.gpu_count * target_replicas
        monthly_cost = hourly_cost * 24 * 30

        # Generate alternatives
        alternatives = self._generate_alternatives(
            benchmark=benchmark,
            target_rps=traffic_profile.requests_per_second,
            slo=slo_requirements,
        )

        return CapacityPlan(
            model_name=benchmark.model_name,
            model_version=benchmark.model_version,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            target_replicas=target_replicas,
            gpu_type=benchmark.gpu_type,
            gpu_count=benchmark.gpu_count,
            gpu_memory_gb=gpu_spec["memory_gb"],
            memory_per_replica=memory_per_replica,
            cpu_per_replica=cpu_per_replica,
            estimated_throughput_tps=total_tps,
            estimated_rps=total_rps,
            estimated_p95_latency_ms=benchmark.p95_latency_ms,
            estimated_p99_latency_ms=benchmark.p99_latency_ms,
            meets_slo=meets_slo,
            slo_violations=violations,
            estimated_monthly_cost=monthly_cost,
            alternatives=alternatives,
            benchmark_source="model_registry",
            assumptions=self._generate_assumptions(benchmark, traffic_profile),
        )

    def estimate_replicas_for_rps(
        self,
        benchmark: BenchmarkData,
        target_rps: float,
        headroom_percent: float = 20.0,
    ) -> int:
        """Estimate replica count needed for target RPS."""
        rps_per_replica = benchmark.requests_per_second
        if rps_per_replica <= 0:
            return 1

        # Add headroom
        effective_rps = target_rps * (1 + headroom_percent / 100)

        return max(1, math.ceil(effective_rps / rps_per_replica))

    def estimate_gpu_memory(
        self,
        model_size_billions: float,
        quantization: str = "fp16",
        context_length: int = 4096,
        batch_size: int = 1,
    ) -> float:
        """Estimate GPU memory required for a model."""
        # Base memory for model weights
        bytes_per_param = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "int8": 1,
            "int4": 0.5,
        }

        param_bytes = bytes_per_param.get(quantization, 2)
        weight_memory_gb = (model_size_billions * 1e9 * param_bytes) / (1024**3)

        # KV cache memory (rough estimate)
        # Assuming 2 bytes per token per layer, 32 layers for 7B model
        layers = int(model_size_billions * 4.5)  # Rough layer count
        kv_cache_gb = (context_length * batch_size * layers * 2 * 2) / (1024**3)

        # Activation memory (10-20% of weights)
        activation_gb = weight_memory_gb * 0.15

        total_gb = weight_memory_gb + kv_cache_gb + activation_gb

        # Add 20% buffer
        return total_gb * 1.2

    def _calculate_min_replicas(
        self,
        benchmark: BenchmarkData,
        target_rps: float,
        slo: SLORequirements,
    ) -> int:
        """Calculate minimum replicas for SLO compliance."""
        # Base calculation from RPS
        base_replicas = self.estimate_replicas_for_rps(benchmark, target_rps)

        # Ensure at least 2 for high availability if required
        if slo.availability_percent >= 99.9:
            base_replicas = max(base_replicas, 2)
        if slo.availability_percent >= 99.99:
            base_replicas = max(base_replicas, 3)

        return base_replicas

    def _calculate_max_replicas(
        self,
        benchmark: BenchmarkData,
        peak_rps: float,
    ) -> int:
        """Calculate maximum replicas for peak traffic."""
        peak_replicas = self.estimate_replicas_for_rps(benchmark, peak_rps, headroom_percent=10)
        return max(peak_replicas, 3)  # At least 3 for autoscaling headroom

    def _calculate_target_replicas(
        self,
        min_replicas: int,
        max_replicas: int,
        traffic_pattern: str,
    ) -> int:
        """Calculate target replicas for steady-state."""
        from ai_navigator.models.workflow import TrafficPattern

        if traffic_pattern == TrafficPattern.STEADY:
            return min_replicas
        elif traffic_pattern == TrafficPattern.BURST:
            # Keep more replicas warm for bursts
            return min(min_replicas + 1, max_replicas)
        elif traffic_pattern == TrafficPattern.GROWTH:
            # Plan for growth
            return min((min_replicas + max_replicas) // 2, max_replicas)
        else:
            return min_replicas

    def _calculate_memory_allocation(self, benchmark: BenchmarkData) -> str:
        """Calculate memory allocation per replica."""
        # GPU memory + overhead for system
        total_gb = int(benchmark.gpu_memory_gb * 1.3)
        return f"{total_gb}Gi"

    def _calculate_cpu_allocation(self, benchmark: BenchmarkData) -> str:
        """Calculate CPU allocation per replica."""
        # Base CPU allocation scales with GPU count
        base_cpu = 4 + (benchmark.gpu_count * 2)
        return str(base_cpu)

    def _check_slo_compliance(
        self,
        benchmark: BenchmarkData,
        slo: SLORequirements,
    ) -> tuple[bool, list[str]]:
        """Check if benchmark meets SLO requirements."""
        violations = []

        if benchmark.p50_latency_ms > slo.p50_latency_ms:
            violations.append(
                f"p50 latency {benchmark.p50_latency_ms}ms exceeds target {slo.p50_latency_ms}ms"
            )

        if benchmark.p95_latency_ms > slo.p95_latency_ms:
            violations.append(
                f"p95 latency {benchmark.p95_latency_ms}ms exceeds target {slo.p95_latency_ms}ms"
            )

        if benchmark.p99_latency_ms > slo.p99_latency_ms:
            violations.append(
                f"p99 latency {benchmark.p99_latency_ms}ms exceeds target {slo.p99_latency_ms}ms"
            )

        if slo.max_tokens_per_second:
            if benchmark.tokens_per_second < slo.max_tokens_per_second:
                violations.append(
                    f"throughput {benchmark.tokens_per_second} tps below target {slo.max_tokens_per_second} tps"
                )

        return len(violations) == 0, violations

    def _generate_alternatives(
        self,
        benchmark: BenchmarkData,
        target_rps: float,
        slo: SLORequirements,
    ) -> list[GPURecommendation]:
        """Generate alternative GPU configurations."""
        alternatives = []

        for gpu_type, spec in self._gpu_catalog.items():
            if gpu_type == benchmark.gpu_type:
                continue

            # Only suggest if memory is sufficient
            if spec["memory_gb"] < benchmark.gpu_memory_gb:
                continue

            # Estimate performance scaling (rough)
            perf_ratio = spec["fp16_tflops"] / self._gpu_catalog.get(
                benchmark.gpu_type, {"fp16_tflops": 312}
            )["fp16_tflops"]

            estimated_latency = benchmark.p95_latency_ms / perf_ratio
            meets_slo = estimated_latency <= slo.p95_latency_ms

            notes = []
            if spec["memory_gb"] > benchmark.gpu_memory_gb * 1.5:
                notes.append("Significantly more memory available")
            if perf_ratio > 1.2:
                notes.append(f"~{int((perf_ratio - 1) * 100)}% faster")
            elif perf_ratio < 0.8:
                notes.append(f"~{int((1 - perf_ratio) * 100)}% slower")

            alternatives.append(
                GPURecommendation(
                    gpu_type=gpu_type,
                    gpu_count=benchmark.gpu_count,
                    estimated_cost_per_hour=spec["cost_per_hour"] * benchmark.gpu_count,
                    meets_slo=meets_slo,
                    headroom_percent=max(0, (slo.p95_latency_ms - estimated_latency) / slo.p95_latency_ms * 100),
                    notes=notes,
                )
            )

        # Sort by cost-effectiveness (meets SLO first, then by cost)
        alternatives.sort(key=lambda x: (not x.meets_slo, x.estimated_cost_per_hour or 0))

        return alternatives[:5]

    def _generate_assumptions(
        self,
        benchmark: BenchmarkData,
        traffic: TrafficProfile,
    ) -> list[str]:
        """Generate list of assumptions for the capacity plan."""
        return [
            f"Based on benchmark with {benchmark.input_tokens} input tokens, {benchmark.output_tokens} output tokens",
            f"Assuming {traffic.average_input_tokens} avg input tokens, {traffic.average_output_tokens} avg output tokens",
            f"GPU utilization target: {benchmark.gpu_utilization_percent}%",
            "Includes 20% headroom for traffic spikes",
            f"Benchmark concurrency: {benchmark.concurrency}",
        ]
