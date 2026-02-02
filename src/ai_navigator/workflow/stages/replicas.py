"""Replica planning stage - Step 6."""

import math
from typing import Optional

import structlog

from ai_navigator.models.capacity import BenchmarkData, CapacityPlan, GPURecommendation
from ai_navigator.models.workflow import WorkflowState
from ai_navigator.workflow.stages.base import BaseStage

logger = structlog.get_logger(__name__)


class StageResult:
    def __init__(
        self,
        success: bool,
        message: str,
        advance: bool = False,
        data: Optional[dict] = None,
        prompt_user: Optional[str] = None,
    ):
        self.success = success
        self.message = message
        self.advance = advance
        self.data = data or {}
        self.prompt_user = prompt_user


# GPU specifications for cost/resource calculations
GPU_SPECS = {
    "A100-40GB": {"memory_gb": 40, "cost_per_hour": 2.50},
    "A100-80GB": {"memory_gb": 80, "cost_per_hour": 4.00},
    "A10": {"memory_gb": 24, "cost_per_hour": 1.00},
    "L4": {"memory_gb": 24, "cost_per_hour": 0.70},
    "T4": {"memory_gb": 16, "cost_per_hour": 0.35},
}


class ReplicasStage(BaseStage):
    """Stage 6: Calculate required replicas and resources."""

    async def process(self, state: WorkflowState, user_input: str) -> StageResult:
        """Calculate capacity plan based on traffic and SLO requirements."""
        if not state.traffic_profile or not state.slo_requirements:
            return StageResult(
                success=False,
                message="Traffic profile and SLO requirements are needed for capacity planning.",
            )

        # Get selected benchmark data
        benchmarks_data = state.metadata.get("benchmarks", [])
        selected_names = state.selected_models

        if not benchmarks_data:
            # Use default estimation
            return self._estimate_without_benchmarks(state)

        # Find the best matching benchmark
        benchmarks = [BenchmarkData.model_validate(b) for b in benchmarks_data]
        selected_benchmark = None

        for b in benchmarks:
            if b.model_name in selected_names or not selected_names:
                selected_benchmark = b
                break

        if not selected_benchmark:
            selected_benchmark = benchmarks[0]

        # Calculate capacity plan
        plan = self._calculate_capacity_plan(state, selected_benchmark)

        # Store in state
        state.capacity_plan_id = plan.id
        state.metadata["capacity_plan"] = plan.model_dump()

        # Build response
        response = self._format_capacity_plan(plan)

        return StageResult(
            success=True,
            message=response,
            advance=True,
            data={"capacity_plan": plan.model_dump()},
        )

    def _calculate_capacity_plan(
        self,
        state: WorkflowState,
        benchmark: BenchmarkData,
    ) -> CapacityPlan:
        """Calculate capacity plan from benchmark and requirements."""
        traffic = state.traffic_profile
        slo = state.slo_requirements

        # Calculate required replicas for RPS target
        rps_per_replica = benchmark.requests_per_second
        target_rps = traffic.requests_per_second if traffic else 10

        # Add headroom for SLO compliance (20%)
        target_with_headroom = target_rps * 1.2

        min_replicas_for_rps = math.ceil(target_with_headroom / rps_per_replica)

        # Account for burst traffic
        peak_rps = traffic.peak_rps if traffic and traffic.peak_rps else target_rps * 2
        max_replicas_for_peak = math.ceil((peak_rps * 1.2) / rps_per_replica)

        # Ensure minimum for availability
        min_replicas = max(min_replicas_for_rps, 2)  # At least 2 for HA

        # Calculate max replicas
        max_replicas = max(max_replicas_for_peak, min_replicas + 2)

        # Target replicas (steady state)
        target_replicas = min_replicas

        # Get GPU specs
        gpu_spec = GPU_SPECS.get(benchmark.gpu_type, GPU_SPECS["A100-40GB"])

        # Calculate estimated performance
        total_rps = target_replicas * rps_per_replica
        total_tps = target_replicas * benchmark.tokens_per_second

        # Check SLO compliance
        meets_slo = (
            benchmark.p95_latency_ms <= (slo.p95_latency_ms if slo else 3000)
            and benchmark.p99_latency_ms <= (slo.p99_latency_ms if slo else 5000)
        )

        violations = []
        if slo:
            if benchmark.p95_latency_ms > slo.p95_latency_ms:
                violations.append(f"p95 latency exceeds target by {benchmark.p95_latency_ms - slo.p95_latency_ms}ms")
            if benchmark.p99_latency_ms > slo.p99_latency_ms:
                violations.append(f"p99 latency exceeds target by {benchmark.p99_latency_ms - slo.p99_latency_ms}ms")

        # Calculate memory requirements
        memory_per_replica = f"{int(benchmark.gpu_memory_gb * 1.2)}Gi"
        cpu_per_replica = "8"  # Standard CPU allocation

        # Cost estimate
        gpu_cost = gpu_spec["cost_per_hour"] * benchmark.gpu_count * target_replicas
        monthly_cost = gpu_cost * 24 * 30

        # Generate alternatives
        alternatives = self._generate_alternatives(benchmark, state)

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
            assumptions=[
                f"Based on benchmark with {benchmark.input_tokens} input tokens, {benchmark.output_tokens} output tokens",
                f"GPU utilization target: {benchmark.gpu_utilization_percent}%",
                "Includes 20% headroom for traffic spikes",
            ],
        )

    def _generate_alternatives(
        self,
        current: BenchmarkData,
        state: WorkflowState,
    ) -> list[GPURecommendation]:
        """Generate alternative GPU configurations."""
        alternatives = []

        # Suggest scaling options
        for gpu_type, spec in GPU_SPECS.items():
            if gpu_type == current.gpu_type:
                continue

            # Only suggest if memory is sufficient
            if spec["memory_gb"] < current.gpu_memory_gb:
                continue

            meets_slo = True
            notes = []

            if spec["memory_gb"] > current.gpu_memory_gb:
                notes.append("More GPU memory - may support larger batch sizes")

            alternatives.append(
                GPURecommendation(
                    gpu_type=gpu_type,
                    gpu_count=current.gpu_count,
                    estimated_cost_per_hour=spec["cost_per_hour"] * current.gpu_count,
                    meets_slo=meets_slo,
                    headroom_percent=0,
                    notes=notes,
                )
            )

        return alternatives[:3]  # Return top 3

    def _estimate_without_benchmarks(self, state: WorkflowState) -> StageResult:
        """Provide estimation when no benchmark data is available."""
        traffic = state.traffic_profile
        target_rps = traffic.requests_per_second if traffic else 10

        # Conservative estimates
        estimated_rps_per_replica = 5  # Conservative
        min_replicas = max(math.ceil(target_rps / estimated_rps_per_replica), 2)

        return StageResult(
            success=True,
            message=(
                "No benchmark data available. Using conservative estimates:\n\n"
                f"- Estimated replicas needed: {min_replicas}-{min_replicas + 2}\n"
                f"- Recommended GPU: A100-40GB\n"
                f"- Recommended memory: 32Gi per replica\n\n"
                "Note: These are estimates. Monitor actual performance and adjust as needed."
            ),
            advance=True,
            data={
                "estimated": True,
                "min_replicas": min_replicas,
                "max_replicas": min_replicas + 2,
            },
        )

    def _format_capacity_plan(self, plan: CapacityPlan) -> str:
        """Format capacity plan as readable message."""
        lines = [
            "**Capacity Plan**\n",
            f"Model: {plan.model_name}",
            f"GPU: {plan.gpu_type} x{plan.gpu_count}",
            "",
            "**Replica Configuration**",
            f"- Minimum: {plan.min_replicas} replicas",
            f"- Target: {plan.target_replicas} replicas",
            f"- Maximum: {plan.max_replicas} replicas",
            "",
            "**Resources per Replica**",
            f"- Memory: {plan.memory_per_replica}",
            f"- CPU: {plan.cpu_per_replica}",
            f"- GPU Memory: {plan.gpu_memory_gb}GB",
            "",
            "**Expected Performance**",
            f"- Throughput: {plan.estimated_throughput_tps:.0f} tokens/sec total",
            f"- Request capacity: {plan.estimated_rps:.0f} RPS",
            f"- p95 latency: {plan.estimated_p95_latency_ms}ms",
            f"- p99 latency: {plan.estimated_p99_latency_ms}ms",
        ]

        if plan.estimated_monthly_cost:
            lines.extend([
                "",
                f"**Estimated Cost**: ${plan.estimated_monthly_cost:.2f}/month",
            ])

        if plan.meets_slo:
            lines.append("\nSLO Status: All requirements met")
        else:
            lines.append(f"\nSLO Violations: {', '.join(plan.slo_violations)}")

        if plan.alternatives:
            lines.extend(["\n**Alternative Configurations**"])
            for alt in plan.alternatives[:2]:
                lines.append(
                    f"- {alt.gpu_type} x{alt.gpu_count}: ${alt.estimated_cost_per_hour:.2f}/hour"
                )

        return "\n".join(lines)
