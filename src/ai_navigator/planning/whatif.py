"""What-if analysis for capacity planning."""

from typing import Optional

import structlog

from ai_navigator.models.capacity import (
    BenchmarkData,
    CapacityPlan,
    WhatIfScenario,
    WhatIfResult,
)
from ai_navigator.models.workflow import SLORequirements, TrafficProfile
from ai_navigator.planning.capacity import CapacityPlanner

logger = structlog.get_logger(__name__)


class WhatIfAnalyzer:
    """Analyzes what-if scenarios for capacity planning."""

    def __init__(self, planner: Optional[CapacityPlanner] = None) -> None:
        """Initialize analyzer."""
        self._planner = planner or CapacityPlanner()

    def analyze_scenario(
        self,
        original_plan: CapacityPlan,
        scenario: WhatIfScenario,
        benchmark: BenchmarkData,
        traffic_profile: TrafficProfile,
        slo_requirements: SLORequirements,
    ) -> WhatIfResult:
        """Analyze a what-if scenario against the original plan."""
        # Apply scenario modifications
        modified_traffic = self._apply_traffic_changes(traffic_profile, scenario)
        modified_slo = self._apply_slo_changes(slo_requirements, scenario)
        modified_benchmark = self._apply_benchmark_changes(benchmark, scenario)

        # Calculate new capacity plan
        modified_plan = self._planner.calculate_capacity_plan(
            benchmark=modified_benchmark,
            traffic_profile=modified_traffic,
            slo_requirements=modified_slo,
        )

        # Calculate deltas
        replica_delta = modified_plan.target_replicas - original_plan.target_replicas

        cost_delta = None
        if original_plan.estimated_monthly_cost and modified_plan.estimated_monthly_cost:
            cost_delta = (
                (modified_plan.estimated_monthly_cost - original_plan.estimated_monthly_cost)
                / original_plan.estimated_monthly_cost
                * 100
            )

        latency_delta = (
            (modified_plan.estimated_p95_latency_ms - original_plan.estimated_p95_latency_ms)
            / original_plan.estimated_p95_latency_ms
            * 100
        )

        throughput_delta = (
            (modified_plan.estimated_throughput_tps - original_plan.estimated_throughput_tps)
            / original_plan.estimated_throughput_tps
            * 100
        )

        # Check feasibility
        is_feasible, warnings, recommendations = self._assess_feasibility(
            original_plan, modified_plan, scenario
        )

        return WhatIfResult(
            scenario=scenario,
            original_plan=original_plan,
            modified_plan=modified_plan,
            replica_delta=replica_delta,
            cost_delta_percent=cost_delta,
            latency_delta_percent=latency_delta,
            throughput_delta_percent=throughput_delta,
            is_feasible=is_feasible,
            warnings=warnings,
            recommendations=recommendations,
        )

    def analyze_traffic_growth(
        self,
        original_plan: CapacityPlan,
        benchmark: BenchmarkData,
        traffic_profile: TrafficProfile,
        slo_requirements: SLORequirements,
        growth_percentages: list[float] = [50, 100, 200, 500],
    ) -> list[WhatIfResult]:
        """Analyze multiple traffic growth scenarios."""
        results = []

        for growth in growth_percentages:
            scenario = WhatIfScenario(
                name=f"{int(growth)}% traffic growth",
                description=f"Traffic increases by {growth}%",
                rps_multiplier=1 + growth / 100,
            )

            result = self.analyze_scenario(
                original_plan=original_plan,
                scenario=scenario,
                benchmark=benchmark,
                traffic_profile=traffic_profile,
                slo_requirements=slo_requirements,
            )
            results.append(result)

        return results

    def analyze_slo_tightening(
        self,
        original_plan: CapacityPlan,
        benchmark: BenchmarkData,
        traffic_profile: TrafficProfile,
        slo_requirements: SLORequirements,
        latency_reductions: list[float] = [25, 50, 75],
    ) -> list[WhatIfResult]:
        """Analyze scenarios with tighter SLO requirements."""
        results = []

        for reduction in latency_reductions:
            new_p95 = int(slo_requirements.p95_latency_ms * (1 - reduction / 100))
            new_p99 = int(slo_requirements.p99_latency_ms * (1 - reduction / 100))

            scenario = WhatIfScenario(
                name=f"{int(reduction)}% latency reduction",
                description=f"Reduce p95 latency target to {new_p95}ms",
                p95_latency_ms=new_p95,
                p99_latency_ms=new_p99,
            )

            result = self.analyze_scenario(
                original_plan=original_plan,
                scenario=scenario,
                benchmark=benchmark,
                traffic_profile=traffic_profile,
                slo_requirements=slo_requirements,
            )
            results.append(result)

        return results

    def analyze_gpu_alternatives(
        self,
        original_plan: CapacityPlan,
        benchmark: BenchmarkData,
        traffic_profile: TrafficProfile,
        slo_requirements: SLORequirements,
        gpu_types: list[str] = ["A100-40GB", "A100-80GB", "A10", "L4"],
    ) -> list[WhatIfResult]:
        """Analyze scenarios with different GPU types."""
        results = []

        for gpu_type in gpu_types:
            if gpu_type == benchmark.gpu_type:
                continue

            scenario = WhatIfScenario(
                name=f"Switch to {gpu_type}",
                description=f"Use {gpu_type} instead of {benchmark.gpu_type}",
                gpu_type=gpu_type,
            )

            result = self.analyze_scenario(
                original_plan=original_plan,
                scenario=scenario,
                benchmark=benchmark,
                traffic_profile=traffic_profile,
                slo_requirements=slo_requirements,
            )
            results.append(result)

        return results

    def _apply_traffic_changes(
        self,
        traffic: TrafficProfile,
        scenario: WhatIfScenario,
    ) -> TrafficProfile:
        """Apply scenario changes to traffic profile."""
        new_rps = traffic.requests_per_second

        if scenario.new_rps is not None:
            new_rps = scenario.new_rps
        elif scenario.rps_multiplier != 1.0:
            new_rps = traffic.requests_per_second * scenario.rps_multiplier

        return TrafficProfile(
            pattern=traffic.pattern,
            requests_per_second=new_rps,
            peak_rps=traffic.peak_rps * scenario.rps_multiplier if traffic.peak_rps else None,
            average_input_tokens=traffic.average_input_tokens,
            average_output_tokens=traffic.average_output_tokens,
            concurrent_users=traffic.concurrent_users,
        )

    def _apply_slo_changes(
        self,
        slo: SLORequirements,
        scenario: WhatIfScenario,
    ) -> SLORequirements:
        """Apply scenario changes to SLO requirements."""
        return SLORequirements(
            p50_latency_ms=slo.p50_latency_ms,
            p95_latency_ms=scenario.p95_latency_ms or slo.p95_latency_ms,
            p99_latency_ms=scenario.p99_latency_ms or slo.p99_latency_ms,
            availability_percent=slo.availability_percent,
            max_tokens_per_second=slo.max_tokens_per_second,
        )

    def _apply_benchmark_changes(
        self,
        benchmark: BenchmarkData,
        scenario: WhatIfScenario,
    ) -> BenchmarkData:
        """Apply scenario changes to benchmark data."""
        if scenario.gpu_type and scenario.gpu_type != benchmark.gpu_type:
            # Estimate performance change for different GPU
            perf_ratio = self._estimate_gpu_performance_ratio(
                benchmark.gpu_type, scenario.gpu_type
            )

            return BenchmarkData(
                model_name=benchmark.model_name,
                model_version=benchmark.model_version,
                gpu_type=scenario.gpu_type,
                gpu_count=benchmark.gpu_count,
                p50_latency_ms=benchmark.p50_latency_ms / perf_ratio,
                p95_latency_ms=benchmark.p95_latency_ms / perf_ratio,
                p99_latency_ms=benchmark.p99_latency_ms / perf_ratio,
                tokens_per_second=benchmark.tokens_per_second * perf_ratio,
                requests_per_second=benchmark.requests_per_second * perf_ratio,
                gpu_memory_gb=benchmark.gpu_memory_gb,
                gpu_utilization_percent=benchmark.gpu_utilization_percent,
                input_tokens=benchmark.input_tokens,
                output_tokens=benchmark.output_tokens,
                batch_size=benchmark.batch_size,
                concurrency=benchmark.concurrency,
            )

        return benchmark

    def _estimate_gpu_performance_ratio(
        self,
        from_gpu: str,
        to_gpu: str,
    ) -> float:
        """Estimate performance ratio between GPU types."""
        # Rough TFLOPS-based estimation
        gpu_tflops = {
            "T4": 65,
            "L4": 121,
            "A10": 125,
            "A100-40GB": 312,
            "A100-80GB": 312,
            "H100-80GB": 989,
        }

        from_tflops = gpu_tflops.get(from_gpu, 312)
        to_tflops = gpu_tflops.get(to_gpu, 312)

        return to_tflops / from_tflops

    def _assess_feasibility(
        self,
        original: CapacityPlan,
        modified: CapacityPlan,
        scenario: WhatIfScenario,
    ) -> tuple[bool, list[str], list[str]]:
        """Assess if scenario is feasible."""
        warnings = []
        recommendations = []
        is_feasible = True

        # Check SLO compliance
        if not modified.meets_slo:
            is_feasible = False
            warnings.append("Modified plan does not meet SLO requirements")
            recommendations.append("Consider using more powerful GPUs or increasing replicas")

        # Check for large replica increases
        if modified.target_replicas > original.target_replicas * 3:
            warnings.append(
                f"Significant replica increase: {original.target_replicas} -> {modified.target_replicas}"
            )
            recommendations.append("Consider GPU upgrade to reduce replica count")

        # Check cost increase
        if original.estimated_monthly_cost and modified.estimated_monthly_cost:
            cost_increase = (
                modified.estimated_monthly_cost - original.estimated_monthly_cost
            ) / original.estimated_monthly_cost * 100

            if cost_increase > 100:
                warnings.append(f"Cost increase: {cost_increase:.0f}%")

        # GPU-specific warnings
        if scenario.gpu_type:
            if scenario.gpu_type in ["T4", "L4"]:
                if original.gpu_type in ["A100-80GB", "A100-40GB", "H100-80GB"]:
                    warnings.append("Downgrading GPU may significantly impact latency")

        return is_feasible, warnings, recommendations
