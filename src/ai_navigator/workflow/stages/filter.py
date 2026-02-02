"""Model filter stage - Step 5."""

from typing import Optional

import structlog

from ai_navigator.models.capacity import BenchmarkData
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


class FilterStage(BaseStage):
    """Stage 5: Filter models that meet SLO requirements."""

    async def process(self, state: WorkflowState, user_input: str) -> StageResult:
        """Filter candidate models based on SLO requirements."""
        if not state.slo_requirements:
            return StageResult(
                success=False,
                message="SLO requirements not defined. Please complete the SLO stage.",
            )

        # Get benchmarks from state
        benchmarks_data = state.metadata.get("benchmarks", [])
        benchmarks = [BenchmarkData.model_validate(b) for b in benchmarks_data]

        if not benchmarks:
            # No benchmarks - use model requirements to suggest
            return self._suggest_without_benchmarks(state)

        # Filter models that meet SLO
        slo = state.slo_requirements
        passing_models: list[tuple[BenchmarkData, dict]] = []
        failing_models: list[tuple[BenchmarkData, list[str]]] = []

        for benchmark in benchmarks:
            violations = []
            headroom = {}

            # Check p95 latency
            if benchmark.p95_latency_ms > slo.p95_latency_ms:
                violations.append(
                    f"p95 latency {benchmark.p95_latency_ms}ms > target {slo.p95_latency_ms}ms"
                )
            else:
                headroom["p95"] = (
                    (slo.p95_latency_ms - benchmark.p95_latency_ms) / slo.p95_latency_ms * 100
                )

            # Check p99 latency
            if benchmark.p99_latency_ms > slo.p99_latency_ms:
                violations.append(
                    f"p99 latency {benchmark.p99_latency_ms}ms > target {slo.p99_latency_ms}ms"
                )
            else:
                headroom["p99"] = (
                    (slo.p99_latency_ms - benchmark.p99_latency_ms) / slo.p99_latency_ms * 100
                )

            # Check throughput if specified
            if slo.max_tokens_per_second:
                if benchmark.tokens_per_second < slo.max_tokens_per_second:
                    violations.append(
                        f"throughput {benchmark.tokens_per_second} tps < "
                        f"target {slo.max_tokens_per_second} tps"
                    )

            if violations:
                failing_models.append((benchmark, violations))
            else:
                passing_models.append((benchmark, headroom))

        # Build response
        if passing_models:
            state.selected_models = [b.model_name for b, _ in passing_models]

            response_lines = [
                f"Found {len(passing_models)} model configuration(s) meeting your SLO requirements:"
            ]

            for benchmark, headroom in passing_models:
                avg_headroom = sum(headroom.values()) / len(headroom) if headroom else 0
                response_lines.append(
                    f"\n**{benchmark.model_name}** ({benchmark.gpu_type} x{benchmark.gpu_count})\n"
                    f"  - p95 latency: {benchmark.p95_latency_ms}ms "
                    f"(target: {slo.p95_latency_ms}ms, headroom: {headroom.get('p95', 0):.0f}%)\n"
                    f"  - p99 latency: {benchmark.p99_latency_ms}ms "
                    f"(target: {slo.p99_latency_ms}ms, headroom: {headroom.get('p99', 0):.0f}%)\n"
                    f"  - Throughput: {benchmark.tokens_per_second} tokens/sec"
                )

            if failing_models:
                response_lines.append(
                    f"\n\n{len(failing_models)} configuration(s) did not meet requirements:"
                )
                for benchmark, violations in failing_models[:2]:  # Show first 2
                    response_lines.append(f"- {benchmark.model_name}: {', '.join(violations)}")

            return StageResult(
                success=True,
                message="\n".join(response_lines),
                advance=True,
                data={
                    "selected": [b.model_dump() for b, _ in passing_models],
                    "rejected": [b.model_dump() for b, _ in failing_models],
                },
            )

        # No models pass SLO
        response_lines = [
            "No model configurations meet your SLO requirements. Issues found:"
        ]
        for benchmark, violations in failing_models:
            response_lines.append(f"\n- {benchmark.model_name}: {', '.join(violations)}")

        response_lines.append("\n\nOptions:")
        response_lines.append("1. Relax your SLO requirements (increase latency targets)")
        response_lines.append("2. Consider a smaller, faster model")
        response_lines.append("3. Use more powerful hardware (more GPUs)")

        return StageResult(
            success=True,
            message="\n".join(response_lines),
            prompt_user="Would you like to adjust your requirements, or proceed with the closest match?",
        )

    def _suggest_without_benchmarks(self, state: WorkflowState) -> StageResult:
        """Suggest models when no benchmark data is available."""
        model_req = state.model_requirements
        slo = state.slo_requirements

        suggestions = []

        # Suggest based on latency requirements
        if slo and slo.p95_latency_ms < 500:
            suggestions.append(
                "For low latency (<500ms), consider smaller models: granite-3b, phi-2, mistral-7b"
            )
        elif slo and slo.p95_latency_ms < 2000:
            suggestions.append(
                "For moderate latency (<2s), models like llama-2-13b or mistral-7b work well"
            )
        else:
            suggestions.append(
                "For batch workloads, larger models like llama-2-70b can be used"
            )

        # Add capability-based suggestions
        if model_req and "code" in model_req.capabilities:
            suggestions.append("For code tasks: codellama, starcoder, or granite-code models")

        state.selected_models = []  # Will be populated based on user choice

        return StageResult(
            success=True,
            message=(
                "No benchmark data available. Based on your requirements:\n\n"
                + "\n".join(f"- {s}" for s in suggestions)
            ),
            advance=True,
            data={"suggestions": suggestions, "needs_selection": True},
        )
