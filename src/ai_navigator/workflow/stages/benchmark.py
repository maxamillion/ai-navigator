"""Benchmark analysis stage - Step 4."""

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


# Default benchmark data for common models (fallback when registry unavailable)
DEFAULT_BENCHMARKS: dict[str, list[BenchmarkData]] = {
    "llama-2-7b": [
        BenchmarkData(
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
        ),
    ],
    "llama-2-13b": [
        BenchmarkData(
            model_name="llama-2-13b",
            model_version="1.0",
            gpu_type="A100-40GB",
            gpu_count=1,
            p50_latency_ms=200,
            p95_latency_ms=350,
            p99_latency_ms=500,
            tokens_per_second=85,
            requests_per_second=10,
            gpu_memory_gb=26,
            gpu_utilization_percent=90,
        ),
    ],
    "llama-2-70b": [
        BenchmarkData(
            model_name="llama-2-70b",
            model_version="1.0",
            gpu_type="A100-80GB",
            gpu_count=4,
            p50_latency_ms=400,
            p95_latency_ms=700,
            p99_latency_ms=1000,
            tokens_per_second=45,
            requests_per_second=5,
            gpu_memory_gb=140,
            gpu_utilization_percent=92,
        ),
    ],
    "mistral-7b": [
        BenchmarkData(
            model_name="mistral-7b",
            model_version="1.0",
            gpu_type="A100-40GB",
            gpu_count=1,
            p50_latency_ms=120,
            p95_latency_ms=200,
            p99_latency_ms=350,
            tokens_per_second=140,
            requests_per_second=18,
            gpu_memory_gb=14,
            gpu_utilization_percent=82,
        ),
    ],
    "granite-3b": [
        BenchmarkData(
            model_name="granite-3b",
            model_version="1.0",
            gpu_type="A10",
            gpu_count=1,
            p50_latency_ms=80,
            p95_latency_ms=150,
            p99_latency_ms=250,
            tokens_per_second=200,
            requests_per_second=25,
            gpu_memory_gb=6,
            gpu_utilization_percent=75,
        ),
    ],
}


class BenchmarkStage(BaseStage):
    """Stage 4: Query model performance benchmark data."""

    def __init__(self, benchmark_extractor: Optional[object] = None) -> None:
        """Initialize with optional benchmark extractor from registry."""
        self._benchmark_extractor = benchmark_extractor

    async def process(self, state: WorkflowState, user_input: str) -> StageResult:
        """Look up benchmark data for candidate models."""
        if not state.model_requirements:
            return StageResult(
                success=False,
                message="Model requirements not defined. Please go back to the intent stage.",
            )

        model_family = state.model_requirements.model_family
        model_name = state.model_requirements.model_name

        # Find relevant benchmarks
        benchmarks = await self._get_benchmarks(model_family, model_name)

        if not benchmarks:
            return StageResult(
                success=True,
                message=(
                    f"No benchmark data found for {model_family or model_name}. "
                    "I'll use estimated performance values for capacity planning."
                ),
                advance=True,
                data={"benchmarks": [], "estimated": True},
            )

        # Store in state metadata
        state.metadata["benchmarks"] = [b.model_dump() for b in benchmarks]

        # Format benchmark summary
        summary_lines = ["Found benchmark data for the following configurations:"]
        for i, b in enumerate(benchmarks, 1):
            summary_lines.append(
                f"\n{i}. **{b.model_name}** on {b.gpu_type} (x{b.gpu_count}):\n"
                f"   - Latency: p50={b.p50_latency_ms}ms, p95={b.p95_latency_ms}ms, "
                f"p99={b.p99_latency_ms}ms\n"
                f"   - Throughput: {b.tokens_per_second} tokens/sec, {b.requests_per_second} RPS\n"
                f"   - GPU Memory: {b.gpu_memory_gb}GB"
            )

        return StageResult(
            success=True,
            message="\n".join(summary_lines),
            advance=True,
            data={"benchmarks": [b.model_dump() for b in benchmarks]},
        )

    async def _get_benchmarks(
        self,
        model_family: Optional[str],
        model_name: Optional[str],
    ) -> list[BenchmarkData]:
        """Get benchmarks from registry or defaults."""
        benchmarks: list[BenchmarkData] = []

        # Try registry first if available
        if self._benchmark_extractor:
            try:
                # This would call the registry's benchmark extractor
                # benchmarks = await self._benchmark_extractor.get_all_benchmarks_for_model(...)
                pass
            except Exception as e:
                logger.warning("Failed to get benchmarks from registry", error=str(e))

        # Fallback to defaults
        if not benchmarks:
            if model_name:
                # Normalize model name
                normalized = model_name.lower().replace(" ", "-")
                for key, data in DEFAULT_BENCHMARKS.items():
                    if key in normalized or normalized in key:
                        benchmarks.extend(data)

            if not benchmarks and model_family:
                # Try model family
                for key, data in DEFAULT_BENCHMARKS.items():
                    if model_family.lower() in key:
                        benchmarks.extend(data)

        return benchmarks
