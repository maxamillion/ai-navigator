"""Benchmark data extraction from Model Registry."""

from typing import Optional

import structlog

from ai_navigator.models.capacity import BenchmarkData
from ai_navigator.registry.client import ModelRegistryClient
from ai_navigator.registry.models import (
    RegisteredModel,
    ModelVersion,
    ValidationMetrics,
    CustomProperties,
)

logger = structlog.get_logger(__name__)


# Known custom property keys for benchmark data
BENCHMARK_PROPERTY_KEYS = {
    "p50_latency_ms": ["p50_latency_ms", "latency_p50", "p50_latency"],
    "p95_latency_ms": ["p95_latency_ms", "latency_p95", "p95_latency"],
    "p99_latency_ms": ["p99_latency_ms", "latency_p99", "p99_latency"],
    "tokens_per_second": ["tokens_per_second", "tps", "throughput_tps"],
    "requests_per_second": ["requests_per_second", "rps", "throughput_rps"],
    "gpu_type": ["gpu_type", "gpu", "accelerator_type"],
    "gpu_count": ["gpu_count", "num_gpus", "accelerator_count"],
    "gpu_memory_gb": ["gpu_memory_gb", "gpu_memory", "vram_gb"],
    "gpu_utilization": ["gpu_utilization", "gpu_util_percent"],
    "input_tokens": ["input_tokens", "input_length", "prompt_tokens"],
    "output_tokens": ["output_tokens", "output_length", "completion_tokens"],
    "batch_size": ["batch_size", "bs"],
    "concurrency": ["concurrency", "concurrent_requests"],
    "tensor_parallel_size": ["tensor_parallel_size", "tp_size", "tp"],
}


def _find_property(props: CustomProperties, key_variants: list[str]) -> Optional[str]:
    """Find property value using multiple possible key names."""
    for key in key_variants:
        value = props.get(key)
        if value is not None:
            return str(value)
    return None


class BenchmarkExtractor:
    """Extracts benchmark data from Model Registry entities."""

    def __init__(self, client: ModelRegistryClient) -> None:
        """Initialize with registry client."""
        self._client = client

    async def get_benchmark_for_model(
        self,
        model_name: str,
        version_name: Optional[str] = None,
        gpu_type: Optional[str] = None,
    ) -> Optional[BenchmarkData]:
        """Get benchmark data for a model."""
        model = await self._client.get_registered_model_by_name(model_name)
        if not model:
            logger.warning("Model not found in registry", model_name=model_name)
            return None

        # Get versions
        versions = await self._client.get_model_versions(model.id)
        if not versions:
            logger.warning("No versions found for model", model_name=model_name)
            return None

        # Select version
        if version_name:
            version = next((v for v in versions if v.name == version_name), None)
        else:
            # Get latest version
            version = max(versions, key=lambda v: v.create_time or v.id)

        if not version:
            logger.warning(
                "Version not found",
                model_name=model_name,
                version_name=version_name,
            )
            return None

        return self._extract_from_version(model, version, gpu_type)

    async def get_all_benchmarks_for_model(
        self,
        model_name: str,
    ) -> list[BenchmarkData]:
        """Get all benchmark data across GPU types for a model."""
        model = await self._client.get_registered_model_by_name(model_name)
        if not model:
            return []

        versions = await self._client.get_model_versions(model.id)
        benchmarks = []

        for version in versions:
            benchmark = self._extract_from_version(model, version, None)
            if benchmark:
                benchmarks.append(benchmark)

        return benchmarks

    async def find_benchmarks_by_gpu(
        self,
        gpu_type: str,
    ) -> list[BenchmarkData]:
        """Find all benchmarks for a specific GPU type."""
        models = await self._client.list_registered_models()
        benchmarks = []

        for model in models:
            versions = await self._client.get_model_versions(model.id)
            for version in versions:
                benchmark = self._extract_from_version(model, version, gpu_type)
                if benchmark and benchmark.gpu_type == gpu_type:
                    benchmarks.append(benchmark)

        return benchmarks

    def _extract_from_version(
        self,
        model: RegisteredModel,
        version: ModelVersion,
        gpu_filter: Optional[str],
    ) -> Optional[BenchmarkData]:
        """Extract benchmark data from model version properties."""
        props = version.custom_properties

        # Check if benchmark data exists
        p50 = _find_property(props, BENCHMARK_PROPERTY_KEYS["p50_latency_ms"])
        tps = _find_property(props, BENCHMARK_PROPERTY_KEYS["tokens_per_second"])

        if not p50 and not tps:
            # No benchmark data in this version
            return None

        gpu_type = _find_property(props, BENCHMARK_PROPERTY_KEYS["gpu_type"]) or "unknown"

        # Apply GPU filter if specified
        if gpu_filter and gpu_type != gpu_filter:
            return None

        try:
            return BenchmarkData(
                model_name=model.name,
                model_version=version.name,
                gpu_type=gpu_type,
                gpu_count=int(
                    _find_property(props, BENCHMARK_PROPERTY_KEYS["gpu_count"]) or "1"
                ),
                p50_latency_ms=float(p50) if p50 else 0.0,
                p95_latency_ms=float(
                    _find_property(props, BENCHMARK_PROPERTY_KEYS["p95_latency_ms"]) or "0"
                ),
                p99_latency_ms=float(
                    _find_property(props, BENCHMARK_PROPERTY_KEYS["p99_latency_ms"]) or "0"
                ),
                tokens_per_second=float(tps) if tps else 0.0,
                requests_per_second=float(
                    _find_property(props, BENCHMARK_PROPERTY_KEYS["requests_per_second"]) or "0"
                ),
                gpu_memory_gb=float(
                    _find_property(props, BENCHMARK_PROPERTY_KEYS["gpu_memory_gb"]) or "0"
                ),
                gpu_utilization_percent=float(
                    _find_property(props, BENCHMARK_PROPERTY_KEYS["gpu_utilization"]) or "0"
                ),
                input_tokens=int(
                    _find_property(props, BENCHMARK_PROPERTY_KEYS["input_tokens"]) or "512"
                ),
                output_tokens=int(
                    _find_property(props, BENCHMARK_PROPERTY_KEYS["output_tokens"]) or "256"
                ),
                batch_size=int(
                    _find_property(props, BENCHMARK_PROPERTY_KEYS["batch_size"]) or "1"
                ),
                concurrency=int(
                    _find_property(props, BENCHMARK_PROPERTY_KEYS["concurrency"]) or "1"
                ),
                source="model_registry",
            )
        except (ValueError, TypeError) as e:
            logger.warning(
                "Failed to parse benchmark data",
                model=model.name,
                version=version.name,
                error=str(e),
            )
            return None

    def extract_validation_metrics(
        self,
        version: ModelVersion,
        model_name: str,
    ) -> ValidationMetrics:
        """Extract validation metrics from version properties."""
        props = version.custom_properties

        return ValidationMetrics(
            model_name=model_name,
            model_version=version.name,
            p50_latency_ms=props.get_float("p50_latency_ms"),
            p95_latency_ms=props.get_float("p95_latency_ms"),
            p99_latency_ms=props.get_float("p99_latency_ms"),
            mean_latency_ms=props.get_float("mean_latency_ms"),
            tokens_per_second=props.get_float("tokens_per_second"),
            requests_per_second=props.get_float("requests_per_second"),
            gpu_memory_gb=props.get_float("gpu_memory_gb"),
            gpu_utilization_percent=props.get_float("gpu_utilization"),
            gpu_type=props.get("gpu_type"),
            gpu_count=props.get_int("gpu_count", 1),
            input_tokens=props.get_int("input_tokens", 512),
            output_tokens=props.get_int("output_tokens", 256),
            batch_size=props.get_int("batch_size", 1),
            concurrency=props.get_int("concurrency", 1),
            tensor_parallel_size=props.get_int("tensor_parallel_size", 1),
        )
