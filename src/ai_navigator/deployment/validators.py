"""Pre-flight validation for deployments."""

from dataclasses import dataclass, field
from typing import Optional

import structlog

from ai_navigator.models.deployment import DeploymentConfig, InferenceServiceSpec

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of validation check."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)


class DeploymentValidator:
    """Validates deployment configurations before execution."""

    def __init__(self, mcp_client: Optional[object] = None) -> None:
        """Initialize validator with optional MCP client for cluster checks."""
        self._mcp_client = mcp_client

    async def validate(self, config: DeploymentConfig) -> ValidationResult:
        """Run all validation checks on a deployment configuration."""
        errors: list[str] = []
        warnings: list[str] = []
        info: list[str] = []

        # Run all validation checks
        spec_result = self._validate_spec(config.inference_service)
        errors.extend(spec_result.errors)
        warnings.extend(spec_result.warnings)

        resource_result = self._validate_resources(config.inference_service)
        errors.extend(resource_result.errors)
        warnings.extend(resource_result.warnings)

        runtime_result = self._validate_runtime(config.inference_service)
        errors.extend(runtime_result.errors)
        warnings.extend(runtime_result.warnings)

        # Cluster checks if MCP client available
        if self._mcp_client:
            cluster_result = await self._validate_cluster(config)
            errors.extend(cluster_result.errors)
            warnings.extend(cluster_result.warnings)
            info.extend(cluster_result.info)

        # Update config validation status
        config.validation_passed = len(errors) == 0
        config.validation_errors = errors
        config.validation_warnings = warnings

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info,
        )

    def _validate_spec(self, spec: InferenceServiceSpec) -> ValidationResult:
        """Validate InferenceService specification."""
        errors: list[str] = []
        warnings: list[str] = []

        # Name validation
        if not spec.name:
            errors.append("Service name is required")
        elif len(spec.name) > 63:
            errors.append("Service name must be 63 characters or less")
        elif not spec.name.replace("-", "").isalnum():
            errors.append("Service name must be alphanumeric with hyphens only")

        # Namespace validation
        if not spec.namespace:
            errors.append("Namespace is required")

        # Model validation
        if not spec.model_name:
            errors.append("Model name is required")

        # Replica validation
        if spec.min_replicas < 0:
            errors.append("min_replicas must be non-negative")
        if spec.max_replicas < spec.min_replicas:
            errors.append("max_replicas must be >= min_replicas")
        if spec.max_replicas > 100:
            warnings.append("max_replicas > 100 may be excessive")

        # GPU validation
        if spec.gpu_count < 0:
            errors.append("gpu_count must be non-negative")
        if spec.gpu_count > 8:
            warnings.append("gpu_count > 8 requires multi-node setup")

        return ValidationResult(passed=len(errors) == 0, errors=errors, warnings=warnings)

    def _validate_resources(self, spec: InferenceServiceSpec) -> ValidationResult:
        """Validate resource requests."""
        errors: list[str] = []
        warnings: list[str] = []

        # Memory validation
        try:
            memory_value = spec.memory.rstrip("GiMi")
            memory_num = float(memory_value)
            if "Gi" in spec.memory:
                memory_gb = memory_num
            else:
                memory_gb = memory_num / 1024

            if memory_gb < 4:
                warnings.append("Memory < 4Gi may be insufficient for most models")
            if memory_gb > 512:
                warnings.append("Memory > 512Gi is unusually high")

        except (ValueError, AttributeError):
            errors.append(f"Invalid memory format: {spec.memory}")

        # CPU validation
        try:
            cpu_value = float(spec.cpu)
            if cpu_value < 1:
                warnings.append("CPU < 1 may cause slow model loading")
            if cpu_value > 64:
                warnings.append("CPU > 64 is unusually high")
        except ValueError:
            errors.append(f"Invalid CPU format: {spec.cpu}")

        # GPU type validation
        known_gpus = ["A100-40GB", "A100-80GB", "A10", "L4", "T4", "H100-80GB"]
        if spec.gpu_type and spec.gpu_type not in known_gpus:
            warnings.append(f"Unknown GPU type: {spec.gpu_type}")

        return ValidationResult(passed=len(errors) == 0, errors=errors, warnings=warnings)

    def _validate_runtime(self, spec: InferenceServiceSpec) -> ValidationResult:
        """Validate runtime configuration."""
        errors: list[str] = []
        warnings: list[str] = []

        if not spec.runtime:
            return ValidationResult(passed=True, errors=[], warnings=[])

        runtime = spec.runtime

        # Runtime name validation
        known_runtimes = ["vllm", "openvino", "triton", "tgis"]
        if runtime.runtime_name not in known_runtimes:
            warnings.append(f"Unknown runtime: {runtime.runtime_name}")

        # vLLM-specific validation
        if runtime.runtime_name == "vllm":
            # Tensor parallelism
            if runtime.tensor_parallel_size > 1:
                if spec.gpu_count < runtime.tensor_parallel_size:
                    errors.append(
                        f"tensor_parallel_size ({runtime.tensor_parallel_size}) "
                        f"exceeds gpu_count ({spec.gpu_count})"
                    )

            # GPU memory utilization
            if runtime.gpu_memory_utilization < 0.5:
                warnings.append("gpu_memory_utilization < 0.5 may waste GPU memory")
            if runtime.gpu_memory_utilization > 0.95:
                warnings.append("gpu_memory_utilization > 0.95 may cause OOM errors")

            # Data type
            valid_dtypes = ["auto", "float16", "bfloat16", "float32"]
            if runtime.dtype not in valid_dtypes:
                errors.append(f"Invalid dtype: {runtime.dtype}")

        return ValidationResult(passed=len(errors) == 0, errors=errors, warnings=warnings)

    async def _validate_cluster(self, config: DeploymentConfig) -> ValidationResult:
        """Validate against cluster state (requires MCP client)."""
        errors: list[str] = []
        warnings: list[str] = []
        info: list[str] = []

        spec = config.inference_service

        try:
            # Check if namespace exists
            # result = await self._mcp_client.call_tool("get_project_status", ...)

            # Check for existing service with same name
            # result = await self._mcp_client.call_tool("get_inference_service", ...)

            # Check GPU availability
            # This would require resource quota checks

            info.append("Cluster validation skipped (simulated mode)")

        except Exception as e:
            warnings.append(f"Cluster validation failed: {e}")

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info,
        )


def format_validation_report(result: ValidationResult) -> str:
    """Format validation result as human-readable report."""
    lines = []

    if result.passed:
        lines.append("**Validation Passed**")
    else:
        lines.append("**Validation Failed**")

    if result.errors:
        lines.append("\n**Errors:**")
        for error in result.errors:
            lines.append(f"- {error}")

    if result.warnings:
        lines.append("\n**Warnings:**")
        for warning in result.warnings:
            lines.append(f"- {warning}")

    if result.info:
        lines.append("\n**Info:**")
        for item in result.info:
            lines.append(f"- {item}")

    return "\n".join(lines)
