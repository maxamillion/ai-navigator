"""Model and GPU recommendation engine."""

from dataclasses import dataclass
from typing import Optional

import structlog

from ai_navigator.models.capacity import BenchmarkData, GPURecommendation
from ai_navigator.models.workflow import ModelRequirements, SLORequirements

logger = structlog.get_logger(__name__)


@dataclass
class ModelRecommendation:
    """Model recommendation with rationale."""

    model_name: str
    model_family: str
    estimated_size_gb: float
    recommended_gpu: str
    gpu_count: int
    meets_requirements: bool
    score: float  # 0-100
    rationale: list[str]


# Model catalog with characteristics
MODEL_CATALOG = {
    "llama-2-7b": {
        "family": "llama",
        "params_b": 7,
        "context": 4096,
        "capabilities": ["chat", "instruct"],
        "quality_score": 75,
    },
    "llama-2-13b": {
        "family": "llama",
        "params_b": 13,
        "context": 4096,
        "capabilities": ["chat", "instruct"],
        "quality_score": 82,
    },
    "llama-2-70b": {
        "family": "llama",
        "params_b": 70,
        "context": 4096,
        "capabilities": ["chat", "instruct"],
        "quality_score": 92,
    },
    "codellama-7b": {
        "family": "codellama",
        "params_b": 7,
        "context": 16384,
        "capabilities": ["code", "instruct"],
        "quality_score": 78,
    },
    "codellama-34b": {
        "family": "codellama",
        "params_b": 34,
        "context": 16384,
        "capabilities": ["code", "instruct"],
        "quality_score": 88,
    },
    "mistral-7b": {
        "family": "mistral",
        "params_b": 7,
        "context": 32768,
        "capabilities": ["chat", "instruct"],
        "quality_score": 85,
    },
    "mixtral-8x7b": {
        "family": "mistral",
        "params_b": 47,  # Active params
        "context": 32768,
        "capabilities": ["chat", "instruct"],
        "quality_score": 90,
    },
    "granite-3b-code": {
        "family": "granite",
        "params_b": 3,
        "context": 8192,
        "capabilities": ["code"],
        "quality_score": 70,
    },
    "granite-8b-code": {
        "family": "granite",
        "params_b": 8,
        "context": 8192,
        "capabilities": ["code"],
        "quality_score": 80,
    },
    "phi-2": {
        "family": "phi",
        "params_b": 2.7,
        "context": 2048,
        "capabilities": ["chat", "code"],
        "quality_score": 72,
    },
}

# GPU to model size mapping
GPU_MODEL_LIMITS = {
    "T4": {"max_params_fp16": 7, "max_params_int8": 14},
    "L4": {"max_params_fp16": 13, "max_params_int8": 26},
    "A10": {"max_params_fp16": 13, "max_params_int8": 26},
    "A100-40GB": {"max_params_fp16": 34, "max_params_int8": 70},
    "A100-80GB": {"max_params_fp16": 70, "max_params_int8": 140},
    "H100-80GB": {"max_params_fp16": 70, "max_params_int8": 140},
}


class ModelRecommender:
    """Recommends models and GPUs based on requirements."""

    def __init__(self) -> None:
        """Initialize recommender."""
        self._model_catalog = MODEL_CATALOG
        self._gpu_limits = GPU_MODEL_LIMITS

    def recommend_models(
        self,
        requirements: ModelRequirements,
        slo: Optional[SLORequirements] = None,
        max_recommendations: int = 5,
    ) -> list[ModelRecommendation]:
        """Recommend models based on requirements."""
        candidates: list[tuple[str, dict, float]] = []

        for model_name, model_info in self._model_catalog.items():
            score = self._score_model(model_name, model_info, requirements, slo)
            if score > 0:
                candidates.append((model_name, model_info, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[2], reverse=True)

        recommendations = []
        for model_name, model_info, score in candidates[:max_recommendations]:
            gpu_rec = self._recommend_gpu_for_model(model_info["params_b"])

            recommendations.append(
                ModelRecommendation(
                    model_name=model_name,
                    model_family=model_info["family"],
                    estimated_size_gb=model_info["params_b"] * 2,  # FP16 estimate
                    recommended_gpu=gpu_rec.gpu_type,
                    gpu_count=gpu_rec.gpu_count,
                    meets_requirements=score >= 50,
                    score=score,
                    rationale=self._generate_rationale(model_name, model_info, requirements),
                )
            )

        return recommendations

    def recommend_gpu(
        self,
        model_size_billions: float,
        quantization: str = "fp16",
        target_latency_ms: Optional[int] = None,
    ) -> GPURecommendation:
        """Recommend GPU for a model size."""
        return self._recommend_gpu_for_model(
            model_size_billions, quantization, target_latency_ms
        )

    def _score_model(
        self,
        model_name: str,
        model_info: dict,
        requirements: ModelRequirements,
        slo: Optional[SLORequirements],
    ) -> float:
        """Score a model against requirements (0-100)."""
        score = model_info["quality_score"]

        # Family match bonus
        if requirements.model_family:
            if model_info["family"] == requirements.model_family:
                score += 20
            else:
                score -= 30

        # Specific model match
        if requirements.model_name:
            if requirements.model_name.lower() in model_name.lower():
                score += 30
            else:
                score -= 20

        # Size constraint
        if requirements.max_parameters:
            if model_info["params_b"] > requirements.max_parameters:
                score -= 50  # Too large
            elif model_info["params_b"] < requirements.max_parameters * 0.5:
                score -= 10  # Much smaller than max

        # Capability match
        if requirements.capabilities:
            matched_caps = set(requirements.capabilities) & set(model_info["capabilities"])
            required_caps = set(requirements.capabilities)
            if required_caps:
                cap_score = len(matched_caps) / len(required_caps) * 30
                score += cap_score
                if not matched_caps:
                    score -= 40  # No capability match

        # Context length
        if requirements.min_context_length:
            if model_info["context"] < requirements.min_context_length:
                score -= 50  # Context too small

        # Latency consideration
        if slo and slo.p95_latency_ms:
            if slo.p95_latency_ms < 500:
                # Prefer smaller models for low latency
                if model_info["params_b"] > 13:
                    score -= 20
                elif model_info["params_b"] <= 7:
                    score += 10

        return max(0, min(100, score))

    def _recommend_gpu_for_model(
        self,
        model_size_billions: float,
        quantization: str = "fp16",
        target_latency_ms: Optional[int] = None,
    ) -> GPURecommendation:
        """Recommend GPU configuration for model size."""
        quant_key = "max_params_fp16" if quantization in ["fp16", "bf16", "fp32"] else "max_params_int8"

        for gpu_type in ["T4", "L4", "A10", "A100-40GB", "A100-80GB"]:
            limits = self._gpu_limits[gpu_type]
            if limits[quant_key] >= model_size_billions:
                notes = [f"Supports {quantization} for {model_size_billions}B params"]

                # Check if faster GPU needed for latency
                if target_latency_ms and target_latency_ms < 500:
                    if gpu_type in ["T4", "L4"]:
                        notes.append("Consider A10/A100 for lower latency")

                return GPURecommendation(
                    gpu_type=gpu_type,
                    gpu_count=1,
                    meets_slo=True,
                    headroom_percent=20,
                    notes=notes,
                )

        # Model too large for single GPU
        gpu_count = 2
        while model_size_billions / gpu_count > 70:  # A100-80GB limit
            gpu_count *= 2

        return GPURecommendation(
            gpu_type="A100-80GB",
            gpu_count=gpu_count,
            meets_slo=True,
            headroom_percent=10,
            notes=[f"Requires {gpu_count}x GPUs with tensor parallelism"],
        )

    def _generate_rationale(
        self,
        model_name: str,
        model_info: dict,
        requirements: ModelRequirements,
    ) -> list[str]:
        """Generate rationale for model recommendation."""
        rationale = []

        # Family match
        if requirements.model_family and model_info["family"] == requirements.model_family:
            rationale.append(f"Matches requested {requirements.model_family} family")

        # Capabilities
        matched_caps = set(requirements.capabilities or []) & set(model_info["capabilities"])
        if matched_caps:
            rationale.append(f"Supports: {', '.join(matched_caps)}")

        # Size
        rationale.append(f"{model_info['params_b']}B parameters")

        # Context
        rationale.append(f"{model_info['context']:,} token context window")

        # Quality
        if model_info["quality_score"] >= 85:
            rationale.append("High quality benchmark scores")
        elif model_info["quality_score"] >= 75:
            rationale.append("Good quality benchmark scores")

        return rationale
