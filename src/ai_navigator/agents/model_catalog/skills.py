"""Skills for the Model Catalog Agent."""

from typing import TYPE_CHECKING, Any

import structlog

from ai_navigator.a2a.skills import SkillInput, SkillResult

if TYPE_CHECKING:
    from ai_navigator.agents.model_catalog.agent import ModelCatalogAgent

logger = structlog.get_logger(__name__)


def register_skills(agent: "ModelCatalogAgent") -> None:
    """Register all skills for the Model Catalog Agent."""

    @agent.skills.register(
        id="query_models",
        name="Query Models",
        description="Search for models in the OpenShift Model Registry",
        tags=["models", "search", "registry"],
        examples=[
            "List all available LLM models",
            "Find models with the 'text-generation' tag",
            "Search for granite models",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "name_filter": {"type": "string", "description": "Filter by model name"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags",
                },
                "limit": {"type": "integer", "description": "Maximum results"},
            },
        },
    )
    async def query_models(input: SkillInput) -> SkillResult:
        """Query the model registry for available models."""
        params = input.params
        name_filter = params.get("name_filter")
        tags = params.get("tags")
        limit = params.get("limit", 100)

        # Extract filter from message if not in params
        if not name_filter and not tags:
            text = input.message.get_text().lower()
            # Try to extract model name from message
            for keyword in ["granite", "llama", "mistral", "gpt"]:
                if keyword in text:
                    name_filter = keyword
                    break
            # Try to extract tags
            for tag in ["llm", "text-generation", "classifier", "embedding"]:
                if tag in text:
                    tags = [tag] if not tags else tags + [tag]

        try:
            models = await agent.openshift_ai.query_model_registry(
                name_filter=name_filter,
                tags=tags,
                limit=limit,
            )

            if not models:
                return SkillResult.ok(
                    message="No models found matching your criteria.",
                    data={"models": []},
                )

            model_list = [
                {
                    "id": m.id,
                    "name": m.name,
                    "description": m.description,
                    "parameters": f"{m.parameters / 1e9:.1f}B" if m.parameters else "Unknown",
                    "tags": m.tags,
                }
                for m in models
            ]

            message = f"Found {len(models)} models:\n\n"
            for m in models:
                params_str = f"{m.parameters / 1e9:.1f}B" if m.parameters else "Unknown"
                message += f"• **{m.name}** ({params_str}): {m.description[:100]}...\n"

            return SkillResult.ok(
                message=message,
                data={"models": model_list, "count": len(models)},
            )
        except Exception as e:
            logger.exception("query_models_failed", error=str(e))
            return SkillResult.error(f"Failed to query models: {e}")

    @agent.skills.register(
        id="get_model_details",
        name="Get Model Details",
        description="Get detailed information about a specific model",
        tags=["models", "details"],
        examples=[
            "Get details for granite-4.0-h-tiny",
            "Show information about Llama-3-8B",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "Model ID to look up"},
            },
            "required": ["model_id"],
        },
    )
    async def get_model_details(input: SkillInput) -> SkillResult:
        """Get detailed information about a specific model."""
        model_id = input.params.get("model_id")

        if not model_id:
            # Try to extract from message
            text = input.message.get_text()
            # Simple extraction - in production would use NLP
            for known_id in ["granite-4-tiny", "granite-guardian-hap", "llama-3-8b", "mistral-7b"]:
                if known_id.replace("-", " ") in text.lower() or known_id in text.lower():
                    model_id = known_id
                    break

        if not model_id:
            return SkillResult.need_input("Please specify a model ID to get details for.")

        try:
            model = await agent.openshift_ai.get_model_details(model_id)

            if not model:
                return SkillResult.error(f"Model not found: {model_id}")

            message = f"""
## {model.name}

**ID:** {model.id}
**Version:** {model.version}
**Format:** {model.format}
**Parameters:** {model.parameters / 1e9:.1f}B
**Size:** {model.size_bytes / 1e9:.1f} GB

### Description
{model.description}

### Tags
{', '.join(model.tags)}

### Metadata
"""
            for key, value in model.metadata.items():
                message += f"- **{key}:** {value}\n"

            return SkillResult.ok(
                message=message,
                data=model.model_dump(),
            )
        except Exception as e:
            logger.exception("get_model_details_failed", error=str(e))
            return SkillResult.error(f"Failed to get model details: {e}")

    @agent.skills.register(
        id="get_benchmarks",
        name="Get Model Benchmarks",
        description="Get benchmark performance data for models",
        tags=["models", "benchmarks", "performance"],
        examples=[
            "Show benchmarks for granite-4.0-h-tiny",
            "Compare model performance",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "Model ID"},
            },
        },
    )
    async def get_benchmarks(input: SkillInput) -> SkillResult:
        """Get benchmark data for a model."""
        model_id = input.params.get("model_id", "granite-4-tiny")

        # Mock benchmark data - in production would query actual benchmarks
        benchmarks: dict[str, dict[str, Any]] = {
            "granite-4-tiny": {
                "inference_speed": {
                    "tokens_per_second": 85,
                    "latency_p50_ms": 125,
                    "latency_p95_ms": 350,
                    "latency_p99_ms": 750,
                },
                "quality": {
                    "mmlu": 0.68,
                    "hellaswag": 0.72,
                    "winogrande": 0.65,
                },
                "efficiency": {
                    "gpu_memory_gb": 14.2,
                    "gpu_utilization": 0.72,
                    "power_watts": 120,
                },
                "hardware": {
                    "gpu": "NVIDIA T4",
                    "vram_gb": 16,
                    "precision": "float16",
                },
            },
            "llama-3-8b": {
                "inference_speed": {
                    "tokens_per_second": 75,
                    "latency_p50_ms": 145,
                    "latency_p95_ms": 400,
                    "latency_p99_ms": 850,
                },
                "quality": {
                    "mmlu": 0.72,
                    "hellaswag": 0.78,
                    "winogrande": 0.70,
                },
                "efficiency": {
                    "gpu_memory_gb": 15.5,
                    "gpu_utilization": 0.78,
                    "power_watts": 135,
                },
                "hardware": {
                    "gpu": "NVIDIA T4",
                    "vram_gb": 16,
                    "precision": "float16",
                },
            },
        }

        data = benchmarks.get(model_id, benchmarks["granite-4-tiny"])

        message = f"""
## Benchmarks for {model_id}

### Inference Performance
- **Tokens/second:** {data['inference_speed']['tokens_per_second']}
- **Latency (p50):** {data['inference_speed']['latency_p50_ms']}ms
- **Latency (p95):** {data['inference_speed']['latency_p95_ms']}ms
- **Latency (p99):** {data['inference_speed']['latency_p99_ms']}ms

### Quality Metrics
- **MMLU:** {data['quality']['mmlu']:.1%}
- **HellaSwag:** {data['quality']['hellaswag']:.1%}
- **WinoGrande:** {data['quality']['winogrande']:.1%}

### Resource Usage
- **GPU Memory:** {data['efficiency']['gpu_memory_gb']} GB
- **GPU Utilization:** {data['efficiency']['gpu_utilization']:.0%}
- **Power:** {data['efficiency']['power_watts']}W

### Hardware Configuration
- **GPU:** {data['hardware']['gpu']}
- **VRAM:** {data['hardware']['vram_gb']} GB
- **Precision:** {data['hardware']['precision']}
"""

        return SkillResult.ok(message=message, data={"benchmarks": data})

    @agent.skills.register(
        id="get_trustyai_scores",
        name="Get TrustyAI Scores",
        description="Get TrustyAI evaluation scores for model safety and fairness",
        tags=["models", "trustyai", "safety", "evaluation"],
        examples=[
            "Get TrustyAI scores for granite-4.0-h-tiny",
            "Check model safety evaluation",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "Model ID"},
            },
        },
    )
    async def get_trustyai_scores(input: SkillInput) -> SkillResult:
        """Get TrustyAI evaluation scores for a model."""
        model_id = input.params.get("model_id", "granite-4-tiny")

        try:
            evaluation = await agent.trustyai.get_model_evaluation(model_id)

            message = f"""
## TrustyAI Evaluation: {evaluation.model_name}

### Overall Trust Score: {evaluation.overall_trust_score:.1%}

### Category Scores
- **HAP (Harmful/Abusive/Prejudiced):** {evaluation.hap_score:.1%}
- **Bias Detection:** {evaluation.bias_score:.1%}
- **Fairness:** {evaluation.fairness_score:.1%}
- **Explainability:** {evaluation.explainability_score:.1%}

### Evaluation Date
{evaluation.evaluation_date}
"""

            if evaluation.details:
                message += "\n### Detailed Breakdown\n"
                for category, scores in evaluation.details.items():
                    if isinstance(scores, dict):
                        message += f"\n**{category.replace('_', ' ').title()}:**\n"
                        for key, value in scores.items():
                            if isinstance(value, float):
                                message += f"  - {key}: {value:.1%}\n"
                            else:
                                message += f"  - {key}: {value}\n"

            return SkillResult.ok(
                message=message,
                data=evaluation.model_dump(),
            )
        except Exception as e:
            logger.exception("get_trustyai_scores_failed", error=str(e))
            return SkillResult.error(f"Failed to get TrustyAI scores: {e}")

    @agent.skills.register(
        id="recommend_for_workload",
        name="Recommend Model for Workload",
        description="Recommend the best model for a specific workload type",
        tags=["models", "recommendation", "workload"],
        examples=[
            "Recommend a model for text classification",
            "What's the best model for code generation?",
            "Suggest a model for our chatbot",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "workload_type": {
                    "type": "string",
                    "description": "Type of workload (e.g., text-generation, classification, embedding)",
                },
                "constraints": {
                    "type": "object",
                    "description": "Resource constraints",
                    "properties": {
                        "max_memory_gb": {"type": "number"},
                        "max_latency_ms": {"type": "number"},
                        "min_trust_score": {"type": "number"},
                    },
                },
            },
        },
    )
    async def recommend_for_workload(input: SkillInput) -> SkillResult:
        """Recommend the best model for a workload."""
        workload_type = input.params.get("workload_type")
        constraints = input.params.get("constraints", {})

        # Extract workload type from message if not provided
        if not workload_type:
            text = input.message.get_text().lower()
            if any(w in text for w in ["chat", "conversation", "assistant"]):
                workload_type = "text-generation"
            elif any(w in text for w in ["classify", "classification", "sentiment"]):
                workload_type = "classification"
            elif any(w in text for w in ["code", "programming", "development"]):
                workload_type = "code-generation"
            elif any(w in text for w in ["embed", "embedding", "similarity"]):
                workload_type = "embedding"
            else:
                workload_type = "text-generation"

        # Get available models
        models = await agent.openshift_ai.query_model_registry(
            tags=[workload_type] if workload_type else None,
        )

        # Get TrustyAI scores for each model
        recommendations = []
        for model in models[:5]:  # Top 5
            evaluation = await agent.trustyai.get_model_evaluation(model.id)
            recommendations.append(
                {
                    "model": model,
                    "trust_score": evaluation.overall_trust_score,
                    "evaluation": evaluation,
                }
            )

        # Sort by trust score
        recommendations.sort(key=lambda x: x["trust_score"], reverse=True)

        # Apply constraints
        max_memory = constraints.get("max_memory_gb", 16)
        min_trust = constraints.get("min_trust_score", 0.8)

        filtered = [
            r
            for r in recommendations
            if r["model"].size_bytes / 1e9 <= max_memory and r["trust_score"] >= min_trust
        ]

        if not filtered:
            filtered = recommendations[:3]  # Fall back to top 3

        message = f"""
## Model Recommendations for {workload_type}

Based on your requirements and TrustyAI evaluations:

"""
        for i, rec in enumerate(filtered[:3], 1):
            m = rec["model"]
            message += f"""
### {i}. {m.name}

- **Parameters:** {m.parameters / 1e9:.1f}B
- **Trust Score:** {rec['trust_score']:.1%}
- **Description:** {m.description[:150]}...

"""

        if filtered:
            top = filtered[0]
            message += f"""
---

**Recommended:** {top['model'].name} with a trust score of {top['trust_score']:.1%}

This model best balances performance, safety, and resource efficiency for your {workload_type} workload.
"""

        return SkillResult.ok(
            message=message,
            data={
                "recommendations": [
                    {
                        "model_id": r["model"].id,
                        "model_name": r["model"].name,
                        "trust_score": r["trust_score"],
                    }
                    for r in filtered[:3]
                ],
                "workload_type": workload_type,
            },
        )
