"""Intent decomposition for the Supervisor Agent."""

import json
from typing import Any

import structlog
from pydantic import BaseModel, Field

from ai_navigator.llm.client import LLMClient
from ai_navigator.llm.prompts import PromptTemplates

logger = structlog.get_logger(__name__)


class SubTask(BaseModel):
    """A sub-task to be delegated to an agent."""

    id: str = Field(..., description="Unique task ID")
    agent: str = Field(..., description="Target agent name")
    skill: str = Field(..., description="Skill to invoke")
    params: dict[str, Any] = Field(default_factory=dict, description="Skill parameters")
    depends_on: list[str] = Field(default_factory=list, description="Task dependencies")


class DecompositionResult(BaseModel):
    """Result of intent decomposition."""

    intent: str = Field(..., description="Detected user intent")
    subtasks: list[SubTask] = Field(default_factory=list, description="Decomposed sub-tasks")
    parallel_groups: list[list[str]] = Field(
        default_factory=list, description="Groups of parallel tasks"
    )


class IntentDecomposer:
    """
    Decomposes user intents into sub-tasks for delegation.

    Uses LLM to analyze user requests and map them to appropriate
    sub-agent skills with proper dependency ordering.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """
        Initialize the decomposer.

        Args:
            llm_client: LLM client for decomposition
        """
        self.llm = llm_client

    async def decompose(self, user_request: str) -> DecompositionResult:
        """
        Decompose a user request into sub-tasks.

        Args:
            user_request: The user's request text

        Returns:
            DecompositionResult with sub-tasks and execution plan
        """
        try:
            prompt = PromptTemplates.format_decomposition(user_request)

            response = await self.llm.complete(
                prompt=prompt,
                system_prompt="You are a task decomposition expert. Output only valid JSON.",
                temperature=0.1,
            )

            # Parse JSON response
            data = self._parse_json_response(response)

            subtasks = [SubTask(**t) for t in data.get("subtasks", [])]

            return DecompositionResult(
                intent=data.get("intent", "unknown"),
                subtasks=subtasks,
                parallel_groups=data.get("parallel_groups", []),
            )
        except Exception as e:
            logger.warning("llm_decomposition_failed", error=str(e))
            # Fall back to heuristic decomposition
            return self._heuristic_decompose(user_request)

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling common issues."""
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()

        # Clean up common issues
        response = response.strip()
        if response.startswith("'") or response.startswith('"'):
            response = response[1:]
        if response.endswith("'") or response.endswith('"'):
            response = response[:-1]

        return json.loads(response)

    def _heuristic_decompose(self, user_request: str) -> DecompositionResult:
        """
        Fall back to heuristic-based decomposition.

        Used when LLM is unavailable or fails.
        """
        text = user_request.lower()
        subtasks: list[SubTask] = []
        parallel_groups: list[list[str]] = []

        # Detect intent and create appropriate subtasks
        if any(w in text for w in ["recommend", "suggest", "best", "which model"]):
            # Recommendation workflow
            subtasks = [
                SubTask(
                    id="1",
                    agent="model_catalog",
                    skill="recommend_for_workload",
                    params={"workload_type": self._extract_workload_type(text)},
                ),
                SubTask(
                    id="2",
                    agent="resource_provisioning",
                    skill="estimate_cost",
                    params={},
                    depends_on=["1"],
                ),
            ]
            parallel_groups = [["1"], ["2"]]
            intent = "recommend_model"

        elif any(w in text for w in ["deploy", "create", "install", "run"]):
            # Deployment workflow
            model_name = self._extract_model_name(text)
            subtasks = [
                SubTask(
                    id="1",
                    agent="model_catalog",
                    skill="get_model_details",
                    params={"model_id": model_name},
                ),
                SubTask(
                    id="2",
                    agent="resource_provisioning",
                    skill="generate_deployment_config",
                    params={"model_name": model_name},
                    depends_on=["1"],
                ),
                SubTask(
                    id="3",
                    agent="resource_provisioning",
                    skill="validate_slo_compliance",
                    params={},
                    depends_on=["2"],
                ),
                SubTask(
                    id="4",
                    agent="resource_provisioning",
                    skill="generate_guardrails",
                    params={},
                    depends_on=["2"],
                ),
            ]
            parallel_groups = [["1"], ["2"], ["3", "4"]]
            intent = "deploy_model"

        elif any(w in text for w in ["status", "check", "health", "monitor"]):
            # Monitoring workflow
            subtasks = [
                SubTask(
                    id="1",
                    agent="deployment_monitor",
                    skill="get_health_summary",
                    params={},
                ),
            ]
            parallel_groups = [["1"]]
            intent = "check_status"

        elif any(w in text for w in ["list", "show", "find", "search"]):
            # Query workflow
            subtasks = [
                SubTask(
                    id="1",
                    agent="model_catalog",
                    skill="query_models",
                    params={},
                ),
            ]
            parallel_groups = [["1"]]
            intent = "list_models"

        elif any(w in text for w in ["metric", "latency", "performance"]):
            # Metrics workflow
            subtasks = [
                SubTask(
                    id="1",
                    agent="deployment_monitor",
                    skill="query_metrics",
                    params={},
                ),
            ]
            parallel_groups = [["1"]]
            intent = "query_metrics"

        elif any(w in text for w in ["slo", "violation", "compliance"]):
            # SLO workflow
            subtasks = [
                SubTask(
                    id="1",
                    agent="deployment_monitor",
                    skill="check_slo_violations",
                    params={},
                ),
            ]
            parallel_groups = [["1"]]
            intent = "check_slo"

        else:
            # Default to model query
            subtasks = [
                SubTask(
                    id="1",
                    agent="model_catalog",
                    skill="query_models",
                    params={},
                ),
            ]
            parallel_groups = [["1"]]
            intent = "general_query"

        return DecompositionResult(
            intent=intent,
            subtasks=subtasks,
            parallel_groups=parallel_groups,
        )

    def _extract_workload_type(self, text: str) -> str:
        """Extract workload type from text."""
        workloads = {
            "chat": "text-generation",
            "conversation": "text-generation",
            "assistant": "text-generation",
            "classification": "classification",
            "classify": "classification",
            "sentiment": "classification",
            "code": "code-generation",
            "programming": "code-generation",
            "embed": "embedding",
            "similarity": "embedding",
            "summarize": "summarization",
            "summary": "summarization",
        }

        for keyword, workload in workloads.items():
            if keyword in text:
                return workload

        return "text-generation"  # Default

    def _extract_model_name(self, text: str) -> str:
        """Extract model name from text."""
        models = [
            "granite-4.0-h-tiny",
            "granite-4-tiny",
            "granite",
            "llama-3-8b",
            "llama-3",
            "llama",
            "mistral-7b",
            "mistral",
        ]

        for model in models:
            if model.lower() in text:
                # Normalize to standard names
                if "granite" in model:
                    return "granite-4.0-h-tiny"
                elif "llama" in model:
                    return "llama-3-8b"
                elif "mistral" in model:
                    return "mistral-7b"

        return "granite-4.0-h-tiny"  # Default
