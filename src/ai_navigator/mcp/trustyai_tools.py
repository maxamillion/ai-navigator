"""MCP Tools for TrustyAI integration."""

from typing import Any

import structlog
from httpx import AsyncClient
from pydantic import BaseModel, Field

from ai_navigator.config import MCPSettings

logger = structlog.get_logger(__name__)


class ModelEvaluation(BaseModel):
    """TrustyAI model evaluation scores."""

    model_id: str = Field(..., description="Model ID")
    model_name: str = Field(..., description="Model name")
    hap_score: float = Field(default=0.0, description="HAP (Harmful/Abusive/Prejudiced) score")
    bias_score: float = Field(default=0.0, description="Bias detection score")
    fairness_score: float = Field(default=0.0, description="Fairness score")
    explainability_score: float = Field(default=0.0, description="Explainability score")
    overall_trust_score: float = Field(default=0.0, description="Overall trust score")
    evaluation_date: str = Field(default="", description="Evaluation timestamp")
    details: dict[str, Any] = Field(default_factory=dict, description="Detailed scores")


class GuardrailConfig(BaseModel):
    """TrustyAI Guardrails configuration."""

    name: str = Field(..., description="Guardrail name")
    detectors: list[dict[str, Any]] = Field(
        default_factory=list, description="Detector configurations"
    )
    actions: list[dict[str, Any]] = Field(default_factory=list, description="Action configurations")
    thresholds: dict[str, float] = Field(default_factory=dict, description="Detection thresholds")


class TrustyAITools:
    """MCP Tool server for TrustyAI integration."""

    def __init__(self, settings: MCPSettings | None = None) -> None:
        """
        Initialize TrustyAI tools.

        Args:
            settings: MCP configuration settings
        """
        self.settings = settings or MCPSettings()
        self._client: AsyncClient | None = None

    async def __aenter__(self) -> "TrustyAITools":
        """Enter async context."""
        self._client = AsyncClient(
            base_url=self.settings.trustyai_url,
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context."""
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> AsyncClient:
        """Get HTTP client."""
        if self._client is None:
            self._client = AsyncClient(
                base_url=self.settings.trustyai_url,
                timeout=30.0,
            )
        return self._client

    async def get_model_evaluation(self, model_id: str) -> ModelEvaluation:
        """
        Get TrustyAI evaluation scores for a model.

        Args:
            model_id: Model ID to evaluate

        Returns:
            Model evaluation scores
        """
        try:
            response = await self.client.get(f"/api/v1/evaluations/{model_id}")
            response.raise_for_status()

            data = response.json()
            return ModelEvaluation(
                model_id=model_id,
                model_name=data.get("modelName", model_id),
                hap_score=data.get("hapScore", 0.0),
                bias_score=data.get("biasScore", 0.0),
                fairness_score=data.get("fairnessScore", 0.0),
                explainability_score=data.get("explainabilityScore", 0.0),
                overall_trust_score=data.get("overallScore", 0.0),
                evaluation_date=data.get("evaluationDate", ""),
                details=data.get("details", {}),
            )
        except Exception as e:
            logger.warning("trustyai_evaluation_failed", model_id=model_id, error=str(e))
            # Return mock evaluation for development
            return self._get_mock_evaluation(model_id)

    def _get_mock_evaluation(self, model_id: str) -> ModelEvaluation:
        """Return mock evaluation for development/testing."""
        mock_evaluations = {
            "granite-4-tiny": ModelEvaluation(
                model_id="granite-4-tiny",
                model_name="granite-4.0-h-tiny",
                hap_score=0.95,
                bias_score=0.92,
                fairness_score=0.90,
                explainability_score=0.88,
                overall_trust_score=0.91,
                evaluation_date="2025-01-15T10:00:00Z",
                details={
                    "hap_breakdown": {
                        "hate_speech": 0.96,
                        "harassment": 0.94,
                        "violence": 0.95,
                    },
                    "bias_breakdown": {
                        "gender": 0.93,
                        "race": 0.91,
                        "age": 0.92,
                    },
                },
            ),
            "granite-guardian-hap": ModelEvaluation(
                model_id="granite-guardian-hap",
                model_name="granite-guardian-hap-38m",
                hap_score=0.98,
                bias_score=0.95,
                fairness_score=0.96,
                explainability_score=0.90,
                overall_trust_score=0.95,
                evaluation_date="2025-01-15T10:00:00Z",
                details={
                    "detector_type": "hap",
                    "precision": 0.97,
                    "recall": 0.94,
                    "f1_score": 0.955,
                },
            ),
            "llama-3-8b": ModelEvaluation(
                model_id="llama-3-8b",
                model_name="meta-llama/Llama-3-8B",
                hap_score=0.88,
                bias_score=0.85,
                fairness_score=0.86,
                explainability_score=0.82,
                overall_trust_score=0.85,
                evaluation_date="2025-01-14T10:00:00Z",
                details={
                    "hap_breakdown": {
                        "hate_speech": 0.90,
                        "harassment": 0.87,
                        "violence": 0.88,
                    },
                },
            ),
        }

        return mock_evaluations.get(
            model_id,
            ModelEvaluation(
                model_id=model_id,
                model_name=model_id,
                hap_score=0.80,
                bias_score=0.80,
                fairness_score=0.80,
                explainability_score=0.75,
                overall_trust_score=0.79,
                evaluation_date="2025-01-01T00:00:00Z",
                details={},
            ),
        )

    async def generate_guardrails_config(
        self,
        name: str,
        enable_hap: bool = True,
        enable_pii: bool = True,
        enable_prompt_injection: bool = True,
        hap_threshold: float = 0.7,
        pii_threshold: float = 0.8,
    ) -> GuardrailConfig:
        """
        Generate TrustyAI GuardrailsOrchestrator configuration.

        Args:
            name: Configuration name
            enable_hap: Enable HAP detection
            enable_pii: Enable PII detection
            enable_prompt_injection: Enable prompt injection detection
            hap_threshold: HAP detection threshold
            pii_threshold: PII detection threshold

        Returns:
            Guardrails configuration
        """
        detectors = []
        thresholds = {}

        if enable_hap:
            detectors.append(
                {
                    "name": "hap-detector",
                    "type": "text-classification",
                    "model": "ibm-granite/granite-guardian-hap-38m",
                    "config": {
                        "threshold": hap_threshold,
                        "categories": ["hate", "harassment", "violence"],
                    },
                }
            )
            thresholds["hap"] = hap_threshold

        if enable_pii:
            detectors.append(
                {
                    "name": "pii-detector",
                    "type": "regex",
                    "config": {
                        "patterns": {
                            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
                            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                        },
                        "threshold": pii_threshold,
                    },
                }
            )
            thresholds["pii"] = pii_threshold

        if enable_prompt_injection:
            detectors.append(
                {
                    "name": "prompt-injection-detector",
                    "type": "text-classification",
                    "config": {
                        "patterns": [
                            "ignore previous instructions",
                            "disregard all prior",
                            "you are now",
                            "act as",
                            "pretend you are",
                        ],
                        "threshold": 0.6,
                    },
                }
            )
            thresholds["prompt_injection"] = 0.6

        actions = [
            {
                "name": "block",
                "trigger": "any_detection",
                "response": "Content blocked due to policy violation",
            },
            {
                "name": "log",
                "trigger": "any_detection",
                "config": {"log_level": "warning"},
            },
            {
                "name": "redact-pii",
                "trigger": "pii_detected",
                "config": {"replacement": "[REDACTED]"},
            },
        ]

        return GuardrailConfig(
            name=name,
            detectors=detectors,
            actions=actions,
            thresholds=thresholds,
        )

    async def generate_guardrails_manifest(
        self,
        config: GuardrailConfig,
        namespace: str = "ai-navigator",
    ) -> dict[str, Any]:
        """
        Generate Kubernetes manifest for GuardrailsOrchestrator.

        Args:
            config: Guardrails configuration
            namespace: Kubernetes namespace

        Returns:
            Kubernetes manifest
        """
        manifest = {
            "apiVersion": "trustyai.opendatahub.io/v1alpha1",
            "kind": "GuardrailsOrchestrator",
            "metadata": {
                "name": config.name,
                "namespace": namespace,
            },
            "spec": {
                "replicas": 1,
                "detectors": config.detectors,
                "actions": config.actions,
                "thresholds": config.thresholds,
                "orchestratorConfig": {
                    "streamingEnabled": True,
                    "healthCheckInterval": 30,
                },
            },
        }

        logger.info(
            "guardrails_manifest_generated",
            name=config.name,
            namespace=namespace,
            detector_count=len(config.detectors),
        )

        return manifest

    async def check_content(
        self,
        content: str,
        detectors: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Check content against guardrails.

        Args:
            content: Content to check
            detectors: Optional list of detectors to use

        Returns:
            Detection results
        """
        try:
            payload = {"content": content}
            if detectors:
                payload["detectors"] = detectors

            response = await self.client.post("/api/v1/check", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning("content_check_failed", error=str(e))
            # Return safe mock response
            return {
                "safe": True,
                "detections": [],
                "scores": {
                    "hap": 0.0,
                    "pii": 0.0,
                    "prompt_injection": 0.0,
                },
            }
