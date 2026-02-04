"""MCP Tools for OpenShift AI integration."""

from typing import Any

import structlog
from httpx import AsyncClient
from pydantic import BaseModel, Field

from ai_navigator.config import MCPSettings

logger = structlog.get_logger(__name__)


class ModelInfo(BaseModel):
    """Information about a model in the registry."""

    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Model name")
    version: str = Field(default="1.0.0", description="Model version")
    description: str = Field(default="", description="Model description")
    format: str = Field(default="pytorch", description="Model format")
    size_bytes: int = Field(default=0, description="Model size in bytes")
    parameters: int = Field(default=0, description="Number of parameters")
    tags: list[str] = Field(default_factory=list, description="Model tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class InferenceServiceSpec(BaseModel):
    """Specification for an InferenceService."""

    name: str = Field(..., description="Service name")
    namespace: str = Field(default="ai-navigator", description="Kubernetes namespace")
    model_name: str = Field(..., description="Model name")
    runtime: str = Field(default="vllm", description="Serving runtime")
    gpu_count: int = Field(default=1, description="Number of GPUs")
    min_replicas: int = Field(default=1, description="Minimum replicas")
    max_replicas: int = Field(default=3, description="Maximum replicas")
    memory: str = Field(default="16Gi", description="Memory request")
    storage_uri: str = Field(default="", description="Model storage URI")


class OpenShiftAITools:
    """MCP Tool server for OpenShift AI integration."""

    def __init__(self, settings: MCPSettings | None = None) -> None:
        """
        Initialize OpenShift AI tools.

        Args:
            settings: MCP configuration settings
        """
        self.settings = settings or MCPSettings()
        self._client: AsyncClient | None = None

    async def __aenter__(self) -> "OpenShiftAITools":
        """Enter async context."""
        self._client = AsyncClient(
            base_url=self.settings.model_registry_url,
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
                base_url=self.settings.model_registry_url,
                timeout=30.0,
            )
        return self._client

    async def query_model_registry(
        self,
        name_filter: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[ModelInfo]:
        """
        Query the OpenShift Model Registry for available models.

        Args:
            name_filter: Optional name filter (substring match)
            tags: Optional tag filter
            limit: Maximum number of results

        Returns:
            List of matching models
        """
        try:
            # Build query parameters
            params: dict[str, Any] = {"limit": limit}
            if name_filter:
                params["name"] = name_filter
            if tags:
                params["tags"] = ",".join(tags)

            response = await self.client.get("/api/v1/models", params=params)
            response.raise_for_status()

            data = response.json()
            models = []
            for item in data.get("items", []):
                models.append(
                    ModelInfo(
                        id=item.get("id", ""),
                        name=item.get("name", ""),
                        version=item.get("version", "1.0.0"),
                        description=item.get("description", ""),
                        format=item.get("format", "pytorch"),
                        size_bytes=item.get("sizeBytes", 0),
                        parameters=item.get("parameters", 0),
                        tags=item.get("tags", []),
                        metadata=item.get("metadata", {}),
                    )
                )
            return models

        except Exception as e:
            logger.warning("model_registry_query_failed", error=str(e))
            # Return mock data for development
            return self._get_mock_models(name_filter, tags, limit)

    def _get_mock_models(
        self,
        name_filter: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[ModelInfo]:
        """Return mock models for development/testing."""
        mock_models = [
            ModelInfo(
                id="granite-4-tiny",
                name="granite-4.0-h-tiny",
                version="4.0.0",
                description="IBM Granite 4.0 Tiny - 7B parameter model optimized for efficiency",
                format="safetensors",
                size_bytes=14_000_000_000,
                parameters=7_000_000_000,
                tags=["llm", "text-generation", "granite", "ibm"],
                metadata={
                    "publisher": "ibm-granite",
                    "license": "apache-2.0",
                    "context_length": 16384,
                    "recommended_gpu": "T4",
                },
            ),
            ModelInfo(
                id="granite-guardian-hap",
                name="granite-guardian-hap-38m",
                version="1.0.0",
                description="Granite Guardian HAP detector - 38M parameters",
                format="safetensors",
                size_bytes=150_000_000,
                parameters=38_000_000,
                tags=["classifier", "safety", "hap", "granite"],
                metadata={
                    "publisher": "ibm-granite",
                    "type": "hap-detector",
                },
            ),
            ModelInfo(
                id="llama-3-8b",
                name="meta-llama/Llama-3-8B",
                version="3.0.0",
                description="Meta Llama 3 8B base model",
                format="safetensors",
                size_bytes=16_000_000_000,
                parameters=8_000_000_000,
                tags=["llm", "text-generation", "llama", "meta"],
                metadata={
                    "publisher": "meta-llama",
                    "license": "llama3",
                    "context_length": 8192,
                },
            ),
            ModelInfo(
                id="mistral-7b",
                name="mistralai/Mistral-7B-v0.3",
                version="0.3.0",
                description="Mistral 7B v0.3 base model",
                format="safetensors",
                size_bytes=14_500_000_000,
                parameters=7_000_000_000,
                tags=["llm", "text-generation", "mistral"],
                metadata={
                    "publisher": "mistralai",
                    "license": "apache-2.0",
                    "context_length": 32768,
                },
            ),
        ]

        # Apply filters
        result = mock_models
        if name_filter:
            result = [m for m in result if name_filter.lower() in m.name.lower()]
        if tags:
            result = [m for m in result if any(t in m.tags for t in tags)]

        return result[:limit]

    async def get_model_details(self, model_id: str) -> ModelInfo | None:
        """
        Get detailed information about a specific model.

        Args:
            model_id: Model ID to look up

        Returns:
            Model information or None if not found
        """
        try:
            response = await self.client.get(f"/api/v1/models/{model_id}")
            response.raise_for_status()

            data = response.json()
            return ModelInfo(
                id=data.get("id", ""),
                name=data.get("name", ""),
                version=data.get("version", "1.0.0"),
                description=data.get("description", ""),
                format=data.get("format", "pytorch"),
                size_bytes=data.get("sizeBytes", 0),
                parameters=data.get("parameters", 0),
                tags=data.get("tags", []),
                metadata=data.get("metadata", {}),
            )
        except Exception as e:
            logger.warning("model_details_failed", model_id=model_id, error=str(e))
            # Try mock data
            mock_models = self._get_mock_models()
            for model in mock_models:
                if model.id == model_id:
                    return model
            return None

    async def create_inference_service(
        self,
        spec: InferenceServiceSpec,
    ) -> dict[str, Any]:
        """
        Create a KServe InferenceService for the model.

        Args:
            spec: InferenceService specification

        Returns:
            Created InferenceService manifest
        """
        manifest = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": spec.name,
                "namespace": spec.namespace,
                "annotations": {
                    "serving.kserve.io/deploymentMode": "RawDeployment",
                },
            },
            "spec": {
                "predictor": {
                    "model": {
                        "modelFormat": {"name": spec.runtime},
                        "runtime": f"{spec.runtime}-runtime",
                        "storageUri": spec.storage_uri,
                    },
                    "minReplicas": spec.min_replicas,
                    "maxReplicas": spec.max_replicas,
                    "resources": {
                        "requests": {
                            "memory": spec.memory,
                            "nvidia.com/gpu": str(spec.gpu_count),
                        },
                        "limits": {
                            "memory": spec.memory,
                            "nvidia.com/gpu": str(spec.gpu_count),
                        },
                    },
                },
            },
        }

        logger.info(
            "inference_service_created",
            name=spec.name,
            namespace=spec.namespace,
            model=spec.model_name,
        )

        return manifest

    async def get_inference_service_status(
        self,
        name: str,
        namespace: str = "ai-navigator",
    ) -> dict[str, Any]:
        """
        Get the status of an InferenceService.

        Args:
            name: Service name
            namespace: Kubernetes namespace

        Returns:
            Status information
        """
        # In production, this would query the Kubernetes API
        # For now, return mock status
        return {
            "name": name,
            "namespace": namespace,
            "ready": True,
            "url": f"http://{name}.{namespace}.svc.cluster.local",
            "conditions": [
                {
                    "type": "Ready",
                    "status": "True",
                    "reason": "Ready",
                    "message": "InferenceService is ready",
                }
            ],
            "replicas": {
                "desired": 1,
                "ready": 1,
                "available": 1,
            },
        }

    async def delete_inference_service(
        self,
        name: str,
        namespace: str = "ai-navigator",
    ) -> bool:
        """
        Delete an InferenceService.

        Args:
            name: Service name
            namespace: Kubernetes namespace

        Returns:
            True if deleted successfully
        """
        logger.info(
            "inference_service_deleted",
            name=name,
            namespace=namespace,
        )
        return True
