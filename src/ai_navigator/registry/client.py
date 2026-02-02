"""Model Registry REST API client."""

from typing import Any, Optional

import httpx
import structlog

from ai_navigator.config import ModelRegistrySettings
from ai_navigator.registry.models import (
    ModelArtifact,
    ModelVersion,
    RegisteredModel,
    MetricHistory,
    MetricHistoryPoint,
    CustomProperties,
)

logger = structlog.get_logger(__name__)


class ModelRegistryError(Exception):
    """Base exception for Model Registry errors."""

    pass


class ModelNotFoundError(ModelRegistryError):
    """Model not found in registry."""

    pass


class ModelRegistryClient:
    """Client for OpenShift AI Model Registry REST API."""

    def __init__(self, settings: Optional[ModelRegistrySettings] = None) -> None:
        """Initialize client."""
        self._settings = settings or ModelRegistrySettings()
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self._settings.url,
                timeout=self._settings.timeout_seconds,
            )
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def list_registered_models(
        self,
        page_size: int = 100,
        order_by: str = "UPDATE_TIME",
    ) -> list[RegisteredModel]:
        """List all registered models."""
        client = await self._get_client()
        params = {
            "pageSize": page_size,
            "orderBy": order_by,
        }

        try:
            response = await client.get("/api/model_registry/v1alpha3/registered_models", params=params)
            response.raise_for_status()
            data = response.json()

            models = []
            for item in data.get("items", []):
                models.append(self._parse_registered_model(item))
            return models

        except httpx.HTTPStatusError as e:
            raise ModelRegistryError(f"Failed to list models: {e}") from e

    async def get_registered_model(self, model_id: str) -> RegisteredModel:
        """Get a registered model by ID."""
        client = await self._get_client()

        try:
            response = await client.get(f"/api/model_registry/v1alpha3/registered_models/{model_id}")
            if response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {model_id}")
            response.raise_for_status()
            return self._parse_registered_model(response.json())

        except httpx.HTTPStatusError as e:
            raise ModelRegistryError(f"Failed to get model: {e}") from e

    async def get_registered_model_by_name(self, name: str) -> Optional[RegisteredModel]:
        """Get a registered model by name."""
        models = await self.list_registered_models()
        for model in models:
            if model.name == name:
                return model
        return None

    async def get_model_versions(self, model_id: str) -> list[ModelVersion]:
        """Get all versions of a registered model."""
        client = await self._get_client()

        try:
            response = await client.get(
                f"/api/model_registry/v1alpha3/registered_models/{model_id}/versions"
            )
            response.raise_for_status()
            data = response.json()

            versions = []
            for item in data.get("items", []):
                versions.append(self._parse_model_version(item))
            return versions

        except httpx.HTTPStatusError as e:
            raise ModelRegistryError(f"Failed to get model versions: {e}") from e

    async def get_model_version(self, version_id: str) -> ModelVersion:
        """Get a specific model version."""
        client = await self._get_client()

        try:
            response = await client.get(f"/api/model_registry/v1alpha3/model_versions/{version_id}")
            if response.status_code == 404:
                raise ModelNotFoundError(f"Version not found: {version_id}")
            response.raise_for_status()
            return self._parse_model_version(response.json())

        except httpx.HTTPStatusError as e:
            raise ModelRegistryError(f"Failed to get model version: {e}") from e

    async def get_model_artifacts(self, version_id: str) -> list[ModelArtifact]:
        """Get artifacts for a model version."""
        client = await self._get_client()

        try:
            response = await client.get(
                f"/api/model_registry/v1alpha3/model_versions/{version_id}/artifacts"
            )
            response.raise_for_status()
            data = response.json()

            artifacts = []
            for item in data.get("items", []):
                artifacts.append(self._parse_model_artifact(item))
            return artifacts

        except httpx.HTTPStatusError as e:
            raise ModelRegistryError(f"Failed to get artifacts: {e}") from e

    async def get_metric_history(
        self,
        run_id: str,
        metric_name: str,
    ) -> MetricHistory:
        """Get metric history from an experiment run."""
        client = await self._get_client()

        try:
            response = await client.get(
                f"/api/model_registry/v1alpha3/experiment_runs/{run_id}/metric_history",
                params={"metric_name": metric_name},
            )
            response.raise_for_status()
            data = response.json()

            history = []
            for point in data.get("metrics", []):
                history.append(
                    MetricHistoryPoint(
                        step=point.get("step", 0),
                        value=point.get("value", 0.0),
                    )
                )

            return MetricHistory(
                metric_name=metric_name,
                run_id=run_id,
                history=history,
            )

        except httpx.HTTPStatusError as e:
            raise ModelRegistryError(f"Failed to get metric history: {e}") from e

    def _parse_registered_model(self, data: dict[str, Any]) -> RegisteredModel:
        """Parse registered model from API response."""
        return RegisteredModel(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description"),
            owner=data.get("owner"),
            state=data.get("state", "LIVE"),
            custom_properties=CustomProperties(
                properties=data.get("customProperties", {})
            ),
        )

    def _parse_model_version(self, data: dict[str, Any]) -> ModelVersion:
        """Parse model version from API response."""
        return ModelVersion(
            id=data.get("id", ""),
            name=data.get("name", ""),
            registered_model_id=data.get("registeredModelId", ""),
            state=data.get("state", "LIVE"),
            description=data.get("description"),
            author=data.get("author"),
            custom_properties=CustomProperties(
                properties=data.get("customProperties", {})
            ),
        )

    def _parse_model_artifact(self, data: dict[str, Any]) -> ModelArtifact:
        """Parse model artifact from API response."""
        return ModelArtifact(
            id=data.get("id", ""),
            name=data.get("name", ""),
            uri=data.get("uri", ""),
            description=data.get("description"),
            model_format_name=data.get("modelFormatName"),
            model_format_version=data.get("modelFormatVersion"),
            storage_key=data.get("storageKey"),
            storage_path=data.get("storagePath"),
            custom_properties=CustomProperties(
                properties=data.get("customProperties", {})
            ),
        )

    async def __aenter__(self) -> "ModelRegistryClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
