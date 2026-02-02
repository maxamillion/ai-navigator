"""Caching layer for Model Registry queries."""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from ai_navigator.registry.models import RegisteredModel, ModelVersion

logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL."""

    value: Any
    created_at: float
    ttl_seconds: float

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds


@dataclass
class RegistryCacheConfig:
    """Cache configuration for registry queries."""

    models_ttl_seconds: float = 300.0  # 5 minutes
    versions_ttl_seconds: float = 300.0
    benchmarks_ttl_seconds: float = 600.0  # 10 minutes
    max_entries: int = 500


class RegistryCache:
    """In-memory cache for Model Registry data."""

    def __init__(self, config: Optional[RegistryCacheConfig] = None) -> None:
        """Initialize cache."""
        self._config = config or RegistryCacheConfig()
        self._models: dict[str, CacheEntry] = {}
        self._models_list: Optional[CacheEntry] = None
        self._versions: dict[str, CacheEntry] = {}
        self._benchmarks: dict[str, CacheEntry] = {}

    async def get_models_list(self) -> Optional[list[RegisteredModel]]:
        """Get cached models list."""
        if self._models_list and not self._models_list.is_expired():
            logger.debug("Cache hit: models list")
            return self._models_list.value
        return None

    async def set_models_list(self, models: list[RegisteredModel]) -> None:
        """Cache models list."""
        self._models_list = CacheEntry(
            value=models,
            created_at=time.time(),
            ttl_seconds=self._config.models_ttl_seconds,
        )

    async def get_model(self, model_id: str) -> Optional[RegisteredModel]:
        """Get cached model by ID."""
        entry = self._models.get(model_id)
        if entry and not entry.is_expired():
            logger.debug("Cache hit: model", model_id=model_id)
            return entry.value
        return None

    async def set_model(self, model: RegisteredModel) -> None:
        """Cache a model."""
        self._ensure_capacity(self._models)
        self._models[model.id] = CacheEntry(
            value=model,
            created_at=time.time(),
            ttl_seconds=self._config.models_ttl_seconds,
        )

    async def get_versions(self, model_id: str) -> Optional[list[ModelVersion]]:
        """Get cached versions for a model."""
        entry = self._versions.get(model_id)
        if entry and not entry.is_expired():
            logger.debug("Cache hit: versions", model_id=model_id)
            return entry.value
        return None

    async def set_versions(self, model_id: str, versions: list[ModelVersion]) -> None:
        """Cache versions for a model."""
        self._ensure_capacity(self._versions)
        self._versions[model_id] = CacheEntry(
            value=versions,
            created_at=time.time(),
            ttl_seconds=self._config.versions_ttl_seconds,
        )

    async def get_benchmark(self, cache_key: str) -> Optional[Any]:
        """Get cached benchmark data."""
        entry = self._benchmarks.get(cache_key)
        if entry and not entry.is_expired():
            logger.debug("Cache hit: benchmark", key=cache_key)
            return entry.value
        return None

    async def set_benchmark(self, cache_key: str, data: Any) -> None:
        """Cache benchmark data."""
        self._ensure_capacity(self._benchmarks)
        self._benchmarks[cache_key] = CacheEntry(
            value=data,
            created_at=time.time(),
            ttl_seconds=self._config.benchmarks_ttl_seconds,
        )

    def make_benchmark_key(
        self,
        model_name: str,
        version: Optional[str] = None,
        gpu_type: Optional[str] = None,
    ) -> str:
        """Create cache key for benchmark data."""
        parts = [model_name]
        if version:
            parts.append(version)
        if gpu_type:
            parts.append(gpu_type)
        return ":".join(parts)

    async def invalidate_model(self, model_id: str) -> None:
        """Invalidate cache for a model."""
        self._models.pop(model_id, None)
        self._versions.pop(model_id, None)
        # Invalidate related benchmarks
        keys_to_remove = [k for k in self._benchmarks if k.startswith(model_id)]
        for key in keys_to_remove:
            del self._benchmarks[key]

    async def invalidate_all(self) -> None:
        """Clear all cached data."""
        self._models.clear()
        self._models_list = None
        self._versions.clear()
        self._benchmarks.clear()

    def _ensure_capacity(self, cache: dict[str, CacheEntry]) -> None:
        """Ensure cache doesn't exceed max entries."""
        # Remove expired entries first
        expired = [k for k, v in cache.items() if v.is_expired()]
        for key in expired:
            del cache[key]

        # If still over capacity, remove oldest entries
        while len(cache) >= self._config.max_entries:
            oldest_key = min(cache.keys(), key=lambda k: cache[k].created_at)
            del cache[oldest_key]

    @property
    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "models": len(self._models),
            "versions": len(self._versions),
            "benchmarks": len(self._benchmarks),
            "has_models_list": self._models_list is not None,
        }
