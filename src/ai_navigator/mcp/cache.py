"""Tool result caching for MCP operations."""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL."""

    value: dict[str, Any]
    created_at: float
    ttl_seconds: float

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds


@dataclass
class CacheConfig:
    """Cache configuration."""

    default_ttl_seconds: float = 300.0  # 5 minutes
    max_entries: int = 1000
    tool_ttls: dict[str, float] = field(default_factory=dict)

    def get_ttl(self, tool_name: str) -> float:
        """Get TTL for a specific tool."""
        return self.tool_ttls.get(tool_name, self.default_ttl_seconds)


# Default cache TTLs for different tool categories
DEFAULT_TOOL_TTLS = {
    # Read operations - longer cache
    "list_data_science_projects": 60.0,
    "list_workbenches": 30.0,
    "list_inference_services": 30.0,
    "get_project_status": 15.0,
    "get_inference_service": 15.0,
    # Write operations - no cache
    "create_data_science_project": 0.0,
    "create_workbench": 0.0,
    "create_inference_service": 0.0,
    "create_s3_data_connection": 0.0,
    "create_storage": 0.0,
    "delete_inference_service": 0.0,
}


class MCPCache:
    """In-memory cache for MCP tool results."""

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        """Initialize cache."""
        self._config = config or CacheConfig(tool_ttls=DEFAULT_TOOL_TTLS)
        self._entries: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []

    def _make_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Create cache key from tool name and arguments."""
        args_json = json.dumps(arguments, sort_keys=True)
        args_hash = hashlib.sha256(args_json.encode()).hexdigest()[:16]
        return f"{tool_name}:{args_hash}"

    async def get(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Get cached result if available and not expired."""
        key = self._make_key(tool_name, arguments)
        entry = self._entries.get(key)

        if entry is None:
            return None

        if entry.is_expired():
            del self._entries[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return None

        # Update access order for LRU
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        logger.debug("Cache hit", tool=tool_name, key=key)
        return entry.value

    async def set(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        ttl_override: Optional[float] = None,
    ) -> None:
        """Cache a tool result."""
        ttl = ttl_override if ttl_override is not None else self._config.get_ttl(tool_name)

        if ttl <= 0:
            # Don't cache write operations
            return

        key = self._make_key(tool_name, arguments)

        # Evict if at capacity
        self._evict_if_needed()

        self._entries[key] = CacheEntry(
            value=result,
            created_at=time.time(),
            ttl_seconds=ttl,
        )
        self._access_order.append(key)

        logger.debug("Cache set", tool=tool_name, key=key, ttl=ttl)

    async def invalidate(
        self,
        tool_name: Optional[str] = None,
        arguments: Optional[dict[str, Any]] = None,
    ) -> int:
        """Invalidate cache entries."""
        count = 0

        if tool_name and arguments:
            # Invalidate specific entry
            key = self._make_key(tool_name, arguments)
            if key in self._entries:
                del self._entries[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                count = 1
        elif tool_name:
            # Invalidate all entries for a tool
            prefix = f"{tool_name}:"
            keys_to_delete = [k for k in self._entries if k.startswith(prefix)]
            for key in keys_to_delete:
                del self._entries[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                count += 1
        else:
            # Invalidate all entries
            count = len(self._entries)
            self._entries.clear()
            self._access_order.clear()

        logger.debug("Cache invalidated", tool=tool_name, count=count)
        return count

    async def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all entries related to a namespace."""
        count = 0
        keys_to_delete = []

        for key, entry in self._entries.items():
            # Check if namespace appears in cached value
            if isinstance(entry.value, dict):
                if entry.value.get("namespace") == namespace:
                    keys_to_delete.append(key)

        for key in keys_to_delete:
            del self._entries[key]
            if key in self._access_order:
                self._access_order.remove(key)
            count += 1

        return count

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if at capacity."""
        while len(self._entries) >= self._config.max_entries:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._entries:
                    del self._entries[oldest_key]

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()
        self._access_order.clear()

    @property
    def size(self) -> int:
        """Get number of cached entries."""
        return len(self._entries)

    async def cleanup_expired(self) -> int:
        """Remove all expired entries."""
        expired_keys = [k for k, v in self._entries.items() if v.is_expired()]
        for key in expired_keys:
            del self._entries[key]
            if key in self._access_order:
                self._access_order.remove(key)
        return len(expired_keys)
