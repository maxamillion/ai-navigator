"""MCP client adapter for rhoai-mcp server."""

import asyncio
from typing import Any, Optional

import httpx
import structlog

from ai_navigator.config import MCPSettings, MCPTransport

logger = structlog.get_logger(__name__)


class MCPClientError(Exception):
    """Base exception for MCP client errors."""

    pass


class MCPConnectionError(MCPClientError):
    """Connection to MCP server failed."""

    pass


class MCPToolError(MCPClientError):
    """Tool execution failed."""

    def __init__(self, message: str, tool_name: str, details: Optional[dict] = None):
        super().__init__(message)
        self.tool_name = tool_name
        self.details = details or {}


class MCPClient:
    """Client adapter for rhoai-mcp server."""

    def __init__(self, settings: Optional[MCPSettings] = None) -> None:
        """Initialize MCP client."""
        self._settings = settings or MCPSettings()
        self._http_client: Optional[httpx.AsyncClient] = None
        self._available_tools: dict[str, dict[str, Any]] = {}
        self._connected = False

    @property
    def base_url(self) -> str:
        """Get MCP server base URL."""
        return f"http://{self._settings.host}:{self._settings.port}"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._settings.timeout_seconds,
            )
        return self._http_client

    async def connect(self) -> None:
        """Connect to MCP server and discover available tools."""
        client = await self._get_client()
        try:
            if self._settings.transport == MCPTransport.SSE:
                response = await client.get("/mcp/tools")
                response.raise_for_status()
                tools_data = response.json()
                self._available_tools = {
                    tool["name"]: tool for tool in tools_data.get("tools", [])
                }
            self._connected = True
            logger.info(
                "Connected to MCP server",
                url=self.base_url,
                tools_count=len(self._available_tools),
            )
        except httpx.HTTPError as e:
            raise MCPConnectionError(f"Failed to connect to MCP server: {e}") from e

    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to MCP server."""
        return self._connected

    def list_tools(self) -> list[str]:
        """List available tool names."""
        return list(self._available_tools.keys())

    def get_tool_schema(self, tool_name: str) -> Optional[dict[str, Any]]:
        """Get schema for a specific tool."""
        return self._available_tools.get(tool_name)

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute a tool on the MCP server."""
        if not self._connected:
            await self.connect()

        client = await self._get_client()
        payload = {
            "name": tool_name,
            "arguments": arguments or {},
        }

        try:
            response = await client.post("/mcp/tools/call", json=payload)
            response.raise_for_status()
            result = response.json()

            if result.get("error"):
                raise MCPToolError(
                    result["error"].get("message", "Unknown error"),
                    tool_name,
                    result["error"],
                )

            logger.debug(
                "Tool executed successfully",
                tool=tool_name,
                arguments=arguments,
            )
            return result.get("result", {})

        except httpx.HTTPStatusError as e:
            raise MCPToolError(
                f"HTTP error calling tool: {e.response.status_code}",
                tool_name,
                {"status_code": e.response.status_code},
            ) from e
        except httpx.HTTPError as e:
            raise MCPConnectionError(f"Connection error calling tool: {e}") from e

    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()


# Tool-specific helper methods for common operations
class MCPToolHelpers:
    """Helper methods for common MCP tool operations."""

    def __init__(self, client: MCPClient) -> None:
        """Initialize with MCP client."""
        self._client = client

    # Project operations
    async def create_project(
        self,
        name: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a Data Science Project."""
        return await self._client.call_tool(
            "create_data_science_project",
            {
                "name": name,
                "display_name": display_name or name,
                "description": description or "",
            },
        )

    async def get_project_status(self, namespace: str) -> dict[str, Any]:
        """Get project status."""
        return await self._client.call_tool(
            "get_project_status",
            {"namespace": namespace},
        )

    async def list_projects(self) -> list[dict[str, Any]]:
        """List all Data Science Projects."""
        result = await self._client.call_tool("list_data_science_projects")
        return result.get("projects", [])

    # Model operations
    async def list_inference_services(self, namespace: str) -> list[dict[str, Any]]:
        """List inference services in a namespace."""
        result = await self._client.call_tool(
            "list_inference_services",
            {"namespace": namespace},
        )
        return result.get("services", [])

    async def create_inference_service(
        self,
        namespace: str,
        name: str,
        model_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create an inference service."""
        return await self._client.call_tool(
            "create_inference_service",
            {
                "namespace": namespace,
                "name": name,
                "model_name": model_name,
                **kwargs,
            },
        )

    async def get_inference_service(
        self, namespace: str, name: str
    ) -> Optional[dict[str, Any]]:
        """Get inference service details."""
        return await self._client.call_tool(
            "get_inference_service",
            {"namespace": namespace, "name": name},
        )

    async def delete_inference_service(self, namespace: str, name: str) -> dict[str, Any]:
        """Delete an inference service."""
        return await self._client.call_tool(
            "delete_inference_service",
            {"namespace": namespace, "name": name},
        )

    # Workbench operations
    async def list_workbenches(self, namespace: str) -> list[dict[str, Any]]:
        """List workbenches in a namespace."""
        result = await self._client.call_tool(
            "list_workbenches",
            {"namespace": namespace},
        )
        return result.get("workbenches", [])

    # Data connection operations
    async def create_s3_connection(
        self,
        namespace: str,
        name: str,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
    ) -> dict[str, Any]:
        """Create S3 data connection."""
        return await self._client.call_tool(
            "create_s3_data_connection",
            {
                "namespace": namespace,
                "name": name,
                "endpoint": endpoint,
                "bucket": bucket,
                "access_key": access_key,
                "secret_key": secret_key,
                "region": region,
            },
        )

    # Storage operations
    async def create_storage(
        self,
        namespace: str,
        name: str,
        size: str = "10Gi",
        storage_class: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create persistent storage."""
        args: dict[str, Any] = {
            "namespace": namespace,
            "name": name,
            "size": size,
        }
        if storage_class:
            args["storage_class"] = storage_class
        return await self._client.call_tool("create_storage", args)
