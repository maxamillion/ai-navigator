"""Configuration management for AI Navigator."""

from enum import Enum
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class MCPTransport(str, Enum):
    """MCP transport protocol."""

    SSE = "sse"
    STDIO = "stdio"


class StateBackend(str, Enum):
    """State storage backend."""

    MEMORY = "memory"
    POSTGRES = "postgres"


class MCPSettings(BaseSettings):
    """MCP client configuration."""

    model_config = SettingsConfigDict(env_prefix="RHOAI_MCP_")

    host: str = Field(default="localhost", description="MCP server host")
    port: int = Field(default=8080, description="MCP server port")
    transport: MCPTransport = Field(default=MCPTransport.SSE, description="Transport protocol")
    timeout_seconds: int = Field(default=30, description="Request timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay_seconds: float = Field(default=1.0, description="Delay between retries")


class ModelRegistrySettings(BaseSettings):
    """Model Registry client configuration."""

    model_config = SettingsConfigDict(env_prefix="MODEL_REGISTRY_")

    url: str = Field(
        default="http://model-registry.odh-model-registries.svc:8080",
        description="Model Registry service URL",
    )
    timeout_seconds: int = Field(default=30, description="Request timeout")
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL for registry queries")


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    provider: str = Field(default="openai", description="LLM provider (openai-compatible)")
    base_url: str = Field(
        default="http://vllm.redhat-ods-applications.svc:8000/v1",
        description="LLM API base URL",
    )
    api_key: SecretStr = Field(default=SecretStr(""), description="API key if required")
    model_name: str = Field(default="granite-3b-code-instruct", description="Model to use")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    max_tokens: int = Field(default=2048, description="Maximum tokens in response")


class StateSettings(BaseSettings):
    """State storage configuration."""

    model_config = SettingsConfigDict(env_prefix="STATE_")

    backend: StateBackend = Field(default=StateBackend.MEMORY, description="Storage backend")
    postgres_dsn: Optional[str] = Field(default=None, description="PostgreSQL connection string")
    ttl_hours: int = Field(default=24, description="State TTL in hours")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="AI_NAVIGATOR_",
        env_nested_delimiter="__",
    )

    enabled: bool = Field(default=True, description="Enable AI Navigator plugin")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Environment")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    mcp: MCPSettings = Field(default_factory=MCPSettings)
    model_registry: ModelRegistrySettings = Field(default_factory=ModelRegistrySettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    state: StateSettings = Field(default_factory=StateSettings)

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
