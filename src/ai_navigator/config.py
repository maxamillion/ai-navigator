"""Configuration management for AI Navigator."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Configuration for an individual agent."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        extra="ignore",
    )

    # Agent identity
    name: str = Field(default="supervisor", description="Agent identifier")
    version: str = Field(default="0.1.0", description="Agent version")
    description: str = Field(
        default="AI Navigator Agent",
        description="Human-readable description",
    )

    # Network configuration
    host: str = Field(default="0.0.0.0", description="Bind address")
    port: int = Field(default=8000, description="HTTP port")
    base_path: str = Field(default="", description="Base URL path prefix")

    # Agent discovery
    supervisor_url: str | None = Field(
        default=None,
        description="URL of supervisor agent (for sub-agents)",
    )
    discovery_mode: str = Field(
        default="kubernetes",
        description="Agent discovery mode: kubernetes, static, or hybrid",
    )

    @property
    def endpoint(self) -> str:
        """Return the agent's endpoint URL."""
        return f"http://{self.host}:{self.port}{self.base_path}"


class LLMSettings(BaseSettings):
    """Configuration for LLM integration."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
    )

    endpoint: str = Field(
        default="http://localhost:8080/v1",
        description="LLM inference endpoint (OpenAI-compatible)",
    )
    model: str = Field(
        default="granite-4.0-h-tiny",
        description="Model name for inference",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authentication",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens in response",
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature",
    )
    timeout: float = Field(
        default=120.0,
        description="Request timeout in seconds",
    )


class KubernetesSettings(BaseSettings):
    """Configuration for Kubernetes integration."""

    model_config = SettingsConfigDict(
        env_prefix="KUBERNETES_",
        env_file=".env",
        extra="ignore",
    )

    namespace: str = Field(
        default="ai-navigator",
        description="Kubernetes namespace for agents",
    )
    in_cluster: bool = Field(
        default=True,
        description="Whether running inside a cluster",
    )
    kubeconfig: str | None = Field(
        default=None,
        description="Path to kubeconfig file (for out-of-cluster)",
    )
    agent_label_selector: str = Field(
        default="app.kubernetes.io/part-of=ai-navigator",
        description="Label selector for discovering agents",
    )


class MCPSettings(BaseSettings):
    """Configuration for MCP tool servers."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_",
        env_file=".env",
        extra="ignore",
    )

    model_registry_url: str = Field(
        default="http://model-registry:8080",
        description="OpenShift Model Registry endpoint",
    )
    prometheus_url: str = Field(
        default="http://prometheus:9090",
        description="Prometheus endpoint",
    )
    trustyai_url: str = Field(
        default="http://trustyai-service:8080",
        description="TrustyAI service endpoint",
    )


class Settings(BaseSettings):
    """Root configuration container."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    agent: AgentSettings = Field(default_factory=AgentSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    kubernetes: KubernetesSettings = Field(default_factory=KubernetesSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format: json or console")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
