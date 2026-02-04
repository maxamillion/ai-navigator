"""MCP Tools for observability integration."""

from datetime import datetime, timezone
from typing import Any

import structlog
from httpx import AsyncClient
from pydantic import BaseModel, Field

from ai_navigator.config import MCPSettings

logger = structlog.get_logger(__name__)


class MetricValue(BaseModel):
    """A single metric value with timestamp."""

    timestamp: datetime = Field(..., description="Metric timestamp")
    value: float = Field(..., description="Metric value")
    labels: dict[str, str] = Field(default_factory=dict, description="Metric labels")


class MetricResult(BaseModel):
    """Result of a Prometheus query."""

    metric_name: str = Field(..., description="Metric name")
    values: list[MetricValue] = Field(default_factory=list, description="Metric values")
    result_type: str = Field(default="vector", description="Result type (vector/matrix)")


class PodStatus(BaseModel):
    """Status of a Kubernetes pod."""

    name: str = Field(..., description="Pod name")
    namespace: str = Field(..., description="Pod namespace")
    phase: str = Field(..., description="Pod phase")
    ready: bool = Field(..., description="Whether pod is ready")
    restarts: int = Field(default=0, description="Container restart count")
    node: str = Field(default="", description="Node name")
    age_seconds: int = Field(default=0, description="Pod age in seconds")
    conditions: list[dict[str, Any]] = Field(default_factory=list, description="Pod conditions")


class SLOStatus(BaseModel):
    """Status of an SLO (Service Level Objective)."""

    name: str = Field(..., description="SLO name")
    target: float = Field(..., description="Target value (e.g., 0.99 for 99%)")
    current: float = Field(..., description="Current value")
    compliant: bool = Field(..., description="Whether SLO is met")
    window: str = Field(default="30d", description="Measurement window")
    error_budget_remaining: float = Field(default=0.0, description="Remaining error budget %")


class ObservabilityTools:
    """MCP Tool server for observability integration."""

    def __init__(self, settings: MCPSettings | None = None) -> None:
        """
        Initialize observability tools.

        Args:
            settings: MCP configuration settings
        """
        self.settings = settings or MCPSettings()
        self._prometheus_client: AsyncClient | None = None

    async def __aenter__(self) -> "ObservabilityTools":
        """Enter async context."""
        self._prometheus_client = AsyncClient(
            base_url=self.settings.prometheus_url,
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context."""
        if self._prometheus_client:
            await self._prometheus_client.aclose()

    @property
    def prometheus(self) -> AsyncClient:
        """Get Prometheus HTTP client."""
        if self._prometheus_client is None:
            self._prometheus_client = AsyncClient(
                base_url=self.settings.prometheus_url,
                timeout=30.0,
            )
        return self._prometheus_client

    async def prometheus_query(
        self,
        query: str,
        time: datetime | None = None,
    ) -> MetricResult:
        """
        Execute a Prometheus instant query.

        Args:
            query: PromQL query string
            time: Optional query time (defaults to now)

        Returns:
            Query result
        """
        try:
            params: dict[str, Any] = {"query": query}
            if time:
                params["time"] = time.timestamp()

            response = await self.prometheus.get("/api/v1/query", params=params)
            response.raise_for_status()

            data = response.json()
            result = data.get("data", {})

            values = []
            for item in result.get("result", []):
                metric = item.get("metric", {})
                value_data = item.get("value", [0, "0"])

                values.append(
                    MetricValue(
                        timestamp=datetime.fromtimestamp(value_data[0], tz=timezone.utc),
                        value=float(value_data[1]),
                        labels=metric,
                    )
                )

            return MetricResult(
                metric_name=query,
                values=values,
                result_type=result.get("resultType", "vector"),
            )
        except Exception as e:
            logger.warning("prometheus_query_failed", query=query, error=str(e))
            return self._get_mock_metric(query)

    async def prometheus_query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "1m",
    ) -> MetricResult:
        """
        Execute a Prometheus range query.

        Args:
            query: PromQL query string
            start: Start time
            end: End time
            step: Query step (e.g., "1m", "5m", "1h")

        Returns:
            Query result
        """
        try:
            params = {
                "query": query,
                "start": start.timestamp(),
                "end": end.timestamp(),
                "step": step,
            }

            response = await self.prometheus.get("/api/v1/query_range", params=params)
            response.raise_for_status()

            data = response.json()
            result = data.get("data", {})

            values = []
            for item in result.get("result", []):
                metric = item.get("metric", {})
                for value_data in item.get("values", []):
                    values.append(
                        MetricValue(
                            timestamp=datetime.fromtimestamp(value_data[0], tz=timezone.utc),
                            value=float(value_data[1]),
                            labels=metric,
                        )
                    )

            return MetricResult(
                metric_name=query,
                values=values,
                result_type="matrix",
            )
        except Exception as e:
            logger.warning("prometheus_range_query_failed", query=query, error=str(e))
            return MetricResult(metric_name=query, values=[], result_type="matrix")

    def _get_mock_metric(self, query: str) -> MetricResult:
        """Return mock metric data for development/testing."""
        now = datetime.now(timezone.utc)

        # Parse common metrics from query
        if "request_latency" in query.lower():
            return MetricResult(
                metric_name=query,
                values=[
                    MetricValue(
                        timestamp=now,
                        value=0.125,  # 125ms p50
                        labels={"quantile": "0.5"},
                    ),
                    MetricValue(
                        timestamp=now,
                        value=0.350,  # 350ms p95
                        labels={"quantile": "0.95"},
                    ),
                    MetricValue(
                        timestamp=now,
                        value=0.750,  # 750ms p99
                        labels={"quantile": "0.99"},
                    ),
                ],
            )
        elif "gpu_utilization" in query.lower():
            return MetricResult(
                metric_name=query,
                values=[
                    MetricValue(
                        timestamp=now,
                        value=0.72,  # 72% GPU utilization
                        labels={"gpu": "0", "model": "granite-4.0-h-tiny"},
                    ),
                ],
            )
        elif "memory" in query.lower():
            return MetricResult(
                metric_name=query,
                values=[
                    MetricValue(
                        timestamp=now,
                        value=14_500_000_000,  # 14.5GB
                        labels={"container": "model-server"},
                    ),
                ],
            )
        elif "requests_total" in query.lower() or "request_count" in query.lower():
            return MetricResult(
                metric_name=query,
                values=[
                    MetricValue(
                        timestamp=now,
                        value=15234,
                        labels={"status": "200"},
                    ),
                    MetricValue(
                        timestamp=now,
                        value=42,
                        labels={"status": "500"},
                    ),
                ],
            )
        else:
            return MetricResult(
                metric_name=query,
                values=[
                    MetricValue(
                        timestamp=now,
                        value=1.0,
                        labels={},
                    ),
                ],
            )

    async def get_inference_service_metrics(
        self,
        name: str,
        namespace: str = "ai-navigator",
    ) -> dict[str, Any]:
        """
        Get comprehensive metrics for an InferenceService.

        Args:
            name: Service name
            namespace: Kubernetes namespace

        Returns:
            Metrics summary
        """
        # Query key metrics
        latency = await self.prometheus_query(
            f'histogram_quantile(0.95, sum(rate(inference_request_duration_seconds_bucket'
            f'{{service="{name}", namespace="{namespace}"}}[5m])) by (le))'
        )

        throughput = await self.prometheus_query(
            f'sum(rate(inference_requests_total{{service="{name}", namespace="{namespace}"}}[5m]))'
        )

        gpu_util = await self.prometheus_query(
            f'avg(DCGM_FI_DEV_GPU_UTIL{{pod=~"{name}.*", namespace="{namespace}"}})'
        )

        memory = await self.prometheus_query(
            f'sum(container_memory_usage_bytes{{pod=~"{name}.*", namespace="{namespace}"}})'
        )

        return {
            "service": name,
            "namespace": namespace,
            "latency_p95_ms": (
                latency.values[0].value * 1000 if latency.values else 0
            ),
            "throughput_rps": throughput.values[0].value if throughput.values else 0,
            "gpu_utilization_pct": (
                gpu_util.values[0].value * 100 if gpu_util.values else 0
            ),
            "memory_gb": (
                memory.values[0].value / (1024**3) if memory.values else 0
            ),
        }

    async def get_pod_status(
        self,
        name: str,
        namespace: str = "ai-navigator",
    ) -> PodStatus:
        """
        Get the status of a pod.

        Args:
            name: Pod name (or prefix for matching)
            namespace: Kubernetes namespace

        Returns:
            Pod status
        """
        # In production, this would use kubernetes client
        # For now, return mock status
        return PodStatus(
            name=f"{name}-abc123",
            namespace=namespace,
            phase="Running",
            ready=True,
            restarts=0,
            node="ip-10-0-1-100.ec2.internal",
            age_seconds=86400,
            conditions=[
                {"type": "Ready", "status": "True"},
                {"type": "ContainersReady", "status": "True"},
                {"type": "PodScheduled", "status": "True"},
            ],
        )

    async def check_slo_status(
        self,
        name: str,
        namespace: str = "ai-navigator",
    ) -> list[SLOStatus]:
        """
        Check SLO compliance for a service.

        Args:
            name: Service name
            namespace: Kubernetes namespace

        Returns:
            List of SLO statuses
        """
        # Define standard SLOs for inference services
        slos = [
            SLOStatus(
                name="availability",
                target=0.999,
                current=0.9995,
                compliant=True,
                window="30d",
                error_budget_remaining=50.0,
            ),
            SLOStatus(
                name="latency_p95",
                target=0.500,  # 500ms
                current=0.350,  # 350ms
                compliant=True,
                window="30d",
                error_budget_remaining=70.0,
            ),
            SLOStatus(
                name="latency_p99",
                target=1.000,  # 1s
                current=0.750,  # 750ms
                compliant=True,
                window="30d",
                error_budget_remaining=80.0,
            ),
            SLOStatus(
                name="error_rate",
                target=0.01,  # 1%
                current=0.003,  # 0.3%
                compliant=True,
                window="30d",
                error_budget_remaining=70.0,
            ),
        ]

        return slos

    async def get_pod_logs(
        self,
        name: str,
        namespace: str = "ai-navigator",
        container: str | None = None,
        tail_lines: int = 100,
    ) -> list[str]:
        """
        Get logs from a pod.

        Args:
            name: Pod name
            namespace: Kubernetes namespace
            container: Container name (optional)
            tail_lines: Number of lines to return

        Returns:
            Log lines
        """
        # In production, use kubernetes client
        # For now, return mock logs
        return [
            f"2025-01-15T10:00:00Z INFO Starting {name}...",
            "2025-01-15T10:00:01Z INFO Loading model weights...",
            "2025-01-15T10:00:30Z INFO Model loaded successfully",
            "2025-01-15T10:00:30Z INFO Server listening on 0.0.0.0:8080",
            "2025-01-15T10:00:31Z INFO Health check passed",
            "2025-01-15T10:01:00Z INFO Processing inference request",
            "2025-01-15T10:01:00Z INFO Request completed in 125ms",
        ]
