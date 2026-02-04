"""Skills for the Deployment Monitor Agent."""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import structlog

from ai_navigator.a2a.skills import SkillInput, SkillResult

if TYPE_CHECKING:
    from ai_navigator.agents.deployment_monitor.agent import DeploymentMonitorAgent

logger = structlog.get_logger(__name__)


def register_skills(agent: "DeploymentMonitorAgent") -> None:
    """Register all skills for the Deployment Monitor Agent."""

    @agent.skills.register(
        id="get_deployment_status",
        name="Get Deployment Status",
        description="Get the current status of an InferenceService deployment",
        tags=["deployment", "status", "health"],
        examples=[
            "Get status of granite-4-0-h-tiny deployment",
            "Check if the model is running",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Service name"},
                "namespace": {"type": "string", "description": "Kubernetes namespace"},
            },
        },
    )
    async def get_deployment_status(input: SkillInput) -> SkillResult:
        """Get deployment status."""
        params = input.params
        name = params.get("name", "granite-4-0-h-tiny")
        namespace = params.get("namespace", "ai-navigator")

        try:
            status = await agent.openshift_ai.get_inference_service_status(name, namespace)
            pod_status = await agent.observability.get_pod_status(name, namespace)

            ready_icon = "Ready" if status.get("ready") else "Not Ready"

            message = f"""
## Deployment Status: {name}

### Overview
- **Status:** {ready_icon}
- **Namespace:** {namespace}
- **URL:** {status.get('url', 'N/A')}

### Replicas
- **Desired:** {status.get('replicas', {}).get('desired', 0)}
- **Ready:** {status.get('replicas', {}).get('ready', 0)}
- **Available:** {status.get('replicas', {}).get('available', 0)}

### Pod Status
- **Name:** {pod_status.name}
- **Phase:** {pod_status.phase}
- **Ready:** {"Yes" if pod_status.ready else "No"}
- **Restarts:** {pod_status.restarts}
- **Node:** {pod_status.node}
- **Age:** {pod_status.age_seconds // 3600}h {(pod_status.age_seconds % 3600) // 60}m

### Conditions
"""
            for condition in status.get("conditions", []):
                status_val = condition.get("status", "Unknown")
                cond_type = condition.get("type", "Unknown")
                message += f"- **{cond_type}:** {status_val}\n"

            return SkillResult.ok(
                message=message,
                data={
                    "status": status,
                    "pod": pod_status.model_dump(),
                },
            )
        except Exception as e:
            logger.exception("get_deployment_status_failed", error=str(e))
            return SkillResult.error(f"Failed to get deployment status: {e}")

    @agent.skills.register(
        id="query_metrics",
        name="Query Metrics",
        description="Query Prometheus metrics for an inference service",
        tags=["metrics", "prometheus", "performance"],
        examples=[
            "Get latency metrics for granite-4-0-h-tiny",
            "Show GPU utilization",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Service name"},
                "namespace": {"type": "string", "description": "Kubernetes namespace"},
                "query": {"type": "string", "description": "Custom PromQL query"},
            },
        },
    )
    async def query_metrics(input: SkillInput) -> SkillResult:
        """Query Prometheus metrics."""
        params = input.params
        name = params.get("name", "granite-4-0-h-tiny")
        namespace = params.get("namespace", "ai-navigator")

        try:
            metrics = await agent.observability.get_inference_service_metrics(name, namespace)

            message = f"""
## Metrics: {name}

### Latency
- **P95:** {metrics['latency_p95_ms']:.1f}ms

### Throughput
- **Requests/sec:** {metrics['throughput_rps']:.2f}

### Resource Utilization
- **GPU:** {metrics['gpu_utilization_pct']:.1f}%
- **Memory:** {metrics['memory_gb']:.2f} GB

### Performance Assessment
"""
            # Add performance assessment
            if metrics["latency_p95_ms"] < 500:
                message += "- Latency: **Good** - Well within acceptable range\n"
            elif metrics["latency_p95_ms"] < 1000:
                message += "- Latency: **Acceptable** - Consider optimization\n"
            else:
                message += "- Latency: **High** - Needs investigation\n"

            if metrics["gpu_utilization_pct"] < 50:
                message += "- GPU: **Underutilized** - Consider batch optimization\n"
            elif metrics["gpu_utilization_pct"] < 80:
                message += "- GPU: **Good** - Healthy utilization\n"
            else:
                message += "- GPU: **High** - Monitor for thermal throttling\n"

            return SkillResult.ok(message=message, data=metrics)
        except Exception as e:
            logger.exception("query_metrics_failed", error=str(e))
            return SkillResult.error(f"Failed to query metrics: {e}")

    @agent.skills.register(
        id="check_slo_violations",
        name="Check SLO Violations",
        description="Check for SLO violations and compliance issues",
        tags=["slo", "compliance", "alerts"],
        examples=[
            "Check for SLO violations",
            "Are we meeting our SLOs?",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Service name"},
                "namespace": {"type": "string", "description": "Kubernetes namespace"},
            },
        },
    )
    async def check_slo_violations(input: SkillInput) -> SkillResult:
        """Check for SLO violations."""
        params = input.params
        name = params.get("name", "granite-4-0-h-tiny")
        namespace = params.get("namespace", "ai-navigator")

        try:
            slos = await agent.observability.check_slo_status(name, namespace)

            all_compliant = all(slo.compliant for slo in slos)
            compliance_status = "All SLOs Met" if all_compliant else "SLO Violations Detected"

            message = f"""
## SLO Compliance Report: {name}

### Overall Status: {compliance_status}

### SLO Details

| SLO | Target | Current | Status | Error Budget |
|-----|--------|---------|--------|--------------|
"""
            for slo in slos:
                status_icon = "OK" if slo.compliant else "VIOLATION"
                # Format values appropriately
                if slo.name == "error_rate":
                    target = f"{slo.target:.1%}"
                    current = f"{slo.current:.1%}"
                elif "latency" in slo.name:
                    target = f"{slo.target * 1000:.0f}ms"
                    current = f"{slo.current * 1000:.0f}ms"
                else:
                    target = f"{slo.target:.2%}"
                    current = f"{slo.current:.3%}"

                message += f"| {slo.name} | {target} | {current} | {status_icon} | {slo.error_budget_remaining:.0f}% |\n"

            # Add recommendations if there are violations
            violations = [slo for slo in slos if not slo.compliant]
            if violations:
                message += "\n### Violations Detected\n"
                for v in violations:
                    message += f"- **{v.name}**: Current {v.current:.2%} exceeds target {v.target:.2%}\n"

                message += "\n### Recommended Actions\n"
                message += "1. Scale up replicas to improve availability\n"
                message += "2. Review recent deployments for regressions\n"
                message += "3. Check resource limits and quotas\n"

            # Error budget summary
            message += "\n### Error Budget Summary\n"
            low_budget = [slo for slo in slos if slo.error_budget_remaining < 30]
            if low_budget:
                message += "\n**Warning:** The following SLOs have low error budget:\n"
                for slo in low_budget:
                    message += f"- {slo.name}: {slo.error_budget_remaining:.0f}% remaining\n"
            else:
                message += "All SLOs have healthy error budgets (>30%)\n"

            return SkillResult.ok(
                message=message,
                data={
                    "compliant": all_compliant,
                    "slos": [slo.model_dump() for slo in slos],
                    "violations": [slo.model_dump() for slo in violations],
                },
            )
        except Exception as e:
            logger.exception("check_slo_violations_failed", error=str(e))
            return SkillResult.error(f"Failed to check SLO violations: {e}")

    @agent.skills.register(
        id="get_pod_logs",
        name="Get Pod Logs",
        description="Retrieve logs from deployment pods",
        tags=["logs", "debugging", "troubleshooting"],
        examples=[
            "Get logs for granite-4-0-h-tiny",
            "Show recent error logs",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Service name"},
                "namespace": {"type": "string", "description": "Kubernetes namespace"},
                "tail_lines": {"type": "integer", "description": "Number of log lines"},
            },
        },
    )
    async def get_pod_logs(input: SkillInput) -> SkillResult:
        """Get pod logs."""
        params = input.params
        name = params.get("name", "granite-4-0-h-tiny")
        namespace = params.get("namespace", "ai-navigator")
        tail_lines = params.get("tail_lines", 50)

        try:
            logs = await agent.observability.get_pod_logs(
                name, namespace, tail_lines=tail_lines
            )

            # Analyze logs for issues
            error_count = sum(1 for line in logs if "ERROR" in line.upper())
            warning_count = sum(1 for line in logs if "WARN" in line.upper())

            message = f"""
## Pod Logs: {name}

### Summary
- **Lines Retrieved:** {len(logs)}
- **Errors:** {error_count}
- **Warnings:** {warning_count}

### Recent Logs
```
{chr(10).join(logs[-20:])}
```
"""

            if error_count > 0:
                message += "\n### Error Lines\n```\n"
                for line in logs:
                    if "ERROR" in line.upper():
                        message += f"{line}\n"
                message += "```\n"

            return SkillResult.ok(
                message=message,
                data={
                    "logs": logs,
                    "error_count": error_count,
                    "warning_count": warning_count,
                },
            )
        except Exception as e:
            logger.exception("get_pod_logs_failed", error=str(e))
            return SkillResult.error(f"Failed to get pod logs: {e}")

    @agent.skills.register(
        id="get_health_summary",
        name="Get Health Summary",
        description="Get a comprehensive health summary for a deployment",
        tags=["health", "summary", "overview"],
        examples=[
            "Give me a health summary",
            "How is the deployment doing?",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Service name"},
                "namespace": {"type": "string", "description": "Kubernetes namespace"},
            },
        },
    )
    async def get_health_summary(input: SkillInput) -> SkillResult:
        """Get comprehensive health summary."""
        params = input.params
        name = params.get("name", "granite-4-0-h-tiny")
        namespace = params.get("namespace", "ai-navigator")

        try:
            # Gather all health information
            status = await agent.openshift_ai.get_inference_service_status(name, namespace)
            metrics = await agent.observability.get_inference_service_metrics(name, namespace)
            slos = await agent.observability.check_slo_status(name, namespace)
            pod_status = await agent.observability.get_pod_status(name, namespace)

            # Calculate overall health score
            health_factors = []

            # Pod health
            if pod_status.ready:
                health_factors.append(1.0)
            else:
                health_factors.append(0.0)

            # SLO compliance
            slo_compliance = sum(1 for s in slos if s.compliant) / len(slos)
            health_factors.append(slo_compliance)

            # Latency health (target < 500ms)
            latency_health = max(0, 1 - (metrics["latency_p95_ms"] / 1000))
            health_factors.append(latency_health)

            # GPU utilization (optimal 40-80%)
            gpu = metrics["gpu_utilization_pct"]
            if 40 <= gpu <= 80:
                gpu_health = 1.0
            elif gpu < 40:
                gpu_health = gpu / 40
            else:
                gpu_health = max(0, 1 - (gpu - 80) / 20)
            health_factors.append(gpu_health)

            overall_health = sum(health_factors) / len(health_factors)

            if overall_health >= 0.9:
                health_status = "Excellent"
                health_icon = "Healthy"
            elif overall_health >= 0.7:
                health_status = "Good"
                health_icon = "Healthy"
            elif overall_health >= 0.5:
                health_status = "Fair"
                health_icon = "Degraded"
            else:
                health_status = "Poor"
                health_icon = "Unhealthy"

            message = f"""
## Health Summary: {name}

### Overall Health: {health_icon} ({overall_health:.0%})
**Assessment:** {health_status}

---

### Deployment Status
- **Ready:** {"Yes" if status.get('ready') else "No"}
- **Replicas:** {status.get('replicas', {}).get('ready', 0)}/{status.get('replicas', {}).get('desired', 0)}
- **Pod Phase:** {pod_status.phase}
- **Restarts:** {pod_status.restarts}

### Performance Metrics
- **Latency (p95):** {metrics['latency_p95_ms']:.1f}ms
- **Throughput:** {metrics['throughput_rps']:.2f} req/s
- **GPU Utilization:** {metrics['gpu_utilization_pct']:.1f}%
- **Memory Usage:** {metrics['memory_gb']:.2f} GB

### SLO Compliance
"""
            for slo in slos:
                status_str = "Met" if slo.compliant else "Violated"
                message += f"- **{slo.name}:** {status_str} ({slo.error_budget_remaining:.0f}% budget)\n"

            # Add recommendations
            message += "\n### Recommendations\n"

            if not status.get("ready"):
                message += "- **Critical:** Service not ready - check pod events\n"
            if pod_status.restarts > 0:
                message += f"- **Warning:** {pod_status.restarts} restarts detected - review logs\n"
            if metrics["latency_p95_ms"] > 500:
                message += "- **Performance:** Consider scaling or optimization\n"
            if metrics["gpu_utilization_pct"] < 40:
                message += "- **Efficiency:** GPU underutilized - consider request batching\n"

            violations = [s for s in slos if not s.compliant]
            if violations:
                message += "- **SLO:** Address violations to maintain service quality\n"

            if overall_health >= 0.9:
                message += "- Service is operating optimally. No immediate action needed.\n"

            return SkillResult.ok(
                message=message,
                data={
                    "overall_health": overall_health,
                    "health_status": health_status,
                    "status": status,
                    "metrics": metrics,
                    "slos": [s.model_dump() for s in slos],
                    "pod": pod_status.model_dump(),
                },
            )
        except Exception as e:
            logger.exception("get_health_summary_failed", error=str(e))
            return SkillResult.error(f"Failed to get health summary: {e}")
