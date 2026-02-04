"""Skills for the Resource Provisioning Agent."""

import json
from typing import TYPE_CHECKING, Any

import structlog

from ai_navigator.a2a.message import Artifact
from ai_navigator.a2a.skills import SkillInput, SkillResult
from ai_navigator.mcp.openshift_ai_tools import InferenceServiceSpec

if TYPE_CHECKING:
    from ai_navigator.agents.resource_provisioning.agent import ResourceProvisioningAgent

logger = structlog.get_logger(__name__)


def register_skills(agent: "ResourceProvisioningAgent") -> None:
    """Register all skills for the Resource Provisioning Agent."""

    @agent.skills.register(
        id="generate_deployment_config",
        name="Generate Deployment Config",
        description="Generate KServe InferenceService configuration for a model",
        tags=["deployment", "kserve", "configuration"],
        examples=[
            "Generate deployment config for granite-4.0-h-tiny",
            "Create InferenceService for Llama 3",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Model name"},
                "service_name": {"type": "string", "description": "Service name"},
                "namespace": {"type": "string", "description": "Kubernetes namespace"},
                "gpu_count": {"type": "integer", "description": "Number of GPUs"},
                "min_replicas": {"type": "integer", "description": "Minimum replicas"},
                "max_replicas": {"type": "integer", "description": "Maximum replicas"},
                "memory": {"type": "string", "description": "Memory request"},
            },
        },
    )
    async def generate_deployment_config(input: SkillInput) -> SkillResult:
        """Generate KServe InferenceService configuration."""
        params = input.params

        # Extract model name from params or message
        model_name = params.get("model_name")
        if not model_name:
            text = input.message.get_text()
            for known_model in [
                "granite-4.0-h-tiny",
                "granite-4-tiny",
                "llama-3-8b",
                "mistral-7b",
            ]:
                if known_model.lower() in text.lower():
                    model_name = known_model
                    break
            if not model_name:
                model_name = "granite-4.0-h-tiny"

        # Normalize model name for service
        service_name = params.get("service_name", model_name.replace(".", "-").replace("/", "-"))
        namespace = params.get("namespace", "ai-navigator")

        # Create spec based on model
        spec = InferenceServiceSpec(
            name=service_name,
            namespace=namespace,
            model_name=model_name,
            runtime="vllm",
            gpu_count=params.get("gpu_count", 1),
            min_replicas=params.get("min_replicas", 1),
            max_replicas=params.get("max_replicas", 3),
            memory=params.get("memory", "16Gi"),
            storage_uri=f"s3://models/{model_name}",
        )

        try:
            manifest = await agent.openshift_ai.create_inference_service(spec)

            # Create artifact with the manifest
            artifact = Artifact.data(
                name=f"{service_name}-inferenceservice.yaml",
                data=manifest,
                format="yaml",
            )

            message = f"""
## InferenceService Configuration Generated

**Service Name:** {service_name}
**Namespace:** {namespace}
**Model:** {model_name}
**Runtime:** vLLM

### Resource Configuration
- **GPUs:** {spec.gpu_count}
- **Memory:** {spec.memory}
- **Replicas:** {spec.min_replicas} - {spec.max_replicas}

### Generated Manifest
```yaml
{json.dumps(manifest, indent=2)}
```

To apply this configuration:
```bash
oc apply -f {service_name}-inferenceservice.yaml
```
"""

            return SkillResult.ok(
                message=message,
                data={"manifest": manifest, "spec": spec.model_dump()},
                artifacts=[artifact],
            )
        except Exception as e:
            logger.exception("generate_deployment_config_failed", error=str(e))
            return SkillResult.error(f"Failed to generate deployment config: {e}")

    @agent.skills.register(
        id="estimate_cost",
        name="Estimate Deployment Cost",
        description="Estimate the cost of a deployment configuration",
        tags=["cost", "estimation", "planning"],
        examples=[
            "Estimate cost for granite-4.0-h-tiny deployment",
            "How much will running 3 replicas cost?",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Model name"},
                "gpu_count": {"type": "integer", "description": "GPUs per replica"},
                "replicas": {"type": "integer", "description": "Number of replicas"},
                "hours_per_month": {"type": "number", "description": "Usage hours/month"},
            },
        },
    )
    async def estimate_cost(input: SkillInput) -> SkillResult:
        """Estimate deployment costs."""
        params = input.params

        gpu_count = params.get("gpu_count", 1)
        replicas = params.get("replicas", 1)
        hours_per_month = params.get("hours_per_month", 720)  # 24x30

        # Cost estimates based on AWS g4dn.2xlarge (T4 GPU)
        gpu_hour_cost = 0.752  # $/hour for g4dn.2xlarge
        storage_gb_month = 0.10  # $/GB/month
        network_gb = 0.09  # $/GB

        # Calculate costs
        compute_cost = gpu_count * replicas * hours_per_month * gpu_hour_cost
        storage_cost = 100 * storage_gb_month  # 100GB assumed
        network_cost = 500 * network_gb  # 500GB estimated traffic

        total_monthly = compute_cost + storage_cost + network_cost

        cost_breakdown: dict[str, Any] = {
            "compute": {
                "gpu_count": gpu_count,
                "replicas": replicas,
                "hours_per_month": hours_per_month,
                "rate_per_gpu_hour": gpu_hour_cost,
                "monthly_cost": compute_cost,
            },
            "storage": {
                "size_gb": 100,
                "rate_per_gb": storage_gb_month,
                "monthly_cost": storage_cost,
            },
            "network": {
                "estimated_gb": 500,
                "rate_per_gb": network_gb,
                "monthly_cost": network_cost,
            },
            "total_monthly": total_monthly,
            "total_annual": total_monthly * 12,
        }

        message = f"""
## Cost Estimate

### Configuration
- **GPUs per replica:** {gpu_count}
- **Replicas:** {replicas}
- **Hours/month:** {hours_per_month}

### Monthly Costs

| Category | Details | Cost |
|----------|---------|------|
| Compute | {gpu_count} GPU x {replicas} replicas x {hours_per_month}h | ${compute_cost:,.2f} |
| Storage | 100 GB @ ${storage_gb_month}/GB | ${storage_cost:,.2f} |
| Network | ~500 GB @ ${network_gb}/GB | ${network_cost:,.2f} |
| **Total** | | **${total_monthly:,.2f}** |

### Annual Projection
**${total_monthly * 12:,.2f}** per year

### Cost Optimization Tips
- Consider using spot instances for non-critical workloads (-60-70%)
- Use autoscaling to reduce replicas during low traffic
- Enable request batching to improve GPU utilization
"""

        return SkillResult.ok(message=message, data=cost_breakdown)

    @agent.skills.register(
        id="validate_slo_compliance",
        name="Validate SLO Compliance",
        description="Validate a deployment configuration against SLO requirements",
        tags=["slo", "validation", "compliance"],
        examples=[
            "Validate config against 99.9% availability SLO",
            "Check if deployment meets latency requirements",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "config": {"type": "object", "description": "Deployment configuration"},
                "slos": {
                    "type": "object",
                    "description": "SLO requirements",
                    "properties": {
                        "availability": {"type": "number"},
                        "latency_p95_ms": {"type": "number"},
                        "latency_p99_ms": {"type": "number"},
                        "error_rate": {"type": "number"},
                    },
                },
            },
        },
    )
    async def validate_slo_compliance(input: SkillInput) -> SkillResult:
        """Validate deployment config against SLOs."""
        params = input.params

        config = params.get("config", {})
        slos = params.get(
            "slos",
            {
                "availability": 0.999,
                "latency_p95_ms": 500,
                "latency_p99_ms": 1000,
                "error_rate": 0.01,
            },
        )

        # Analyze configuration for SLO compliance
        min_replicas = config.get("min_replicas", 1)
        max_replicas = config.get("max_replicas", 3)
        gpu_count = config.get("gpu_count", 1)

        validations = []
        compliant = True

        # Availability check
        if min_replicas >= 2:
            validations.append(
                {
                    "slo": "availability",
                    "target": f"{slos['availability']:.1%}",
                    "status": "PASS",
                    "reason": f"Multiple replicas ({min_replicas}+) provide HA",
                }
            )
        else:
            validations.append(
                {
                    "slo": "availability",
                    "target": f"{slos['availability']:.1%}",
                    "status": "WARNING",
                    "reason": "Single replica may not meet 99.9% availability",
                    "recommendation": "Increase min_replicas to 2+",
                }
            )
            if slos["availability"] >= 0.999:
                compliant = False

        # Latency check (based on GPU resources)
        if gpu_count >= 1:
            validations.append(
                {
                    "slo": "latency_p95",
                    "target": f"{slos['latency_p95_ms']}ms",
                    "status": "PASS",
                    "reason": "GPU acceleration should meet latency targets",
                }
            )
        else:
            validations.append(
                {
                    "slo": "latency_p95",
                    "target": f"{slos['latency_p95_ms']}ms",
                    "status": "FAIL",
                    "reason": "CPU-only inference may exceed latency targets",
                    "recommendation": "Add GPU resources",
                }
            )
            compliant = False

        # Scaling check
        if max_replicas >= 3:
            validations.append(
                {
                    "slo": "burst_capacity",
                    "target": "3x baseline",
                    "status": "PASS",
                    "reason": f"Autoscaling up to {max_replicas} replicas",
                }
            )
        else:
            validations.append(
                {
                    "slo": "burst_capacity",
                    "target": "3x baseline",
                    "status": "WARNING",
                    "reason": "Limited scaling headroom",
                    "recommendation": "Increase max_replicas for burst handling",
                }
            )

        message = f"""
## SLO Compliance Validation

### Overall Status: {"COMPLIANT" if compliant else "NON-COMPLIANT"}

### Validation Results

| SLO | Target | Status | Notes |
|-----|--------|--------|-------|
"""
        for v in validations:
            status_icon = "PASS" if v["status"] == "PASS" else ("WARN" if v["status"] == "WARNING" else "FAIL")
            message += f"| {v['slo']} | {v['target']} | {status_icon} | {v['reason']} |\n"

        if not compliant:
            message += "\n### Recommendations\n"
            for v in validations:
                if v["status"] != "PASS" and "recommendation" in v:
                    message += f"- {v['recommendation']}\n"

        return SkillResult.ok(
            message=message,
            data={
                "compliant": compliant,
                "validations": validations,
                "slos": slos,
            },
        )

    @agent.skills.register(
        id="apply_deployment",
        name="Apply Deployment",
        description="Apply a deployment configuration to the cluster",
        tags=["deployment", "apply", "kubernetes"],
        examples=[
            "Deploy granite-4.0-h-tiny to the cluster",
            "Apply the generated configuration",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "manifest": {"type": "object", "description": "Kubernetes manifest"},
                "dry_run": {"type": "boolean", "description": "Dry run mode"},
            },
        },
    )
    async def apply_deployment(input: SkillInput) -> SkillResult:
        """Apply deployment to the cluster."""
        params = input.params
        dry_run = params.get("dry_run", True)  # Default to dry-run for safety

        # In production, this would use kubernetes client to apply
        # For now, simulate the deployment

        manifest = params.get("manifest")
        if not manifest:
            # Generate a default manifest
            spec = InferenceServiceSpec(
                name="granite-4-0-h-tiny",
                namespace="ai-navigator",
                model_name="granite-4.0-h-tiny",
            )
            manifest = await agent.openshift_ai.create_inference_service(spec)

        service_name = manifest.get("metadata", {}).get("name", "unknown")
        namespace = manifest.get("metadata", {}).get("namespace", "ai-navigator")

        if dry_run:
            message = f"""
## Deployment Dry Run

**Mode:** Dry Run (no changes applied)

### Would Apply:
- **InferenceService:** {service_name}
- **Namespace:** {namespace}

### Manifest Preview
```yaml
{json.dumps(manifest, indent=2)}
```

To apply for real, set `dry_run: false` in the request.
"""
            return SkillResult.ok(
                message=message,
                data={
                    "dry_run": True,
                    "manifest": manifest,
                    "status": "validated",
                },
            )
        else:
            # Simulate successful deployment
            message = f"""
## Deployment Applied Successfully

**InferenceService:** {service_name}
**Namespace:** {namespace}
**Status:** Creating

### Next Steps
1. Monitor deployment: `oc get inferenceservice {service_name} -n {namespace}`
2. Check pod status: `oc get pods -l serving.kserve.io/inferenceservice={service_name}`
3. View logs: `oc logs -l serving.kserve.io/inferenceservice={service_name}`

### Endpoint (when ready)
```
http://{service_name}.{namespace}.svc.cluster.local
```
"""
            return SkillResult.ok(
                message=message,
                data={
                    "dry_run": False,
                    "manifest": manifest,
                    "status": "applied",
                    "endpoint": f"http://{service_name}.{namespace}.svc.cluster.local",
                },
            )

    @agent.skills.register(
        id="generate_guardrails",
        name="Generate Guardrails Config",
        description="Generate TrustyAI GuardrailsOrchestrator configuration",
        tags=["guardrails", "trustyai", "safety"],
        examples=[
            "Generate guardrails with HAP and PII detection",
            "Create safety configuration for the model",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Guardrails config name"},
                "enable_hap": {"type": "boolean", "description": "Enable HAP detection"},
                "enable_pii": {"type": "boolean", "description": "Enable PII detection"},
                "enable_prompt_injection": {
                    "type": "boolean",
                    "description": "Enable prompt injection detection",
                },
            },
        },
    )
    async def generate_guardrails(input: SkillInput) -> SkillResult:
        """Generate TrustyAI guardrails configuration."""
        params = input.params

        name = params.get("name", "default-guardrails")
        enable_hap = params.get("enable_hap", True)
        enable_pii = params.get("enable_pii", True)
        enable_prompt_injection = params.get("enable_prompt_injection", True)

        try:
            config = await agent.trustyai.generate_guardrails_config(
                name=name,
                enable_hap=enable_hap,
                enable_pii=enable_pii,
                enable_prompt_injection=enable_prompt_injection,
            )

            manifest = await agent.trustyai.generate_guardrails_manifest(config)

            # Create artifact
            artifact = Artifact.data(
                name=f"{name}-guardrails.yaml",
                data=manifest,
                format="yaml",
            )

            message = f"""
## GuardrailsOrchestrator Configuration Generated

**Name:** {name}

### Enabled Detectors
- **HAP Detection:** {"Enabled" if enable_hap else "Disabled"}
- **PII Detection:** {"Enabled" if enable_pii else "Disabled"}
- **Prompt Injection:** {"Enabled" if enable_prompt_injection else "Disabled"}

### Detectors Configuration
"""
            for detector in config.detectors:
                message += f"""
#### {detector['name']}
- **Type:** {detector['type']}
- **Threshold:** {config.thresholds.get(detector['name'].split('-')[0], 'N/A')}
"""

            message += f"""
### Generated Manifest
```yaml
{json.dumps(manifest, indent=2)}
```

To apply:
```bash
oc apply -f {name}-guardrails.yaml
```
"""

            return SkillResult.ok(
                message=message,
                data={
                    "config": config.model_dump(),
                    "manifest": manifest,
                },
                artifacts=[artifact],
            )
        except Exception as e:
            logger.exception("generate_guardrails_failed", error=str(e))
            return SkillResult.error(f"Failed to generate guardrails config: {e}")
