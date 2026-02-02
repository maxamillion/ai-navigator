"""YAML generation for KServe/vLLM deployment manifests."""

from pathlib import Path
from typing import Any, Optional

import yaml
from jinja2 import Environment, FileSystemLoader, PackageLoader

import structlog

from ai_navigator.models.deployment import DeploymentConfig, InferenceServiceSpec

logger = structlog.get_logger(__name__)


class YAMLGenerator:
    """Generates Kubernetes YAML manifests from deployment configuration."""

    def __init__(self, templates_dir: Optional[Path] = None) -> None:
        """Initialize generator with templates directory."""
        if templates_dir and templates_dir.exists():
            self._env = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            # Use inline templates as fallback
            self._env = None

    def generate_all(self, config: DeploymentConfig) -> dict[str, str]:
        """Generate all manifests for a deployment."""
        manifests: dict[str, str] = {}

        # Generate InferenceService
        manifests["inferenceservice.yaml"] = self.generate_inference_service(
            config.inference_service
        )

        # Generate HPA if requested
        if config.create_hpa:
            manifests["hpa.yaml"] = self.generate_hpa(config.inference_service)

        # Generate PDB if requested
        if config.create_pdb:
            manifests["pdb.yaml"] = self.generate_pdb(config.inference_service)

        # Store manifests in config
        config.manifests = manifests

        return manifests

    def generate_inference_service(self, spec: InferenceServiceSpec) -> str:
        """Generate KServe InferenceService manifest."""
        manifest = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": spec.name,
                "namespace": spec.namespace,
                "labels": {
                    "app.kubernetes.io/name": spec.name,
                    "app.kubernetes.io/part-of": "ai-navigator",
                    **spec.labels,
                },
                "annotations": {
                    "serving.kserve.io/autoscalerClass": "hpa",
                    "serving.kserve.io/deploymentMode": "RawDeployment",
                    **spec.annotations,
                },
            },
            "spec": {
                "predictor": {
                    "minReplicas": spec.min_replicas,
                    "maxReplicas": spec.max_replicas,
                    "scaleTarget": spec.scale_target,
                    "scaleMetric": spec.scale_metric,
                    "timeout": spec.timeout_seconds,
                    "model": self._build_model_spec(spec),
                },
            },
        }

        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)

    def generate_hpa(self, spec: InferenceServiceSpec) -> str:
        """Generate HorizontalPodAutoscaler manifest."""
        manifest = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{spec.name}-hpa",
                "namespace": spec.namespace,
                "labels": {
                    "app.kubernetes.io/name": spec.name,
                    "app.kubernetes.io/part-of": "ai-navigator",
                },
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{spec.name}-predictor",
                },
                "minReplicas": spec.min_replicas,
                "maxReplicas": spec.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80,
                            },
                        },
                    },
                ],
                "behavior": {
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 10,
                                "periodSeconds": 60,
                            },
                        ],
                    },
                    "scaleUp": {
                        "stabilizationWindowSeconds": 60,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 100,
                                "periodSeconds": 15,
                            },
                            {
                                "type": "Pods",
                                "value": 4,
                                "periodSeconds": 15,
                            },
                        ],
                        "selectPolicy": "Max",
                    },
                },
            },
        }

        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)

    def generate_pdb(self, spec: InferenceServiceSpec) -> str:
        """Generate PodDisruptionBudget manifest."""
        manifest = {
            "apiVersion": "policy/v1",
            "kind": "PodDisruptionBudget",
            "metadata": {
                "name": f"{spec.name}-pdb",
                "namespace": spec.namespace,
                "labels": {
                    "app.kubernetes.io/name": spec.name,
                    "app.kubernetes.io/part-of": "ai-navigator",
                },
            },
            "spec": {
                "minAvailable": 1,
                "selector": {
                    "matchLabels": {
                        "serving.kserve.io/inferenceservice": spec.name,
                    },
                },
            },
        }

        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)

    def generate_service_account(
        self,
        namespace: str,
        name: str,
        pull_secret: Optional[str] = None,
    ) -> str:
        """Generate ServiceAccount manifest."""
        manifest: dict[str, Any] = {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/part-of": "ai-navigator",
                },
            },
        }

        if pull_secret:
            manifest["imagePullSecrets"] = [{"name": pull_secret}]

        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)

    def _build_model_spec(self, spec: InferenceServiceSpec) -> dict[str, Any]:
        """Build the model specification for InferenceService."""
        model_spec: dict[str, Any] = {
            "modelFormat": {"name": spec.model_format},
            "resources": {
                "requests": {
                    "cpu": spec.cpu,
                    "memory": spec.memory,
                },
                "limits": {
                    "cpu": str(int(spec.cpu) * 2),
                    "memory": spec.memory,
                },
            },
        }

        # Add GPU resources
        if spec.gpu_count > 0:
            model_spec["resources"]["requests"]["nvidia.com/gpu"] = str(spec.gpu_count)
            model_spec["resources"]["limits"]["nvidia.com/gpu"] = str(spec.gpu_count)

        # Add storage URI if specified
        if spec.storage_uri:
            model_spec["storageUri"] = spec.storage_uri

        # Add runtime configuration
        if spec.runtime:
            runtime = spec.runtime
            model_spec["runtime"] = runtime.runtime_name

            # Build args for vLLM
            if runtime.runtime_name == "vllm":
                args = [
                    f"--tensor-parallel-size={runtime.tensor_parallel_size}",
                    f"--gpu-memory-utilization={runtime.gpu_memory_utilization}",
                    f"--dtype={runtime.dtype}",
                ]

                if runtime.max_model_len:
                    args.append(f"--max-model-len={runtime.max_model_len}")

                if runtime.enforce_eager:
                    args.append("--enforce-eager")

                args.extend(runtime.extra_args)

                model_spec["args"] = args

            # Add environment variables
            if runtime.env_vars:
                model_spec["env"] = [
                    {"name": k, "value": v} for k, v in runtime.env_vars.items()
                ]

        return model_spec

    def generate_serving_runtime(
        self,
        name: str = "vllm-runtime",
        namespace: str = "default",
        image: str = "quay.io/modh/vllm:0.4.0",
        gpu_type: Optional[str] = None,
    ) -> str:
        """Generate ServingRuntime manifest for vLLM."""
        manifest: dict[str, Any] = {
            "apiVersion": "serving.kserve.io/v1alpha1",
            "kind": "ServingRuntime",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/part-of": "ai-navigator",
                },
            },
            "spec": {
                "annotations": {
                    "prometheus.io/path": "/metrics",
                    "prometheus.io/port": "8080",
                },
                "containers": [
                    {
                        "name": "kserve-container",
                        "image": image,
                        "command": ["python", "-m", "vllm.entrypoints.openai.api_server"],
                        "args": [
                            "--model=$(MODEL_NAME)",
                            "--port=8080",
                        ],
                        "env": [
                            {"name": "MODEL_NAME", "value": ""},
                        ],
                        "ports": [
                            {"containerPort": 8080, "protocol": "TCP"},
                        ],
                        "resources": {
                            "requests": {
                                "cpu": "4",
                                "memory": "16Gi",
                            },
                            "limits": {
                                "cpu": "8",
                                "memory": "32Gi",
                            },
                        },
                    },
                ],
                "multiModel": False,
                "supportedModelFormats": [
                    {
                        "name": "pytorch",
                        "version": "1",
                        "autoSelect": True,
                    },
                ],
            },
        }

        # Add GPU node selector if specified
        if gpu_type:
            manifest["spec"]["nodeSelector"] = {
                "nvidia.com/gpu.product": gpu_type,
            }

        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)
