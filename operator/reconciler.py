"""Reconciliation logic for Agent CRD."""

from datetime import datetime, timezone
from typing import Any

import httpx
import structlog
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ReconcileResult(BaseModel):
    """Result of a reconciliation operation."""

    success: bool = Field(..., description="Whether reconciliation succeeded")
    message: str = Field(default="", description="Status message")
    url: str | None = Field(default=None, description="Agent endpoint URL")
    agent_card: dict[str, Any] | None = Field(default=None, description="Agent card data")


class HealthCheckResult(BaseModel):
    """Result of a health check."""

    healthy: bool = Field(..., description="Whether agent is healthy")
    message: str = Field(default="", description="Health status message")
    timestamp: str = Field(default="", description="Check timestamp")
    agent_card: dict[str, Any] | None = Field(default=None, description="Agent card if available")


class AgentReconciler:
    """
    Reconciles Agent CRs with Kubernetes resources.

    Creates and manages:
    - Deployments for agent pods
    - Services for network access
    - Routes for external access (OpenShift)
    """

    def __init__(self) -> None:
        """Initialize the reconciler."""
        self._k8s_configured = False
        self._apps_v1: client.AppsV1Api | None = None
        self._core_v1: client.CoreV1Api | None = None

    def _configure_k8s(self) -> None:
        """Configure Kubernetes client."""
        if self._k8s_configured:
            return

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        self._apps_v1 = client.AppsV1Api()
        self._core_v1 = client.CoreV1Api()
        self._k8s_configured = True

    @property
    def apps_v1(self) -> client.AppsV1Api:
        """Get Apps V1 API client."""
        self._configure_k8s()
        assert self._apps_v1 is not None
        return self._apps_v1

    @property
    def core_v1(self) -> client.CoreV1Api:
        """Get Core V1 API client."""
        self._configure_k8s()
        assert self._core_v1 is not None
        return self._core_v1

    async def reconcile_create(
        self,
        name: str,
        namespace: str,
        spec: dict[str, Any],
    ) -> ReconcileResult:
        """
        Reconcile Agent CR creation.

        Args:
            name: Agent name
            namespace: Kubernetes namespace
            spec: Agent spec from CR

        Returns:
            ReconcileResult with status
        """
        logger.info("reconcile_create", name=name, namespace=namespace)

        try:
            # Build resources
            deployment = self._build_deployment(name, namespace, spec)
            service = self._build_service(name, namespace, spec)

            # Create deployment
            try:
                self.apps_v1.create_namespaced_deployment(
                    namespace=namespace,
                    body=deployment,
                )
                logger.info("deployment_created", name=name)
            except ApiException as e:
                if e.status == 409:  # Already exists
                    self.apps_v1.patch_namespaced_deployment(
                        name=name,
                        namespace=namespace,
                        body=deployment,
                    )
                    logger.info("deployment_updated", name=name)
                else:
                    raise

            # Create service
            try:
                self.core_v1.create_namespaced_service(
                    namespace=namespace,
                    body=service,
                )
                logger.info("service_created", name=name)
            except ApiException as e:
                if e.status == 409:  # Already exists
                    logger.info("service_exists", name=name)
                else:
                    raise

            # Build URL
            port = spec.get("port", 8000)
            url = f"http://{name}.{namespace}.svc.cluster.local:{port}"

            return ReconcileResult(
                success=True,
                message="Agent deployed successfully",
                url=url,
            )

        except Exception as e:
            logger.exception("reconcile_create_failed", name=name, error=str(e))
            return ReconcileResult(
                success=False,
                message=f"Failed to create agent: {e}",
            )

    async def reconcile_update(
        self,
        name: str,
        namespace: str,
        spec: dict[str, Any],
        old_spec: dict[str, Any],
    ) -> ReconcileResult:
        """
        Reconcile Agent CR update.

        Args:
            name: Agent name
            namespace: Kubernetes namespace
            spec: New agent spec
            old_spec: Previous agent spec

        Returns:
            ReconcileResult with status
        """
        logger.info("reconcile_update", name=name, namespace=namespace)

        try:
            # Rebuild and patch deployment
            deployment = self._build_deployment(name, namespace, spec)

            self.apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=namespace,
                body=deployment,
            )
            logger.info("deployment_patched", name=name)

            # Build URL
            port = spec.get("port", 8000)
            url = f"http://{name}.{namespace}.svc.cluster.local:{port}"

            return ReconcileResult(
                success=True,
                message="Agent updated successfully",
                url=url,
            )

        except Exception as e:
            logger.exception("reconcile_update_failed", name=name, error=str(e))
            return ReconcileResult(
                success=False,
                message=f"Failed to update agent: {e}",
            )

    async def reconcile_delete(
        self,
        name: str,
        namespace: str,
    ) -> None:
        """
        Reconcile Agent CR deletion.

        Args:
            name: Agent name
            namespace: Kubernetes namespace
        """
        logger.info("reconcile_delete", name=name, namespace=namespace)

        # Delete deployment
        try:
            self.apps_v1.delete_namespaced_deployment(
                name=name,
                namespace=namespace,
            )
            logger.info("deployment_deleted", name=name)
        except ApiException as e:
            if e.status != 404:
                raise

        # Delete service
        try:
            self.core_v1.delete_namespaced_service(
                name=name,
                namespace=namespace,
            )
            logger.info("service_deleted", name=name)
        except ApiException as e:
            if e.status != 404:
                raise

    async def check_agent_health(
        self,
        name: str,
        namespace: str,
        url: str,
    ) -> HealthCheckResult:
        """
        Check health of an agent.

        Args:
            name: Agent name
            namespace: Kubernetes namespace
            url: Agent endpoint URL

        Returns:
            HealthCheckResult with status
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check health endpoint
                health_response = await client.get(f"{url}/healthz")
                healthy = health_response.status_code == 200

                # Get agent card
                agent_card = None
                try:
                    card_response = await client.get(f"{url}/.well-known/agent.json")
                    if card_response.status_code == 200:
                        agent_card = card_response.json()
                except Exception:
                    pass

                return HealthCheckResult(
                    healthy=healthy,
                    message="Agent is healthy" if healthy else "Health check failed",
                    timestamp=timestamp,
                    agent_card=agent_card,
                )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Health check error: {e}",
                timestamp=timestamp,
            )

    def _build_deployment(
        self,
        name: str,
        namespace: str,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        """Build Kubernetes Deployment manifest."""
        agent_class = spec.get("class", "supervisor")
        image = spec.get("image", "quay.io/redhat-et/ai-navigator:latest")
        port = spec.get("port", 8000)
        replicas = spec.get("replicas", 1)
        resources = spec.get("resources", {})

        # Default resource requests/limits
        resource_requests = resources.get("requests", {})
        resource_limits = resources.get("limits", {})

        if not resource_requests:
            resource_requests = {"memory": "512Mi", "cpu": "500m"}
        if not resource_limits:
            resource_limits = {"memory": "1Gi", "cpu": "1000m"}

        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": name,
                    "app.kubernetes.io/part-of": "ai-navigator",
                    "app.kubernetes.io/component": agent_class,
                },
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": name,
                    },
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app.kubernetes.io/name": name,
                            "app.kubernetes.io/part-of": "ai-navigator",
                            "app.kubernetes.io/component": agent_class,
                        },
                    },
                    "spec": {
                        "serviceAccountName": "ai-navigator",
                        "containers": [
                            {
                                "name": "agent",
                                "image": image,
                                "ports": [
                                    {
                                        "name": "http",
                                        "containerPort": port,
                                        "protocol": "TCP",
                                    },
                                ],
                                "env": [
                                    {"name": "AGENT_NAME", "value": name},
                                    {"name": "AGENT_PORT", "value": str(port)},
                                    {
                                        "name": "KUBERNETES_NAMESPACE",
                                        "valueFrom": {
                                            "fieldRef": {"fieldPath": "metadata.namespace"}
                                        },
                                    },
                                ],
                                "resources": {
                                    "requests": resource_requests,
                                    "limits": resource_limits,
                                },
                                "livenessProbe": {
                                    "httpGet": {"path": "/healthz", "port": port},
                                    "initialDelaySeconds": 10,
                                    "periodSeconds": 10,
                                },
                                "readinessProbe": {
                                    "httpGet": {"path": "/readyz", "port": port},
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5,
                                },
                            },
                        ],
                    },
                },
            },
        }

    def _build_service(
        self,
        name: str,
        namespace: str,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        """Build Kubernetes Service manifest."""
        port = spec.get("port", 8000)

        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": name,
                    "app.kubernetes.io/part-of": "ai-navigator",
                },
            },
            "spec": {
                "selector": {
                    "app.kubernetes.io/name": name,
                },
                "ports": [
                    {
                        "name": "http",
                        "port": port,
                        "targetPort": port,
                        "protocol": "TCP",
                    },
                ],
            },
        }
