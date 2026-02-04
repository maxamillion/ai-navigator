"""Kopf event handlers for Agent CRD."""

from typing import Any

import kopf
import structlog

from k8s_operator.reconciler import AgentReconciler

logger = structlog.get_logger(__name__)

# Global reconciler instance
_reconciler: AgentReconciler | None = None


def get_reconciler() -> AgentReconciler:
    """Get or create the reconciler instance."""
    global _reconciler
    if _reconciler is None:
        _reconciler = AgentReconciler()
    return _reconciler


@kopf.on.create("ai-navigator.redhat.com", "v1alpha1", "agents")
async def agent_create(
    body: kopf.Body,
    name: str,
    namespace: str,
    spec: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """
    Handle Agent CR creation.

    Creates:
    - Deployment for the agent
    - Service for network access
    - Route for external access (optional)
    """
    logger.info("agent_create", name=name, namespace=namespace)

    reconciler = get_reconciler()

    try:
        result = await reconciler.reconcile_create(
            name=name,
            namespace=namespace,
            spec=spec,
        )

        # Return status to be patched
        return {
            "phase": "Ready" if result.success else "Failed",
            "message": result.message,
            "url": result.url,
            "agentCard": result.agent_card,
        }

    except Exception as e:
        logger.exception("agent_create_failed", name=name, error=str(e))
        return {
            "phase": "Failed",
            "message": str(e),
        }


@kopf.on.update("ai-navigator.redhat.com", "v1alpha1", "agents")
async def agent_update(
    body: kopf.Body,
    name: str,
    namespace: str,
    spec: dict[str, Any],
    old: dict[str, Any],
    new: dict[str, Any],
    diff: kopf.Diff,
    **kwargs,
) -> dict[str, Any]:
    """
    Handle Agent CR updates.

    Updates the deployment configuration when spec changes.
    """
    logger.info("agent_update", name=name, namespace=namespace, diff=list(diff))

    reconciler = get_reconciler()

    try:
        result = await reconciler.reconcile_update(
            name=name,
            namespace=namespace,
            spec=spec,
            old_spec=old.get("spec", {}),
        )

        return {
            "phase": "Ready" if result.success else "Failed",
            "message": result.message,
            "url": result.url,
            "agentCard": result.agent_card,
        }

    except Exception as e:
        logger.exception("agent_update_failed", name=name, error=str(e))
        return {
            "phase": "Failed",
            "message": str(e),
        }


@kopf.on.delete("ai-navigator.redhat.com", "v1alpha1", "agents")
async def agent_delete(
    body: kopf.Body,
    name: str,
    namespace: str,
    **kwargs,
) -> None:
    """
    Handle Agent CR deletion.

    Cleans up:
    - Deployment
    - Service
    - Route (if exists)
    """
    logger.info("agent_delete", name=name, namespace=namespace)

    reconciler = get_reconciler()

    try:
        await reconciler.reconcile_delete(
            name=name,
            namespace=namespace,
        )
        logger.info("agent_deleted", name=name, namespace=namespace)

    except Exception as e:
        logger.exception("agent_delete_failed", name=name, error=str(e))
        raise


@kopf.on.timer("ai-navigator.redhat.com", "v1alpha1", "agents", interval=30.0)
async def health_check_timer(
    body: kopf.Body,
    name: str,
    namespace: str,
    status: dict[str, Any],
    **kwargs,
) -> dict[str, Any] | None:
    """
    Periodic health check for agents.

    Checks:
    - Agent endpoint health
    - Updates status.lastSeen
    - Updates agent card if changed
    """
    url = status.get("url")
    if not url:
        return None

    reconciler = get_reconciler()

    try:
        health_result = await reconciler.check_agent_health(
            name=name,
            namespace=namespace,
            url=url,
        )

        if health_result.healthy:
            return {
                "phase": "Ready",
                "lastSeen": health_result.timestamp,
                "agentCard": health_result.agent_card,
            }
        else:
            return {
                "phase": "Unhealthy",
                "message": health_result.message,
                "lastSeen": health_result.timestamp,
            }

    except Exception as e:
        logger.warning("health_check_failed", name=name, error=str(e))
        return {
            "phase": "Unknown",
            "message": f"Health check failed: {e}",
        }
