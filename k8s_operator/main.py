"""Kopf operator entry point for AI Navigator Agent CRD."""

import kopf
import structlog

from k8s_operator.handlers import (
    agent_create,
    agent_delete,
    agent_update,
    health_check_timer,
)

logger = structlog.get_logger(__name__)


# Configure kopf settings
@kopf.on.startup()
async def configure(settings: kopf.OperatorSettings, **kwargs):
    """Configure operator settings on startup."""
    settings.posting.enabled = True
    settings.watching.server_timeout = 300
    settings.persistence.finalizer = "ai-navigator.redhat.com/finalizer"

    logger.info("operator_starting", version="0.1.0")


@kopf.on.cleanup()
async def cleanup(**kwargs):
    """Cleanup on operator shutdown."""
    logger.info("operator_stopping")


# Register handlers
kopf.on.create("ai-navigator.redhat.com", "v1alpha1", "agents")(agent_create)
kopf.on.update("ai-navigator.redhat.com", "v1alpha1", "agents")(agent_update)
kopf.on.delete("ai-navigator.redhat.com", "v1alpha1", "agents")(agent_delete)
kopf.on.timer("ai-navigator.redhat.com", "v1alpha1", "agents", interval=30.0)(health_check_timer)


if __name__ == "__main__":
    kopf.run()
