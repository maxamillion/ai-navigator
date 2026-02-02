"""Data models for AI Navigator."""

from ai_navigator.models.capacity import (
    BenchmarkData,
    CapacityPlan,
    GPURecommendation,
    WhatIfScenario,
    WhatIfResult,
)
from ai_navigator.models.deployment import (
    DeploymentConfig,
    DeploymentResult,
    DeploymentStatus,
    InferenceServiceSpec,
    RuntimeConfig,
)
from ai_navigator.models.workflow import (
    Message,
    MessageRole,
    ModelRequirements,
    SLORequirements,
    TrafficProfile,
    TrafficPattern,
    WorkflowStage,
    WorkflowState,
)

__all__ = [
    # Workflow models
    "Message",
    "MessageRole",
    "ModelRequirements",
    "SLORequirements",
    "TrafficProfile",
    "TrafficPattern",
    "WorkflowStage",
    "WorkflowState",
    # Capacity models
    "BenchmarkData",
    "CapacityPlan",
    "GPURecommendation",
    "WhatIfScenario",
    "WhatIfResult",
    # Deployment models
    "DeploymentConfig",
    "DeploymentResult",
    "DeploymentStatus",
    "InferenceServiceSpec",
    "RuntimeConfig",
]
