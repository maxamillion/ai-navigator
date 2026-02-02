"""Quickstart engine for guided OpenShift AI tasks."""

from ai_navigator.quickstarts.engine import QuickstartEngine, QuickstartResult
from ai_navigator.quickstarts.tasks.deploy_model import DeployModelTask
from ai_navigator.quickstarts.tasks.serving_runtime import ServingRuntimeTask
from ai_navigator.quickstarts.tasks.test_inference import TestInferenceTask
from ai_navigator.quickstarts.tasks.project import ProjectSetupTask
from ai_navigator.quickstarts.tasks.connection import DataConnectionTask

__all__ = [
    "QuickstartEngine",
    "QuickstartResult",
    "DeployModelTask",
    "ServingRuntimeTask",
    "TestInferenceTask",
    "ProjectSetupTask",
    "DataConnectionTask",
]
