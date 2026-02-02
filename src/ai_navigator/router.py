"""FastAPI router with all AI Navigator API endpoints."""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

import structlog

from ai_navigator.config import get_settings, Settings
from ai_navigator.models.workflow import SLORequirements, TrafficProfile
from ai_navigator.models.capacity import CapacityPlan, WhatIfScenario, WhatIfResult
from ai_navigator.models.deployment import DeploymentConfig, DeploymentResult

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/ai-navigator", tags=["AI Navigator"])


# Request/Response models
class WorkflowStartRequest(BaseModel):
    """Request to start a new workflow."""

    user_id: str = Field(description="User identifier")
    initial_message: str = Field(description="Initial user message")


class WorkflowStartResponse(BaseModel):
    """Response from starting a workflow."""

    workflow_id: str
    stage: str
    response: str


class WorkflowContinueRequest(BaseModel):
    """Request to continue a workflow."""

    message: str = Field(description="User's message")


class WorkflowContinueResponse(BaseModel):
    """Response from continuing a workflow."""

    workflow_id: str
    stage: str
    is_complete: bool
    response: str


class WorkflowStatusResponse(BaseModel):
    """Workflow status response."""

    id: str
    user_id: str
    stage: str
    stage_index: int
    total_stages: int
    is_complete: bool
    message_count: int


class QuickstartStartRequest(BaseModel):
    """Request to start a quickstart task."""

    session_id: str = Field(description="Session identifier")
    context: Optional[dict[str, Any]] = Field(default=None)


class QuickstartContinueRequest(BaseModel):
    """Request to continue a quickstart."""

    message: str = Field(description="User input")


class QuickstartResponse(BaseModel):
    """Response from quickstart execution."""

    task_name: str
    status: str
    message: str
    data: Optional[dict[str, Any]] = None
    next_prompt: Optional[str] = None


class CapacityEstimateRequest(BaseModel):
    """Request for capacity estimation."""

    model_name: str
    traffic_profile: TrafficProfile
    slo_requirements: SLORequirements


class CapacityEstimateResponse(BaseModel):
    """Capacity estimation response."""

    plan: CapacityPlan


class WhatIfRequest(BaseModel):
    """Request for what-if analysis."""

    plan_id: str
    scenarios: list[WhatIfScenario]


class WhatIfResponse(BaseModel):
    """What-if analysis response."""

    results: list[WhatIfResult]


class DeployPreviewRequest(BaseModel):
    """Request for deployment preview."""

    config: DeploymentConfig


class DeployPreviewResponse(BaseModel):
    """Deployment preview response."""

    manifests: dict[str, str]
    validation_passed: bool
    validation_errors: list[str]
    validation_warnings: list[str]


class DeployExecuteRequest(BaseModel):
    """Request to execute deployment."""

    config_id: str
    dry_run: bool = Field(default=False)


class DeployExecuteResponse(BaseModel):
    """Deployment execution response."""

    result: DeploymentResult


# Dependency injection
async def get_workflow_engine():
    """Get workflow engine instance."""
    from ai_navigator.workflow.engine import WorkflowEngine
    from ai_navigator.state.memory import InMemoryStateStore
    from ai_navigator.state.manager import StateManager

    store = InMemoryStateStore()
    state_manager = StateManager(store)
    return WorkflowEngine(state_manager)


async def get_quickstart_engine():
    """Get quickstart engine instance."""
    from ai_navigator.quickstarts.engine import QuickstartEngine
    from ai_navigator.quickstarts.tasks import (
        DeployModelTask,
        ServingRuntimeTask,
        TestInferenceTask,
        ProjectSetupTask,
        DataConnectionTask,
    )

    engine = QuickstartEngine()
    engine.register_task(DeployModelTask())
    engine.register_task(ServingRuntimeTask())
    engine.register_task(TestInferenceTask())
    engine.register_task(ProjectSetupTask())
    engine.register_task(DataConnectionTask())
    return engine


async def get_capacity_planner():
    """Get capacity planner instance."""
    from ai_navigator.planning.capacity import CapacityPlanner

    return CapacityPlanner()


# Health check
@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-navigator"}


# Workflow endpoints
@router.post("/workflow/start", response_model=WorkflowStartResponse)
async def start_workflow(
    request: WorkflowStartRequest,
    engine=Depends(get_workflow_engine),
) -> WorkflowStartResponse:
    """Start a new workflow conversation."""
    try:
        state, response = await engine.start_workflow(
            user_id=request.user_id,
            initial_message=request.initial_message,
        )
        return WorkflowStartResponse(
            workflow_id=state.id,
            stage=state.stage.value,
            response=response,
        )
    except Exception as e:
        logger.error("Failed to start workflow", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/{workflow_id}/continue", response_model=WorkflowContinueResponse)
async def continue_workflow(
    workflow_id: str,
    request: WorkflowContinueRequest,
    engine=Depends(get_workflow_engine),
) -> WorkflowContinueResponse:
    """Continue an existing workflow."""
    try:
        state, response = await engine.continue_workflow(
            workflow_id=workflow_id,
            user_message=request.message,
        )
        return WorkflowContinueResponse(
            workflow_id=state.id,
            stage=state.stage.value,
            is_complete=state.is_complete(),
            response=response,
        )
    except Exception as e:
        logger.error("Failed to continue workflow", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    workflow_id: str,
    engine=Depends(get_workflow_engine),
) -> WorkflowStatusResponse:
    """Get workflow status."""
    status = await engine.get_workflow_status(workflow_id)
    if not status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return WorkflowStatusResponse(**status)


# Quickstart endpoints
@router.get("/quickstarts")
async def list_quickstarts(
    engine=Depends(get_quickstart_engine),
) -> list[dict[str, str]]:
    """List available quickstart tasks."""
    return engine.list_tasks()


@router.post("/quickstart/{task_name}", response_model=QuickstartResponse)
async def start_quickstart(
    task_name: str,
    request: QuickstartStartRequest,
    engine=Depends(get_quickstart_engine),
) -> QuickstartResponse:
    """Start a quickstart task."""
    result = await engine.start_task(
        task_name=task_name,
        session_id=request.session_id,
        initial_context=request.context,
    )
    return QuickstartResponse(
        task_name=result.task_name,
        status=result.status.value,
        message=result.message,
        data=result.data,
        next_prompt=result.next_prompt,
    )


@router.post("/quickstart/{task_name}/continue", response_model=QuickstartResponse)
async def continue_quickstart(
    task_name: str,
    request: QuickstartContinueRequest,
    engine=Depends(get_quickstart_engine),
) -> QuickstartResponse:
    """Continue a quickstart task."""
    # Note: In a real implementation, session_id would come from request
    session_id = "default"
    result = await engine.continue_task(
        session_id=session_id,
        user_input=request.message,
    )
    return QuickstartResponse(
        task_name=result.task_name,
        status=result.status.value,
        message=result.message,
        data=result.data,
        next_prompt=result.next_prompt,
    )


# Capacity planning endpoints
@router.post("/capacity/estimate", response_model=CapacityEstimateResponse)
async def estimate_capacity(
    request: CapacityEstimateRequest,
    planner=Depends(get_capacity_planner),
) -> CapacityEstimateResponse:
    """Get capacity estimate for model deployment."""
    from ai_navigator.models.capacity import BenchmarkData

    # Use default benchmark for estimation
    benchmark = BenchmarkData(
        model_name=request.model_name,
        model_version="1.0",
        gpu_type="A100-40GB",
        gpu_count=1,
        p50_latency_ms=150,
        p95_latency_ms=250,
        p99_latency_ms=400,
        tokens_per_second=100,
        requests_per_second=10,
        gpu_memory_gb=16,
        gpu_utilization_percent=80,
    )

    plan = planner.calculate_capacity_plan(
        benchmark=benchmark,
        traffic_profile=request.traffic_profile,
        slo_requirements=request.slo_requirements,
    )

    return CapacityEstimateResponse(plan=plan)


@router.post("/capacity/whatif", response_model=WhatIfResponse)
async def analyze_whatif(
    request: WhatIfRequest,
) -> WhatIfResponse:
    """Run what-if analysis on capacity plan."""
    from ai_navigator.planning.whatif import WhatIfAnalyzer
    from ai_navigator.planning.capacity import CapacityPlanner
    from ai_navigator.models.capacity import BenchmarkData
    from ai_navigator.models.workflow import TrafficProfile, SLORequirements

    analyzer = WhatIfAnalyzer()

    # Create placeholder data for demo
    benchmark = BenchmarkData(
        model_name="demo-model",
        model_version="1.0",
        gpu_type="A100-40GB",
        gpu_count=1,
        p50_latency_ms=150,
        p95_latency_ms=250,
        p99_latency_ms=400,
        tokens_per_second=100,
        requests_per_second=10,
        gpu_memory_gb=16,
        gpu_utilization_percent=80,
    )

    traffic = TrafficProfile(requests_per_second=10)
    slo = SLORequirements(p50_latency_ms=500, p95_latency_ms=1000, p99_latency_ms=2000)

    planner = CapacityPlanner()
    original_plan = planner.calculate_capacity_plan(benchmark, traffic, slo)

    results = []
    for scenario in request.scenarios:
        result = analyzer.analyze_scenario(
            original_plan=original_plan,
            scenario=scenario,
            benchmark=benchmark,
            traffic_profile=traffic,
            slo_requirements=slo,
        )
        results.append(result)

    return WhatIfResponse(results=results)


# Deployment endpoints
@router.post("/deploy/preview", response_model=DeployPreviewResponse)
async def preview_deployment(
    request: DeployPreviewRequest,
) -> DeployPreviewResponse:
    """Preview deployment manifests."""
    from ai_navigator.deployment.generator import YAMLGenerator
    from ai_navigator.deployment.validators import DeploymentValidator

    generator = YAMLGenerator()
    validator = DeploymentValidator()

    # Validate configuration
    validation = await validator.validate(request.config)

    # Generate manifests
    manifests = generator.generate_all(request.config)

    return DeployPreviewResponse(
        manifests=manifests,
        validation_passed=validation.passed,
        validation_errors=validation.errors,
        validation_warnings=validation.warnings,
    )


@router.post("/deploy/execute", response_model=DeployExecuteResponse)
async def execute_deployment(
    request: DeployExecuteRequest,
) -> DeployExecuteResponse:
    """Execute deployment."""
    from ai_navigator.deployment.orchestrator import DeploymentOrchestrator
    from ai_navigator.models.deployment import DeploymentConfig, DeploymentStatus

    # In a real implementation, we'd retrieve the config from state
    # For now, return a simulated result
    result = DeploymentResult(
        deployment_id=request.config_id,
        status=DeploymentStatus.RUNNING if not request.dry_run else DeploymentStatus.PENDING,
        message="Deployment simulated" if not request.dry_run else "Dry run completed",
        endpoint_url="https://model-service-default.apps.cluster.example.com",
        ready_replicas=2,
        total_replicas=2,
    )

    return DeployExecuteResponse(result=result)


# Plugin registration for OLS
def register_plugin(app) -> None:
    """Register AI Navigator plugin with FastAPI app."""
    app.include_router(router)
    logger.info("AI Navigator plugin registered")
