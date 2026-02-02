"""Workflow engine state machine orchestrator."""

from typing import Any, Optional

import structlog

from ai_navigator.models.workflow import (
    Message,
    MessageRole,
    WorkflowStage,
    WorkflowState,
)
from ai_navigator.state.manager import StateManager
from ai_navigator.workflow.stages.intent import IntentStage
from ai_navigator.workflow.stages.traffic import TrafficStage
from ai_navigator.workflow.stages.slo import SLOStage
from ai_navigator.workflow.stages.benchmark import BenchmarkStage
from ai_navigator.workflow.stages.filter import FilterStage
from ai_navigator.workflow.stages.replicas import ReplicasStage
from ai_navigator.workflow.stages.deploy import DeployStage
from ai_navigator.workflow.stages.monitor import MonitorStage

logger = structlog.get_logger(__name__)


class WorkflowError(Exception):
    """Base workflow exception."""

    pass


class StageExecutionError(WorkflowError):
    """Stage execution failed."""

    def __init__(self, message: str, stage: WorkflowStage, details: Optional[dict] = None):
        super().__init__(message)
        self.stage = stage
        self.details = details or {}


class StageResult:
    """Result of a stage execution."""

    def __init__(
        self,
        success: bool,
        message: str,
        advance: bool = False,
        data: Optional[dict[str, Any]] = None,
        prompt_user: Optional[str] = None,
    ):
        self.success = success
        self.message = message
        self.advance = advance  # Whether to advance to next stage
        self.data = data or {}
        self.prompt_user = prompt_user  # Question to ask user


class WorkflowEngine:
    """Orchestrates the 8-step workflow state machine."""

    def __init__(
        self,
        state_manager: StateManager,
        intent_stage: Optional[IntentStage] = None,
        traffic_stage: Optional[TrafficStage] = None,
        slo_stage: Optional[SLOStage] = None,
        benchmark_stage: Optional[BenchmarkStage] = None,
        filter_stage: Optional[FilterStage] = None,
        replicas_stage: Optional[ReplicasStage] = None,
        deploy_stage: Optional[DeployStage] = None,
        monitor_stage: Optional[MonitorStage] = None,
    ) -> None:
        """Initialize workflow engine with stage handlers."""
        self._state_manager = state_manager

        # Initialize stage handlers
        self._stages: dict[WorkflowStage, Any] = {
            WorkflowStage.INTENT: intent_stage or IntentStage(),
            WorkflowStage.TRAFFIC: traffic_stage or TrafficStage(),
            WorkflowStage.SLO: slo_stage or SLOStage(),
            WorkflowStage.BENCHMARK: benchmark_stage or BenchmarkStage(),
            WorkflowStage.FILTER: filter_stage or FilterStage(),
            WorkflowStage.REPLICAS: replicas_stage or ReplicasStage(),
            WorkflowStage.DEPLOY: deploy_stage or DeployStage(),
            WorkflowStage.MONITOR: monitor_stage or MonitorStage(),
        }

    async def start_workflow(self, user_id: str, initial_message: str) -> tuple[WorkflowState, str]:
        """Start a new workflow for a user."""
        # Create new workflow state
        state = await self._state_manager.create_workflow(user_id)

        # Add initial user message
        state.add_message(MessageRole.USER, initial_message)

        # Process the initial message
        result = await self._process_stage(state, initial_message)

        # Add assistant response
        state.add_message(MessageRole.ASSISTANT, result.message)

        # Save updated state
        await self._state_manager.save_workflow(state)

        logger.info(
            "Workflow started",
            workflow_id=state.id,
            user_id=user_id,
            stage=state.stage.value,
        )

        response = result.message
        if result.prompt_user:
            response = f"{result.message}\n\n{result.prompt_user}"

        return state, response

    async def continue_workflow(
        self,
        workflow_id: str,
        user_message: str,
    ) -> tuple[WorkflowState, str]:
        """Continue an existing workflow with user input."""
        state = await self._state_manager.get_workflow(workflow_id)
        if not state:
            raise WorkflowError(f"Workflow not found: {workflow_id}")

        # Add user message
        state.add_message(MessageRole.USER, user_message)

        # Process current stage
        result = await self._process_stage(state, user_message)

        # Handle stage advancement
        if result.advance:
            next_stage = state.advance_stage()
            if next_stage:
                logger.info(
                    "Advancing to next stage",
                    workflow_id=workflow_id,
                    from_stage=state.stage.value,
                    to_stage=next_stage.value,
                )
                # Auto-process next stage if it doesn't require user input
                if self._is_auto_stage(next_stage):
                    result = await self._process_stage(state, "")

        # Add assistant response
        state.add_message(MessageRole.ASSISTANT, result.message)

        # Save updated state
        await self._state_manager.save_workflow(state)

        response = result.message
        if result.prompt_user:
            response = f"{result.message}\n\n{result.prompt_user}"

        return state, response

    async def get_workflow_status(self, workflow_id: str) -> Optional[dict[str, Any]]:
        """Get current workflow status."""
        state = await self._state_manager.get_workflow(workflow_id)
        if not state:
            return None

        return {
            "id": state.id,
            "user_id": state.user_id,
            "stage": state.stage.value,
            "stage_index": list(WorkflowStage).index(state.stage) + 1,
            "total_stages": len(WorkflowStage),
            "is_complete": state.is_complete(),
            "intent": state.intent,
            "has_traffic_profile": state.traffic_profile is not None,
            "has_slo_requirements": state.slo_requirements is not None,
            "selected_models": state.selected_models,
            "has_capacity_plan": state.capacity_plan_id is not None,
            "has_deployment_config": state.deployment_config_id is not None,
            "message_count": len(state.conversation_history),
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
        }

    async def reset_workflow(self, workflow_id: str) -> Optional[WorkflowState]:
        """Reset workflow to initial state."""
        state = await self._state_manager.get_workflow(workflow_id)
        if not state:
            return None

        # Create new workflow for same user
        new_state = await self._state_manager.create_workflow(state.user_id)

        # Delete old workflow
        await self._state_manager.delete_workflow(workflow_id)

        return new_state

    async def _process_stage(self, state: WorkflowState, user_input: str) -> StageResult:
        """Process the current stage."""
        stage_handler = self._stages.get(state.stage)
        if not stage_handler:
            raise StageExecutionError(
                f"No handler for stage: {state.stage}",
                state.stage,
            )

        try:
            result = await stage_handler.process(state, user_input)
            return result
        except Exception as e:
            logger.error(
                "Stage execution failed",
                stage=state.stage.value,
                error=str(e),
            )
            raise StageExecutionError(
                f"Failed to process {state.stage.value} stage: {e}",
                state.stage,
                {"error": str(e)},
            ) from e

    def _is_auto_stage(self, stage: WorkflowStage) -> bool:
        """Check if stage can auto-process without user input."""
        # Benchmark and Filter stages can auto-process
        return stage in {WorkflowStage.BENCHMARK, WorkflowStage.FILTER}

    def get_stage_description(self, stage: WorkflowStage) -> str:
        """Get human-readable description of a stage."""
        descriptions = {
            WorkflowStage.INTENT: "Understanding your requirements",
            WorkflowStage.TRAFFIC: "Analyzing expected traffic patterns",
            WorkflowStage.SLO: "Defining service level objectives",
            WorkflowStage.BENCHMARK: "Looking up model performance benchmarks",
            WorkflowStage.FILTER: "Finding models that meet your requirements",
            WorkflowStage.REPLICAS: "Calculating optimal resource allocation",
            WorkflowStage.DEPLOY: "Preparing deployment configuration",
            WorkflowStage.MONITOR: "Verifying deployment and testing endpoints",
        }
        return descriptions.get(stage, "Processing...")
