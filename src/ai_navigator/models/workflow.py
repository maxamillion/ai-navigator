"""Workflow state and conversation models."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class WorkflowStage(str, Enum):
    """Workflow stages following NeuralNavigator's 8-step process."""

    INTENT = "intent"  # Step 1: Extract requirements from natural language
    TRAFFIC = "traffic"  # Step 2: Determine traffic patterns
    SLO = "slo"  # Step 3: Set latency targets
    BENCHMARK = "benchmark"  # Step 4: Query model performance data
    FILTER = "filter"  # Step 5: Select models meeting requirements
    REPLICAS = "replicas"  # Step 6: Calculate resources
    DEPLOY = "deploy"  # Step 7: Generate and apply manifests
    MONITOR = "monitor"  # Step 8: Verify deployment


class MessageRole(str, Enum):
    """Message sender role."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Conversation message."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrafficPattern(str, Enum):
    """Traffic pattern classification."""

    STEADY = "steady"  # Consistent load
    BURST = "burst"  # Periodic spikes
    GROWTH = "growth"  # Growing over time
    VARIABLE = "variable"  # Unpredictable


class TrafficProfile(BaseModel):
    """Traffic characteristics for capacity planning."""

    pattern: TrafficPattern = TrafficPattern.STEADY
    requests_per_second: float = Field(ge=0, description="Average RPS")
    peak_rps: Optional[float] = Field(default=None, ge=0, description="Peak RPS during bursts")
    average_input_tokens: int = Field(default=512, ge=1, description="Average input tokens")
    average_output_tokens: int = Field(default=256, ge=1, description="Average output tokens")
    concurrent_users: Optional[int] = Field(default=None, ge=1, description="Concurrent users")


class SLORequirements(BaseModel):
    """Service Level Objectives for model serving."""

    p50_latency_ms: int = Field(ge=0, description="50th percentile latency target")
    p95_latency_ms: int = Field(ge=0, description="95th percentile latency target")
    p99_latency_ms: int = Field(ge=0, description="99th percentile latency target")
    availability_percent: float = Field(
        default=99.9, ge=0, le=100, description="Availability target"
    )
    max_tokens_per_second: Optional[int] = Field(
        default=None, ge=1, description="Throughput target"
    )
    max_queue_time_ms: Optional[int] = Field(default=None, ge=0, description="Max queue time")


class ModelRequirements(BaseModel):
    """Model selection requirements."""

    model_family: Optional[str] = Field(default=None, description="Model family (e.g., llama)")
    model_name: Optional[str] = Field(default=None, description="Specific model name")
    min_context_length: Optional[int] = Field(
        default=None, ge=1, description="Minimum context window"
    )
    max_parameters: Optional[float] = Field(
        default=None, ge=0, description="Maximum parameter count (billions)"
    )
    quantization: Optional[str] = Field(
        default=None, description="Quantization level (e.g., fp16, int8)"
    )
    capabilities: list[str] = Field(
        default_factory=list, description="Required capabilities (chat, code, etc.)"
    )


class WorkflowState(BaseModel):
    """Complete workflow state for a user session."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = Field(description="User identifier")
    stage: WorkflowStage = Field(default=WorkflowStage.INTENT)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Stage-specific data
    intent: Optional[str] = Field(default=None, description="Extracted user intent")
    traffic_profile: Optional[TrafficProfile] = Field(default=None)
    slo_requirements: Optional[SLORequirements] = Field(default=None)
    model_requirements: Optional[ModelRequirements] = Field(default=None)

    # Results
    selected_models: list[str] = Field(default_factory=list)
    capacity_plan_id: Optional[str] = Field(default=None)
    deployment_config_id: Optional[str] = Field(default=None)

    # Conversation history
    conversation_history: list[Message] = Field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_message(self, role: MessageRole, content: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append(Message(role=role, content=content))
        self.updated_at = datetime.utcnow()

    def advance_stage(self) -> Optional[WorkflowStage]:
        """Advance to the next workflow stage."""
        stages = list(WorkflowStage)
        current_index = stages.index(self.stage)
        if current_index < len(stages) - 1:
            self.stage = stages[current_index + 1]
            self.updated_at = datetime.utcnow()
            return self.stage
        return None

    def is_complete(self) -> bool:
        """Check if workflow has completed all stages."""
        return self.stage == WorkflowStage.MONITOR
