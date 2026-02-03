# AI Navigator Architecture

This document provides a detailed architectural overview of the AI Navigator system for developers joining the maintenance team.

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [8-Stage Workflow Engine](#8-stage-workflow-engine)
6. [State Management](#state-management)
7. [External Integrations](#external-integrations)
8. [Deployment Architecture](#deployment-architecture)
9. [Design Patterns](#design-patterns)
10. [Extension Points](#extension-points)

---

## System Overview

AI Navigator is an AI-powered conversational agent that guides users through deploying ML models on OpenShift AI. It provides:

- **SLO-driven capacity planning**: Calculates GPU resources based on latency and throughput requirements
- **Benchmark integration**: Uses performance data from Model Registry to inform decisions
- **Kubernetes manifest generation**: Produces ready-to-deploy InferenceService YAML
- **Conversational workflow**: Natural language interface for deployment planning

### Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Web Framework | FastAPI |
| Data Validation | Pydantic v2 |
| HTTP Client | httpx (async) |
| Template Engine | Jinja2 |
| Database | PostgreSQL (optional) |
| Logging | structlog |
| Retry Logic | tenacity |

---

## High-Level Architecture

```
                                    ┌─────────────────────────────────┐
                                    │         Client (UI/CLI)         │
                                    └───────────────┬─────────────────┘
                                                    │ HTTP/REST
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Router                                    │
│  /ai-navigator/workflow/*  │  /ai-navigator/capacity/*  │  /ai-navigator/*   │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            │                           │                           │
            ▼                           ▼                           ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  Workflow Engine  │     │  Capacity Planner │     │  Quickstart Engine│
│                   │     │                   │     │                   │
│  - 8 Stage FSM    │     │  - GPU Catalog    │     │  - Task Registry  │
│  - State Machine  │     │  - SLO Compliance │     │  - Session Mgmt   │
│  - Conversation   │     │  - Cost Estimates │     │  - Step Tracking  │
└─────────┬─────────┘     └─────────┬─────────┘     └─────────┬─────────┘
          │                         │                         │
          ▼                         ▼                         ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              State Manager                                     │
│                     In-Memory (dev) │ PostgreSQL (prod)                       │
└───────────────────────────────────────────────────────────────────────────────┘
          │                                                   │
          │                                                   │
          ▼                                                   ▼
┌───────────────────┐                             ┌───────────────────┐
│    MCP Client     │                             │  Model Registry   │
│  (rhoai-mcp)      │                             │  Client           │
│                   │                             │                   │
│  - Tool Discovery │                             │  - Model Lookup   │
│  - Orchestration  │                             │  - Benchmarks     │
│  - Retry/Recovery │                             │  - Caching        │
└─────────┬─────────┘                             └─────────┬─────────┘
          │                                                   │
          ▼                                                   ▼
┌───────────────────┐                             ┌───────────────────┐
│   OpenShift AI    │                             │  Model Registry   │
│   (via MCP)       │                             │  Service          │
└───────────────────┘                             └───────────────────┘
```

---

## Core Components

### Directory Structure

```
src/ai_navigator/
├── config.py                 # Pydantic settings with env var support
├── router.py                 # FastAPI endpoints
├── models/                   # Pydantic data models
│   ├── workflow.py           # Workflow state, messages, SLO, traffic
│   ├── capacity.py           # Benchmarks, capacity plans, GPU recs
│   └── deployment.py         # Deployment configs, InferenceService specs
├── state/                    # State persistence layer
│   ├── manager.py            # StateManager abstraction
│   ├── memory.py             # In-memory store (development)
│   └── postgres.py           # PostgreSQL store (production)
├── workflow/                 # 8-stage workflow engine
│   ├── engine.py             # WorkflowEngine orchestrator
│   └── stages/               # Individual stage handlers
│       ├── base.py           # BaseStage abstract class
│       ├── intent.py         # Stage 1: Requirement extraction
│       ├── traffic.py        # Stage 2: Traffic patterns
│       ├── slo.py            # Stage 3: SLO definition
│       ├── benchmark.py      # Stage 4: Performance lookup
│       ├── filter.py         # Stage 5: Model selection
│       ├── replicas.py       # Stage 6: Resource calculation
│       ├── deploy.py         # Stage 7: Manifest generation
│       └── monitor.py        # Stage 8: Deployment verification
├── mcp/                      # MCP integration
│   ├── client.py             # MCPClient HTTP adapter
│   ├── orchestrator.py       # Multi-tool orchestration
│   ├── cache.py              # Response caching
│   └── recovery.py           # Retry policies
├── registry/                 # Model Registry integration
│   ├── client.py             # REST API client
│   ├── models.py             # Registry data models
│   ├── benchmarks.py         # Benchmark extraction
│   └── cache.py              # Response caching
├── planning/                 # Capacity planning
│   ├── capacity.py           # CapacityPlanner engine
│   ├── recommender.py        # Model/GPU recommendation
│   └── whatif.py             # Scenario analysis
├── deployment/               # Kubernetes deployment
│   ├── generator.py          # YAML manifest generation
│   ├── validators.py         # Config validation
│   └── orchestrator.py       # Deployment execution
└── quickstarts/              # Interactive tutorials
    ├── engine.py             # QuickstartEngine
    └── tasks/                # Predefined quickstart tasks
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `WorkflowEngine` | Orchestrates 8-stage FSM, manages stage transitions |
| `StateManager` | Abstracts state persistence (workflow, plans, configs) |
| `CapacityPlanner` | Calculates replicas, GPU allocation, and costs |
| `MCPClient` | Communicates with rhoai-mcp server for OpenShift operations |
| `MCPOrchestrator` | Coordinates multiple MCP tool calls with dependencies |
| `YAMLGenerator` | Generates Kubernetes manifests (InferenceService, HPA, PDB) |
| `QuickstartEngine` | Manages interactive guided tutorials |

---

## Data Flow

### Workflow Conversation Flow

```
User Message
     │
     ▼
┌────────────────┐
│  router.py     │ POST /workflow/{id}/continue
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ WorkflowEngine │
│ continue_      │
│ workflow()     │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ StateManager   │ Load WorkflowState from store
│ get_workflow() │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Current Stage  │ e.g., TrafficStage, SLOStage
│ .process()     │
└───────┬────────┘
        │
        ├── Extract data from user input
        ├── Update WorkflowState
        ├── Return StageResult
        │
        ▼
┌────────────────┐
│ WorkflowEngine │
│ Check .advance │──────► If True: Move to next stage
│                │──────► If auto-stage: Process immediately
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ StateManager   │ Persist updated state
│ save_workflow()│
└───────┬────────┘
        │
        ▼
   Response to User
```

### Capacity Planning Flow

```
Traffic + SLO Requirements
          │
          ▼
┌─────────────────────┐
│   CapacityPlanner   │
│ calculate_capacity_ │
│ plan()              │
└─────────┬───────────┘
          │
          ├── Calculate min replicas from RPS
          ├── Calculate max replicas for peak traffic
          ├── Apply HA constraints
          ├── Check SLO compliance
          ├── Estimate costs
          ├── Generate alternatives
          │
          ▼
┌─────────────────────┐
│   CapacityPlan      │
│                     │
│ - min/max/target    │
│   replicas          │
│ - GPU type/count    │
│ - Cost estimates    │
│ - SLO compliance    │
│ - Alternatives      │
└─────────────────────┘
```

---

## 8-Stage Workflow Engine

The workflow follows a finite state machine pattern with 8 stages:

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌───────────┐
│ INTENT  │───►│ TRAFFIC │───►│   SLO   │───►│ BENCHMARK │
│ Step 1  │    │ Step 2  │    │ Step 3  │    │  Step 4   │
└─────────┘    └─────────┘    └─────────┘    └───────────┘
                                                   │
                                                   ▼ (auto)
┌─────────┐    ┌─────────┐    ┌──────────┐   ┌───────────┐
│ MONITOR │◄───│ DEPLOY  │◄───│ REPLICAS │◄──│  FILTER   │
│ Step 8  │    │ Step 7  │    │  Step 6  │   │  Step 5   │
└─────────┘    └─────────┘    └──────────┘   └───────────┘
                                                   ▲
                                                   │ (auto)
```

### Stage Details

| Stage | Purpose | User Interaction | Key Data |
|-------|---------|------------------|----------|
| **INTENT** | Extract model requirements | Yes - describe needs | `ModelRequirements` |
| **TRAFFIC** | Define traffic patterns | Yes - RPS, patterns | `TrafficProfile` |
| **SLO** | Set performance targets | Yes - latency targets | `SLORequirements` |
| **BENCHMARK** | Fetch model performance data | No - auto | `BenchmarkData` |
| **FILTER** | Select matching models | No - auto | `selected_models` |
| **REPLICAS** | Calculate resources | Yes - confirm plan | `CapacityPlan` |
| **DEPLOY** | Generate manifests | Yes - approve | `DeploymentConfig` |
| **MONITOR** | Verify deployment | Yes - test endpoints | Status |

### Stage Implementation Pattern

Each stage implements `BaseStage`:

```python
class BaseStage(ABC):
    @abstractmethod
    async def process(
        self,
        state: WorkflowState,
        user_input: str
    ) -> StageResult:
        """Process user input and return result."""
        ...

    def get_prompt_template(self) -> str:
        """Optional LLM prompt template."""
        return ""

    def validate_input(
        self,
        state: WorkflowState,
        user_input: str
    ) -> tuple[bool, str]:
        """Validate user input."""
        return True, ""
```

### StageResult Structure

```python
class StageResult:
    success: bool       # Did processing succeed?
    message: str        # Response to user
    advance: bool       # Move to next stage?
    data: dict          # Extracted data
    prompt_user: str    # Follow-up question
```

---

## State Management

### State Store Abstraction

The system uses a pluggable state backend:

```python
class StateStore(ABC):
    async def save_workflow(self, state: WorkflowState) -> None: ...
    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowState]: ...
    async def get_workflow_by_user(self, user_id: str) -> Optional[WorkflowState]: ...
    async def delete_workflow(self, workflow_id: str) -> bool: ...
    async def save_capacity_plan(self, plan: CapacityPlan) -> None: ...
    async def get_capacity_plan(self, plan_id: str) -> Optional[CapacityPlan]: ...
    async def save_deployment_config(self, config: DeploymentConfig) -> None: ...
    async def get_deployment_config(self, config_id: str) -> Optional[DeploymentConfig]: ...
    async def close(self) -> None: ...
```

### Implementations

| Backend | Class | Use Case |
|---------|-------|----------|
| In-Memory | `InMemoryStateStore` | Development, testing |
| PostgreSQL | `PostgresStateStore` | Production |

### WorkflowState Structure

```python
class WorkflowState(BaseModel):
    id: str                                    # UUID
    user_id: str                               # User identifier
    stage: WorkflowStage                       # Current stage
    created_at: datetime
    updated_at: datetime

    # Stage-specific data
    intent: Optional[str]
    traffic_profile: Optional[TrafficProfile]
    slo_requirements: Optional[SLORequirements]
    model_requirements: Optional[ModelRequirements]

    # Results
    selected_models: list[str]
    capacity_plan_id: Optional[str]
    deployment_config_id: Optional[str]

    # Conversation history
    conversation_history: list[Message]

    # Extensibility
    metadata: dict[str, Any]
```

---

## External Integrations

### MCP Integration (rhoai-mcp)

The MCP client connects to the rhoai-mcp server for OpenShift AI operations:

```python
class MCPClient:
    async def connect(self) -> None
    async def disconnect(self) -> None
    async def call_tool(self, tool_name: str, arguments: dict) -> dict
    def list_tools(self) -> list[str]
    def get_tool_schema(self, tool_name: str) -> dict
```

**Available MCP Tools:**
- `create_data_science_project` - Create namespace
- `list_inference_services` - List deployed models
- `create_inference_service` - Deploy model
- `get_inference_service` - Get deployment status
- `delete_inference_service` - Remove deployment
- `create_s3_data_connection` - Configure storage
- `list_workbenches` - List development environments

### MCP Orchestrator

For complex operations requiring multiple tools:

```python
class MCPOrchestrator:
    async def execute_parallel(self, calls: list[ToolCall]) -> OrchestrationResult
    async def execute_sequential(self, calls: list[ToolCall]) -> OrchestrationResult
    async def execute_dag(self, calls: list[ToolCall]) -> OrchestrationResult
```

**Execution Patterns:**
- **Parallel**: Independent tools run concurrently
- **Sequential**: Results passed between tools
- **DAG**: Dependency-aware execution

### Model Registry Integration

```python
class RegistryClient:
    async def list_models(self) -> list[RegisteredModel]
    async def get_model(self, model_id: str) -> RegisteredModel
    async def get_model_version(self, model_id: str, version: str) -> ModelVersion
    async def get_benchmarks(self, model_id: str) -> list[BenchmarkData]
```

---

## Deployment Architecture

### Container Build

```dockerfile
# Multi-stage build
FROM registry.access.redhat.com/ubi9/python-311:latest AS builder
# Install dependencies with uv

FROM registry.access.redhat.com/ubi9/python-311-minimal:latest
# Copy application and dependencies
EXPOSE 8080
CMD ["uvicorn", "ai_navigator.router:router", "--host", "0.0.0.0", "--port", "8080"]
```

### Kubernetes Resources

| Resource | Purpose |
|----------|---------|
| Deployment | Application pods |
| Service | Internal networking |
| Route | External access (OpenShift) |
| HPA | Auto-scaling (2-5 replicas) |
| PDB | Availability during updates |
| ConfigMap | Environment configuration |
| Secret | Sensitive credentials |

### Configuration

Environment variables (prefix `AI_NAVIGATOR_`):

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_NAVIGATOR_ENVIRONMENT` | Environment (dev/staging/prod) | development |
| `AI_NAVIGATOR_DEBUG` | Enable debug mode | false |
| `RHOAI_MCP_HOST` | MCP server host | localhost |
| `RHOAI_MCP_PORT` | MCP server port | 8080 |
| `MODEL_REGISTRY_URL` | Model Registry URL | (cluster internal) |
| `STATE_BACKEND` | State store (memory/postgres) | memory |
| `STATE_POSTGRES_DSN` | PostgreSQL connection string | - |
| `LLM_BASE_URL` | LLM API endpoint | (vLLM internal) |

---

## Design Patterns

### 1. State Machine Pattern (Workflow Engine)

The `WorkflowEngine` implements a finite state machine where:
- States = `WorkflowStage` enum values
- Transitions = `advance_stage()` method
- Actions = Stage `process()` methods

### 2. Strategy Pattern (Stages)

Each stage is a separate class implementing `BaseStage`, allowing:
- Independent development and testing
- Easy addition of new stages
- Customizable behavior per stage

### 3. Factory Pattern (Dependency Injection)

Router uses factory functions for lazy initialization:

```python
async def get_workflow_engine():
    store = InMemoryStateStore()
    state_manager = StateManager(store)
    return WorkflowEngine(state_manager)
```

### 4. Repository Pattern (State Management)

`StateManager` wraps `StateStore` implementations, providing a clean API for state operations.

### 5. Adapter Pattern (MCP Client)

`MCPClient` adapts the rhoai-mcp HTTP API to internal interfaces.

### 6. Template Method Pattern (Quickstarts)

`QuickstartTask` defines the skeleton:
- `get_steps()` - Define steps
- `execute()` - Run the task
- `get_required_inputs()` - Collect inputs

---

## Extension Points

### Adding a New Workflow Stage

1. Create stage handler in `workflow/stages/`:

```python
# workflow/stages/newstage.py
from ai_navigator.workflow.stages.base import BaseStage

class NewStage(BaseStage):
    async def process(self, state, user_input):
        # Implementation
        return StageResult(success=True, message="...", advance=True)
```

2. Add to `WorkflowStage` enum in `models/workflow.py`
3. Register in `WorkflowEngine.__init__()` in `workflow/engine.py`

### Adding a New State Backend

1. Implement `StateStore` interface in `state/`:

```python
class RedisStateStore(StateStore):
    async def save_workflow(self, state): ...
    # ... implement all methods
```

2. Add to `StateBackend` enum in `config.py`
3. Update factory in `router.py`

### Adding a New Quickstart Task

1. Create task in `quickstarts/tasks/`:

```python
class MyTask(QuickstartTask):
    @property
    def name(self): return "my-task"

    @property
    def display_name(self): return "My Task"

    @property
    def description(self): return "Does something useful"

    def get_steps(self): return [QuickstartStep(...)]

    async def execute(self, context, user_input=None):
        return QuickstartResult(...)
```

2. Register in router's `get_quickstart_engine()`

### Adding GPU Types

Extend `GPU_CATALOG` in `planning/capacity.py`:

```python
GPU_CATALOG = {
    "NEW-GPU": {
        "memory_gb": 48,
        "fp16_tflops": 400,
        "cost_per_hour": 3.50,
        "availability": "medium",
    },
    # ... existing GPUs
}
```

---

## Performance Considerations

### Caching

- **MCP responses**: Cached via `MCPCache` with configurable TTL
- **Registry queries**: Cached via `RegistryCache`
- Both use LRU eviction with TTL expiration

### Async Throughout

All I/O operations are async:
- HTTP calls via `httpx.AsyncClient`
- Database operations via `asyncpg`
- MCP tool calls

### Connection Pooling

`MCPClient` and `RegistryClient` maintain connection pools for efficient reuse.

### State Serialization

Pydantic models provide efficient JSON serialization/deserialization for state persistence.

---

## Security Considerations

### Input Validation

- All API inputs validated via Pydantic models
- User input sanitized before processing
- SQL injection prevented via parameterized queries (asyncpg)

### Secrets Management

- API keys stored as `SecretStr` (masked in logs)
- Kubernetes Secrets for production credentials
- No secrets in ConfigMaps or environment variables

### Network Security

- Internal services use ClusterIP
- External access via OpenShift Routes with TLS
- MCP server connection restricted to cluster network

---

## Monitoring and Observability

### Structured Logging

All modules use `structlog` for structured JSON logging:

```python
logger.info("Workflow started", workflow_id=state.id, stage=state.stage.value)
```

### Health Checks

`GET /ai-navigator/health` returns service status for Kubernetes probes.

### Metrics (Future)

Prometheus metrics endpoint planned at `/metrics` for:
- Request latency
- Workflow stage transitions
- MCP tool call success rates
- Model Registry query latency
