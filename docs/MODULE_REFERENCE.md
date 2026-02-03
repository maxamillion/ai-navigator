# AI Navigator Module Reference

Detailed reference for all modules in the AI Navigator codebase.

## Table of Contents

1. [Models](#models)
2. [Workflow Engine](#workflow-engine)
3. [Capacity Planning](#capacity-planning)
4. [MCP Integration](#mcp-integration)
5. [Model Registry](#model-registry)
6. [Deployment](#deployment)
7. [State Management](#state-management)
8. [Quickstarts](#quickstarts)
9. [Configuration](#configuration)

---

## Models

Location: `src/ai_navigator/models/`

### workflow.py

Core workflow data models.

#### WorkflowStage (Enum)

The 8 stages of the deployment workflow:

```python
class WorkflowStage(str, Enum):
    INTENT = "intent"       # Extract requirements
    TRAFFIC = "traffic"     # Define traffic patterns
    SLO = "slo"             # Set performance targets
    BENCHMARK = "benchmark" # Fetch benchmarks (auto)
    FILTER = "filter"       # Select models (auto)
    REPLICAS = "replicas"   # Calculate resources
    DEPLOY = "deploy"       # Generate manifests
    MONITOR = "monitor"     # Verify deployment
```

#### MessageRole (Enum)

Conversation participant types:

```python
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
```

#### Message

Individual conversation message:

| Field | Type | Description |
|-------|------|-------------|
| `role` | MessageRole | Who sent the message |
| `content` | str | Message text |
| `timestamp` | datetime | When sent |
| `metadata` | dict | Additional data |

#### TrafficPattern (Enum)

Traffic classification:

```python
class TrafficPattern(str, Enum):
    STEADY = "steady"     # Consistent load
    BURST = "burst"       # Periodic spikes
    GROWTH = "growth"     # Increasing over time
    VARIABLE = "variable" # Unpredictable
```

#### TrafficProfile

Traffic characteristics for capacity planning:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pattern` | TrafficPattern | STEADY | Traffic pattern |
| `requests_per_second` | float | - | Average RPS |
| `peak_rps` | float | None | Peak RPS during bursts |
| `average_input_tokens` | int | 512 | Avg input tokens |
| `average_output_tokens` | int | 256 | Avg output tokens |
| `concurrent_users` | int | None | Concurrent users |

#### SLORequirements

Service Level Objectives:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `p50_latency_ms` | int | - | 50th percentile latency |
| `p95_latency_ms` | int | - | 95th percentile latency |
| `p99_latency_ms` | int | - | 99th percentile latency |
| `availability_percent` | float | 99.9 | Availability target |
| `max_tokens_per_second` | int | None | Throughput target |
| `max_queue_time_ms` | int | None | Max queue time |

#### ModelRequirements

Model selection criteria:

| Field | Type | Description |
|-------|------|-------------|
| `model_family` | str | Model family (llama, mistral, etc.) |
| `model_name` | str | Specific model name |
| `min_context_length` | int | Minimum context window |
| `max_parameters` | float | Max parameter count (billions) |
| `quantization` | str | Quantization level (fp16, int8) |
| `capabilities` | list[str] | Required capabilities |

#### WorkflowState

Complete workflow session state:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | UUID |
| `user_id` | str | User identifier |
| `stage` | WorkflowStage | Current stage |
| `created_at` | datetime | Creation time |
| `updated_at` | datetime | Last update time |
| `intent` | str | Extracted user intent |
| `traffic_profile` | TrafficProfile | Traffic requirements |
| `slo_requirements` | SLORequirements | Performance targets |
| `model_requirements` | ModelRequirements | Model criteria |
| `selected_models` | list[str] | Matching models |
| `capacity_plan_id` | str | Capacity plan reference |
| `deployment_config_id` | str | Deployment config reference |
| `conversation_history` | list[Message] | Full conversation |
| `metadata` | dict | Extensibility |

**Methods:**
- `add_message(role, content)` - Add message to history
- `advance_stage()` - Move to next stage
- `is_complete()` - Check if workflow finished

---

### capacity.py

Capacity planning models.

#### BenchmarkData

Model performance metrics from Model Registry:

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | str | Model identifier |
| `model_version` | str | Model version |
| `gpu_type` | str | GPU used for benchmark |
| `gpu_count` | int | Number of GPUs |
| `p50_latency_ms` | float | 50th percentile latency |
| `p95_latency_ms` | float | 95th percentile latency |
| `p99_latency_ms` | float | 99th percentile latency |
| `tokens_per_second` | float | Token throughput |
| `requests_per_second` | float | Request throughput |
| `gpu_memory_gb` | float | GPU memory usage |
| `gpu_utilization_percent` | float | GPU utilization |
| `input_tokens` | int | Test input tokens |
| `output_tokens` | int | Test output tokens |
| `batch_size` | int | Test batch size |
| `concurrency` | int | Test concurrency |
| `benchmark_date` | datetime | When benchmarked |
| `source` | str | Data source |

#### GPURecommendation

GPU configuration recommendation:

| Field | Type | Description |
|-------|------|-------------|
| `gpu_type` | str | GPU type (A100-40GB, etc.) |
| `gpu_count` | int | GPUs per replica |
| `estimated_cost_per_hour` | float | Hourly cost |
| `meets_slo` | bool | Meets SLO requirements |
| `headroom_percent` | float | Headroom above SLO |
| `notes` | list[str] | Additional notes |

#### CapacityPlan

Complete capacity plan:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | UUID |
| `created_at` | datetime | Creation time |
| `model_name` | str | Selected model |
| `model_version` | str | Model version |
| `min_replicas` | int | Minimum replicas |
| `max_replicas` | int | Maximum replicas |
| `target_replicas` | int | Target replica count |
| `gpu_type` | str | GPU type |
| `gpu_count` | int | GPUs per replica |
| `gpu_memory_gb` | float | GPU memory |
| `memory_per_replica` | str | Memory request (e.g., "32Gi") |
| `cpu_per_replica` | str | CPU request |
| `estimated_throughput_tps` | float | Expected tokens/sec |
| `estimated_rps` | float | Expected requests/sec |
| `estimated_p95_latency_ms` | float | Expected p95 latency |
| `estimated_p99_latency_ms` | float | Expected p99 latency |
| `meets_slo` | bool | SLO compliance |
| `slo_violations` | list[str] | List of violations |
| `estimated_monthly_cost` | float | Monthly cost estimate |
| `alternatives` | list[GPURecommendation] | Alternative configs |
| `benchmark_source` | str | Benchmark data source |
| `assumptions` | list[str] | Planning assumptions |

#### WhatIfScenario

Scenario for what-if analysis:

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Scenario name |
| `description` | str | Description |
| `rps_multiplier` | float | RPS multiplier |
| `new_rps` | float | Absolute new RPS |
| `p95_latency_ms` | int | New p95 target |
| `p99_latency_ms` | int | New p99 target |
| `gpu_type` | str | New GPU type |
| `max_replicas` | int | New max replicas |

#### WhatIfResult

What-if analysis result:

| Field | Type | Description |
|-------|------|-------------|
| `scenario` | WhatIfScenario | Input scenario |
| `original_plan` | CapacityPlan | Original plan |
| `modified_plan` | CapacityPlan | New plan |
| `replica_delta` | int | Change in replicas |
| `cost_delta_percent` | float | Cost change % |
| `latency_delta_percent` | float | Latency change % |
| `throughput_delta_percent` | float | Throughput change % |
| `is_feasible` | bool | Whether achievable |
| `warnings` | list[str] | Warnings |
| `recommendations` | list[str] | Recommendations |

---

### deployment.py

Deployment configuration models.

#### DeploymentStatus (Enum)

```python
class DeploymentStatus(str, Enum):
    PENDING = "pending"
    CREATING = "creating"
    RUNNING = "running"
    UPDATING = "updating"
    FAILED = "failed"
    DELETED = "deleted"
```

#### RuntimeConfig

vLLM runtime configuration:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `runtime_name` | str | vllm | Runtime name |
| `tensor_parallel_size` | int | 1 | Tensor parallelism GPUs |
| `gpu_memory_utilization` | float | 0.9 | GPU memory fraction |
| `dtype` | str | float16 | Data type |
| `max_model_len` | int | None | Max context length |
| `enforce_eager` | bool | False | Disable CUDA graphs |
| `extra_args` | list[str] | [] | Additional vLLM args |
| `env_vars` | dict | {} | Environment variables |

#### InferenceServiceSpec

KServe InferenceService specification:

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Service name |
| `namespace` | str | Kubernetes namespace |
| `min_replicas` | int | Minimum replicas |
| `max_replicas` | int | Maximum replicas |
| `gpu_count` | int | GPUs per replica |
| `cpu` | str | CPU request |
| `memory` | str | Memory request |
| `model_format` | str | Model format (pytorch) |
| `storage_uri` | str | Model storage location |
| `runtime` | RuntimeConfig | Runtime config |
| `timeout_seconds` | int | Request timeout |
| `scale_target` | int | Autoscaling target |
| `scale_metric` | str | Scaling metric |
| `labels` | dict | Kubernetes labels |
| `annotations` | dict | Kubernetes annotations |

#### DeploymentConfig

Complete deployment configuration:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | UUID |
| `inference_service` | InferenceServiceSpec | Service spec |
| `create_hpa` | bool | Create HPA |
| `create_pdb` | bool | Create PDB |
| `manifests` | dict | Generated YAML |

#### DeploymentResult

Deployment execution result:

| Field | Type | Description |
|-------|------|-------------|
| `deployment_id` | str | Deployment ID |
| `status` | DeploymentStatus | Current status |
| `message` | str | Status message |
| `endpoint_url` | str | Service URL |
| `ready_replicas` | int | Ready replicas |
| `total_replicas` | int | Total replicas |
| `test_results` | dict | Inference test results |

---

## Workflow Engine

Location: `src/ai_navigator/workflow/`

### engine.py

#### WorkflowError

Base exception for workflow errors.

#### StageExecutionError

Stage execution failed:
- `stage`: WorkflowStage that failed
- `details`: Error details dict

#### StageResult

Result of stage execution:

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Did processing succeed |
| `message` | str | Response to user |
| `advance` | bool | Move to next stage |
| `data` | dict | Extracted data |
| `prompt_user` | str | Follow-up question |

#### WorkflowEngine

Main workflow orchestrator:

```python
class WorkflowEngine:
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
    ) -> None: ...
```

**Methods:**

| Method | Description |
|--------|-------------|
| `start_workflow(user_id, initial_message)` | Start new workflow |
| `continue_workflow(workflow_id, user_message)` | Continue existing workflow |
| `get_workflow_status(workflow_id)` | Get workflow status |
| `reset_workflow(workflow_id)` | Reset to initial state |
| `get_stage_description(stage)` | Human-readable stage name |

---

### stages/base.py

#### BaseStage (ABC)

Abstract base class for stage handlers:

```python
class BaseStage(ABC):
    @abstractmethod
    async def process(
        self,
        state: WorkflowState,
        user_input: str
    ) -> StageResult:
        """Process user input for this stage."""
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

---

### stages/intent.py

#### IntentStage

Stage 1: Extract model requirements from natural language.

**Model Pattern Recognition:**
- llama, mistral, granite, phi, falcon, codellama, starcoder

**Capability Detection:**
- chat, code, instruct, embedding, vision

**Size Extraction:**
- Parses "7B", "70B", etc.

**Methods:**
- `process(state, user_input)` - Extract intent
- `_extract_intent(text)` - Parse natural language

---

### stages/traffic.py

#### TrafficStage

Stage 2: Determine traffic patterns.

**Parses:**
- RPS (requests per second)
- Peak traffic expectations
- Concurrent users
- Token count expectations

**Presets:**
- Low traffic (~10 RPS)
- Medium traffic (~50 RPS)
- High traffic (~200 RPS)
- Very high traffic (~1000 RPS)

---

### stages/slo.py

#### SLOStage

Stage 3: Set latency targets.

**Presets:**
- Interactive: p50=200ms, p95=500ms, p99=1000ms
- Batch: p50=2000ms, p95=5000ms, p99=10000ms
- Realtime: p50=100ms, p95=200ms, p99=500ms
- Standard: p50=500ms, p95=1000ms, p99=2000ms

**Parses:**
- Custom latency specifications
- Availability requirements

---

### stages/benchmark.py

#### BenchmarkStage

Stage 4: Fetch model performance benchmarks.

**Auto-processing:** Does not require user input.

**Data Sources:**
- Model Registry (primary)
- Default benchmarks (fallback)

---

### stages/filter.py

#### FilterStage

Stage 5: Select matching models.

**Auto-processing:** Does not require user input.

**Filtering by:**
- Model family
- Capabilities
- Size constraints
- Context length

---

### stages/replicas.py

#### ReplicasStage

Stage 6: Calculate GPU resources and replicas.

**Uses:**
- CapacityPlanner for calculations
- Benchmark data from Stage 4
- Traffic and SLO from Stages 2-3

**Generates:**
- CapacityPlan with alternatives
- Cost estimates

---

### stages/deploy.py

#### DeployStage

Stage 7: Generate Kubernetes manifests.

**Generates:**
- InferenceService YAML
- HPA YAML
- PDB YAML

**Uses:**
- YAMLGenerator
- Capacity plan from Stage 6

---

### stages/monitor.py

#### MonitorStage

Stage 8: Verify deployment and test endpoints.

**Checks:**
- Deployment status
- Replica readiness
- Inference endpoint health
- Latency validation

---

## Capacity Planning

Location: `src/ai_navigator/planning/`

### capacity.py

#### GPU_CATALOG

GPU specifications and costs:

```python
GPU_CATALOG = {
    "A100-80GB": {
        "memory_gb": 80,
        "fp16_tflops": 312,
        "cost_per_hour": 4.00,
        "availability": "high",
    },
    "A100-40GB": {"memory_gb": 40, "fp16_tflops": 312, "cost_per_hour": 2.50, ...},
    "A10": {"memory_gb": 24, "fp16_tflops": 125, "cost_per_hour": 1.00, ...},
    "L4": {"memory_gb": 24, "fp16_tflops": 121, "cost_per_hour": 0.70, ...},
    "T4": {"memory_gb": 16, "fp16_tflops": 65, "cost_per_hour": 0.35, ...},
    "H100-80GB": {"memory_gb": 80, "fp16_tflops": 989, "cost_per_hour": 8.00, ...},
}
```

#### MODEL_MEMORY_ESTIMATES

Approximate memory requirements by model size:

```python
MODEL_MEMORY_ESTIMATES = {
    "7b": {"fp16_gb": 14, "int8_gb": 7, "int4_gb": 4},
    "13b": {"fp16_gb": 26, "int8_gb": 13, "int4_gb": 7},
    "34b": {"fp16_gb": 68, "int8_gb": 34, "int4_gb": 17},
    "70b": {"fp16_gb": 140, "int8_gb": 70, "int4_gb": 35},
}
```

#### CapacityPlanner

SLO-driven capacity planning engine:

**Methods:**

| Method | Description |
|--------|-------------|
| `calculate_capacity_plan(benchmark, traffic, slo, availability_zones)` | Full capacity plan |
| `estimate_replicas_for_rps(benchmark, target_rps, headroom_percent)` | Replica estimation |
| `estimate_gpu_memory(model_size, quantization, context_length, batch_size)` | GPU memory estimation |

**Internal Methods:**
- `_calculate_min_replicas()` - Minimum for SLO
- `_calculate_max_replicas()` - Maximum for peak traffic
- `_calculate_target_replicas()` - Steady-state target
- `_calculate_memory_allocation()` - Memory per replica
- `_calculate_cpu_allocation()` - CPU per replica
- `_check_slo_compliance()` - Verify SLO met
- `_generate_alternatives()` - Alternative GPU configs
- `_generate_assumptions()` - Document assumptions

---

### recommender.py

#### ModelRecommender

Model and GPU recommendation logic:

**Methods:**

| Method | Description |
|--------|-------------|
| `recommend_models(requirements)` | Find matching models |
| `recommend_gpu(model_size, quantization, target_latency)` | Recommend GPU |

---

### whatif.py

#### WhatIfAnalyzer

Scenario analysis:

**Methods:**

| Method | Description |
|--------|-------------|
| `analyze_scenario(original_plan, scenario, benchmark, traffic, slo)` | Compare scenarios |

**Scenario Types:**
- Traffic changes (RPS multiplier, absolute RPS)
- SLO changes (tighter/looser latency)
- Resource changes (GPU type, max replicas)

---

## MCP Integration

Location: `src/ai_navigator/mcp/`

### client.py

#### MCPClient

HTTP adapter for rhoai-mcp server:

```python
class MCPClient:
    def __init__(self, settings: Optional[MCPSettings] = None) -> None: ...
```

**Methods:**

| Method | Description |
|--------|-------------|
| `connect()` | Connect and discover tools |
| `disconnect()` | Close connection |
| `call_tool(tool_name, arguments)` | Execute MCP tool |
| `list_tools()` | List available tools |
| `get_tool_schema(tool_name)` | Get tool schema |

**Properties:**
- `is_connected` - Connection status
- `base_url` - Server URL

**Context Manager Support:**
```python
async with MCPClient() as client:
    result = await client.call_tool("tool_name", args)
```

#### MCPToolHelpers

Convenience methods for common operations:

| Method | Description |
|--------|-------------|
| `create_project(name, display_name, description)` | Create DS Project |
| `get_project_status(namespace)` | Get project status |
| `list_projects()` | List all projects |
| `list_inference_services(namespace)` | List deployed models |
| `create_inference_service(namespace, name, model_name)` | Deploy model |
| `get_inference_service(namespace, name)` | Get deployment |
| `delete_inference_service(namespace, name)` | Remove deployment |
| `list_workbenches(namespace)` | List workbenches |
| `create_s3_connection(...)` | Create S3 connection |
| `create_storage(namespace, name, size)` | Create PVC |

---

### orchestrator.py

#### ToolCall

Represents a tool call in an orchestration:

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | str | MCP tool name |
| `arguments` | dict | Tool arguments |
| `depends_on` | list[str] | Dependency keys |
| `result_key` | str | Key for result storage |
| `transform` | Callable | Result transformer |

#### OrchestrationResult

Result of orchestration:

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | All calls succeeded |
| `results` | dict | Tool results by key |
| `errors` | dict | Errors by key |
| `execution_order` | list | Order of execution |

#### MCPOrchestrator

Multi-tool orchestration:

**Methods:**

| Method | Description |
|--------|-------------|
| `execute_parallel(calls, stop_on_error)` | Run independent calls concurrently |
| `execute_sequential(calls, context, stop_on_error)` | Run calls in order |
| `execute_dag(calls, context)` | Run with dependency resolution |

**Argument Resolution:**
- `$result_key` - Reference previous result
- `$result_key.field` - Nested access

**Pre-defined Orchestrations:**
- `setup_project_orchestration()` - Create project with storage
- `get_project_overview_orchestration()` - Parallel project info fetch

---

### cache.py

#### MCPCache

LRU cache with TTL for MCP responses.

---

### recovery.py

#### RetryPolicy

Retry configuration:

| Field | Type | Default |
|-------|------|---------|
| `max_retries` | int | 3 |
| `delay_seconds` | float | 1.0 |
| `exponential_base` | float | 2.0 |

#### with_retry

Decorator/function for retry logic:

```python
result = await with_retry(async_func, retry_policy)
```

---

## Model Registry

Location: `src/ai_navigator/registry/`

### client.py

#### RegistryClient

REST client for OpenShift AI Model Registry:

**Methods:**

| Method | Description |
|--------|-------------|
| `list_models()` | List registered models |
| `get_model(model_id)` | Get model by ID |
| `get_model_version(model_id, version)` | Get specific version |
| `get_artifacts(model_id, version)` | Get model artifacts |

---

### models.py

Registry data models:

- `RegisteredModel` - Model metadata
- `ModelVersion` - Version information
- `ModelArtifact` - Artifact locations

---

### benchmarks.py

Benchmark extraction from registry properties.

---

### cache.py

Registry response caching with TTL.

---

## Deployment

Location: `src/ai_navigator/deployment/`

### generator.py

#### YAMLGenerator

Kubernetes manifest generation:

**Methods:**

| Method | Description |
|--------|-------------|
| `generate_all(config)` | Generate all manifests |
| `generate_inference_service(spec)` | InferenceService YAML |
| `generate_hpa(spec)` | HPA YAML |
| `generate_pdb(spec)` | PDB YAML |
| `generate_service_account(namespace, name)` | ServiceAccount YAML |
| `generate_serving_runtime(name, namespace, image)` | ServingRuntime YAML |

**Generated Manifests:**
- `serving.kserve.io/v1beta1 InferenceService`
- `autoscaling/v2 HorizontalPodAutoscaler`
- `policy/v1 PodDisruptionBudget`
- `serving.kserve.io/v1alpha1 ServingRuntime`

---

### validators.py

#### DeploymentValidator

Configuration validation:

**Methods:**

| Method | Description |
|--------|-------------|
| `validate(config)` | Validate deployment config |

**Checks:**
- Resource constraints
- SLO feasibility
- Network policy

---

### orchestrator.py

#### DeploymentOrchestrator

Deployment execution:

**Methods:**

| Method | Description |
|--------|-------------|
| `deploy(config, dry_run)` | Execute deployment |
| `get_status(deployment_id)` | Check status |
| `delete(deployment_id)` | Remove deployment |

---

## State Management

Location: `src/ai_navigator/state/`

### manager.py

#### StateStore (ABC)

Abstract base for state backends:

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

#### StateManager

High-level state operations:

**Methods:**

| Method | Description |
|--------|-------------|
| `create_workflow(user_id)` | Create new workflow |
| `get_workflow(workflow_id)` | Get by ID |
| `get_active_workflow(user_id)` | Get user's active workflow |
| `save_workflow(state)` | Persist workflow |
| `delete_workflow(workflow_id)` | Delete workflow |
| `save_capacity_plan(plan)` | Save capacity plan |
| `get_capacity_plan(plan_id)` | Get capacity plan |
| `save_deployment_config(config)` | Save deployment config |
| `get_deployment_config(config_id)` | Get deployment config |
| `close()` | Close connections |

---

### memory.py

#### InMemoryStateStore

Development/testing state store using dictionaries.

---

### postgres.py

#### PostgresStateStore

Production state store using asyncpg.

---

## Quickstarts

Location: `src/ai_navigator/quickstarts/`

### engine.py

#### QuickstartStatus (Enum)

```python
class QuickstartStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

#### QuickstartStep

Individual step in a task:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Step identifier |
| `name` | str | Step name |
| `description` | str | Step description |
| `status` | QuickstartStatus | Step status |
| `result` | str | Step result |
| `error` | str | Error message |

#### QuickstartResult

Task execution result:

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | str | Task name |
| `status` | QuickstartStatus | Overall status |
| `steps` | list | Step details |
| `message` | str | Message to user |
| `data` | dict | Result data |
| `next_prompt` | str | Next question |

#### QuickstartTask (ABC)

Abstract base for quickstart tasks:

```python
class QuickstartTask(ABC):
    @property
    def name(self) -> str: ...
    @property
    def display_name(self) -> str: ...
    @property
    def description(self) -> str: ...
    def get_steps(self) -> list[QuickstartStep]: ...
    async def execute(self, context, user_input) -> QuickstartResult: ...
    def get_required_inputs(self) -> list[dict]: ...
```

#### QuickstartEngine

Task execution engine:

**Methods:**

| Method | Description |
|--------|-------------|
| `register_task(task)` | Register a task |
| `list_tasks()` | List available tasks |
| `get_task(task_name)` | Get task by name |
| `start_task(task_name, session_id, initial_context)` | Start task |
| `continue_task(session_id, user_input)` | Continue task |
| `cancel_task(session_id)` | Cancel task |
| `get_session_status(session_id)` | Get session status |

---

### tasks/

Pre-defined quickstart tasks:

| Task | File | Description |
|------|------|-------------|
| `deploy-model` | deploy_model.py | Deploy an AI model |
| `setup-project` | project.py | Create DS Project |
| `test-inference` | test_inference.py | Test endpoint |
| `serving-runtime` | serving_runtime.py | Configure runtime |
| `data-connection` | connection.py | Setup S3 connection |

---

## Configuration

Location: `src/ai_navigator/config.py`

### Environment (Enum)

```python
class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
```

### MCPTransport (Enum)

```python
class MCPTransport(str, Enum):
    SSE = "sse"
    STDIO = "stdio"
```

### StateBackend (Enum)

```python
class StateBackend(str, Enum):
    MEMORY = "memory"
    POSTGRES = "postgres"
```

### MCPSettings

MCP client configuration (env prefix: `RHOAI_MCP_`):

| Field | Type | Default | Env Var |
|-------|------|---------|---------|
| `host` | str | localhost | RHOAI_MCP_HOST |
| `port` | int | 8080 | RHOAI_MCP_PORT |
| `transport` | MCPTransport | sse | RHOAI_MCP_TRANSPORT |
| `timeout_seconds` | int | 30 | RHOAI_MCP_TIMEOUT_SECONDS |
| `max_retries` | int | 3 | RHOAI_MCP_MAX_RETRIES |
| `retry_delay_seconds` | float | 1.0 | RHOAI_MCP_RETRY_DELAY_SECONDS |

### ModelRegistrySettings

Registry client configuration (env prefix: `MODEL_REGISTRY_`):

| Field | Type | Default | Env Var |
|-------|------|---------|---------|
| `url` | str | (cluster internal) | MODEL_REGISTRY_URL |
| `timeout_seconds` | int | 30 | MODEL_REGISTRY_TIMEOUT_SECONDS |
| `cache_ttl_seconds` | int | 300 | MODEL_REGISTRY_CACHE_TTL_SECONDS |

### LLMSettings

LLM provider configuration (env prefix: `LLM_`):

| Field | Type | Default | Env Var |
|-------|------|---------|---------|
| `provider` | str | openai | LLM_PROVIDER |
| `base_url` | str | (cluster internal) | LLM_BASE_URL |
| `api_key` | SecretStr | (empty) | LLM_API_KEY |
| `model_name` | str | granite-3b-code-instruct | LLM_MODEL_NAME |
| `temperature` | float | 0.1 | LLM_TEMPERATURE |
| `max_tokens` | int | 2048 | LLM_MAX_TOKENS |

### StateSettings

State backend configuration (env prefix: `STATE_`):

| Field | Type | Default | Env Var |
|-------|------|---------|---------|
| `backend` | StateBackend | memory | STATE_BACKEND |
| `postgres_dsn` | str | None | STATE_POSTGRES_DSN |
| `ttl_hours` | int | 24 | STATE_TTL_HOURS |

### Settings

Main application settings (env prefix: `AI_NAVIGATOR_`):

| Field | Type | Default | Env Var |
|-------|------|---------|---------|
| `enabled` | bool | true | AI_NAVIGATOR_ENABLED |
| `environment` | Environment | development | AI_NAVIGATOR_ENVIRONMENT |
| `debug` | bool | false | AI_NAVIGATOR_DEBUG |
| `log_level` | str | INFO | AI_NAVIGATOR_LOG_LEVEL |
| `mcp` | MCPSettings | (defaults) | (nested) |
| `model_registry` | ModelRegistrySettings | (defaults) | (nested) |
| `llm` | LLMSettings | (defaults) | (nested) |
| `state` | StateSettings | (defaults) | (nested) |

**Properties:**
- `is_production` - Check if in production

### get_settings()

Get application settings singleton:

```python
from ai_navigator.config import get_settings

settings = get_settings()
print(settings.environment)
```
