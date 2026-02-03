# AI Navigator API Reference

Complete API documentation for AI Navigator endpoints.

## Base URL

```
/ai-navigator
```

## Authentication

Currently, no authentication is required. In production, integrate with OpenShift OAuth.

---

## Health & Status

### Health Check

Check if the service is running.

```
GET /ai-navigator/health
```

**Response 200 OK**
```json
{
  "status": "healthy",
  "service": "ai-navigator"
}
```

---

## Workflow Endpoints

The workflow API provides a conversational interface for deploying ML models.

### Start Workflow

Begin a new deployment workflow conversation.

```
POST /ai-navigator/workflow/start
```

**Request Body**
```json
{
  "user_id": "string",       // Required: Unique user identifier
  "initial_message": "string" // Required: User's initial request
}
```

**Example Request**
```json
{
  "user_id": "user-123",
  "initial_message": "I want to deploy a Llama 2 70B model for a chatbot"
}
```

**Response 200 OK**
```json
{
  "workflow_id": "uuid-string",
  "stage": "intent",
  "response": "I understand you want to deploy a Llama 2 70B model...\n\nI'll now help you define your traffic expectations."
}
```

**Response Fields**
| Field | Type | Description |
|-------|------|-------------|
| `workflow_id` | string | Unique workflow identifier for future calls |
| `stage` | string | Current workflow stage (intent, traffic, slo, etc.) |
| `response` | string | Assistant's response message |

---

### Continue Workflow

Continue an existing workflow with user input.

```
POST /ai-navigator/workflow/{workflow_id}/continue
```

**Path Parameters**
| Parameter | Type | Description |
|-----------|------|-------------|
| `workflow_id` | string | Workflow ID from start response |

**Request Body**
```json
{
  "message": "string"  // Required: User's message
}
```

**Example Request**
```json
{
  "message": "We expect about 100 requests per second with peaks of 500 during promotions"
}
```

**Response 200 OK**
```json
{
  "workflow_id": "uuid-string",
  "stage": "slo",
  "is_complete": false,
  "response": "I've captured your traffic expectations:\n- Average: 100 RPS\n- Peak: 500 RPS\n\nNow let's define your performance requirements..."
}
```

**Response Fields**
| Field | Type | Description |
|-------|------|-------------|
| `workflow_id` | string | Workflow identifier |
| `stage` | string | Current workflow stage |
| `is_complete` | boolean | Whether workflow has finished |
| `response` | string | Assistant's response |

**Workflow Stages**
| Stage | Description |
|-------|-------------|
| `intent` | Extracting model requirements |
| `traffic` | Defining traffic patterns |
| `slo` | Setting performance targets |
| `benchmark` | Fetching model benchmarks (auto) |
| `filter` | Selecting matching models (auto) |
| `replicas` | Calculating resource allocation |
| `deploy` | Generating deployment config |
| `monitor` | Verifying deployment |

---

### Get Workflow Status

Get current status of a workflow.

```
GET /ai-navigator/workflow/{workflow_id}/status
```

**Path Parameters**
| Parameter | Type | Description |
|-----------|------|-------------|
| `workflow_id` | string | Workflow ID |

**Response 200 OK**
```json
{
  "id": "uuid-string",
  "user_id": "user-123",
  "stage": "replicas",
  "stage_index": 6,
  "total_stages": 8,
  "is_complete": false,
  "message_count": 12
}
```

**Response 404 Not Found**
```json
{
  "detail": "Workflow not found"
}
```

---

## Quickstart Endpoints

Interactive tutorials for common tasks.

### List Quickstarts

Get all available quickstart tasks.

```
GET /ai-navigator/quickstarts
```

**Response 200 OK**
```json
[
  {
    "name": "deploy-model",
    "display_name": "Deploy a Model",
    "description": "Step-by-step guide to deploy an AI model on OpenShift AI"
  },
  {
    "name": "setup-project",
    "display_name": "Setup Data Science Project",
    "description": "Create and configure a new Data Science Project"
  },
  {
    "name": "test-inference",
    "display_name": "Test Inference Endpoint",
    "description": "Verify your model deployment is working correctly"
  },
  {
    "name": "serving-runtime",
    "display_name": "Configure Serving Runtime",
    "description": "Set up vLLM or other serving runtimes"
  },
  {
    "name": "data-connection",
    "display_name": "Setup Data Connection",
    "description": "Configure S3 or other data connections"
  }
]
```

---

### Start Quickstart

Begin a quickstart task.

```
POST /ai-navigator/quickstart/{task_name}
```

**Path Parameters**
| Parameter | Type | Description |
|-----------|------|-------------|
| `task_name` | string | Name of the quickstart task |

**Request Body**
```json
{
  "session_id": "string",          // Required: Session identifier
  "context": {                     // Optional: Initial context
    "namespace": "my-project"
  }
}
```

**Response 200 OK**
```json
{
  "task_name": "deploy-model",
  "status": "waiting_input",
  "message": "**Deploy a Model**\n\nStep-by-step guide to deploy an AI model.\n\nPlease provide:\n- **model_name**: Name of the model to deploy",
  "data": null,
  "next_prompt": "Provide the required values, or type 'cancel' to abort."
}
```

**Status Values**
| Status | Description |
|--------|-------------|
| `pending` | Task not started |
| `in_progress` | Task executing |
| `waiting_input` | Waiting for user input |
| `completed` | Task finished successfully |
| `failed` | Task failed |
| `cancelled` | Task was cancelled |

---

### Continue Quickstart

Continue a quickstart with user input.

```
POST /ai-navigator/quickstart/{task_name}/continue
```

**Path Parameters**
| Parameter | Type | Description |
|-----------|------|-------------|
| `task_name` | string | Name of the quickstart task |

**Request Body**
```json
{
  "message": "string"  // Required: User input (or "cancel" to abort)
}
```

**Response 200 OK**
```json
{
  "task_name": "deploy-model",
  "status": "completed",
  "message": "Model deployed successfully!",
  "data": {
    "endpoint_url": "https://model-service.apps.cluster.example.com"
  },
  "next_prompt": null
}
```

---

## Capacity Planning Endpoints

### Estimate Capacity

Calculate resource requirements for a model deployment.

```
POST /ai-navigator/capacity/estimate
```

**Request Body**
```json
{
  "model_name": "string",      // Required: Model identifier
  "traffic_profile": {         // Required: Traffic expectations
    "pattern": "steady",       // steady, burst, growth, variable
    "requests_per_second": 100,
    "peak_rps": 500,
    "average_input_tokens": 512,
    "average_output_tokens": 256,
    "concurrent_users": 50
  },
  "slo_requirements": {        // Required: Performance targets
    "p50_latency_ms": 500,
    "p95_latency_ms": 1000,
    "p99_latency_ms": 2000,
    "availability_percent": 99.9,
    "max_tokens_per_second": 10000
  }
}
```

**Response 200 OK**
```json
{
  "plan": {
    "id": "uuid-string",
    "created_at": "2024-01-15T10:30:00Z",
    "model_name": "llama-2-70b",
    "model_version": "1.0",
    "min_replicas": 3,
    "max_replicas": 8,
    "target_replicas": 4,
    "gpu_type": "A100-80GB",
    "gpu_count": 4,
    "gpu_memory_gb": 80,
    "memory_per_replica": "160Gi",
    "cpu_per_replica": "12",
    "estimated_throughput_tps": 480,
    "estimated_rps": 120,
    "estimated_p95_latency_ms": 250,
    "estimated_p99_latency_ms": 400,
    "meets_slo": true,
    "slo_violations": [],
    "estimated_monthly_cost": 23040.00,
    "alternatives": [
      {
        "gpu_type": "H100-80GB",
        "gpu_count": 2,
        "estimated_cost_per_hour": 16.00,
        "meets_slo": true,
        "headroom_percent": 35.0,
        "notes": ["~50% faster", "Significantly more memory available"]
      }
    ],
    "benchmark_source": "model_registry",
    "assumptions": [
      "Based on benchmark with 512 input tokens, 256 output tokens",
      "Includes 20% headroom for traffic spikes"
    ]
  }
}
```

**Capacity Plan Fields**
| Field | Type | Description |
|-------|------|-------------|
| `min_replicas` | integer | Minimum replicas for SLO compliance |
| `max_replicas` | integer | Maximum replicas for peak traffic |
| `target_replicas` | integer | Recommended steady-state replicas |
| `gpu_type` | string | GPU type (T4, L4, A10, A100-40GB, A100-80GB, H100-80GB) |
| `gpu_count` | integer | GPUs per replica |
| `meets_slo` | boolean | Whether configuration meets all SLO requirements |
| `slo_violations` | array | List of SLO violations if any |
| `estimated_monthly_cost` | number | Estimated monthly cost in USD |
| `alternatives` | array | Alternative GPU configurations |

---

### What-If Analysis

Analyze capacity under different scenarios.

```
POST /ai-navigator/capacity/whatif
```

**Request Body**
```json
{
  "plan_id": "uuid-string",    // Required: Original capacity plan ID
  "scenarios": [               // Required: Scenarios to analyze
    {
      "name": "Double Traffic",
      "description": "Holiday season traffic spike",
      "rps_multiplier": 2.0
    },
    {
      "name": "Tighter SLO",
      "description": "Stricter latency requirements",
      "p95_latency_ms": 500
    },
    {
      "name": "GPU Upgrade",
      "description": "Switch to H100 GPUs",
      "gpu_type": "H100-80GB"
    }
  ]
}
```

**Response 200 OK**
```json
{
  "results": [
    {
      "scenario": {
        "name": "Double Traffic",
        "rps_multiplier": 2.0
      },
      "original_plan": { /* original plan */ },
      "modified_plan": { /* new plan for scenario */ },
      "replica_delta": 4,
      "cost_delta_percent": 100.0,
      "latency_delta_percent": 0.0,
      "throughput_delta_percent": 100.0,
      "is_feasible": true,
      "warnings": [],
      "recommendations": ["Consider pre-scaling before traffic spike"]
    }
  ]
}
```

**Scenario Options**
| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Scenario name |
| `description` | string | Optional description |
| `rps_multiplier` | number | Multiply current RPS by this factor |
| `new_rps` | number | Set absolute new RPS |
| `p95_latency_ms` | integer | New p95 latency target |
| `p99_latency_ms` | integer | New p99 latency target |
| `gpu_type` | string | Change GPU type |
| `max_replicas` | integer | Change maximum replicas |

---

## Deployment Endpoints

### Preview Deployment

Generate and validate deployment manifests.

```
POST /ai-navigator/deploy/preview
```

**Request Body**
```json
{
  "config": {
    "inference_service": {
      "name": "my-model",
      "namespace": "my-project",
      "min_replicas": 2,
      "max_replicas": 5,
      "gpu_count": 1,
      "cpu": "4",
      "memory": "32Gi",
      "model_format": "pytorch",
      "storage_uri": "s3://bucket/model",
      "runtime": {
        "runtime_name": "vllm",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "dtype": "float16",
        "max_model_len": 4096
      },
      "labels": {},
      "annotations": {}
    },
    "create_hpa": true,
    "create_pdb": true
  }
}
```

**Response 200 OK**
```json
{
  "manifests": {
    "inferenceservice.yaml": "apiVersion: serving.kserve.io/v1beta1\nkind: InferenceService\n...",
    "hpa.yaml": "apiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler\n...",
    "pdb.yaml": "apiVersion: policy/v1\nkind: PodDisruptionBudget\n..."
  },
  "validation_passed": true,
  "validation_errors": [],
  "validation_warnings": ["Consider increasing memory for production use"]
}
```

**InferenceService Spec Fields**
| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Service name |
| `namespace` | string | Kubernetes namespace |
| `min_replicas` | integer | Minimum replicas |
| `max_replicas` | integer | Maximum replicas |
| `gpu_count` | integer | GPUs per replica |
| `cpu` | string | CPU request |
| `memory` | string | Memory request |
| `model_format` | string | Model format (pytorch, tensorflow) |
| `storage_uri` | string | Model storage location |
| `runtime` | object | Runtime configuration (vLLM settings) |
| `timeout_seconds` | integer | Request timeout |
| `scale_target` | integer | Target concurrency for autoscaling |
| `scale_metric` | string | Autoscaling metric |

**Runtime Config (vLLM)**
| Field | Type | Description |
|-------|------|-------------|
| `runtime_name` | string | Runtime name (vllm) |
| `tensor_parallel_size` | integer | Number of GPUs for tensor parallelism |
| `gpu_memory_utilization` | number | GPU memory fraction (0.0-1.0) |
| `dtype` | string | Data type (float16, bfloat16, float32) |
| `max_model_len` | integer | Maximum context length |
| `enforce_eager` | boolean | Disable CUDA graphs |
| `extra_args` | array | Additional vLLM arguments |
| `env_vars` | object | Environment variables |

---

### Execute Deployment

Deploy the generated manifests.

```
POST /ai-navigator/deploy/execute
```

**Request Body**
```json
{
  "config_id": "uuid-string",  // Required: Config ID from preview
  "dry_run": false             // Optional: Simulate deployment only
}
```

**Response 200 OK**
```json
{
  "result": {
    "deployment_id": "uuid-string",
    "status": "running",
    "message": "Deployment completed successfully",
    "endpoint_url": "https://my-model-my-project.apps.cluster.example.com",
    "ready_replicas": 2,
    "total_replicas": 2,
    "test_results": {
      "health_check": "passed",
      "inference_test": "passed",
      "latency_ms": 245
    }
  }
}
```

**Deployment Status Values**
| Status | Description |
|--------|-------------|
| `pending` | Deployment not started |
| `creating` | Resources being created |
| `running` | Deployment successful |
| `updating` | Deployment being updated |
| `failed` | Deployment failed |
| `deleted` | Deployment removed |

---

## Error Responses

All endpoints return errors in this format:

**4xx Client Errors**
```json
{
  "detail": "Error message describing the problem"
}
```

**5xx Server Errors**
```json
{
  "detail": "Internal server error"
}
```

**Common Error Codes**
| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource doesn't exist |
| 422 | Unprocessable Entity - Validation failed |
| 500 | Internal Server Error - Server-side error |

---

## Data Types

### Traffic Patterns

```
steady   - Consistent load throughout the day
burst    - Periodic spikes in traffic
growth   - Traffic increasing over time
variable - Unpredictable traffic patterns
```

### GPU Types

| Type | Memory | FP16 TFLOPS | Cost/Hour |
|------|--------|-------------|-----------|
| T4 | 16 GB | 65 | $0.35 |
| L4 | 24 GB | 121 | $0.70 |
| A10 | 24 GB | 125 | $1.00 |
| A100-40GB | 40 GB | 312 | $2.50 |
| A100-80GB | 80 GB | 312 | $4.00 |
| H100-80GB | 80 GB | 989 | $8.00 |

### Model Families

```
llama      - Meta Llama models
mistral    - Mistral AI models
granite    - IBM Granite models
phi        - Microsoft Phi models
falcon     - TII Falcon models
codellama  - Code-specialized Llama
starcoder  - Code generation models
```

---

## Rate Limits

No rate limits are currently enforced. In production:
- Consider implementing per-user rate limits
- Use OpenShift's rate limiting features for the Route

---

## Examples

### Complete Workflow Example

```bash
# 1. Start workflow
curl -X POST http://localhost:8080/ai-navigator/workflow/start \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "developer-1",
    "initial_message": "I need to deploy Llama 2 70B for a customer support chatbot"
  }'

# Response: {"workflow_id": "abc-123", "stage": "traffic", "response": "..."}

# 2. Continue - provide traffic info
curl -X POST http://localhost:8080/ai-navigator/workflow/abc-123/continue \
  -H "Content-Type: application/json" \
  -d '{"message": "About 50 RPS average, with peaks of 200 during business hours"}'

# 3. Continue - provide SLO
curl -X POST http://localhost:8080/ai-navigator/workflow/abc-123/continue \
  -H "Content-Type: application/json" \
  -d '{"message": "We need p95 latency under 1 second"}'

# 4. Continue - approve capacity plan
curl -X POST http://localhost:8080/ai-navigator/workflow/abc-123/continue \
  -H "Content-Type: application/json" \
  -d '{"message": "The plan looks good, lets deploy"}'

# 5. Continue - confirm deployment
curl -X POST http://localhost:8080/ai-navigator/workflow/abc-123/continue \
  -H "Content-Type: application/json" \
  -d '{"message": "Yes, deploy to the production namespace"}'
```

### Direct Capacity Planning

```bash
curl -X POST http://localhost:8080/ai-navigator/capacity/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "granite-3b-code-instruct",
    "traffic_profile": {
      "pattern": "steady",
      "requests_per_second": 200,
      "average_input_tokens": 1024,
      "average_output_tokens": 512
    },
    "slo_requirements": {
      "p50_latency_ms": 300,
      "p95_latency_ms": 800,
      "p99_latency_ms": 1500,
      "availability_percent": 99.95
    }
  }'
```

### Generate Deployment Manifests

```bash
curl -X POST http://localhost:8080/ai-navigator/deploy/preview \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "inference_service": {
        "name": "granite-code",
        "namespace": "ai-apps",
        "min_replicas": 2,
        "max_replicas": 10,
        "gpu_count": 1,
        "cpu": "8",
        "memory": "32Gi",
        "model_format": "pytorch",
        "storage_uri": "s3://models/granite-3b",
        "runtime": {
          "runtime_name": "vllm",
          "tensor_parallel_size": 1,
          "gpu_memory_utilization": 0.9,
          "dtype": "float16"
        }
      },
      "create_hpa": true,
      "create_pdb": true
    }
  }'
```
