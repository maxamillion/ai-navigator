# AI Navigator

AI-powered agent for OpenShift AI capacity planning and deployment. AI Navigator provides a conversational interface that guides users through deploying ML models on OpenShift AI with SLO-driven capacity planning.

## Features

- **Conversational Workflow**: Natural language interface for deployment planning
- **SLO-Driven Capacity Planning**: Calculate GPU resources based on latency and throughput requirements
- **Benchmark Integration**: Use Model Registry performance data for accurate sizing
- **Kubernetes Manifest Generation**: Produce ready-to-deploy InferenceService YAML
- **What-If Analysis**: Explore capacity under different traffic and SLO scenarios
- **Interactive Quickstarts**: Step-by-step tutorials for common tasks

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ai-navigator.git
cd ai-navigator

# Install with uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Running the Server

```bash
# Development server with auto-reload
uvicorn ai_navigator.router:router --reload --port 8080

# The API is available at http://localhost:8080/ai-navigator/
```

### Verify Installation

```bash
# Health check
curl http://localhost:8080/ai-navigator/health

# Expected response:
# {"status": "healthy", "service": "ai-navigator"}
```

## Usage

### Starting a Deployment Workflow

```bash
# Start a new workflow
curl -X POST http://localhost:8080/ai-navigator/workflow/start \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "developer-1",
    "initial_message": "I need to deploy Llama 2 70B for a chatbot"
  }'
```

The workflow guides you through 8 stages:
1. **Intent** - Describe what model you need
2. **Traffic** - Define expected request patterns
3. **SLO** - Set performance targets
4. **Benchmark** - Fetch model performance data (automatic)
5. **Filter** - Select matching models (automatic)
6. **Replicas** - Review capacity plan
7. **Deploy** - Generate and apply manifests
8. **Monitor** - Verify deployment

### Direct Capacity Planning

```bash
curl -X POST http://localhost:8080/ai-navigator/capacity/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "llama-2-70b",
    "traffic_profile": {
      "pattern": "steady",
      "requests_per_second": 100
    },
    "slo_requirements": {
      "p50_latency_ms": 500,
      "p95_latency_ms": 1000,
      "p99_latency_ms": 2000
    }
  }'
```

### Running Quickstarts

```bash
# List available quickstarts
curl http://localhost:8080/ai-navigator/quickstarts

# Start a quickstart
curl -X POST http://localhost:8080/ai-navigator/quickstart/deploy-model \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session"}'
```

## Development

### Prerequisites

- Python 3.11+
- uv or pip

### Setup

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check src tests
ruff format src tests

# Type checking
mypy src
```

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/unit/test_capacity.py

# Run specific test
pytest tests/unit/test_capacity.py::TestCapacityPlanner::test_calculate_capacity_plan

# Run with verbose output
pytest -v
```

### Project Structure

```
ai-navigator/
├── src/ai_navigator/
│   ├── config.py           # Configuration management
│   ├── router.py           # FastAPI endpoints
│   ├── models/             # Pydantic data models
│   │   ├── workflow.py     # Workflow state, traffic, SLO
│   │   ├── capacity.py     # Benchmarks, capacity plans
│   │   └── deployment.py   # Deployment configurations
│   ├── workflow/           # 8-stage workflow engine
│   │   ├── engine.py       # Workflow orchestrator
│   │   └── stages/         # Stage handlers
│   ├── planning/           # Capacity planning
│   │   ├── capacity.py     # Core planning algorithms
│   │   ├── recommender.py  # Model/GPU recommendations
│   │   └── whatif.py       # Scenario analysis
│   ├── mcp/                # MCP integration
│   ├── registry/           # Model Registry client
│   ├── deployment/         # Kubernetes manifest generation
│   ├── quickstarts/        # Interactive tutorials
│   └── state/              # State persistence
├── tests/
│   └── unit/               # Unit tests
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # System architecture
│   ├── DEVELOPER_GUIDE.md  # Developer onboarding
│   └── API_REFERENCE.md    # API documentation
├── deploy/kubernetes/      # Kubernetes manifests
├── pyproject.toml          # Project configuration
├── Containerfile           # Container build
└── CLAUDE.md               # AI assistant guidelines
```

## Configuration

AI Navigator is configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_NAVIGATOR_ENVIRONMENT` | Environment (development/staging/production) | development |
| `AI_NAVIGATOR_DEBUG` | Enable debug mode | false |
| `RHOAI_MCP_HOST` | MCP server host | localhost |
| `RHOAI_MCP_PORT` | MCP server port | 8080 |
| `MODEL_REGISTRY_URL` | Model Registry URL | (cluster internal) |
| `STATE_BACKEND` | State backend (memory/postgres) | memory |
| `STATE_POSTGRES_DSN` | PostgreSQL connection string | - |
| `LLM_BASE_URL` | LLM API endpoint | (cluster internal) |
| `LLM_API_KEY` | LLM API key | - |
| `LLM_MODEL_NAME` | LLM model name | granite-3b-code-instruct |

## Container Build

```bash
# Build container image
podman build -t ai-navigator:latest -f Containerfile .

# Run container
podman run -p 8080:8080 ai-navigator:latest
```

## Deployment

Deploy to OpenShift using the provided Kubernetes manifests:

```bash
cd deploy/kubernetes

# Apply manifests
oc apply -k .

# Or apply individually
oc apply -f namespace.yaml
oc apply -f configmap.yaml
oc apply -f secret.yaml
oc apply -f deployment.yaml
oc apply -f service.yaml
oc apply -f route.yaml
```

## API Reference

See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for complete API documentation.

### Key Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /ai-navigator/health` | Health check |
| `POST /ai-navigator/workflow/start` | Start deployment workflow |
| `POST /ai-navigator/workflow/{id}/continue` | Continue workflow |
| `GET /ai-navigator/workflow/{id}/status` | Get workflow status |
| `POST /ai-navigator/capacity/estimate` | Calculate capacity plan |
| `POST /ai-navigator/capacity/whatif` | What-if analysis |
| `POST /ai-navigator/deploy/preview` | Preview deployment manifests |
| `GET /ai-navigator/quickstarts` | List quickstart tutorials |

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

### Key Components

- **WorkflowEngine**: Orchestrates the 8-stage deployment workflow
- **CapacityPlanner**: Calculates GPU resources and replicas from SLO requirements
- **MCPClient**: Integrates with rhoai-mcp for OpenShift AI operations
- **YAMLGenerator**: Produces KServe InferenceService manifests
- **QuickstartEngine**: Manages interactive tutorial sessions

### Technology Stack

- **Python 3.11+** with async/await throughout
- **FastAPI** for REST API
- **Pydantic v2** for data validation
- **httpx** for async HTTP client
- **structlog** for structured logging
- **PostgreSQL** for production state persistence

## Contributing

See [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for development guidelines.

### Development Principles

1. **Test-Driven Development**: Write tests first
2. **Type Safety**: Use type hints everywhere (mypy --strict)
3. **Async First**: All I/O operations are async
4. **Simplicity**: Prefer the simplest solution that works
5. **Clean Code**: Follow PEP 8 via ruff

### Adding Features

- **New Workflow Stage**: See [Developer Guide](docs/DEVELOPER_GUIDE.md#adding-a-workflow-stage)
- **New Quickstart**: See [Developer Guide](docs/DEVELOPER_GUIDE.md#adding-a-quickstart-task)
- **New GPU Type**: Edit `GPU_CATALOG` in `planning/capacity.py`

## Support

- Report issues at [GitHub Issues](https://github.com/your-org/ai-navigator/issues)
- Check [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for troubleshooting

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.
