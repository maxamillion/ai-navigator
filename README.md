# AI Navigator

Kubernetes-native Supervisor/Sub-Agent system for OpenShift AI capacity planning.

## Overview

AI Navigator provides intelligent model deployment orchestration for OpenShift AI using a multi-agent architecture:

- **Supervisor Agent**: Orchestrates user requests, decomposes tasks, and delegates to sub-agents
- **Model Catalog Agent**: Queries model registry, retrieves benchmarks and TrustyAI scores
- **Resource Provisioning Agent**: Generates deployment configs, estimates costs, validates SLOs
- **Deployment Monitor Agent**: Monitors deployments, queries metrics, checks SLO violations

## Architecture

The system uses two key protocols:
- **A2A (Agent-to-Agent)**: For inter-agent communication and task delegation
- **MCP (Model Context Protocol)**: For agent access to external tools and APIs

Agents are managed as Kubernetes custom resources via a Kopf-based operator.

## Quick Start

### Prerequisites

- Python 3.11+
- OpenShift 4.15+ with OpenShift AI 3.2
- NVIDIA T4 GPU (for model serving)

### Installation

```bash
# Clone the repository
git clone https://github.com/redhat-et/ai-navigator.git
cd ai-navigator

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/unit/ -v
```

### Local Development

```bash
# Start the supervisor agent
uvicorn ai_navigator.agents.supervisor.agent:app --port 8000

# Test the agent card endpoint
curl http://localhost:8000/.well-known/agent.json
```

### OpenShift Deployment

```bash
# Apply the Agent CRD
oc apply -f manifests/crds/agent-crd.yaml

# Deploy the operator
oc apply -f manifests/operator/

# Deploy agents
oc apply -f manifests/agents/

# Verify deployment
curl http://supervisor-route/.well-known/agent.json
```

## Configuration

Configuration is managed via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_NAME` | Agent identifier | `supervisor` |
| `AGENT_PORT` | HTTP port | `8000` |
| `LLM_ENDPOINT` | LLM inference endpoint | Required |
| `LLM_MODEL` | Model name | `granite-4.0-h-tiny` |
| `KUBERNETES_NAMESPACE` | Deployment namespace | `ai-navigator` |

## API

All agents expose A2A-compliant endpoints:

- `GET /.well-known/agent.json` - Agent card with capabilities
- `POST /` - JSON-RPC message handling
- `GET /healthz` - Health check
- `GET /metrics` - Prometheus metrics

### Example Request

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "Recommend a model for text classification"}]
      }
    }
  }'
```

## Development

```bash
# Run linting
ruff check src/ operator/ tests/
ruff format src/ operator/ tests/

# Type checking
mypy src/ operator/

# Run all tests
pytest tests/ -v --cov=src/ai_navigator
```

## License

Apache License 2.0
