# AI Navigator - Claude Code Guidelines

## Project Overview

AI Navigator is a Kubernetes-native Supervisor/Sub-Agent system for OpenShift AI capacity planning. It uses:
- **A2A Protocol** for agent-to-agent communication
- **MCP Protocol** for agent-to-tool access
- **Agent CRD & Operator** (Kopf-based) for lifecycle management
- **KServe RawDeployment** for Granite-4.0-H-Tiny (7B) model serving
- **TrustyAI GuardrailsOrchestrator** for safety enforcement

## Key Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run linting
ruff check src/ operator/ tests/
ruff format src/ operator/ tests/

# Type checking
mypy src/ operator/

# Start supervisor locally
uvicorn ai_navigator.agents.supervisor.agent:app --port 8000

# Run operator locally
kopf run operator/main.py --verbose
```

## Architecture

```
User / MCP Client
        │
        ▼ A2A
┌─────────────────────────────────────┐
│       SUPERVISOR AGENT              │
│   "RHOAI Deployment Orchestrator"   │
└──────────────────┬──────────────────┘
                   │ A2A Task Delegation
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ MODEL       │ │ RESOURCE    │ │ DEPLOYMENT  │
│ CATALOG     │ │ PROVISIONING│ │ MONITOR     │
│ SUB-AGENT   │ │ SUB-AGENT   │ │ SUB-AGENT   │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │ MCP           │ MCP           │ MCP
       ▼               ▼               ▼
   Tools/APIs      Tools/APIs      Tools/APIs
```

## Directory Structure

- `src/ai_navigator/a2a/` - A2A Protocol core (BaseAgent, messages, tasks)
- `src/ai_navigator/agents/` - Agent implementations (supervisor, sub-agents)
- `src/ai_navigator/mcp/` - MCP tool servers
- `src/ai_navigator/llm/` - LLM integration for decomposition/aggregation
- `operator/` - Kopf-based Kubernetes operator
- `manifests/` - Kubernetes manifests (CRDs, deployments, etc.)

## Development Principles

1. **A2A Protocol Compliance**: All agents must expose `/.well-known/agent.json`
2. **Skill-Based Design**: Each agent capability is a discrete skill
3. **Kubernetes-Native**: Use CRDs for agent lifecycle management
4. **Observability**: Every agent exposes `/healthz` and `/metrics`
5. **Type Safety**: Use Pydantic models for all data structures

## Hardware Target

- GPU: NVIDIA T4 (16GB VRAM)
- Model: granite-4.0-h-tiny (7B parameters)
- Precision: float16 (T4 Turing architecture)
- Max Context: ~16K tokens

## Key Dependencies

- `a2a-sdk>=0.3.22` - A2A Protocol SDK
- `fastapi>=0.115.0` - Web framework
- `kopf>=1.38.0` - Kubernetes operator framework
- `mcp>=1.6.0` - Model Context Protocol
- `pydantic>=2.10.0` - Data validation
