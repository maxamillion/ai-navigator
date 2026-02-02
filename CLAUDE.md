# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Navigator is an AI-powered agent for OpenShift AI capacity planning and deployment. It provides conversational workflow guidance for deploying ML models on OpenShift AI, with SLO-driven capacity planning, benchmark integration, and Kubernetes manifest generation.

## Build and Test Commands

```bash
# Install dependencies (using uv or pip)
uv pip install -e ".[dev]"
# or
pip install -e ".[dev]"

# Run all tests with coverage
pytest

# Run a single test file
pytest tests/unit/test_capacity.py

# Run a specific test
pytest tests/unit/test_capacity.py::TestCapacityPlanner::test_calculate_capacity_plan

# Lint and format
ruff check src tests
ruff format src tests

# Type checking
mypy src
```

## Architecture

### Core Flow: 8-Stage Workflow Engine

The system implements an 8-stage conversational workflow in `src/ai_navigator/workflow/`:

1. **INTENT** - Extract requirements from natural language
2. **TRAFFIC** - Determine traffic patterns (steady, burst, growth)
3. **SLO** - Set latency targets (p50, p95, p99)
4. **BENCHMARK** - Query model performance data from registry
5. **FILTER** - Select models meeting requirements
6. **REPLICAS** - Calculate GPU resources and replicas
7. **DEPLOY** - Generate Kubernetes manifests
8. **MONITOR** - Verify deployment and test endpoints

The `WorkflowEngine` (`workflow/engine.py`) orchestrates stage transitions. Each stage has its own handler in `workflow/stages/`. State is managed via `StateManager` with pluggable backends (memory or postgres).

### Key Subsystems

**MCP Integration** (`src/ai_navigator/mcp/`):
- `MCPClient` connects to rhoai-mcp server for OpenShift AI operations
- `MCPOrchestrator` handles multi-tool calls with dependency management (parallel, sequential, DAG execution)

**Model Registry** (`src/ai_navigator/registry/`):
- REST client for OpenShift AI Model Registry v1alpha3 API
- Fetches model metadata, versions, artifacts, and benchmark data

**Capacity Planning** (`src/ai_navigator/planning/`):
- `CapacityPlanner` calculates replicas, GPU requirements, and costs from SLO requirements and benchmarks
- `WhatIfAnalyzer` enables scenario analysis (traffic growth, GPU changes, SLO tightening)
- GPU catalog with specs for T4, L4, A10, A100, H100

**Deployment** (`src/ai_navigator/deployment/`):
- YAML generator for InferenceService manifests
- Validation and orchestration for Kubernetes deployments

**Quickstarts** (`src/ai_navigator/quickstarts/`):
- Interactive task engine for guided tutorials (deploy model, setup project, test inference)

### API Layer

FastAPI router in `src/ai_navigator/router.py` exposes:
- `/ai-navigator/workflow/*` - Conversational workflow endpoints
- `/ai-navigator/capacity/*` - Capacity estimation and what-if analysis
- `/ai-navigator/deploy/*` - Deployment preview and execution
- `/ai-navigator/quickstarts` - Guided tutorials

### Configuration

Settings via pydantic-settings with environment variables:
- `AI_NAVIGATOR_*` - Main app settings
- `RHOAI_MCP_*` - MCP server connection
- `MODEL_REGISTRY_*` - Model Registry URL
- `LLM_*` - LLM provider for conversation (OpenAI-compatible)
- `STATE_*` - State backend (memory/postgres)

### Key Models

- `WorkflowState` - Complete session state including stage, traffic profile, SLO requirements
- `TrafficProfile` - RPS, burst patterns, token counts
- `SLORequirements` - Latency percentiles, availability targets
- `CapacityPlan` - Calculated replicas, GPU allocation, cost estimates
- `BenchmarkData` - Model performance metrics from registry
