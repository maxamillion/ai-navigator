# AI Navigator Developer Guide

This guide helps new developers get up to speed with contributing to AI Navigator.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Code Organization](#code-organization)
4. [Testing Guide](#testing-guide)
5. [API Development](#api-development)
6. [Adding New Features](#adding-new-features)
7. [Common Tasks](#common-tasks)
8. [Debugging](#debugging)
9. [Code Style](#code-style)
10. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- Python 3.11+
- `uv` package manager (recommended) or `pip`
- Docker/Podman (for container builds)
- Access to OpenShift cluster (for integration testing)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/your-org/ai-navigator.git
cd ai-navigator

# Create virtual environment (uv recommended)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Verify installation
pytest --version
ruff --version
mypy --version
```

### Running Locally

```bash
# Start the development server
uvicorn ai_navigator.router:router --reload --port 8080

# The API is now available at http://localhost:8080/ai-navigator/
```

### Environment Variables

Create a `.env` file for local development:

```bash
# .env
AI_NAVIGATOR_ENVIRONMENT=development
AI_NAVIGATOR_DEBUG=true

# MCP Server (optional for workflow stages to work fully)
RHOAI_MCP_HOST=localhost
RHOAI_MCP_PORT=8081

# Model Registry (optional)
MODEL_REGISTRY_URL=http://localhost:8082

# State backend (memory for development)
STATE_BACKEND=memory

# LLM (optional - for enhanced NLP)
LLM_BASE_URL=http://localhost:8083/v1
LLM_API_KEY=your-key
LLM_MODEL_NAME=granite-3b-code-instruct
```

---

## Development Workflow

### Daily Development Cycle

1. **Pull latest changes**
   ```bash
   git pull origin main
   ```

2. **Run tests to verify clean state**
   ```bash
   pytest
   ```

3. **Make changes**

4. **Run linting and type checking**
   ```bash
   ruff check src tests
   ruff format src tests
   mypy src
   ```

5. **Run tests**
   ```bash
   pytest
   ```

6. **Commit with descriptive message**
   ```bash
   git add -A
   git commit -m "Add feature X to workflow engine"
   ```

### Branch Strategy

- `main` - Production-ready code
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Pre-commit Hooks

Install pre-commit hooks for automatic linting:

```bash
pre-commit install
```

This runs ruff and mypy before each commit.

---

## Code Organization

### Module Overview

```
src/ai_navigator/
├── config.py           # All configuration in one place
├── router.py           # FastAPI endpoints - API surface
├── models/             # Data models (Pydantic)
├── state/              # State persistence
├── workflow/           # Core workflow engine
├── mcp/                # MCP integration
├── registry/           # Model Registry client
├── planning/           # Capacity planning algorithms
├── deployment/         # Kubernetes manifest generation
└── quickstarts/        # Interactive tutorials
```

### Key Files to Understand

| File | Purpose | Priority |
|------|---------|----------|
| `router.py` | All API endpoints | High |
| `workflow/engine.py` | Workflow orchestration | High |
| `models/workflow.py` | Core data models | High |
| `config.py` | Configuration | Medium |
| `planning/capacity.py` | Capacity algorithms | Medium |
| `mcp/client.py` | MCP integration | Medium |

### Import Conventions

```python
# Standard library
from datetime import datetime
from typing import Optional

# Third-party
from fastapi import APIRouter
from pydantic import BaseModel
import structlog

# Local imports (use absolute imports)
from ai_navigator.models.workflow import WorkflowState
from ai_navigator.workflow.engine import WorkflowEngine
```

---

## Testing Guide

### Test Structure

```
tests/
└── unit/
    ├── test_capacity.py     # Capacity planner tests
    ├── test_deployment.py   # Deployment generator tests
    ├── test_models.py       # Data model validation tests
    └── test_state.py        # State management tests
```

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/unit/test_capacity.py

# Run specific test class
pytest tests/unit/test_capacity.py::TestCapacityPlanner

# Run specific test
pytest tests/unit/test_capacity.py::TestCapacityPlanner::test_calculate_capacity_plan

# Run with verbose output
pytest -v

# Run with print statements visible
pytest -s

# Run only tests matching pattern
pytest -k "capacity"
```

### Writing Tests

Follow these patterns from the existing test suite:

```python
"""Tests for my_module."""

import pytest
from ai_navigator.my_module import MyClass


class TestMyClass:
    """Test suite for MyClass."""

    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return MyClass()

    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        return {"key": "value"}

    def test_method_does_something(self, instance, sample_data):
        """Test that method produces expected result."""
        result = instance.method(sample_data)

        assert result.field == "expected"

    def test_method_handles_edge_case(self, instance):
        """Test edge case handling."""
        result = instance.method({})

        assert result.is_valid is False
```

### Test-Driven Development

The project follows TDD principles:

1. **Write the test first**
   ```python
   def test_new_feature(self, instance):
       result = instance.new_feature("input")
       assert result == "expected"
   ```

2. **Run the test (should fail)**
   ```bash
   pytest tests/unit/test_my_module.py::TestMyClass::test_new_feature
   ```

3. **Implement the feature**

4. **Run the test (should pass)**

5. **Refactor if needed**

### Fixtures

Use fixtures for common setup:

```python
@pytest.fixture
def planner():
    """Create capacity planner."""
    return CapacityPlanner()

@pytest.fixture
def benchmark():
    """Create sample benchmark data."""
    return BenchmarkData(
        model_name="llama-2-7b",
        model_version="1.0",
        gpu_type="A100-40GB",
        gpu_count=1,
        p50_latency_ms=150,
        p95_latency_ms=250,
        p99_latency_ms=400,
        tokens_per_second=120,
        requests_per_second=15,
        gpu_memory_gb=14,
        gpu_utilization_percent=85,
    )
```

### Async Tests

Use `pytest-asyncio` for async tests:

```python
import pytest

class TestAsyncFeature:
    @pytest.fixture
    async def async_resource(self):
        """Create async resource."""
        client = AsyncClient()
        yield client
        await client.close()

    async def test_async_operation(self, async_resource):
        """Test async operation."""
        result = await async_resource.fetch_data()
        assert result is not None
```

### Mocking External Services

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestWithMocking:
    async def test_mcp_call(self):
        """Test with mocked MCP client."""
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = {"status": "success"}

        with patch("ai_navigator.mcp.client.MCPClient", return_value=mock_client):
            # Test code that uses MCPClient
            pass
```

---

## API Development

### Adding a New Endpoint

1. **Define request/response models in `router.py`**:

```python
class MyRequest(BaseModel):
    """Request for my endpoint."""
    field: str = Field(description="Field description")

class MyResponse(BaseModel):
    """Response from my endpoint."""
    result: str
```

2. **Create the endpoint**:

```python
@router.post("/my-endpoint", response_model=MyResponse)
async def my_endpoint(
    request: MyRequest,
    engine=Depends(get_workflow_engine),
) -> MyResponse:
    """My endpoint description."""
    result = await engine.do_something(request.field)
    return MyResponse(result=result)
```

3. **Add error handling**:

```python
from fastapi import HTTPException

@router.post("/my-endpoint")
async def my_endpoint(request: MyRequest):
    try:
        result = await do_operation(request)
        return MyResponse(result=result)
    except OperationError as e:
        logger.error("Operation failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal error")
```

### Dependency Injection

Use FastAPI's `Depends` for dependency injection:

```python
async def get_my_service():
    """Factory for MyService."""
    return MyService()

@router.get("/data")
async def get_data(service=Depends(get_my_service)):
    return await service.fetch()
```

---

## Adding New Features

### Adding a Workflow Stage

1. **Create the stage handler** (`workflow/stages/newstage.py`):

```python
"""New stage - Step N."""

from ai_navigator.models.workflow import WorkflowState
from ai_navigator.workflow.stages.base import BaseStage


class StageResult:
    def __init__(self, success, message, advance=False, data=None, prompt_user=None):
        self.success = success
        self.message = message
        self.advance = advance
        self.data = data or {}
        self.prompt_user = prompt_user


class NewStage(BaseStage):
    """Stage N: Description of what this stage does."""

    async def process(self, state: WorkflowState, user_input: str) -> StageResult:
        """Process user input for this stage."""
        if not user_input.strip():
            return StageResult(
                success=True,
                message="Introduction message",
                prompt_user="What information do you need?",
            )

        # Extract and validate data from user input
        extracted = self._extract_data(user_input)

        # Update state
        state.new_field = extracted

        # Check if ready to advance
        if self._is_complete(extracted):
            return StageResult(
                success=True,
                message="Got it! Moving to next stage.",
                advance=True,
                data=extracted,
            )

        return StageResult(
            success=True,
            message="Partial data received.",
            prompt_user="Please also provide X.",
        )

    def _extract_data(self, text: str) -> dict:
        """Extract relevant data from user input."""
        # Implementation
        return {}

    def _is_complete(self, data: dict) -> bool:
        """Check if all required data is present."""
        return True
```

2. **Update the enum** (`models/workflow.py`):

```python
class WorkflowStage(str, Enum):
    INTENT = "intent"
    TRAFFIC = "traffic"
    SLO = "slo"
    BENCHMARK = "benchmark"
    FILTER = "filter"
    NEW_STAGE = "new_stage"  # Add here
    REPLICAS = "replicas"
    DEPLOY = "deploy"
    MONITOR = "monitor"
```

3. **Register in engine** (`workflow/engine.py`):

```python
from ai_navigator.workflow.stages.newstage import NewStage

class WorkflowEngine:
    def __init__(self, ..., new_stage: Optional[NewStage] = None):
        self._stages = {
            # ...
            WorkflowStage.NEW_STAGE: new_stage or NewStage(),
            # ...
        }
```

4. **Write tests** (`tests/unit/test_newstage.py`):

```python
import pytest
from ai_navigator.workflow.stages.newstage import NewStage
from ai_navigator.models.workflow import WorkflowState

class TestNewStage:
    @pytest.fixture
    def stage(self):
        return NewStage()

    @pytest.fixture
    def state(self):
        return WorkflowState(user_id="test-user")

    async def test_initial_prompt(self, stage, state):
        result = await stage.process(state, "")
        assert result.success
        assert result.prompt_user is not None

    async def test_extracts_data(self, stage, state):
        result = await stage.process(state, "user input with data")
        assert result.success
        assert result.advance
```

### Adding a Quickstart Task

1. **Create the task** (`quickstarts/tasks/mytask.py`):

```python
"""My quickstart task."""

from typing import Any, Optional

from ai_navigator.quickstarts.engine import (
    QuickstartTask,
    QuickstartResult,
    QuickstartStep,
    QuickstartStatus,
)


class MyTask(QuickstartTask):
    """Guides user through doing something."""

    @property
    def name(self) -> str:
        return "my-task"

    @property
    def display_name(self) -> str:
        return "My Task"

    @property
    def description(self) -> str:
        return "Step-by-step guide to do something"

    def get_steps(self) -> list[QuickstartStep]:
        return [
            QuickstartStep(
                id="step-1",
                name="First Step",
                description="Do the first thing",
            ),
            QuickstartStep(
                id="step-2",
                name="Second Step",
                description="Do the second thing",
            ),
        ]

    def get_required_inputs(self) -> list[dict[str, str]]:
        return [
            {"name": "namespace", "description": "Target namespace"},
        ]

    async def execute(
        self,
        context: dict[str, Any],
        user_input: Optional[str] = None,
    ) -> QuickstartResult:
        current_step = context.get("current_step", 0)
        steps = self.get_steps()

        # Process based on current step
        if current_step == 0:
            # Execute step 1
            steps[0].status = QuickstartStatus.COMPLETED
            context["current_step"] = 1
            return QuickstartResult(
                task_name=self.name,
                status=QuickstartStatus.IN_PROGRESS,
                steps=steps,
                message="Step 1 completed!",
                next_prompt="Ready for step 2?",
            )

        # Complete
        for step in steps:
            step.status = QuickstartStatus.COMPLETED

        return QuickstartResult(
            task_name=self.name,
            status=QuickstartStatus.COMPLETED,
            steps=steps,
            message="All done!",
        )
```

2. **Register the task** (`router.py`):

```python
async def get_quickstart_engine():
    from ai_navigator.quickstarts.tasks.mytask import MyTask

    engine = QuickstartEngine()
    # ... existing tasks
    engine.register_task(MyTask())
    return engine
```

### Adding a New GPU Type

Edit `planning/capacity.py`:

```python
GPU_CATALOG = {
    # ... existing GPUs
    "NEW-GPU-48GB": {
        "memory_gb": 48,
        "fp16_tflops": 350,
        "cost_per_hour": 3.00,
        "availability": "medium",
    },
}
```

---

## Common Tasks

### Updating Data Models

When updating Pydantic models:

1. **Add new fields with defaults** (backward compatible):
   ```python
   class WorkflowState(BaseModel):
       # ... existing fields
       new_field: Optional[str] = Field(default=None, description="New field")
   ```

2. **Update serialization if needed** in state stores

3. **Update tests** to cover new fields

### Adding Configuration Options

1. **Add to settings class** (`config.py`):
   ```python
   class Settings(BaseSettings):
       new_option: str = Field(default="value", description="My option")
   ```

2. **Environment variable** will be `AI_NAVIGATOR_NEW_OPTION`

### Generating Kubernetes Manifests

Use `YAMLGenerator` in `deployment/generator.py`:

```python
from ai_navigator.deployment.generator import YAMLGenerator
from ai_navigator.models.deployment import DeploymentConfig, InferenceServiceSpec

generator = YAMLGenerator()
spec = InferenceServiceSpec(
    name="my-model",
    namespace="my-namespace",
    # ... other fields
)
config = DeploymentConfig(inference_service=spec)
manifests = generator.generate_all(config)
# manifests["inferenceservice.yaml"], manifests["hpa.yaml"], etc.
```

---

## Debugging

### Logging

Use structlog throughout:

```python
import structlog

logger = structlog.get_logger(__name__)

def my_function(data):
    logger.info("Processing data", data_id=data.id)
    try:
        result = process(data)
        logger.debug("Processing complete", result=result)
        return result
    except Exception as e:
        logger.error("Processing failed", error=str(e), data_id=data.id)
        raise
```

### Debug Mode

Enable debug mode for verbose logging:

```bash
AI_NAVIGATOR_DEBUG=true uvicorn ai_navigator.router:router --reload
```

### Interactive Debugging

Use `pdb` or `ipdb`:

```python
def problematic_function():
    import pdb; pdb.set_trace()  # Execution pauses here
    # or
    import ipdb; ipdb.set_trace()  # Better interface
```

### Testing API Manually

```bash
# Start server
uvicorn ai_navigator.router:router --reload --port 8080

# Health check
curl http://localhost:8080/ai-navigator/health

# Start workflow
curl -X POST http://localhost:8080/ai-navigator/workflow/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "initial_message": "I need a Llama 2 model for chat"}'

# Continue workflow
curl -X POST http://localhost:8080/ai-navigator/workflow/{workflow_id}/continue \
  -H "Content-Type: application/json" \
  -d '{"message": "About 100 requests per second"}'
```

---

## Code Style

### Python Style (enforced by ruff)

- Follow PEP 8
- Line length: 100 characters
- Use type hints everywhere
- Use `pathlib.Path` for file paths
- Prefer comprehensions over loops

### Docstrings

```python
def calculate_replicas(
    benchmark: BenchmarkData,
    target_rps: float,
    headroom_percent: float = 20.0,
) -> int:
    """Calculate replica count for target RPS.

    Args:
        benchmark: Performance benchmark data for the model.
        target_rps: Target requests per second.
        headroom_percent: Extra capacity headroom (default 20%).

    Returns:
        Minimum number of replicas needed.

    Raises:
        ValueError: If benchmark has invalid RPS data.
    """
    pass
```

### Error Handling

```python
# Custom exceptions in module
class MyModuleError(Exception):
    """Base exception for my_module."""
    pass

class SpecificError(MyModuleError):
    """Specific error condition."""
    def __init__(self, message: str, details: dict):
        super().__init__(message)
        self.details = details

# Usage
def my_function():
    try:
        result = risky_operation()
    except ExternalError as e:
        raise SpecificError("Operation failed", {"original": str(e)}) from e
```

### Async Patterns

```python
# Use async context managers
async with MCPClient() as client:
    result = await client.call_tool("tool_name", args)

# Gather for parallel operations
results = await asyncio.gather(
    operation1(),
    operation2(),
    operation3(),
)

# Handle exceptions in gather
results = await asyncio.gather(
    operation1(),
    operation2(),
    return_exceptions=True,
)
for result in results:
    if isinstance(result, Exception):
        logger.error("Operation failed", error=str(result))
```

---

## Troubleshooting

### Common Issues

**Import errors**
```bash
# Ensure package is installed in editable mode
uv pip install -e ".[dev]"
```

**Type checking errors**
```bash
# Check specific file
mypy src/ai_navigator/my_module.py

# Ignore specific error (use sparingly)
result: Any = untyped_function()  # type: ignore
```

**Test failures**
```bash
# Run with verbose output
pytest -v tests/unit/test_file.py

# Run with debugging on failure
pytest --pdb tests/unit/test_file.py
```

**Circular imports**
```python
# Use TYPE_CHECKING guard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_navigator.other_module import OtherClass

def my_function(obj: "OtherClass") -> None:  # Quote the type hint
    pass
```

**Async issues**
```bash
# Ensure pytest-asyncio is installed
pytest --version
# Should show pytest-asyncio plugin

# Check asyncio_mode in pyproject.toml
# [tool.pytest.ini_options]
# asyncio_mode = "auto"
```

### Getting Help

1. Check existing tests for usage examples
2. Read the docstrings in source code
3. Check `CLAUDE.md` for project guidelines
4. Look at `docs/ARCHITECTURE.md` for design decisions
5. Ask team members in Slack/Teams

---

## Next Steps

After reading this guide:

1. **Explore the codebase** - Read through key files listed above
2. **Run the tests** - `pytest` to see how the system works
3. **Try the API** - Start the server and test endpoints manually
4. **Pick a small task** - Start with a bug fix or documentation update
5. **Ask questions** - Don't hesitate to ask for clarification
