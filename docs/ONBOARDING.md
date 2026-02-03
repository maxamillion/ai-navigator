# AI Navigator Onboarding Guide

Welcome to the AI Navigator maintenance team. This guide helps you get productive quickly.

## Day 1: Setup and Orientation

### 1. Environment Setup (30 min)

```bash
# Clone and install
git clone https://github.com/your-org/ai-navigator.git
cd ai-navigator
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify setup
pytest --version    # pytest 7.4+
ruff --version      # ruff 0.1+
mypy --version      # mypy 1.8+
```

### 2. Run Tests (10 min)

```bash
# All tests should pass
pytest

# View coverage report
pytest --cov=ai_navigator --cov-report=html
open htmlcov/index.html
```

### 3. Start the Server (5 min)

```bash
uvicorn ai_navigator.router:router --reload --port 8080

# Test it works
curl http://localhost:8080/ai-navigator/health
```

### 4. Read Key Documentation (1 hour)

| Document | Priority | Purpose |
|----------|----------|---------|
| [README.md](../README.md) | High | Project overview |
| [ARCHITECTURE.md](ARCHITECTURE.md) | High | System design |
| [CLAUDE.md](../CLAUDE.md) | High | Development principles |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Medium | Development workflow |
| [API_REFERENCE.md](API_REFERENCE.md) | Medium | API endpoints |
| [MODULE_REFERENCE.md](MODULE_REFERENCE.md) | Low | Module details |

---

## Day 2: Understand the Codebase

### 1. Follow a Request Through the Code

Start a workflow and trace the code path:

```bash
# Terminal 1: Start server with debug logging
AI_NAVIGATOR_DEBUG=true uvicorn ai_navigator.router:router --reload

# Terminal 2: Make a request
curl -X POST http://localhost:8080/ai-navigator/workflow/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "initial_message": "I need Llama 2 for chat"}'
```

**Trace the path:**
1. `router.py:start_workflow()` - API entry point
2. `workflow/engine.py:start_workflow()` - Workflow orchestration
3. `state/manager.py:create_workflow()` - State creation
4. `workflow/stages/intent.py:process()` - Intent extraction
5. `state/manager.py:save_workflow()` - State persistence

### 2. Explore Key Files

Read these files in order:

```bash
# Core models - understand the data structures
cat src/ai_navigator/models/workflow.py

# Workflow engine - understand the FSM
cat src/ai_navigator/workflow/engine.py

# A stage implementation
cat src/ai_navigator/workflow/stages/intent.py

# Capacity planning - the algorithm
cat src/ai_navigator/planning/capacity.py

# YAML generation
cat src/ai_navigator/deployment/generator.py
```

### 3. Run Individual Tests

Understand test patterns:

```bash
# Run a specific test with output
pytest -v -s tests/unit/test_capacity.py::TestCapacityPlanner::test_calculate_capacity_plan

# Read the test file
cat tests/unit/test_capacity.py
```

---

## Week 1: First Tasks

### Suggested Starter Tasks

| Task | Difficulty | Files |
|------|------------|-------|
| Add a new GPU type | Easy | `planning/capacity.py` |
| Add validation to a stage | Easy | `workflow/stages/*.py` |
| Add a new quickstart step | Medium | `quickstarts/tasks/*.py` |
| Add a new API endpoint | Medium | `router.py` |
| Add a new workflow stage | Hard | Multiple files |

### Example: Add a New GPU Type

1. Edit `src/ai_navigator/planning/capacity.py`:

```python
GPU_CATALOG = {
    # Add new GPU
    "A10G": {
        "memory_gb": 24,
        "fp16_tflops": 125,
        "cost_per_hour": 0.80,
        "availability": "high",
    },
    # ... existing GPUs
}
```

2. Write a test:

```python
# tests/unit/test_capacity.py
def test_a10g_in_alternatives(self, planner, benchmark, traffic, slo):
    plan = planner.calculate_capacity_plan(benchmark, traffic, slo)
    gpu_types = [alt.gpu_type for alt in plan.alternatives]
    assert "A10G" in gpu_types
```

3. Run tests:

```bash
pytest tests/unit/test_capacity.py -v
```

---

## Key Concepts Quick Reference

### 8-Stage Workflow

```
INTENT → TRAFFIC → SLO → BENCHMARK* → FILTER* → REPLICAS → DEPLOY → MONITOR
                          (*auto)       (*auto)
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `WorkflowEngine` | Orchestrates stages |
| `WorkflowState` | Session state |
| `StageResult` | Stage output |
| `CapacityPlanner` | Calculates resources |
| `MCPClient` | OpenShift AI operations |
| `YAMLGenerator` | Kubernetes manifests |

### Common Patterns

**Reading state:**
```python
state = await state_manager.get_workflow(workflow_id)
```

**Processing a stage:**
```python
result = await stage.process(state, user_input)
if result.advance:
    state.advance_stage()
```

**Generating YAML:**
```python
generator = YAMLGenerator()
manifests = generator.generate_all(config)
```

### Configuration

| Env Var | What it controls |
|---------|------------------|
| `AI_NAVIGATOR_DEBUG` | Verbose logging |
| `STATE_BACKEND` | memory or postgres |
| `RHOAI_MCP_HOST` | MCP server |

---

## Development Commands Cheatsheet

```bash
# Development
uvicorn ai_navigator.router:router --reload --port 8080

# Testing
pytest                                    # All tests
pytest -v tests/unit/test_capacity.py    # Specific file
pytest -k "capacity"                     # Pattern match
pytest --pdb                             # Debug on failure

# Linting
ruff check src tests                     # Check for issues
ruff format src tests                    # Auto-format
ruff check --fix src tests              # Auto-fix

# Type checking
mypy src                                 # Full check
mypy src/ai_navigator/planning/         # Specific module

# Git
git status
git add -p                               # Interactive staging
git commit -m "Brief description"
```

---

## Getting Help

### Where to Look

1. **Code behavior**: Read the tests in `tests/unit/`
2. **Data models**: Look at `models/*.py`
3. **API usage**: See `docs/API_REFERENCE.md`
4. **Architecture**: See `docs/ARCHITECTURE.md`

### Who to Ask

- **Code questions**: Check git blame for recent authors
- **Architecture questions**: Senior team members
- **Process questions**: Team lead

### Debugging Tips

1. **Enable debug logging:**
   ```bash
   AI_NAVIGATOR_DEBUG=true uvicorn ...
   ```

2. **Add breakpoints:**
   ```python
   import pdb; pdb.set_trace()
   ```

3. **Check recent changes:**
   ```bash
   git log --oneline -20
   git show <commit>
   ```

---

## Checklist

### Day 1
- [ ] Clone repository
- [ ] Install dependencies
- [ ] Run tests successfully
- [ ] Start server locally
- [ ] Read README.md
- [ ] Read CLAUDE.md

### Week 1
- [ ] Read ARCHITECTURE.md
- [ ] Read DEVELOPER_GUIDE.md
- [ ] Trace a request through the code
- [ ] Complete one starter task
- [ ] Submit first PR

### Month 1
- [ ] Understand all workflow stages
- [ ] Know the capacity planning algorithm
- [ ] Comfortable with MCP integration
- [ ] Can add new API endpoints
- [ ] Can add new workflow stages
