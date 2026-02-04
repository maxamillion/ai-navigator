# AI Navigator Makefile
# Comprehensive build, test, and deployment automation

# =============================================================================
# Configuration Variables
# =============================================================================

# Container registry settings
IMAGE_REGISTRY ?= quay.io/redhat-et
IMAGE_NAME ?= ai-navigator
IMAGE_TAG ?= latest
FULL_IMAGE := $(IMAGE_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# Auto-detect container engine (prefer podman)
CONTAINER_ENGINE ?= $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)

# Kubernetes settings
NAMESPACE ?= ai-navigator
KUBECTL ?= kubectl

# Python/uv settings
UV ?= uv

# Paths
MANIFESTS_DIR := manifests
CRD_DIR := $(MANIFESTS_DIR)/crds
OPERATOR_DIR := $(MANIFESTS_DIR)/operator
AGENTS_DIR := $(MANIFESTS_DIR)/agents
SERVING_DIR := $(MANIFESTS_DIR)/serving
TRUSTYAI_DIR := $(MANIFESTS_DIR)/trustyai

# =============================================================================
# Default Target
# =============================================================================

.DEFAULT_GOAL := help

# =============================================================================
# Development Setup
# =============================================================================

.PHONY: install
install: ## Install package in editable mode
	$(UV) sync --no-dev

.PHONY: install-dev
install-dev: ## Install package with dev dependencies
	$(UV) sync

# =============================================================================
# Testing
# =============================================================================

.PHONY: test
test: ## Run all tests
	$(UV) run pytest tests/ -v

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(UV) run pytest tests/unit/ -v

.PHONY: test-integration
test-integration: ## Run integration tests only
	$(UV) run pytest tests/integration/ -v

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	$(UV) run pytest tests/ -v --cov=src/ai_navigator --cov-report=term-missing --cov-report=html

# =============================================================================
# Code Quality
# =============================================================================

.PHONY: lint
lint: ## Run ruff linter
	$(UV) run ruff check src/ operator/ tests/

.PHONY: format
format: ## Format code with ruff
	$(UV) run ruff format src/ operator/ tests/

.PHONY: format-check
format-check: ## Check code formatting without changes
	$(UV) run ruff format --check src/ operator/ tests/

.PHONY: typecheck
typecheck: ## Run mypy type checker
	$(UV) run mypy src/ operator/

.PHONY: check
check: lint typecheck format-check ## Run all quality checks (lint + typecheck + format-check)

# =============================================================================
# Container Images
# =============================================================================

.PHONY: build
build: ## Build container image
	$(CONTAINER_ENGINE) build -t $(FULL_IMAGE) -f Containerfile .

.PHONY: push
push: ## Push container image to registry
	$(CONTAINER_ENGINE) push $(FULL_IMAGE)

.PHONY: build-push
build-push: build push ## Build and push container image

# =============================================================================
# Kubernetes Deployment
# =============================================================================

.PHONY: deploy-crds
deploy-crds: ## Apply Agent CRD
	$(KUBECTL) apply -f $(CRD_DIR)/agent-crd.yaml

.PHONY: deploy-operator
deploy-operator: ## Deploy operator (namespace, RBAC, deployment)
	$(KUBECTL) apply -f $(OPERATOR_DIR)/namespace.yaml
	$(KUBECTL) apply -f $(OPERATOR_DIR)/serviceaccount.yaml
	$(KUBECTL) apply -f $(OPERATOR_DIR)/rbac.yaml
	$(KUBECTL) apply -f $(OPERATOR_DIR)/deployment.yaml

.PHONY: deploy-agents
deploy-agents: ## Deploy all agent CRs
	$(KUBECTL) apply -f $(AGENTS_DIR)/supervisor/agent-cr.yaml
	$(KUBECTL) apply -f $(AGENTS_DIR)/model-catalog/agent-cr.yaml
	$(KUBECTL) apply -f $(AGENTS_DIR)/resource-provisioning/agent-cr.yaml
	$(KUBECTL) apply -f $(AGENTS_DIR)/deployment-monitor/agent-cr.yaml

.PHONY: deploy-serving
deploy-serving: ## Deploy KServe runtime and InferenceService
	$(KUBECTL) apply -f $(SERVING_DIR)/serving-runtime.yaml
	$(KUBECTL) apply -f $(SERVING_DIR)/inference-service.yaml

.PHONY: deploy-trustyai
deploy-trustyai: ## Deploy TrustyAI guardrails orchestrator
	$(KUBECTL) apply -f $(TRUSTYAI_DIR)/guardrails-orchestrator.yaml

.PHONY: deploy
deploy: deploy-crds deploy-operator deploy-agents deploy-serving deploy-trustyai ## Full deployment (CRDs, operator, agents, serving, trustyai)

.PHONY: undeploy-agents
undeploy-agents: ## Remove agent CRs only
	-$(KUBECTL) delete -f $(AGENTS_DIR)/supervisor/agent-cr.yaml
	-$(KUBECTL) delete -f $(AGENTS_DIR)/model-catalog/agent-cr.yaml
	-$(KUBECTL) delete -f $(AGENTS_DIR)/resource-provisioning/agent-cr.yaml
	-$(KUBECTL) delete -f $(AGENTS_DIR)/deployment-monitor/agent-cr.yaml

.PHONY: undeploy
undeploy: ## Remove all deployed resources
	-$(KUBECTL) delete -f $(TRUSTYAI_DIR)/guardrails-orchestrator.yaml
	-$(KUBECTL) delete -f $(SERVING_DIR)/inference-service.yaml
	-$(KUBECTL) delete -f $(SERVING_DIR)/serving-runtime.yaml
	-$(KUBECTL) delete -f $(AGENTS_DIR)/supervisor/agent-cr.yaml
	-$(KUBECTL) delete -f $(AGENTS_DIR)/model-catalog/agent-cr.yaml
	-$(KUBECTL) delete -f $(AGENTS_DIR)/resource-provisioning/agent-cr.yaml
	-$(KUBECTL) delete -f $(AGENTS_DIR)/deployment-monitor/agent-cr.yaml
	-$(KUBECTL) delete -f $(OPERATOR_DIR)/deployment.yaml
	-$(KUBECTL) delete -f $(OPERATOR_DIR)/rbac.yaml
	-$(KUBECTL) delete -f $(OPERATOR_DIR)/serviceaccount.yaml
	-$(KUBECTL) delete -f $(OPERATOR_DIR)/namespace.yaml
	-$(KUBECTL) delete -f $(CRD_DIR)/agent-crd.yaml

# =============================================================================
# Local Development
# =============================================================================

.PHONY: run-supervisor
run-supervisor: ## Run supervisor agent locally
	$(UV) run uvicorn ai_navigator.agents.supervisor.agent:app --reload --port 8000

.PHONY: run-operator
run-operator: ## Run operator locally with kopf
	$(UV) run kopf run operator/main.py --verbose

# =============================================================================
# Utility
# =============================================================================

.PHONY: clean
clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf uv.lock
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-venv
clean-venv: clean ## Clean build artifacts and virtual environment
	rm -rf .venv/

.PHONY: help
help: ## Show available targets
	@echo "AI Navigator - Available Make Targets"
	@echo "======================================"
	@echo ""
	@echo "Configuration:"
	@echo "  IMAGE_REGISTRY   = $(IMAGE_REGISTRY)"
	@echo "  IMAGE_NAME       = $(IMAGE_NAME)"
	@echo "  IMAGE_TAG        = $(IMAGE_TAG)"
	@echo "  CONTAINER_ENGINE = $(CONTAINER_ENGINE)"
	@echo "  NAMESPACE        = $(NAMESPACE)"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
