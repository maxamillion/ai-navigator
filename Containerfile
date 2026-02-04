# AI Navigator Container Image
# Multi-stage build for minimal runtime image

# Build stage
FROM registry.access.redhat.com/ubi9/python-311:latest AS builder

WORKDIR /opt/app-root/src

# Copy dependency files
COPY pyproject.toml README.md ./

# Install build dependencies and package
RUN pip install --no-cache-dir build && \
    pip install --no-cache-dir .

# Copy source code
COPY src/ ./src/
COPY k8s_operator/ ./k8s_operator/

# Build wheel
RUN python -m build --wheel

# Runtime stage
FROM registry.access.redhat.com/ubi9/python-311:latest

LABEL name="ai-navigator" \
      vendor="Red Hat" \
      version="0.1.0" \
      summary="AI Navigator - Kubernetes-native Supervisor/Sub-Agent system" \
      description="Multi-agent system for OpenShift AI capacity planning"

WORKDIR /opt/app-root/src

# Copy wheel from builder
COPY --from=builder /opt/app-root/src/dist/*.whl ./

# Install the package
RUN pip install --no-cache-dir ./*.whl && \
    rm -f ./*.whl

# Copy manifests for operator
COPY manifests/ ./manifests/

# Default environment variables
ENV AGENT_NAME="supervisor" \
    AGENT_PORT="8000" \
    PYTHONUNBUFFERED="1"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${AGENT_PORT}/healthz || exit 1

# Expose port
EXPOSE 8000

# Default command - run supervisor agent
CMD ["uvicorn", "ai_navigator.agents.supervisor.agent:app", "--host", "0.0.0.0", "--port", "8000"]
