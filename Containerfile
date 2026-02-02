# AI Navigator Container Image
# Multi-stage build for optimized production image

# Build stage
FROM registry.access.redhat.com/ubi9/python-311:latest AS builder

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install --no-cache .

# Copy source code
COPY src/ src/

# Install the package
RUN . /app/.venv/bin/activate && \
    uv pip install --no-cache .

# Production stage
FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY --from=builder /app/src /app/src

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create non-root user
USER 1001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/ai-navigator/health')" || exit 1

# Default port
EXPOSE 8080

# Entry point
CMD ["uvicorn", "ai_navigator.router:router", "--host", "0.0.0.0", "--port", "8080"]
