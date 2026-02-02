"""Error handling and retry logic for MCP operations."""

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, TypeVar

import structlog

from ai_navigator.mcp.client import MCPConnectionError, MCPToolError

logger = structlog.get_logger(__name__)

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    retryable_errors: tuple[type[Exception], ...] = (MCPConnectionError,)
    retry_on_tool_errors: bool = False


class RetryExhaustedError(Exception):
    """All retry attempts have been exhausted."""

    def __init__(self, message: str, last_error: Exception, attempts: int):
        super().__init__(message)
        self.last_error = last_error
        self.attempts = attempts


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    policy: Optional[RetryPolicy] = None,
) -> T:
    """Execute an async operation with retry logic."""
    policy = policy or RetryPolicy()
    last_error: Optional[Exception] = None
    delay = policy.initial_delay_seconds

    for attempt in range(policy.max_retries + 1):
        try:
            return await operation()
        except policy.retryable_errors as e:
            last_error = e
            if attempt < policy.max_retries:
                logger.warning(
                    "Operation failed, retrying",
                    attempt=attempt + 1,
                    max_retries=policy.max_retries,
                    delay=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)
                delay = min(delay * policy.exponential_base, policy.max_delay_seconds)
            continue
        except MCPToolError as e:
            if policy.retry_on_tool_errors and attempt < policy.max_retries:
                last_error = e
                logger.warning(
                    "Tool error, retrying",
                    attempt=attempt + 1,
                    tool=e.tool_name,
                    error=str(e),
                )
                await asyncio.sleep(delay)
                delay = min(delay * policy.exponential_base, policy.max_delay_seconds)
                continue
            raise

    raise RetryExhaustedError(
        f"Operation failed after {policy.max_retries + 1} attempts",
        last_error,  # type: ignore[arg-type]
        policy.max_retries + 1,
    )


class CircuitBreaker:
    """Circuit breaker pattern for MCP operations."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 60.0,
        half_open_requests: int = 1,
    ) -> None:
        """Initialize circuit breaker."""
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._half_open_requests = half_open_requests

        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state: str = "closed"
        self._half_open_successes = 0

    @property
    def state(self) -> str:
        """Get current circuit state."""
        return self._state

    def _should_attempt(self) -> bool:
        """Check if an operation should be attempted."""
        if self._state == "closed":
            return True

        if self._state == "open":
            if self._last_failure_time is None:
                return True
            elapsed = asyncio.get_event_loop().time() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._state = "half-open"
                self._half_open_successes = 0
                return True
            return False

        # half-open: allow limited requests
        return True

    def record_success(self) -> None:
        """Record a successful operation."""
        if self._state == "half-open":
            self._half_open_successes += 1
            if self._half_open_successes >= self._half_open_requests:
                self._state = "closed"
                self._failure_count = 0
                logger.info("Circuit breaker closed")
        else:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        self._failure_count += 1
        self._last_failure_time = asyncio.get_event_loop().time()

        if self._state == "half-open":
            self._state = "open"
            logger.warning("Circuit breaker opened (half-open failure)")
        elif self._failure_count >= self._failure_threshold:
            self._state = "open"
            logger.warning(
                "Circuit breaker opened",
                failures=self._failure_count,
                threshold=self._failure_threshold,
            )

    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        """Execute operation with circuit breaker protection."""
        if not self._should_attempt():
            raise MCPConnectionError(
                f"Circuit breaker is open. Recovery in {self._recovery_timeout}s"
            )

        try:
            result = await operation()
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


class RecoveryManager:
    """Manages error recovery strategies for MCP operations."""

    def __init__(
        self,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        """Initialize recovery manager."""
        self._retry_policy = retry_policy or RetryPolicy()
        self._circuit_breaker = circuit_breaker or CircuitBreaker()

    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        fallback: Optional[Callable[[], Awaitable[T]]] = None,
    ) -> T:
        """Execute operation with full recovery support."""
        try:
            return await self._circuit_breaker.execute(
                lambda: with_retry(operation, self._retry_policy)
            )
        except (RetryExhaustedError, MCPConnectionError) as e:
            if fallback:
                logger.warning(
                    "Primary operation failed, using fallback",
                    error=str(e),
                )
                return await fallback()
            raise
