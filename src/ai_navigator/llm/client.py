"""LLM Client for AI Navigator."""

from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_navigator.config import LLMSettings

logger = structlog.get_logger(__name__)


class ChatMessage(BaseModel):
    """A chat message."""

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletion(BaseModel):
    """Response from chat completion."""

    content: str = Field(..., description="Generated content")
    finish_reason: str = Field(default="stop", description="Why generation stopped")
    usage: dict[str, int] = Field(default_factory=dict, description="Token usage")


class LLMClientError(Exception):
    """Error during LLM communication."""

    pass


class LLMClient:
    """
    Client for LLM inference using OpenAI-compatible API.

    Supports vLLM, TGI, and other OpenAI-compatible endpoints.
    """

    def __init__(self, settings: LLMSettings | None = None) -> None:
        """
        Initialize LLM client.

        Args:
            settings: LLM configuration
        """
        self.settings = settings or LLMSettings()
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "LLMClient":
        """Enter async context."""
        self._client = httpx.AsyncClient(
            base_url=self.settings.endpoint,
            timeout=self.settings.timeout,
            headers=self._get_headers(),
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context."""
        if self._client:
            await self._client.aclose()

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.endpoint,
                timeout=self.settings.timeout,
                headers=self._get_headers(),
            )
        return self._client

    @retry(
        retry=retry_if_exception_type((httpx.RequestError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def chat(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> ChatCompletion:
        """
        Send a chat completion request.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature (overrides settings)
            max_tokens: Max tokens to generate (overrides settings)
            stop: Stop sequences

        Returns:
            ChatCompletion response

        Raises:
            LLMClientError: If the request fails
        """
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature or self.settings.temperature,
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        if stop:
            payload["stop"] = stop

        try:
            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()

            data = response.json()
            choice = data.get("choices", [{}])[0]

            return ChatCompletion(
                content=choice.get("message", {}).get("content", ""),
                finish_reason=choice.get("finish_reason", "stop"),
                usage=data.get("usage", {}),
            )
        except httpx.HTTPStatusError as e:
            logger.error("llm_request_failed", status=e.response.status_code, error=str(e))
            raise LLMClientError(f"LLM request failed: {e}") from e
        except httpx.RequestError as e:
            logger.error("llm_connection_error", error=str(e))
            raise LLMClientError(f"Connection error: {e}") from e

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Simple completion with optional system prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        result = await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return result.content

    async def health_check(self) -> bool:
        """
        Check if the LLM endpoint is healthy.

        Returns:
            True if endpoint is reachable
        """
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except httpx.RequestError:
            # Try a simple models list as fallback
            try:
                response = await self.client.get("/models")
                return response.status_code == 200
            except httpx.RequestError:
                return False
