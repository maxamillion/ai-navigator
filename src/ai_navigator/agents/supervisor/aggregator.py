"""Result aggregation for the Supervisor Agent."""

import structlog
from pydantic import BaseModel, Field

from ai_navigator.agents.supervisor.delegator import DelegationResult
from ai_navigator.llm.client import LLMClient
from ai_navigator.llm.prompts import PromptTemplates

logger = structlog.get_logger(__name__)


class AggregatedResponse(BaseModel):
    """Aggregated response from multiple sub-agents."""

    summary: str = Field(..., description="Summary response for the user")
    details: list[dict] = Field(default_factory=list, description="Detailed results")
    success: bool = Field(..., description="Overall success status")
    failed_agents: list[str] = Field(default_factory=list, description="Agents that failed")


class ResultAggregator:
    """
    Aggregates results from multiple sub-agents into a coherent response.

    Uses LLM for natural language synthesis when available,
    falls back to template-based aggregation otherwise.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """
        Initialize the aggregator.

        Args:
            llm_client: Optional LLM client for synthesis
        """
        self.llm = llm_client

    async def aggregate(
        self,
        user_request: str,
        results: list[DelegationResult],
    ) -> AggregatedResponse:
        """
        Aggregate results from sub-agents.

        Args:
            user_request: Original user request
            results: Results from delegation

        Returns:
            AggregatedResponse with synthesized response
        """
        # Collect successful and failed results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        # Try LLM-based synthesis
        if self.llm and successful:
            try:
                summary = await self._llm_synthesize(user_request, successful)
            except Exception as e:
                logger.warning("llm_synthesis_failed", error=str(e))
                summary = self._template_synthesize(user_request, successful, failed)
        else:
            summary = self._template_synthesize(user_request, successful, failed)

        return AggregatedResponse(
            summary=summary,
            details=[r.model_dump() for r in results],
            success=len(failed) == 0 and len(successful) > 0,
            failed_agents=[r.agent for r in failed],
        )

    async def _llm_synthesize(
        self,
        user_request: str,
        results: list[DelegationResult],
    ) -> str:
        """Use LLM to synthesize a natural response."""
        if self.llm is None:
            raise ValueError("LLM client not configured")

        results_for_prompt = [
            {
                "agent": r.agent,
                "skill": r.skill,
                "message": r.message or "",
                "data": r.data or {},
            }
            for r in results
        ]

        prompt = PromptTemplates.format_aggregation(user_request, results_for_prompt)

        response = await self.llm.complete(
            prompt=prompt,
            system_prompt="You are a helpful assistant synthesizing results for the user.",
            temperature=0.3,
        )

        return response

    def _template_synthesize(
        self,
        user_request: str,
        successful: list[DelegationResult],
        failed: list[DelegationResult],
    ) -> str:
        """Template-based synthesis when LLM is unavailable."""
        parts = []

        # Add header
        if len(failed) == 0:
            parts.append("## Request Completed Successfully\n")
        elif len(successful) == 0:
            parts.append("## Request Failed\n")
        else:
            parts.append("## Request Partially Completed\n")

        # Add successful results
        if successful:
            for result in successful:
                if result.message:
                    parts.append(result.message)
                    parts.append("\n---\n")

        # Add failure information
        if failed:
            parts.append("\n### Issues Encountered\n")
            for result in failed:
                parts.append(f"- **{result.agent}** ({result.skill}): {result.error}\n")

        return "\n".join(parts)

    def aggregate_sync(
        self,
        user_request: str,
        results: list[DelegationResult],
    ) -> AggregatedResponse:
        """
        Synchronous aggregation using templates only.

        Useful for contexts where async is not available.
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        summary = self._template_synthesize(user_request, successful, failed)

        return AggregatedResponse(
            summary=summary,
            details=[r.model_dump() for r in results],
            success=len(failed) == 0 and len(successful) > 0,
            failed_agents=[r.agent for r in failed],
        )
