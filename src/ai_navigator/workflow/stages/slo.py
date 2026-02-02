"""SLO definition stage - Step 3."""

import re
from typing import Optional

import structlog

from ai_navigator.models.workflow import SLORequirements, WorkflowState
from ai_navigator.workflow.stages.base import BaseStage

logger = structlog.get_logger(__name__)


class StageResult:
    def __init__(
        self,
        success: bool,
        message: str,
        advance: bool = False,
        data: Optional[dict] = None,
        prompt_user: Optional[str] = None,
    ):
        self.success = success
        self.message = message
        self.advance = advance
        self.data = data or {}
        self.prompt_user = prompt_user


# SLO presets for common use cases
SLO_PRESETS = {
    "interactive": {
        "p50_latency_ms": 500,
        "p95_latency_ms": 1000,
        "p99_latency_ms": 2000,
        "availability_percent": 99.9,
        "desc": "Interactive chat (fast responses)",
    },
    "batch": {
        "p50_latency_ms": 5000,
        "p95_latency_ms": 10000,
        "p99_latency_ms": 30000,
        "availability_percent": 99.5,
        "desc": "Batch processing (higher latency OK)",
    },
    "realtime": {
        "p50_latency_ms": 200,
        "p95_latency_ms": 500,
        "p99_latency_ms": 1000,
        "availability_percent": 99.99,
        "desc": "Real-time applications (very fast)",
    },
    "standard": {
        "p50_latency_ms": 1000,
        "p95_latency_ms": 3000,
        "p99_latency_ms": 5000,
        "availability_percent": 99.9,
        "desc": "Standard API (balanced)",
    },
}


class SLOStage(BaseStage):
    """Stage 3: Define Service Level Objectives."""

    async def process(self, state: WorkflowState, user_input: str) -> StageResult:
        """Gather SLO requirements."""
        # First interaction - ask for SLO info
        if state.slo_requirements is None and not user_input.strip():
            presets_text = "\n".join(
                f"- **{name}**: {info['desc']} (p95: {info['p95_latency_ms']}ms)"
                for name, info in SLO_PRESETS.items()
            )

            return StageResult(
                success=True,
                message="Now let's define your performance requirements (SLOs).",
                prompt_user=(
                    f"Choose a preset or specify custom targets:\n\n"
                    f"{presets_text}\n\n"
                    "Or specify custom values:\n"
                    "- Target p95 latency (e.g., '2 seconds', '500ms')\n"
                    "- Availability requirement (e.g., '99.9%')"
                ),
            )

        # Parse user input
        slo = self._parse_slo_requirements(user_input)

        if slo:
            state.slo_requirements = slo

            response = (
                f"SLO requirements configured:\n"
                f"- p50 latency: {slo.p50_latency_ms}ms\n"
                f"- p95 latency: {slo.p95_latency_ms}ms\n"
                f"- p99 latency: {slo.p99_latency_ms}ms\n"
                f"- Availability: {slo.availability_percent}%"
            )
            if slo.max_tokens_per_second:
                response += f"\n- Throughput target: {slo.max_tokens_per_second} tokens/sec"

            return StageResult(
                success=True,
                message=response,
                advance=True,
                data=slo.model_dump(),
            )

        # Couldn't parse - ask for clarification
        return StageResult(
            success=True,
            message="I need clearer SLO requirements.",
            prompt_user=(
                "Please specify either:\n"
                "- A preset (interactive, batch, realtime, standard)\n"
                "- Or a target latency (e.g., 'under 2 seconds')"
            ),
        )

    def _parse_slo_requirements(self, text: str) -> Optional[SLORequirements]:
        """Parse SLO requirements from user input."""
        text_lower = text.lower()

        # Check for presets
        for preset_name, preset in SLO_PRESETS.items():
            if preset_name in text_lower:
                return SLORequirements(
                    p50_latency_ms=preset["p50_latency_ms"],
                    p95_latency_ms=preset["p95_latency_ms"],
                    p99_latency_ms=preset["p99_latency_ms"],
                    availability_percent=preset["availability_percent"],
                )

        # Try to parse custom values
        p95 = self._extract_latency(text_lower, "p95")
        p99 = self._extract_latency(text_lower, "p99")
        p50 = self._extract_latency(text_lower, "p50")

        # Also check for generic "latency" or "response time"
        if p95 is None:
            generic = self._extract_generic_latency(text_lower)
            if generic:
                p95 = generic
                p99 = int(generic * 1.5)
                p50 = int(generic * 0.5)

        if p95 is None:
            return None

        # Default p50 and p99 if not specified
        if p50 is None:
            p50 = int(p95 * 0.5)
        if p99 is None:
            p99 = int(p95 * 1.5)

        # Extract availability
        availability = self._extract_availability(text_lower)

        # Extract throughput target
        throughput = self._extract_throughput(text_lower)

        return SLORequirements(
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            availability_percent=availability,
            max_tokens_per_second=throughput,
        )

    def _extract_latency(self, text: str, percentile: str) -> Optional[int]:
        """Extract latency for specific percentile."""
        patterns = [
            rf"{percentile}[:\s]+(\d+\.?\d*)\s*(?:ms|milliseconds?)",
            rf"{percentile}[:\s]+(\d+\.?\d*)\s*(?:s|seconds?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))
                if "second" in pattern or "\\s*s" in pattern:
                    value *= 1000
                return int(value)

        return None

    def _extract_generic_latency(self, text: str) -> Optional[int]:
        """Extract generic latency value."""
        patterns = [
            r"(?:under|less than|below|max|maximum)?\s*(\d+\.?\d*)\s*(?:ms|milliseconds?)",
            r"(?:under|less than|below|max|maximum)?\s*(\d+\.?\d*)\s*(?:s|seconds?)",
            r"(\d+\.?\d*)\s*(?:ms|milliseconds?)\s*(?:latency|response)",
            r"(\d+\.?\d*)\s*(?:s|seconds?)\s*(?:latency|response)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))
                if "second" in pattern:
                    value *= 1000
                return int(value)

        return None

    def _extract_availability(self, text: str) -> float:
        """Extract availability percentage."""
        match = re.search(r"(\d+\.?\d*)\s*%?\s*(?:availability|uptime|available)", text)
        if match:
            return float(match.group(1))

        # Check for "three nines" style
        if "four nines" in text or "4 nines" in text:
            return 99.99
        if "three nines" in text or "3 nines" in text:
            return 99.9
        if "two nines" in text or "2 nines" in text:
            return 99.0

        return 99.9  # Default

    def _extract_throughput(self, text: str) -> Optional[int]:
        """Extract throughput target."""
        match = re.search(r"(\d+)\s*(?:tokens?)?/?(?:s|sec|second)", text)
        if match:
            return int(match.group(1))
        return None
