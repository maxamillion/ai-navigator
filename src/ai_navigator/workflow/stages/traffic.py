"""Traffic profile stage - Step 2."""

import re
from typing import Optional

import structlog

from ai_navigator.models.workflow import TrafficPattern, TrafficProfile, WorkflowState
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


# Traffic pattern keywords
PATTERN_KEYWORDS = {
    TrafficPattern.STEADY: ["steady", "constant", "consistent", "stable", "uniform"],
    TrafficPattern.BURST: ["burst", "spike", "peak", "surge", "sudden"],
    TrafficPattern.GROWTH: ["growing", "increasing", "scaling", "ramp"],
    TrafficPattern.VARIABLE: ["variable", "unpredictable", "fluctuating", "random"],
}

# Common RPS expectations
RPS_PRESETS = {
    "low": {"rps": 1, "desc": "Low traffic (1-5 RPS)"},
    "medium": {"rps": 10, "desc": "Medium traffic (5-20 RPS)"},
    "high": {"rps": 50, "desc": "High traffic (20-100 RPS)"},
    "very_high": {"rps": 200, "desc": "Very high traffic (100+ RPS)"},
}


class TrafficStage(BaseStage):
    """Stage 2: Determine traffic profile and request patterns."""

    async def process(self, state: WorkflowState, user_input: str) -> StageResult:
        """Gather traffic profile information."""
        # First interaction - ask for traffic info
        if state.traffic_profile is None and not user_input.strip():
            return StageResult(
                success=True,
                message="Now let's understand your expected traffic.",
                prompt_user=(
                    "Please describe your expected usage:\n"
                    "1. How many requests per second (RPS) do you expect?\n"
                    "   - Low: 1-5 RPS (development, small team)\n"
                    "   - Medium: 5-20 RPS (internal app, moderate usage)\n"
                    "   - High: 20-100 RPS (production app, significant usage)\n"
                    "   - Very High: 100+ RPS (high-scale production)\n\n"
                    "2. Is the traffic steady, bursty, or growing over time?\n\n"
                    "3. How many concurrent users do you expect?"
                ),
            )

        # Parse user input
        profile = self._parse_traffic_profile(user_input)

        if profile:
            state.traffic_profile = profile

            response = (
                f"Traffic profile configured:\n"
                f"- Pattern: {profile.pattern.value}\n"
                f"- Expected RPS: {profile.requests_per_second}"
            )
            if profile.peak_rps:
                response += f" (peak: {profile.peak_rps})"
            if profile.concurrent_users:
                response += f"\n- Concurrent users: {profile.concurrent_users}"
            response += (
                f"\n- Average input tokens: {profile.average_input_tokens}\n"
                f"- Average output tokens: {profile.average_output_tokens}"
            )

            return StageResult(
                success=True,
                message=response,
                advance=True,
                data=profile.model_dump(),
            )

        # Couldn't parse - ask for clarification
        return StageResult(
            success=True,
            message="I need a bit more information about your traffic.",
            prompt_user=(
                "Please specify:\n"
                "- Expected requests per second (e.g., '10 RPS' or 'medium')\n"
                "- Traffic pattern (steady, bursty, or growing)"
            ),
        )

    def _parse_traffic_profile(self, text: str) -> Optional[TrafficProfile]:
        """Parse traffic profile from user input."""
        text_lower = text.lower()

        # Try to extract RPS
        rps = self._extract_rps(text_lower)
        if rps is None:
            return None

        # Determine pattern
        pattern = self._extract_pattern(text_lower)

        # Extract peak RPS for bursty traffic
        peak_rps = None
        if pattern == TrafficPattern.BURST:
            peak_match = re.search(r"peak[:\s]+(\d+)", text_lower)
            if peak_match:
                peak_rps = float(peak_match.group(1))
            else:
                peak_rps = rps * 3  # Default: 3x normal for bursts

        # Extract concurrent users
        concurrent_users = self._extract_concurrent_users(text_lower)

        # Extract token counts
        input_tokens = self._extract_token_count(text_lower, "input")
        output_tokens = self._extract_token_count(text_lower, "output")

        return TrafficProfile(
            pattern=pattern,
            requests_per_second=rps,
            peak_rps=peak_rps,
            average_input_tokens=input_tokens or 512,
            average_output_tokens=output_tokens or 256,
            concurrent_users=concurrent_users,
        )

    def _extract_rps(self, text: str) -> Optional[float]:
        """Extract RPS from text."""
        # Check for preset levels
        for level, info in RPS_PRESETS.items():
            if level in text:
                return float(info["rps"])

        # Check for explicit RPS number
        rps_match = re.search(r"(\d+\.?\d*)\s*(?:rps|requests?\s*(?:per|/)\s*second)", text)
        if rps_match:
            return float(rps_match.group(1))

        # Check for just a number with context
        num_match = re.search(r"(\d+\.?\d*)\s*(?:requests?|rps)", text)
        if num_match:
            return float(num_match.group(1))

        # Check for descriptions
        if any(word in text for word in ["few", "small", "light", "minimal"]):
            return 2.0
        if any(word in text for word in ["moderate", "average", "typical"]):
            return 15.0
        if any(word in text for word in ["heavy", "significant", "lots"]):
            return 50.0

        return None

    def _extract_pattern(self, text: str) -> TrafficPattern:
        """Extract traffic pattern from text."""
        for pattern, keywords in PATTERN_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return pattern
        return TrafficPattern.STEADY

    def _extract_concurrent_users(self, text: str) -> Optional[int]:
        """Extract concurrent users from text."""
        match = re.search(r"(\d+)\s*(?:concurrent|simultaneous)\s*users?", text)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)\s*users?", text)
        if match:
            return int(match.group(1))
        return None

    def _extract_token_count(self, text: str, token_type: str) -> Optional[int]:
        """Extract token count from text."""
        pattern = rf"(\d+)\s*(?:{token_type}|prompt|completion)?\s*tokens?"
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
        return None
