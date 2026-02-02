"""Intent extraction stage - Step 1."""

import re
from typing import Optional

import structlog

from ai_navigator.models.workflow import ModelRequirements, WorkflowState
from ai_navigator.workflow.stages.base import BaseStage

logger = structlog.get_logger(__name__)


# Import StageResult here to avoid circular imports
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


# Common model families and their variants
MODEL_PATTERNS = {
    "llama": r"\b(llama[-\s]?[23]?|llama[-\s]?\d+b?)\b",
    "mistral": r"\b(mistral|mixtral)\b",
    "granite": r"\b(granite[-\s]?\d*b?)\b",
    "phi": r"\b(phi[-\s]?[23]?)\b",
    "falcon": r"\b(falcon[-\s]?\d*b?)\b",
    "codellama": r"\b(code[-\s]?llama)\b",
    "starcoder": r"\b(star[-\s]?coder\d*)\b",
}

# Capability keywords
CAPABILITY_PATTERNS = {
    "chat": r"\b(chat|conversation|dialogue|assistant)\b",
    "code": r"\b(code|coding|programming|developer)\b",
    "instruct": r"\b(instruct|instruction|following)\b",
    "embedding": r"\b(embed|embedding|vector)\b",
    "vision": r"\b(vision|image|visual|multimodal)\b",
}

# Size patterns
SIZE_PATTERN = r"\b(\d+\.?\d*)\s*[bB]\b"


class IntentStage(BaseStage):
    """Stage 1: Extract user intent and model requirements."""

    async def process(self, state: WorkflowState, user_input: str) -> StageResult:
        """Extract intent from user's natural language request."""
        if not user_input.strip():
            return StageResult(
                success=True,
                message="Welcome! I'll help you deploy an AI model on OpenShift AI.",
                prompt_user=(
                    "What would you like to deploy? Please describe:\n"
                    "- What model or type of model you need (e.g., Llama 2 70B, a code assistant)\n"
                    "- What you'll use it for (e.g., chatbot, code completion, embeddings)"
                ),
            )

        # Extract intent components
        extracted = self._extract_intent(user_input)

        # Update state with extracted information
        state.intent = user_input
        state.model_requirements = ModelRequirements(
            model_family=extracted.get("model_family"),
            model_name=extracted.get("model_name"),
            max_parameters=extracted.get("max_parameters"),
            capabilities=extracted.get("capabilities", []),
        )

        # Build response
        response_parts = ["I understand you want to deploy an AI model."]

        if extracted.get("model_family"):
            response_parts.append(f"Model family: {extracted['model_family']}")
        if extracted.get("model_name"):
            response_parts.append(f"Specific model: {extracted['model_name']}")
        if extracted.get("max_parameters"):
            response_parts.append(f"Size: up to {extracted['max_parameters']}B parameters")
        if extracted.get("capabilities"):
            response_parts.append(f"Capabilities: {', '.join(extracted['capabilities'])}")

        # Check if we have enough to proceed
        if extracted.get("model_family") or extracted.get("capabilities"):
            response_parts.append("\nI'll now help you define your traffic expectations.")
            return StageResult(
                success=True,
                message="\n".join(response_parts),
                advance=True,
                data=extracted,
            )

        # Need more information
        return StageResult(
            success=True,
            message="\n".join(response_parts),
            prompt_user=(
                "Could you provide more details about:\n"
                "- The specific model you want (e.g., Llama 2 70B, Mistral 7B)\n"
                "- Or the task type (chat, code completion, embeddings)?"
            ),
        )

    def _extract_intent(self, text: str) -> dict:
        """Extract model requirements from natural language."""
        text_lower = text.lower()
        extracted: dict = {
            "model_family": None,
            "model_name": None,
            "max_parameters": None,
            "capabilities": [],
        }

        # Extract model family
        for family, pattern in MODEL_PATTERNS.items():
            if re.search(pattern, text_lower):
                extracted["model_family"] = family
                # Try to extract specific model name
                match = re.search(pattern, text_lower)
                if match:
                    extracted["model_name"] = match.group(0).replace(" ", "-")
                break

        # Extract capabilities
        for capability, pattern in CAPABILITY_PATTERNS.items():
            if re.search(pattern, text_lower):
                extracted["capabilities"].append(capability)

        # Extract size
        size_match = re.search(SIZE_PATTERN, text_lower)
        if size_match:
            extracted["max_parameters"] = float(size_match.group(1))

        # Infer capabilities from model family
        if extracted["model_family"] == "codellama":
            if "code" not in extracted["capabilities"]:
                extracted["capabilities"].append("code")
        elif extracted["model_family"] == "starcoder":
            if "code" not in extracted["capabilities"]:
                extracted["capabilities"].append("code")

        logger.debug("Extracted intent", extracted=extracted, input=text)
        return extracted

    def get_prompt_template(self) -> str:
        """LLM prompt template for intent extraction."""
        return """You are an AI assistant helping users deploy models on OpenShift AI.
Extract the following information from the user's request:

1. Model family (llama, mistral, granite, phi, falcon, etc.)
2. Specific model name if mentioned
3. Model size (in billions of parameters)
4. Required capabilities (chat, code, instruct, embedding, vision)
5. Use case description

User request: {user_input}

Extract the information as JSON:
"""
