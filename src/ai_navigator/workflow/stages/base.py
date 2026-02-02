"""Base stage handler interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_navigator.models.workflow import WorkflowState
    from ai_navigator.workflow.engine import StageResult


class BaseStage(ABC):
    """Abstract base class for workflow stages."""

    @abstractmethod
    async def process(self, state: "WorkflowState", user_input: str) -> "StageResult":
        """Process user input for this stage.

        Args:
            state: Current workflow state
            user_input: User's message/input

        Returns:
            StageResult indicating success, message, and whether to advance
        """
        ...

    def get_prompt_template(self) -> str:
        """Get the prompt template for this stage."""
        return ""

    def validate_input(self, state: "WorkflowState", user_input: str) -> tuple[bool, str]:
        """Validate user input for this stage.

        Returns:
            Tuple of (is_valid, error_message)
        """
        return True, ""
