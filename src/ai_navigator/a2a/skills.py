"""A2A Skill registry and dispatch."""

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from ai_navigator.a2a.agent_card import AgentSkill
from ai_navigator.a2a.message import Message
from ai_navigator.a2a.task import Task


class SkillInput(BaseModel):
    """Input to a skill handler."""

    task: Task = Field(..., description="The task being processed")
    message: Message = Field(..., description="The triggering message")
    params: dict[str, Any] = Field(default_factory=dict, description="Skill parameters")


class SkillResult(BaseModel):
    """Result from a skill handler."""

    success: bool = Field(..., description="Whether the skill succeeded")
    message: str | None = Field(default=None, description="Result message")
    data: dict[str, Any] | None = Field(default=None, description="Structured result data")
    artifacts: list[Any] = Field(default_factory=list, description="Produced artifacts")
    requires_input: bool = Field(default=False, description="Whether more input is needed")
    input_prompt: str | None = Field(default=None, description="Prompt for required input")

    @classmethod
    def ok(
        cls,
        message: str | None = None,
        data: dict[str, Any] | None = None,
        artifacts: list[Any] | None = None,
    ) -> "SkillResult":
        """Create a successful result."""
        return cls(
            success=True,
            message=message,
            data=data,
            artifacts=artifacts or [],
        )

    @classmethod
    def error(cls, message: str) -> "SkillResult":
        """Create an error result."""
        return cls(success=False, message=message)

    @classmethod
    def need_input(cls, prompt: str) -> "SkillResult":
        """Create a result requesting more input."""
        return cls(
            success=True,
            requires_input=True,
            input_prompt=prompt,
        )


# Type alias for skill handler functions
SkillHandler = Callable[[SkillInput], Awaitable[SkillResult]]


class Skill(BaseModel):
    """A registered skill with its handler."""

    id: str = Field(..., description="Unique skill identifier")
    name: str = Field(..., description="Human-readable skill name")
    description: str = Field(..., description="What the skill does")
    tags: list[str] = Field(default_factory=list, description="Skill tags")
    examples: list[str] = Field(default_factory=list, description="Example invocations")
    input_schema: dict[str, Any] | None = Field(default=None, description="JSON Schema for input")
    output_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema for output"
    )

    # Handler is not serialized
    _handler: SkillHandler | None = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, handler: SkillHandler | None = None, **data: Any) -> None:
        super().__init__(**data)
        self._handler = handler

    @property
    def handler(self) -> SkillHandler | None:
        """Get the skill handler."""
        return self._handler

    def to_agent_skill(self) -> AgentSkill:
        """Convert to AgentSkill for agent card."""
        return AgentSkill(
            id=self.id,
            name=self.name,
            description=self.description,
            tags=self.tags,
            examples=self.examples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
        )

    async def execute(self, input: SkillInput) -> SkillResult:
        """Execute the skill handler."""
        if self._handler is None:
            return SkillResult.error(f"No handler registered for skill: {self.id}")
        return await self._handler(input)


class SkillRegistry:
    """Registry for agent skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(
        self,
        id: str,
        name: str,
        description: str,
        handler: SkillHandler,
        tags: list[str] | None = None,
        examples: list[str] | None = None,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> Skill:
        """Register a skill with its handler."""
        skill = Skill(
            id=id,
            name=name,
            description=description,
            tags=tags or [],
            examples=examples or [],
            input_schema=input_schema,
            output_schema=output_schema,
            handler=handler,
        )
        self._skills[id] = skill
        return skill

    def get(self, skill_id: str) -> Skill | None:
        """Get a skill by ID."""
        return self._skills.get(skill_id)

    def list(self) -> list[Skill]:
        """List all registered skills."""
        return list(self._skills.values())

    def to_agent_skills(self) -> list[AgentSkill]:
        """Convert all skills to AgentSkill format for agent card."""
        return [skill.to_agent_skill() for skill in self._skills.values()]

    async def dispatch(self, skill_id: str, input: SkillInput) -> SkillResult:
        """Dispatch a skill by ID."""
        skill = self.get(skill_id)
        if skill is None:
            return SkillResult.error(f"Unknown skill: {skill_id}")
        return await skill.execute(input)

    def find_by_tag(self, tag: str) -> list[Skill]:
        """Find skills by tag."""
        return [s for s in self._skills.values() if tag in s.tags]

    def find_by_text(self, text: str) -> list[Skill]:
        """Find skills matching text in name or description."""
        text_lower = text.lower()
        return [
            s
            for s in self._skills.values()
            if text_lower in s.name.lower() or text_lower in s.description.lower()
        ]


def skill(
    id: str,
    name: str,
    description: str,
    tags: list[str] | None = None,
    examples: list[str] | None = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Callable[[SkillHandler], Skill]:
    """Decorator to create a Skill from a handler function."""

    def decorator(handler: SkillHandler) -> Skill:
        return Skill(
            id=id,
            name=name,
            description=description,
            tags=tags or [],
            examples=examples or [],
            input_schema=input_schema,
            output_schema=output_schema,
            handler=handler,
        )

    return decorator
