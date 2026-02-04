"""A2A Agent Card definition and builder."""

from typing import Any

from pydantic import BaseModel, Field


class AgentProvider(BaseModel):
    """Information about the agent provider."""

    organization: str = Field(..., description="Organization name")
    url: str | None = Field(default=None, description="Organization URL")


class AgentSkill(BaseModel):
    """A skill exposed by an agent."""

    id: str = Field(..., description="Unique skill identifier")
    name: str = Field(..., description="Human-readable skill name")
    description: str = Field(..., description="What the skill does")
    tags: list[str] = Field(default_factory=list, description="Skill tags for discovery")
    examples: list[str] = Field(default_factory=list, description="Example invocations")
    input_schema: dict[str, Any] | None = Field(
        default=None,
        alias="inputSchema",
        description="JSON Schema for skill input",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        alias="outputSchema",
        description="JSON Schema for skill output",
    )


class AgentCapabilities(BaseModel):
    """Agent capabilities declaration."""

    streaming: bool = Field(default=False, description="Supports streaming responses")
    push_notifications: bool = Field(
        default=False,
        alias="pushNotifications",
        description="Supports push notifications",
    )
    state_transition_history: bool = Field(
        default=True,
        alias="stateTransitionHistory",
        description="Maintains task history",
    )


class AgentCard(BaseModel):
    """A2A Agent Card for agent discovery and capability advertisement."""

    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="What the agent does")
    url: str = Field(..., description="Agent endpoint URL")
    version: str = Field(default="0.1.0", description="Agent version")
    provider: AgentProvider = Field(..., description="Agent provider info")
    capabilities: AgentCapabilities = Field(
        default_factory=AgentCapabilities,
        description="Agent capabilities",
    )
    skills: list[AgentSkill] = Field(
        default_factory=list,
        description="Skills offered by this agent",
    )
    documentation_url: str | None = Field(
        default=None,
        alias="documentationUrl",
        description="Documentation URL",
    )
    default_input_modes: list[str] = Field(
        default_factory=lambda: ["text"],
        alias="defaultInputModes",
        description="Supported input modes",
    )
    default_output_modes: list[str] = Field(
        default_factory=lambda: ["text"],
        alias="defaultOutputModes",
        description="Supported output modes",
    )

    class Config:
        populate_by_name = True

    def model_dump_json_ld(self) -> dict[str, Any]:
        """Dump the agent card in JSON-LD format for /.well-known/agent.json."""
        return {
            "@context": "https://schema.org/",
            "@type": "SoftwareApplication",
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "provider": {
                "@type": "Organization",
                "name": self.provider.organization,
                "url": self.provider.url,
            },
            "capabilities": {
                "streaming": self.capabilities.streaming,
                "pushNotifications": self.capabilities.push_notifications,
                "stateTransitionHistory": self.capabilities.state_transition_history,
            },
            "skills": [
                {
                    "id": skill.id,
                    "name": skill.name,
                    "description": skill.description,
                    "tags": skill.tags,
                    "examples": skill.examples,
                    "inputSchema": skill.input_schema,
                    "outputSchema": skill.output_schema,
                }
                for skill in self.skills
            ],
            "documentationUrl": self.documentation_url,
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
        }


class AgentCardBuilder:
    """Builder for constructing AgentCard instances."""

    def __init__(self, name: str, description: str, url: str) -> None:
        self._name = name
        self._description = description
        self._url = url
        self._version = "0.1.0"
        self._provider = AgentProvider(organization="Red Hat")
        self._capabilities = AgentCapabilities()
        self._skills: list[AgentSkill] = []
        self._documentation_url: str | None = None

    def version(self, version: str) -> "AgentCardBuilder":
        """Set agent version."""
        self._version = version
        return self

    def provider(
        self,
        organization: str,
        url: str | None = None,
    ) -> "AgentCardBuilder":
        """Set agent provider."""
        self._provider = AgentProvider(organization=organization, url=url)
        return self

    def capabilities(
        self,
        streaming: bool = False,
        push_notifications: bool = False,
        state_transition_history: bool = True,
    ) -> "AgentCardBuilder":
        """Set agent capabilities."""
        self._capabilities = AgentCapabilities(
            streaming=streaming,
            push_notifications=push_notifications,
            state_transition_history=state_transition_history,
        )
        return self

    def skill(
        self,
        id: str,
        name: str,
        description: str,
        tags: list[str] | None = None,
        examples: list[str] | None = None,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> "AgentCardBuilder":
        """Add a skill to the agent."""
        self._skills.append(
            AgentSkill(
                id=id,
                name=name,
                description=description,
                tags=tags or [],
                examples=examples or [],
                input_schema=input_schema,
                output_schema=output_schema,
            )
        )
        return self

    def documentation(self, url: str) -> "AgentCardBuilder":
        """Set documentation URL."""
        self._documentation_url = url
        return self

    def build(self) -> AgentCard:
        """Build the AgentCard."""
        return AgentCard(
            name=self._name,
            description=self._description,
            url=self._url,
            version=self._version,
            provider=self._provider,
            capabilities=self._capabilities,
            skills=self._skills,
            documentation_url=self._documentation_url,
        )
