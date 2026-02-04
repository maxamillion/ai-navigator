"""A2A Message types for agent communication."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class PartType(str, Enum):
    """Types of message parts."""

    TEXT = "text"
    FILE = "file"
    DATA = "data"


class TextPart(BaseModel):
    """Text content part of a message."""

    type: Literal["text"] = "text"
    text: str = Field(..., description="The text content")


class FilePart(BaseModel):
    """File content part of a message."""

    type: Literal["file"] = "file"
    file: dict[str, Any] = Field(
        ...,
        description="File data with name, mimeType, and either bytes or uri",
    )

    @classmethod
    def from_bytes(
        cls,
        name: str,
        content: bytes,
        mime_type: str = "application/octet-stream",
    ) -> "FilePart":
        """Create a FilePart from bytes."""
        import base64

        return cls(
            file={
                "name": name,
                "mimeType": mime_type,
                "bytes": base64.b64encode(content).decode("utf-8"),
            }
        )

    @classmethod
    def from_uri(
        cls,
        name: str,
        uri: str,
        mime_type: str = "application/octet-stream",
    ) -> "FilePart":
        """Create a FilePart from a URI."""
        return cls(
            file={
                "name": name,
                "mimeType": mime_type,
                "uri": uri,
            }
        )


class DataPart(BaseModel):
    """Structured data part of a message."""

    type: Literal["data"] = "data"
    data: dict[str, Any] = Field(..., description="Structured JSON data")


# Union type for all part types
Part = Annotated[TextPart | FilePart | DataPart, Field(discriminator="type")]


class MessageRole(str, Enum):
    """Role of the message sender."""

    USER = "user"
    AGENT = "agent"


class Message(BaseModel):
    """A2A Message containing parts from user or agent."""

    role: MessageRole = Field(..., description="Role of the message sender")
    parts: list[Part] = Field(default_factory=list, description="Message content parts")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")

    @classmethod
    def user(cls, text: str, **metadata: Any) -> "Message":
        """Create a user message with text content."""
        return cls(
            role=MessageRole.USER,
            parts=[TextPart(text=text)],
            metadata=metadata,
        )

    @classmethod
    def agent(cls, text: str, **metadata: Any) -> "Message":
        """Create an agent message with text content."""
        return cls(
            role=MessageRole.AGENT,
            parts=[TextPart(text=text)],
            metadata=metadata,
        )

    def get_text(self) -> str:
        """Extract all text content from message parts."""
        texts = []
        for part in self.parts:
            if isinstance(part, TextPart):
                texts.append(part.text)
        return "\n".join(texts)


class Artifact(BaseModel):
    """An artifact produced by an agent during task execution."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique artifact ID")
    name: str = Field(..., description="Human-readable artifact name")
    parts: list[Part] = Field(default_factory=list, description="Artifact content parts")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

    @classmethod
    def text(cls, name: str, content: str, **metadata: Any) -> "Artifact":
        """Create a text artifact."""
        return cls(
            name=name,
            parts=[TextPart(text=content)],
            metadata=metadata,
        )

    @classmethod
    def data(cls, name: str, data: dict[str, Any], **metadata: Any) -> "Artifact":
        """Create a structured data artifact."""
        return cls(
            name=name,
            parts=[DataPart(data=data)],
            metadata=metadata,
        )

    @classmethod
    def file(
        cls,
        name: str,
        content: bytes,
        filename: str,
        mime_type: str = "application/octet-stream",
        **metadata: Any,
    ) -> "Artifact":
        """Create a file artifact."""
        return cls(
            name=name,
            parts=[FilePart.from_bytes(filename, content, mime_type)],
            metadata=metadata,
        )
