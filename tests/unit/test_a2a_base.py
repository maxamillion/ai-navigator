"""Unit tests for A2A protocol base components."""

import pytest
from datetime import datetime, timezone

from ai_navigator.a2a.message import (
    Message,
    MessageRole,
    TextPart,
    FilePart,
    DataPart,
    Artifact,
)
from ai_navigator.a2a.task import Task, TaskState, TaskStatus, TaskStore
from ai_navigator.a2a.agent_card import AgentCard, AgentCardBuilder, AgentSkill
from ai_navigator.a2a.skills import Skill, SkillRegistry, SkillInput, SkillResult


class TestMessage:
    """Tests for Message types."""

    def test_text_part_creation(self):
        """Test creating a text part."""
        part = TextPart(text="Hello, world!")
        assert part.type == "text"
        assert part.text == "Hello, world!"

    def test_data_part_creation(self):
        """Test creating a data part."""
        data = {"key": "value", "count": 42}
        part = DataPart(data=data)
        assert part.type == "data"
        assert part.data == data

    def test_file_part_from_bytes(self):
        """Test creating a file part from bytes."""
        content = b"file content"
        part = FilePart.from_bytes("test.txt", content, "text/plain")
        assert part.type == "file"
        assert part.file["name"] == "test.txt"
        assert part.file["mimeType"] == "text/plain"
        assert "bytes" in part.file

    def test_file_part_from_uri(self):
        """Test creating a file part from URI."""
        part = FilePart.from_uri("doc.pdf", "s3://bucket/doc.pdf", "application/pdf")
        assert part.type == "file"
        assert part.file["name"] == "doc.pdf"
        assert part.file["uri"] == "s3://bucket/doc.pdf"

    def test_user_message_creation(self):
        """Test creating a user message."""
        msg = Message.user("Hello")
        assert msg.role == MessageRole.USER
        assert len(msg.parts) == 1
        assert isinstance(msg.parts[0], TextPart)
        assert msg.get_text() == "Hello"

    def test_agent_message_creation(self):
        """Test creating an agent message."""
        msg = Message.agent("Response", task_id="123")
        assert msg.role == MessageRole.AGENT
        assert msg.get_text() == "Response"
        assert msg.metadata["task_id"] == "123"

    def test_message_get_text_multiple_parts(self):
        """Test extracting text from multiple parts."""
        msg = Message(
            role=MessageRole.USER,
            parts=[
                TextPart(text="First"),
                DataPart(data={"key": "value"}),
                TextPart(text="Second"),
            ],
        )
        assert msg.get_text() == "First\nSecond"


class TestArtifact:
    """Tests for Artifact type."""

    def test_text_artifact(self):
        """Test creating a text artifact."""
        artifact = Artifact.text("result", "Some content")
        assert artifact.name == "result"
        assert len(artifact.parts) == 1
        assert isinstance(artifact.parts[0], TextPart)

    def test_data_artifact(self):
        """Test creating a data artifact."""
        data = {"models": ["a", "b", "c"]}
        artifact = Artifact.data("models", data)
        assert artifact.name == "models"
        assert isinstance(artifact.parts[0], DataPart)

    def test_artifact_has_id(self):
        """Test that artifacts have unique IDs."""
        a1 = Artifact.text("a", "content")
        a2 = Artifact.text("b", "content")
        assert a1.id != a2.id


class TestTask:
    """Tests for Task type."""

    def test_task_creation(self):
        """Test creating a task."""
        task = Task()
        assert task.id is not None
        assert task.status.state == TaskState.SUBMITTED
        assert len(task.messages) == 0
        assert len(task.artifacts) == 0

    def test_task_with_session(self):
        """Test task with session ID."""
        task = Task(session_id="session-123")
        assert task.session_id == "session-123"

    def test_task_add_message(self):
        """Test adding a message to a task."""
        task = Task()
        original_updated = task.updated_at

        msg = Message.user("Test")
        task.add_message(msg)

        assert len(task.messages) == 1
        assert task.messages[0] == msg
        assert task.updated_at >= original_updated

    def test_task_add_artifact(self):
        """Test adding an artifact to a task."""
        task = Task()
        artifact = Artifact.text("output", "result")
        task.add_artifact(artifact)

        assert len(task.artifacts) == 1
        assert task.artifacts[0] == artifact

    def test_task_state_transitions(self):
        """Test task state transitions."""
        task = Task()

        task.set_working("Processing")
        assert task.status.state == TaskState.WORKING

        task.set_completed("Done")
        assert task.status.state == TaskState.COMPLETED
        assert task.is_terminal

    def test_task_failed_state(self):
        """Test task failure."""
        task = Task()
        task.set_failed("Error occurred")

        assert task.status.state == TaskState.FAILED
        assert task.is_terminal
        assert "Error occurred" in task.status.message.get_text()

    def test_task_input_required(self):
        """Test input required state."""
        task = Task()
        task.set_input_required("Please provide more details")

        assert task.status.state == TaskState.INPUT_REQUIRED
        assert not task.is_terminal

    def test_task_jsonrpc_dump(self):
        """Test JSON-RPC serialization."""
        task = Task(session_id="sess-1")
        task.set_completed("Done")

        data = task.model_dump_jsonrpc()
        assert "id" in data
        assert data["sessionId"] == "sess-1"
        assert data["status"]["state"] == "completed"


class TestTaskStore:
    """Tests for TaskStore."""

    def test_create_and_get(self):
        """Test creating and retrieving a task."""
        store = TaskStore()
        task = Task()
        store.create(task)

        retrieved = store.get(task.id)
        assert retrieved is not None
        assert retrieved.id == task.id

    def test_get_nonexistent(self):
        """Test getting a nonexistent task."""
        store = TaskStore()
        assert store.get("nonexistent") is None

    def test_update(self):
        """Test updating a task."""
        store = TaskStore()
        task = Task()
        store.create(task)

        task.set_working("Processing")
        store.update(task)

        retrieved = store.get(task.id)
        assert retrieved.status.state == TaskState.WORKING

    def test_delete(self):
        """Test deleting a task."""
        store = TaskStore()
        task = Task()
        store.create(task)

        assert store.delete(task.id) is True
        assert store.get(task.id) is None
        assert store.delete(task.id) is False

    def test_list_by_session(self):
        """Test listing tasks by session."""
        store = TaskStore()
        task1 = Task(session_id="session-a")
        task2 = Task(session_id="session-a")
        task3 = Task(session_id="session-b")

        store.create(task1)
        store.create(task2)
        store.create(task3)

        session_a_tasks = store.list_by_session("session-a")
        assert len(session_a_tasks) == 2


class TestAgentCard:
    """Tests for AgentCard."""

    def test_agent_card_creation(self):
        """Test creating an agent card."""
        card = AgentCard(
            name="Test Agent",
            description="A test agent",
            url="http://localhost:8000",
            provider={"organization": "Test Org"},
        )
        assert card.name == "Test Agent"
        assert card.url == "http://localhost:8000"

    def test_agent_card_builder(self):
        """Test using AgentCardBuilder."""
        card = (
            AgentCardBuilder(
                name="Builder Agent",
                description="Built with builder",
                url="http://localhost:8001",
            )
            .version("1.0.0")
            .provider("Red Hat", "https://www.redhat.com")
            .capabilities(streaming=True, push_notifications=False)
            .skill(
                id="test_skill",
                name="Test Skill",
                description="A test skill",
                tags=["test"],
            )
            .documentation("https://docs.example.com")
            .build()
        )

        assert card.name == "Builder Agent"
        assert card.version == "1.0.0"
        assert card.capabilities.streaming is True
        assert len(card.skills) == 1
        assert card.skills[0].id == "test_skill"

    def test_agent_card_json_ld(self):
        """Test JSON-LD serialization."""
        card = AgentCard(
            name="JSON-LD Agent",
            description="For JSON-LD testing",
            url="http://localhost:8000",
            provider={"organization": "Test"},
        )

        json_ld = card.model_dump_json_ld()
        assert json_ld["@context"] == "https://schema.org/"
        assert json_ld["@type"] == "SoftwareApplication"
        assert json_ld["name"] == "JSON-LD Agent"


class TestSkillRegistry:
    """Tests for SkillRegistry."""

    def test_register_skill(self):
        """Test registering a skill."""
        registry = SkillRegistry()

        async def handler(input: SkillInput) -> SkillResult:
            return SkillResult.ok("Success")

        skill = registry.register(
            id="my_skill",
            name="My Skill",
            description="Does something",
            handler=handler,
            tags=["test"],
        )

        assert skill.id == "my_skill"
        assert registry.get("my_skill") is not None

    def test_get_nonexistent_skill(self):
        """Test getting a nonexistent skill."""
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_list_skills(self):
        """Test listing all skills."""
        registry = SkillRegistry()

        async def handler(input: SkillInput) -> SkillResult:
            return SkillResult.ok()

        registry.register("skill1", "Skill 1", "First", handler)
        registry.register("skill2", "Skill 2", "Second", handler)

        skills = registry.list()
        assert len(skills) == 2

    def test_find_by_tag(self):
        """Test finding skills by tag."""
        registry = SkillRegistry()

        async def handler(input: SkillInput) -> SkillResult:
            return SkillResult.ok()

        registry.register("s1", "S1", "D1", handler, tags=["models", "query"])
        registry.register("s2", "S2", "D2", handler, tags=["deployment"])
        registry.register("s3", "S3", "D3", handler, tags=["models"])

        model_skills = registry.find_by_tag("models")
        assert len(model_skills) == 2

    @pytest.mark.asyncio
    async def test_dispatch_skill(self):
        """Test dispatching to a skill."""
        registry = SkillRegistry()

        async def handler(input: SkillInput) -> SkillResult:
            text = input.message.get_text()
            return SkillResult.ok(message=f"Received: {text}")

        registry.register("echo", "Echo", "Echoes input", handler)

        task = Task()
        message = Message.user("Hello")
        skill_input = SkillInput(task=task, message=message, params={})

        result = await registry.dispatch("echo", skill_input)
        assert result.success
        assert "Hello" in result.message

    @pytest.mark.asyncio
    async def test_dispatch_unknown_skill(self):
        """Test dispatching to unknown skill."""
        registry = SkillRegistry()

        task = Task()
        message = Message.user("Test")
        skill_input = SkillInput(task=task, message=message, params={})

        result = await registry.dispatch("unknown", skill_input)
        assert not result.success
        assert "Unknown skill" in result.message


class TestSkillResult:
    """Tests for SkillResult."""

    def test_ok_result(self):
        """Test creating success result."""
        result = SkillResult.ok(message="Success", data={"key": "value"})
        assert result.success is True
        assert result.message == "Success"
        assert result.data == {"key": "value"}

    def test_error_result(self):
        """Test creating error result."""
        result = SkillResult.error("Something went wrong")
        assert result.success is False
        assert result.message == "Something went wrong"

    def test_need_input_result(self):
        """Test creating input-required result."""
        result = SkillResult.need_input("Please provide more details")
        assert result.success is True
        assert result.requires_input is True
        assert result.input_prompt == "Please provide more details"
