"""A2A Protocol implementation for AI Navigator."""

from ai_navigator.a2a.agent_card import AgentCard, AgentCapabilities, AgentProvider, AgentSkill
from ai_navigator.a2a.base_agent import BaseAgent
from ai_navigator.a2a.message import Artifact, DataPart, FilePart, Message, Part, TextPart
from ai_navigator.a2a.skills import Skill, SkillRegistry
from ai_navigator.a2a.task import Task, TaskState, TaskStatus

__all__ = [
    # Agent
    "BaseAgent",
    # Agent Card
    "AgentCard",
    "AgentCapabilities",
    "AgentProvider",
    "AgentSkill",
    # Message
    "Message",
    "Part",
    "TextPart",
    "FilePart",
    "DataPart",
    "Artifact",
    # Task
    "Task",
    "TaskState",
    "TaskStatus",
    # Skills
    "Skill",
    "SkillRegistry",
]
