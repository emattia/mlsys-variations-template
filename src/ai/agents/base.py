"""Base agent classes and interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..monitoring import AgentMonitor


class AgentType(Enum):
    """Types of agents supported."""

    REACT = "react"
    LANGRAPH = "langraph"
    CREWAI = "crewai"
    CUSTOM = "custom"


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    agent_type: AgentType
    llm_provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    tools: list[str] = None
    system_prompt: str = ""
    max_iterations: int = 10
    verbose: bool = False

    def __post_init__(self):
        if self.tools is None:
            self.tools = []


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, config: AgentConfig, monitor: AgentMonitor | None = None):
        self.config = config
        self.monitor = monitor or AgentMonitor()
        self.tools = {}

    @abstractmethod
    async def run(self, task: str, context: dict[str, Any] | None = None) -> str:
        """Execute a task and return the result."""
        pass

    @abstractmethod
    def add_tool(self, name: str, tool: Any) -> None:
        """Add a tool to the agent."""
        pass

    def _log_execution(self, task: str, result: str, metadata: dict[str, Any] = None):
        """Log agent execution for monitoring."""
        if self.monitor:
            self.monitor.log_execution(
                agent_name=self.config.name,
                task=task,
                result=result,
                metadata=metadata or {},
            )


class Agent:
    """Factory class for creating agents."""

    @staticmethod
    def create(config: AgentConfig, monitor: AgentMonitor | None = None) -> BaseAgent:
        """Create an agent based on configuration."""
        if config.agent_type == AgentType.REACT:
            from .react import ReactAgent

            return ReactAgent(config, monitor)
        elif config.agent_type == AgentType.LANGRAPH:
            from .langraph import LangGraphAgent

            return LangGraphAgent(config, monitor)
        elif config.agent_type == AgentType.CREWAI:
            from .crewai import CrewAIAgent

            return CrewAIAgent(config, monitor)
        else:
            raise ValueError(f"Unsupported agent type: {config.agent_type}")
