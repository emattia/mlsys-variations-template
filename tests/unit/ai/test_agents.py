"""Tests for AI agent framework components."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.ai.agents import Agent, AgentConfig, AgentRegistry, AgentType, BaseAgent
from src.ai.agents.react import ReactAgent
from src.ai.monitoring import AgentMonitor


class TestAgentConfig:
    """Test AgentConfig dataclass."""

    def test_agent_config_creation(self):
        """Test basic agent configuration creation."""
        config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.REACT,
            llm_provider="openai",
            model="gpt-3.5-turbo",
        )

        assert config.name == "test_agent"
        assert config.agent_type == AgentType.REACT
        assert config.llm_provider == "openai"
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.7  # default
        assert config.tools == []  # default empty list

    def test_agent_config_with_tools(self):
        """Test agent config with custom tools."""
        config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.REACT,
            llm_provider="openai",
            model="gpt-4",
            tools=["calculator", "web_search"],
        )

        assert config.tools == ["calculator", "web_search"]


class TestBaseAgent:
    """Test BaseAgent abstract class."""

    def test_base_agent_instantiation(self):
        """Test that BaseAgent cannot be instantiated directly."""
        config = AgentConfig(
            name="test",
            agent_type=AgentType.REACT,
            llm_provider="openai",
            model="gpt-4",
        )

        with pytest.raises(TypeError):
            BaseAgent(config)


class TestReactAgent:
    """Test ReactAgent implementation."""

    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration."""
        return AgentConfig(
            name="test_react_agent",
            agent_type=AgentType.REACT,
            llm_provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.5,
        )

    @pytest.fixture
    def mock_monitor(self):
        """Create mock agent monitor."""
        return Mock(spec=AgentMonitor)

    def test_react_agent_creation(self, agent_config, mock_monitor):
        """Test ReactAgent creation."""
        with patch("src.ai.agents.react.LLMProvider") as mock_llm_provider:
            mock_llm_provider.create.return_value = Mock()

            agent = ReactAgent(agent_config, mock_monitor)

            assert agent.config == agent_config
            assert agent.monitor == mock_monitor
            assert agent.tools == {}

    @pytest.mark.asyncio
    async def test_react_agent_run(self, agent_config, mock_monitor):
        """Test ReactAgent run method."""
        with patch("src.ai.agents.react.LLMProvider") as mock_llm_provider:
            # Setup mock LLM
            mock_llm = AsyncMock()
            mock_llm.generate.return_value = "Test response from agent"
            mock_llm_provider.create.return_value = mock_llm

            agent = ReactAgent(agent_config, mock_monitor)

            # Test run method
            result = await agent.run("Test task")

            assert result == "Test response from agent"
            mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_react_agent_run_with_error(self, agent_config, mock_monitor):
        """Test ReactAgent run method with error."""
        with patch("src.ai.agents.react.LLMProvider") as mock_llm_provider:
            # Setup mock LLM that raises an error
            mock_llm = AsyncMock()
            mock_llm.generate.side_effect = Exception("LLM error")
            mock_llm_provider.create.return_value = mock_llm

            agent = ReactAgent(agent_config, mock_monitor)

            # Test run method with error
            result = await agent.run("Test task")

            assert "Error executing task" in result

    def test_react_agent_add_tool(self, agent_config, mock_monitor):
        """Test adding tools to ReactAgent."""
        with patch("src.ai.agents.react.LLMProvider") as mock_llm_provider:
            mock_llm_provider.create.return_value = Mock()

            agent = ReactAgent(agent_config, mock_monitor)

            # Add a mock tool
            mock_tool = Mock()
            mock_tool.description = "Test tool"

            agent.add_tool("test_tool", mock_tool)

            assert "test_tool" in agent.tools
            assert agent.tools["test_tool"] == mock_tool


class TestAgentFactory:
    """Test Agent factory class."""

    def test_create_react_agent(self):
        """Test creating a ReactAgent via factory."""
        config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.REACT,
            llm_provider="openai",
            model="gpt-4",
        )

        with patch("src.ai.agents.react.LLMProvider"):
            agent = Agent.create(config)

            assert isinstance(agent, ReactAgent)
            assert agent.config == config

    def test_create_langraph_agent(self):
        """Test creating a LangGraphAgent via factory."""
        config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.LANGRAPH,
            llm_provider="openai",
            model="gpt-4",
        )

        agent = Agent.create(config)

        # Should create LangGraphAgent (placeholder implementation)
        assert agent.config == config

    def test_create_crewai_agent(self):
        """Test creating a CrewAIAgent via factory."""
        config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.CREWAI,
            llm_provider="openai",
            model="gpt-4",
        )

        agent = Agent.create(config)

        # Should create CrewAIAgent (placeholder implementation)
        assert agent.config == config

    def test_create_unknown_agent_type(self):
        """Test creating agent with unknown type raises error."""
        config = AgentConfig(
            name="test_agent",
            agent_type="unknown_type",  # Invalid type
            llm_provider="openai",
            model="gpt-4",
        )

        with pytest.raises(ValueError, match="Unsupported agent type"):
            Agent.create(config)


class TestAgentRegistry:
    """Test AgentRegistry for managing multiple agents."""

    @pytest.fixture
    def registry(self):
        """Create test agent registry."""
        return AgentRegistry()

    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.REACT,
            llm_provider="openai",
            model="gpt-4",
        )
        mock_agent.run = AsyncMock(return_value="Test result")
        return mock_agent

    def test_register_agent(self, registry, mock_agent):
        """Test registering an agent."""
        registry.register_agent(mock_agent)

        assert "test_agent" in registry.agents
        assert registry.get_agent("test_agent") == mock_agent

    def test_list_agents(self, registry, mock_agent):
        """Test listing agents."""
        registry.register_agent(mock_agent)

        agent_names = registry.list_agents()
        assert agent_names == ["test_agent"]

    def test_get_nonexistent_agent(self, registry):
        """Test getting non-existent agent returns None."""
        agent = registry.get_agent("nonexistent")
        assert agent is None

    @pytest.mark.asyncio
    async def test_run_agent(self, registry, mock_agent):
        """Test running an agent through registry."""
        registry.register_agent(mock_agent)

        result = await registry.run_agent("test_agent", "Test task")

        assert result == "Test result"
        mock_agent.run.assert_called_once_with("Test task", None)

    @pytest.mark.asyncio
    async def test_run_nonexistent_agent(self, registry):
        """Test running non-existent agent raises error."""
        with pytest.raises(ValueError, match="Agent 'nonexistent' not found"):
            await registry.run_agent("nonexistent", "Test task")

    @pytest.mark.asyncio
    async def test_multi_agent_task(self, registry):
        """Test running task across multiple agents."""
        # Create multiple mock agents
        agent1 = Mock(spec=BaseAgent)
        agent1.config = AgentConfig(
            name="agent1",
            agent_type=AgentType.REACT,
            llm_provider="openai",
            model="gpt-4",
        )
        agent1.run = AsyncMock(return_value="Result 1")

        agent2 = Mock(spec=BaseAgent)
        agent2.config = AgentConfig(
            name="agent2",
            agent_type=AgentType.REACT,
            llm_provider="openai",
            model="gpt-4",
        )
        agent2.run = AsyncMock(return_value="Result 2")

        registry.register_agent(agent1)
        registry.register_agent(agent2)

        results = await registry.multi_agent_task("Test task", ["agent1", "agent2"])

        assert results == {"agent1": "Result 1", "agent2": "Result 2"}

    @pytest.mark.asyncio
    async def test_multi_agent_task_with_error(self, registry):
        """Test multi-agent task with one agent failing."""
        # Create agents where one fails
        agent1 = Mock(spec=BaseAgent)
        agent1.config = AgentConfig(
            name="agent1",
            agent_type=AgentType.REACT,
            llm_provider="openai",
            model="gpt-4",
        )
        agent1.run = AsyncMock(return_value="Result 1")

        agent2 = Mock(spec=BaseAgent)
        agent2.config = AgentConfig(
            name="agent2",
            agent_type=AgentType.REACT,
            llm_provider="openai",
            model="gpt-4",
        )
        agent2.run = AsyncMock(side_effect=Exception("Agent failed"))

        registry.register_agent(agent1)
        registry.register_agent(agent2)

        results = await registry.multi_agent_task("Test task", ["agent1", "agent2"])

        assert results["agent1"] == "Result 1"
        assert "Error:" in results["agent2"]

    def test_get_agent_stats(self, registry):
        """Test getting agent statistics."""
        # Create mock agent with monitor
        mock_monitor = Mock()
        mock_monitor.get_stats.return_value = {"total_calls": 5, "avg_duration": 1.5}

        mock_agent = Mock(spec=BaseAgent)
        mock_agent.config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.REACT,
            llm_provider="openai",
            model="gpt-4",
        )
        mock_agent.monitor = mock_monitor

        registry.register_agent(mock_agent)

        stats = registry.get_agent_stats()

        assert "test_agent" in stats
        assert stats["test_agent"]["total_calls"] == 5


class TestAgentIntegration:
    """Integration tests for agent components."""

    @pytest.mark.asyncio
    async def test_agent_with_monitor_integration(self):
        """Test agent working with monitor."""
        config = AgentConfig(
            name="integration_test_agent",
            agent_type=AgentType.REACT,
            llm_provider="openai",
            model="gpt-3.5-turbo",
        )

        monitor = AgentMonitor()

        with patch("src.ai.agents.react.LLMProvider") as mock_llm_provider:
            mock_llm = AsyncMock()
            mock_llm.generate.return_value = "Integration test response"
            mock_llm_provider.create.return_value = mock_llm

            agent = ReactAgent(config, monitor)

            # Run task and check monitoring
            result = await agent.run("Integration test task")

            assert result == "Integration test response"
            # Monitor should have recorded the execution
            # (Note: This would need actual monitor implementation to verify)


if __name__ == "__main__":
    pytest.main([__file__])
