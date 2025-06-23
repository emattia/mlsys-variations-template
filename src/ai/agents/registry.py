"""Agent registry for managing multiple agents."""

from .base import AgentConfig, BaseAgent


class AgentRegistry:
    """Registry for managing and orchestrating multiple agents."""

    def __init__(self):
        self.agents: dict[str, BaseAgent] = {}
        self.configs: dict[str, AgentConfig] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent in the registry."""
        self.agents[agent.config.name] = agent
        self.configs[agent.config.name] = agent.config

    def get_agent(self, name: str) -> BaseAgent | None:
        """Get an agent by name."""
        return self.agents.get(name)

    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        return list(self.agents.keys())

    async def run_agent(self, name: str, task: str, context: dict | None = None) -> str:
        """Run a specific agent by name."""
        agent = self.get_agent(name)
        if not agent:
            raise ValueError(f"Agent '{name}' not found in registry")

        return await agent.run(task, context)

    async def multi_agent_task(
        self, task: str, agent_names: list[str]
    ) -> dict[str, str]:
        """Run a task across multiple agents and return their results."""
        results = {}

        for agent_name in agent_names:
            agent = self.get_agent(agent_name)
            if agent:
                try:
                    result = await agent.run(task)
                    results[agent_name] = result
                except Exception as e:
                    results[agent_name] = f"Error: {str(e)}"
            else:
                results[agent_name] = f"Agent '{agent_name}' not found"

        return results

    def get_agent_stats(self) -> dict[str, dict]:
        """Get statistics for all agents."""
        stats = {}
        for name, agent in self.agents.items():
            if hasattr(agent, "monitor") and agent.monitor:
                stats[name] = agent.monitor.get_stats()
        return stats
