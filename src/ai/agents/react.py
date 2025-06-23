"""ReAct (Reasoning + Acting) agent implementation."""

from typing import Any

from ..llm import LLMProvider
from .base import AgentConfig, BaseAgent


class ReactAgent(BaseAgent):
    """
    ReAct agent that uses reasoning and acting pattern.

    This is a simple implementation that can be extended with more
    sophisticated reasoning patterns.
    """

    def __init__(self, config: AgentConfig, monitor=None):
        super().__init__(config, monitor)
        self.llm = LLMProvider.create(
            config.llm_provider,
            {
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            },
        )

    async def run(self, task: str, context: dict[str, Any] | None = None) -> str:
        """Execute a task using ReAct pattern."""
        context = context or {}

        system_prompt = (
            self.config.system_prompt
            or """
You are a helpful AI agent that uses a Reasoning and Acting (ReAct) pattern.
For each task:
1. Think about what you need to do
2. Act by taking steps or using tools if available
3. Observe the results
4. Reflect on whether you've completed the task

Available tools: {tools}
"""
        )

        prompt = system_prompt.format(tools=list(self.tools.keys()))
        prompt += f"\n\nTask: {task}\n\nPlease complete this task step by step."

        try:
            result = await self.llm.generate(prompt)

            # Log execution
            self._log_execution(
                task,
                result,
                {
                    "agent_type": "react",
                    "tools_used": list(self.tools.keys()),
                    "context": context,
                },
            )

            return result
        except Exception as e:
            error_msg = f"Error executing task: {str(e)}"
            self._log_execution(task, error_msg, {"error": True})
            return error_msg

    def add_tool(self, name: str, tool: Any) -> None:
        """Add a tool to the agent."""
        self.tools[name] = tool

        # Update system prompt to include new tool
        if hasattr(tool, "description"):
            tool_desc = f"{name}: {tool.description}"
        else:
            tool_desc = f"{name}: {type(tool).__name__}"

        if not hasattr(self, "_tool_descriptions"):
            self._tool_descriptions = []
        self._tool_descriptions.append(tool_desc)
