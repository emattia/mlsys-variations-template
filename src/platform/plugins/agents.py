"""Concrete implementations of Agent plugins."""

import re
import textwrap

from .base import Agent, ExecutionContext
from .registry import get_plugin, register_plugin


@register_plugin(
    name="react_agent",
    category="agent",
    description="A simple ReAct-style agent.",
)
class ReActAgent(Agent):
    """A simple agent that uses the ReAct (Reasoning and Acting) framework."""

    def initialize(self, context: ExecutionContext) -> None:
        """Initializes the agent and its tools."""
        self.context = context
        self.llm_provider = get_plugin(
            name=context.config.llm_provider.plugin,
            category="llm_provider",
            config=context.config.llm_provider.config,
        )
        self.llm_provider.initialize(context)

        self.tools = {}
        tool_names = self.config.get("tools", [])
        for tool_name in tool_names:
            tool_plugin = get_plugin(name=tool_name, category="tool")
            tool_plugin.initialize(context)
            self.tools[tool_name] = tool_plugin

        self.logger.info(
            f"ReActAgent initialized with tools: {list(self.tools.keys())}"
        )

    def run(self, task: str, max_steps: int = None) -> str:
        """Runs the ReAct loop to accomplish the task."""
        # Get max_steps from config, with fallback to default
        if max_steps is None:
            max_steps = self.config.get("max_steps", 5)

        prompt = textwrap.dedent(f"""
            You are a helpful assistant. You have access to the following tools:

            {self._get_tool_descriptions()}

            To use a tool, use the following format:
            Thought: I need to use a tool to find information.
            Action: [tool_name](tool_input)

            If you have the final answer, use the format:
            Thought: I have the final answer.
            Final Answer: The final answer is...

            Begin!

            Question: {task}
        """)

        for _ in range(max_steps):
            thought_process = self.llm_provider.generate(prompt, self.context)

            action_match = re.search(r"Action: \[([^\]]+)\]\((.*)\)", thought_process)
            final_answer_match = re.search(r"Final Answer: (.*)", thought_process)

            if final_answer_match:
                return final_answer_match.group(1)

            if action_match:
                tool_name = action_match.group(1)
                tool_input = action_match.group(2)

                if tool_name in self.tools:
                    tool_result = self.tools[tool_name].run(tool_input)
                    observation = f"Observation: {tool_result}"
                else:
                    observation = f"Error: Tool '{tool_name}' not found."

                prompt += f"\\n{thought_process}\\n{observation}"
            else:
                # If no action, assume it's a final answer
                return thought_process

        return "Agent could not finish the task in the given number of steps."

    def _get_tool_descriptions(self) -> str:
        """Returns a string describing the available tools."""
        return "\\n".join(
            [f"- {name}: {tool.__class__.__doc__}" for name, tool in self.tools.items()]
        )
