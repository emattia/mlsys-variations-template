"""Base classes for agent tools."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolParameter:
    """Defines a tool parameter."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    result: Any
    error: str | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTool(ABC):
    """Abstract base class for agent tools."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """Tool parameters."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def validate_parameters(self, **kwargs) -> bool:
        """Validate provided parameters."""
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"Required parameter '{param.name}' missing")

            if param.name in kwargs:
                # Basic type validation
                value = kwargs[param.name]
                if param.type == "str" and not isinstance(value, str):
                    raise ValueError(f"Parameter '{param.name}' must be a string")
                elif param.type == "int" and not isinstance(value, int):
                    raise ValueError(f"Parameter '{param.name}' must be an integer")
                elif param.type == "float" and not isinstance(value, int | float):
                    raise ValueError(f"Parameter '{param.name}' must be a number")
                elif param.type == "bool" and not isinstance(value, bool):
                    raise ValueError(f"Parameter '{param.name}' must be a boolean")

        return True

    def get_schema(self) -> dict[str, Any]:
        """Get tool schema for LLM function calling."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}

            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self):
        self.tools: dict[str, BaseTool] = {}
        self.logger = logging.getLogger(__name__)

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool."""
        if name in self.tools:
            del self.tools[name]
            self.logger.info(f"Unregistered tool: {name}")

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    def get_all_tools(self) -> dict[str, BaseTool]:
        """Get all registered tools."""
        return self.tools.copy()

    async def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False, result=None, error=f"Tool '{name}' not found"
            )

        try:
            tool.validate_parameters(**kwargs)
            result = await tool.execute(**kwargs)
            self.logger.info(f"Executed tool '{name}' successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error executing tool '{name}': {e}")
            return ToolResult(success=False, result=None, error=str(e))

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Get schemas for all tools (for LLM function calling)."""
        return [tool.get_schema() for tool in self.tools.values()]

    def register_builtin_tools(self) -> None:
        """Register built-in tools."""
        from .builtin import (
            CalculatorTool,
            CodeExecutorTool,
            FileSystemTool,
            WebSearchTool,
        )

        builtin_tools = [
            WebSearchTool(),
            CalculatorTool(),
            FileSystemTool(),
            CodeExecutorTool(),
        ]

        for tool in builtin_tools:
            self.register_tool(tool)

        self.logger.info(f"Registered {len(builtin_tools)} built-in tools")

    def filter_tools(
        self, category: str | None = None, **criteria
    ) -> dict[str, BaseTool]:
        """Filter tools by criteria."""
        filtered = {}

        for name, tool in self.tools.items():
            # Check category if specified
            if category and hasattr(tool, "category") and tool.category != category:
                continue

            # Check other criteria
            match = True
            for key, value in criteria.items():
                if not hasattr(tool, key) or getattr(tool, key) != value:
                    match = False
                    break

            if match:
                filtered[name] = tool

        return filtered
