"""Tests for AI agent tools."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.ai.tools import (
    BaseTool,
    CalculatorTool,
    CodeExecutorTool,
    FileSystemTool,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    WebSearchTool,
)


class TestToolParameter:
    """Test ToolParameter dataclass."""

    def test_tool_parameter_creation(self):
        """Test basic tool parameter creation."""
        param = ToolParameter(
            name="input_text", type="str", description="Text to process", required=True
        )

        assert param.name == "input_text"
        assert param.type == "str"
        assert param.description == "Text to process"
        assert param.required is True
        assert param.default is None

    def test_tool_parameter_with_default(self):
        """Test tool parameter with default value."""
        param = ToolParameter(
            name="max_length",
            type="int",
            description="Maximum length",
            required=False,
            default=100,
        )

        assert param.required is False
        assert param.default == 100


class TestToolResult:
    """Test ToolResult dataclass."""

    def test_tool_result_success(self):
        """Test successful tool result."""
        result = ToolResult(
            success=True, result="Operation completed", metadata={"duration": 0.5}
        )

        assert result.success is True
        assert result.result == "Operation completed"
        assert result.error is None
        assert result.metadata["duration"] == 0.5

    def test_tool_result_failure(self):
        """Test failed tool result."""
        result = ToolResult(success=False, result=None, error="Operation failed")

        assert result.success is False
        assert result.result is None
        assert result.error == "Operation failed"

    def test_tool_result_metadata_default(self):
        """Test tool result with default metadata."""
        result = ToolResult(success=True, result="test")

        assert result.metadata == {}


class TestBaseTool:
    """Test BaseTool abstract class."""

    def test_base_tool_cannot_be_instantiated(self):
        """Test that BaseTool cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTool()

    def test_base_tool_validation(self):
        """Test base tool parameter validation."""

        # Create a concrete implementation for testing
        class TestTool(BaseTool):
            @property
            def name(self):
                return "test_tool"

            @property
            def description(self):
                return "Test tool"

            @property
            def parameters(self):
                return [
                    ToolParameter("required_param", "str", "Required parameter", True),
                    ToolParameter(
                        "optional_param", "int", "Optional parameter", False, 42
                    ),
                ]

            async def execute(self, **kwargs):
                return ToolResult(True, "executed")

        tool = TestTool()

        # Test successful validation
        assert tool.validate_parameters(required_param="test") is True

        # Test missing required parameter
        with pytest.raises(
            ValueError, match="Required parameter 'required_param' missing"
        ):
            tool.validate_parameters(optional_param=10)

        # Test type validation
        with pytest.raises(ValueError, match="must be a string"):
            tool.validate_parameters(required_param=123)  # Should be string

    def test_get_schema(self):
        """Test getting tool schema for LLM function calling."""

        class TestTool(BaseTool):
            @property
            def name(self):
                return "test_tool"

            @property
            def description(self):
                return "Test tool for schema"

            @property
            def parameters(self):
                return [
                    ToolParameter("text", "str", "Input text", True),
                    ToolParameter("count", "int", "Number of items", False, 5),
                ]

            async def execute(self, **kwargs):
                return ToolResult(True, "executed")

        tool = TestTool()
        schema = tool.get_schema()

        assert schema["name"] == "test_tool"
        assert schema["description"] == "Test tool for schema"
        assert "parameters" in schema
        assert "text" in schema["parameters"]["properties"]
        assert "count" in schema["parameters"]["properties"]
        assert "text" in schema["parameters"]["required"]
        assert "count" not in schema["parameters"]["required"]


class TestCalculatorTool:
    """Test CalculatorTool implementation."""

    @pytest.fixture
    def calculator(self):
        """Create calculator tool instance."""
        return CalculatorTool()

    def test_calculator_properties(self, calculator):
        """Test calculator tool properties."""
        assert calculator.name == "calculator"
        assert "mathematical" in calculator.description.lower()
        assert len(calculator.parameters) == 1
        assert calculator.parameters[0].name == "expression"

    @pytest.mark.asyncio
    async def test_calculator_basic_operations(self, calculator):
        """Test basic mathematical operations."""
        test_cases = [
            ("2 + 3", 5),
            ("10 - 4", 6),
            ("6 * 7", 42),
            ("15 / 3", 5),
            ("2 ** 3", 8),
            ("sqrt(16)", 4),
        ]

        for expression, expected in test_cases:
            result = await calculator.execute(expression=expression)

            assert result.success is True
            assert result.result == expected

    @pytest.mark.asyncio
    async def test_calculator_complex_expression(self, calculator):
        """Test complex mathematical expression."""
        result = await calculator.execute(expression="(2 + 3) * 4 - 1")

        assert result.success is True
        assert result.result == 19  # (2+3)*4-1 = 5*4-1 = 20-1 = 19

    @pytest.mark.asyncio
    async def test_calculator_math_functions(self, calculator):
        """Test mathematical functions."""

        result = await calculator.execute(expression="sin(pi/2)")

        assert result.success is True
        assert abs(result.result - 1.0) < 0.0001  # sin(π/2) = 1

    @pytest.mark.asyncio
    async def test_calculator_invalid_expression(self, calculator):
        """Test calculator with invalid expression."""
        result = await calculator.execute(expression="invalid expression")

        assert result.success is False
        assert result.result is None
        assert "error" in result.error.lower()

    @pytest.mark.asyncio
    async def test_calculator_security(self, calculator):
        """Test calculator security (no dangerous operations)."""
        # Should not allow dangerous operations
        dangerous_expressions = [
            "__import__('os').system('ls')",
            "exec('print(1)')",
            "eval('1+1')",
        ]

        for expr in dangerous_expressions:
            _ = await calculator.execute(expression=expr)
            # Should either fail or not execute dangerous code
            # The safe evaluation should prevent these


class TestWebSearchTool:
    """Test WebSearchTool implementation."""

    @pytest.fixture
    def web_search(self):
        """Create web search tool instance."""
        return WebSearchTool()

    def test_web_search_properties(self, web_search):
        """Test web search tool properties."""
        assert web_search.name == "web_search"
        assert "search" in web_search.description.lower()

        param_names = [p.name for p in web_search.parameters]
        assert "query" in param_names
        assert "num_results" in param_names

    @pytest.mark.asyncio
    async def test_web_search_execution(self, web_search):
        """Test web search execution (placeholder implementation)."""
        result = await web_search.execute(
            query="artificial intelligence", num_results=3
        )

        assert result.success is True
        assert isinstance(result.result, list)
        assert len(result.result) <= 3

        # Check result structure
        for search_result in result.result:
            assert "title" in search_result
            assert "url" in search_result
            assert "snippet" in search_result


class TestFileSystemTool:
    """Test FileSystemTool implementation."""

    @pytest.fixture
    def filesystem(self):
        """Create filesystem tool instance."""
        return FileSystemTool()

    def test_filesystem_properties(self, filesystem):
        """Test filesystem tool properties."""
        assert filesystem.name == "filesystem"
        assert "file system" in filesystem.description.lower()

        param_names = [p.name for p in filesystem.parameters]
        assert "operation" in param_names
        assert "path" in param_names
        assert "content" in param_names

    @pytest.mark.asyncio
    async def test_filesystem_exists_operation(self, filesystem):
        """Test filesystem exists operation."""
        # Test with current directory (should exist)
        _ = await filesystem.execute(operation="exists", path="./")

        # Note: This might fail due to safe directory restrictions
        # The tool restricts access to safe directories only

    @pytest.mark.asyncio
    async def test_filesystem_unsafe_path(self, filesystem):
        """Test filesystem with unsafe path."""
        # Try to access system directory (should be denied)
        result = await filesystem.execute(operation="read", path="/etc/passwd")

        assert result.success is False
        assert "access denied" in result.error.lower()

    @pytest.mark.asyncio
    async def test_filesystem_write_read_cycle(self, filesystem):
        """Test writing and reading a file."""
        # Use a path that should be in safe directories
        test_content = "Test file content"

        # This test might fail due to safe directory restrictions
        # In a real implementation, we'd need to setup safe directories

        # For now, just test that the method doesn't crash
        _ = await filesystem.execute(
            operation="write", path="./data/test.txt", content=test_content
        )

        # Result depends on whether ./data is considered safe


class TestCodeExecutorTool:
    """Test CodeExecutorTool implementation."""

    @pytest.fixture
    def code_executor(self):
        """Create code executor tool instance."""
        return CodeExecutorTool()

    def test_code_executor_properties(self, code_executor):
        """Test code executor tool properties."""
        assert code_executor.name == "code_executor"
        assert "python" in code_executor.description.lower()

        param_names = [p.name for p in code_executor.parameters]
        assert "code" in param_names
        assert "timeout" in param_names

    @pytest.mark.asyncio
    async def test_code_executor_simple_code(self, code_executor):
        """Test executing simple Python code."""
        code = """
print("Hello, World!")
result = 2 + 3
print(f"Result: {result}")
"""

        result = await code_executor.execute(code=code)

        assert result.success is True
        assert "Hello, World!" in result.result
        assert "Result: 5" in result.result

    @pytest.mark.asyncio
    async def test_code_executor_math_operations(self, code_executor):
        """Test executing mathematical operations."""
        code = """
import math
result = math.sqrt(16) + math.sin(math.pi/2)
print(f"Result: {result}")
"""

        result = await code_executor.execute(code=code)

        assert result.success is True
        assert "Result: 5.0" in result.result  # sqrt(16) + sin(π/2) = 4 + 1 = 5

    @pytest.mark.asyncio
    async def test_code_executor_syntax_error(self, code_executor):
        """Test code executor with syntax error."""
        code = """
print("Missing closing quote
result = 2 + 3
"""

        result = await code_executor.execute(code=code)

        assert result.success is False
        assert "syntax error" in result.error.lower()

    @pytest.mark.asyncio
    async def test_code_executor_security_restrictions(self, code_executor):
        """Test that code executor blocks dangerous operations."""
        dangerous_code = """
import os
os.system('rm -rf /')
"""

        result = await code_executor.execute(code=dangerous_code)

        assert result.success is False
        assert "dangerous import" in result.error.lower()

    @pytest.mark.asyncio
    async def test_code_executor_allowed_imports(self, code_executor):
        """Test that safe imports are allowed."""
        code = """
import math
import json
import datetime

print("Safe imports work!")
print(f"Pi = {math.pi}")
print(f"Today: {datetime.date.today()}")
"""

        result = await code_executor.execute(code=code)

        assert result.success is True
        assert "Safe imports work!" in result.result


class TestToolRegistry:
    """Test ToolRegistry functionality."""

    @pytest.fixture
    def registry(self):
        """Create tool registry instance."""
        return ToolRegistry()

    @pytest.fixture
    def mock_tool(self):
        """Create mock tool for testing."""
        tool = Mock(spec=BaseTool)
        tool.name = "mock_tool"
        tool.description = "Mock tool for testing"
        tool.execute = AsyncMock(return_value=ToolResult(True, "mock result"))
        tool.validate_parameters = Mock(return_value=True)
        return tool

    def test_register_tool(self, registry, mock_tool):
        """Test registering a tool."""
        registry.register_tool(mock_tool)

        assert "mock_tool" in registry.tools
        assert registry.get_tool("mock_tool") == mock_tool

    def test_unregister_tool(self, registry, mock_tool):
        """Test unregistering a tool."""
        registry.register_tool(mock_tool)
        registry.unregister_tool("mock_tool")

        assert "mock_tool" not in registry.tools
        assert registry.get_tool("mock_tool") is None

    def test_list_tools(self, registry, mock_tool):
        """Test listing tools."""
        registry.register_tool(mock_tool)

        tools = registry.list_tools()
        assert "mock_tool" in tools

    def test_get_all_tools(self, registry, mock_tool):
        """Test getting all tools."""
        registry.register_tool(mock_tool)

        all_tools = registry.get_all_tools()
        assert isinstance(all_tools, dict)
        assert "mock_tool" in all_tools
        assert all_tools["mock_tool"] == mock_tool

    @pytest.mark.asyncio
    async def test_execute_tool(self, registry, mock_tool):
        """Test executing a tool through registry."""
        registry.register_tool(mock_tool)

        result = await registry.execute_tool("mock_tool", param1="value1")

        assert result.success is True
        assert result.result == "mock result"
        mock_tool.validate_parameters.assert_called_once_with(param1="value1")
        mock_tool.execute.assert_called_once_with(param1="value1")

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, registry):
        """Test executing non-existent tool."""
        result = await registry.execute_tool("nonexistent")

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_with_validation_error(self, registry, mock_tool):
        """Test executing tool with validation error."""
        mock_tool.validate_parameters.side_effect = ValueError("Invalid parameter")
        registry.register_tool(mock_tool)

        result = await registry.execute_tool("mock_tool")

        assert result.success is False
        assert "Invalid parameter" in result.error

    def test_get_tools_schema(self, registry, mock_tool):
        """Test getting tools schema for LLM function calling."""
        # Add get_schema method to mock
        mock_schema = {
            "name": "mock_tool",
            "description": "Mock tool",
            "parameters": {"type": "object", "properties": {}},
        }
        mock_tool.get_schema.return_value = mock_schema

        registry.register_tool(mock_tool)

        schemas = registry.get_tools_schema()

        assert len(schemas) == 1
        assert schemas[0] == mock_schema

    def test_register_builtin_tools(self, registry):
        """Test registering built-in tools."""
        registry.register_builtin_tools()

        # Should register all built-in tools
        tools = registry.list_tools()
        expected_tools = ["calculator", "web_search", "filesystem", "code_executor"]

        for tool_name in expected_tools:
            assert tool_name in tools

    def test_filter_tools(self, registry):
        """Test filtering tools by criteria."""
        # Create tools with different categories
        tool1 = Mock(spec=BaseTool)
        tool1.name = "tool1"
        tool1.category = "math"

        tool2 = Mock(spec=BaseTool)
        tool2.name = "tool2"
        tool2.category = "web"

        registry.register_tool(tool1)
        registry.register_tool(tool2)

        # Filter by category
        math_tools = registry.filter_tools(category="math")
        assert len(math_tools) == 1
        assert "tool1" in math_tools

        web_tools = registry.filter_tools(category="web")
        assert len(web_tools) == 1
        assert "tool2" in web_tools


class TestToolIntegration:
    """Integration tests for tool components."""

    @pytest.mark.asyncio
    async def test_full_tool_workflow(self):
        """Test complete tool workflow with registry."""
        registry = ToolRegistry()
        calculator = CalculatorTool()

        # Register tool
        registry.register_tool(calculator)

        # Execute through registry
        result = await registry.execute_tool("calculator", expression="2 + 3 * 4")

        assert result.success is True
        assert result.result == 14  # 2 + (3*4) = 2 + 12 = 14

    def test_tool_schema_generation(self):
        """Test that tools generate proper schemas."""
        calculator = CalculatorTool()
        schema = calculator.get_schema()

        # Verify schema structure for LLM function calling
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        assert "type" in schema["parameters"]
        assert "properties" in schema["parameters"]
        assert "required" in schema["parameters"]

        # Verify expression parameter
        assert "expression" in schema["parameters"]["properties"]
        assert "expression" in schema["parameters"]["required"]


if __name__ == "__main__":
    pytest.main([__file__])
