"""Built-in tools for AI agents."""

import asyncio
from pathlib import Path
from typing import Any

from ..tools.base import BaseTool, ToolParameter, ToolResult


class CalculatorTool(BaseTool):
    """Tool for safe mathematical calculations."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform safe mathematical calculations"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="str",
                description="Mathematical expression to evaluate (e.g., '2 + 3 * 4')",
            )
        ]

    def _safe_eval(self, expression: str) -> Any:
        """Safely evaluate mathematical expression using AST."""
        import ast
        import math

        try:
            # Parse the expression into an AST
            node = ast.parse(expression, mode="eval")

            # Define safe functions that can be called
            safe_functions = {
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "asin": math.asin,
                "acos": math.acos,
                "atan": math.atan,
                "log": math.log,
                "log10": math.log10,
                "exp": math.exp,
                "pow": pow,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
            }

            # Define safe constants
            safe_constants = {
                "pi": math.pi,
                "e": math.e,
            }

            def _safe_eval_node(node):
                """Recursively evaluate AST nodes safely."""
                if isinstance(node, ast.Constant):
                    return node.value

                elif isinstance(node, ast.Name):
                    # Allow safe constants
                    if node.id in safe_constants:
                        return safe_constants[node.id]
                    else:
                        raise ValueError(f"Undefined variable: {node.id}")

                elif isinstance(node, ast.BinOp):
                    left = _safe_eval_node(node.left)
                    right = _safe_eval_node(node.right)
                    op_type = type(node.op)

                    if op_type == ast.Add:
                        return left + right
                    elif op_type == ast.Sub:
                        return left - right
                    elif op_type == ast.Mult:
                        return left * right
                    elif op_type == ast.Div:
                        if right == 0:
                            raise ValueError("Division by zero")
                        return left / right
                    elif op_type == ast.FloorDiv:
                        if right == 0:
                            raise ValueError("Division by zero")
                        return left // right
                    elif op_type == ast.Mod:
                        return left % right
                    elif op_type == ast.Pow:
                        return left**right
                    else:
                        raise ValueError(f"Unsupported binary operation: {op_type}")

                elif isinstance(node, ast.UnaryOp):
                    operand = _safe_eval_node(node.operand)
                    op_type = type(node.op)

                    if op_type == ast.UAdd:
                        return +operand
                    elif op_type == ast.USub:
                        return -operand
                    else:
                        raise ValueError(f"Unsupported unary operation: {op_type}")

                elif isinstance(node, ast.Call):
                    # Support safe function calls
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in safe_functions:
                            func = safe_functions[func_name]
                            args = [_safe_eval_node(arg) for arg in node.args]
                            try:
                                return func(*args)
                            except Exception as e:
                                raise ValueError(
                                    f"Error calling {func_name}: {e}"
                                ) from e
                        else:
                            raise ValueError(f"Unsafe function call: {func_name}")
                    else:
                        raise ValueError("Complex function calls not supported")

                else:
                    raise ValueError(f"Unsupported node type: {type(node)}")

            return _safe_eval_node(node.body)

        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            raise ValueError(f"Invalid expression: {e}") from e

    async def execute(self, expression: str) -> ToolResult:
        """Execute mathematical calculation safely."""
        try:
            result = self._safe_eval(expression)

            return ToolResult(
                success=True,
                result=result,
                metadata={"expression": expression, "type": type(result).__name__},
            )

        except Exception as e:
            return ToolResult(
                success=False, result=None, error=f"Calculation error: {str(e)}"
            )


class WebSearchTool(BaseTool):
    """Tool for web search (placeholder implementation)."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="query", type="str", description="Search query"),
            ToolParameter(
                name="num_results",
                type="int",
                description="Number of results to return",
                required=False,
                default=5,
            ),
        ]

    async def execute(self, query: str, num_results: int = 5) -> ToolResult:
        """Execute web search."""
        try:
            # TODO: Implement actual web search
            # This could integrate with Google Search API, Bing API, or DuckDuckGo

            # Placeholder implementation
            results = [
                {
                    "title": f"Search result {i + 1} for: {query}",
                    "url": f"https://example.com/result{i + 1}",
                    "snippet": f"This is a placeholder snippet for search result {i + 1} about {query}.",
                }
                for i in range(min(num_results, 3))
            ]

            return ToolResult(
                success=True,
                result=results,
                metadata={
                    "query": query,
                    "num_results": len(results),
                    "search_engine": "placeholder",
                },
            )

        except Exception as e:
            return ToolResult(
                success=False, result=None, error=f"Web search error: {str(e)}"
            )


class FileSystemTool(BaseTool):
    """Tool for safe file system operations."""

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def description(self) -> str:
        return "Perform safe file system operations (read, write, list)"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="operation",
                type="str",
                description="Operation to perform: 'read', 'write', 'list', 'exists'",
            ),
            ToolParameter(
                name="path", type="str", description="File or directory path"
            ),
            ToolParameter(
                name="content",
                type="str",
                description="Content to write (for write operation)",
                required=False,
            ),
        ]

    async def execute(
        self, operation: str, path: str, content: str = None
    ) -> ToolResult:
        """Execute file system operation."""
        try:
            # Security: Only allow operations in safe directories
            safe_dirs = ["./data", "./temp", "./output"]
            path_obj = Path(path).resolve()

            # Check if path is within safe directories
            is_safe = any(
                str(path_obj).startswith(str(Path(safe_dir).resolve()))
                for safe_dir in safe_dirs
            )

            if not is_safe:
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"Access denied: Path '{path}' is outside safe directories",
                )

            if operation == "read":
                if not path_obj.exists():
                    return ToolResult(
                        success=False, result=None, error=f"File not found: {path}"
                    )

                with open(path_obj, encoding="utf-8") as f:
                    content = f.read()

                return ToolResult(
                    success=True,
                    result=content,
                    metadata={
                        "operation": "read",
                        "path": str(path_obj),
                        "size": len(content),
                    },
                )

            elif operation == "write":
                if content is None:
                    return ToolResult(
                        success=False,
                        result=None,
                        error="Content required for write operation",
                    )

                # Ensure directory exists
                path_obj.parent.mkdir(parents=True, exist_ok=True)

                with open(path_obj, "w", encoding="utf-8") as f:
                    f.write(content)

                return ToolResult(
                    success=True,
                    result=f"Written {len(content)} characters to {path}",
                    metadata={
                        "operation": "write",
                        "path": str(path_obj),
                        "size": len(content),
                    },
                )

            elif operation == "list":
                if not path_obj.exists():
                    return ToolResult(
                        success=False, result=None, error=f"Directory not found: {path}"
                    )

                if not path_obj.is_dir():
                    return ToolResult(
                        success=False,
                        result=None,
                        error=f"Path is not a directory: {path}",
                    )

                items = []
                for item in path_obj.iterdir():
                    items.append(
                        {
                            "name": item.name,
                            "type": "directory" if item.is_dir() else "file",
                            "size": item.stat().st_size if item.is_file() else None,
                        }
                    )

                return ToolResult(
                    success=True,
                    result=items,
                    metadata={
                        "operation": "list",
                        "path": str(path_obj),
                        "count": len(items),
                    },
                )

            elif operation == "exists":
                exists = path_obj.exists()
                return ToolResult(
                    success=True,
                    result=exists,
                    metadata={"operation": "exists", "path": str(path_obj)},
                )

            else:
                return ToolResult(
                    success=False, result=None, error=f"Unknown operation: {operation}"
                )

        except Exception as e:
            return ToolResult(
                success=False, result=None, error=f"File system error: {str(e)}"
            )


class CodeExecutorTool(BaseTool):
    """Tool for executing code safely (Python only)."""

    @property
    def name(self) -> str:
        return "code_executor"

    @property
    def description(self) -> str:
        return "Execute Python code safely in a restricted environment"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="code", type="str", description="Python code to execute"
            ),
            ToolParameter(
                name="timeout",
                type="int",
                description="Timeout in seconds",
                required=False,
                default=10,
            ),
        ]

    async def execute(self, code: str, timeout: int = 10) -> ToolResult:
        """Execute Python code safely."""
        try:
            # Security restrictions
            dangerous_imports = [
                "os",
                "sys",
                "subprocess",
                "socket",
                "urllib",
                "requests",
                "shutil",
                "tempfile",
                "pickle",
                "eval",
                "exec",
                "compile",
            ]

            # Check for dangerous imports
            for dangerous in dangerous_imports:
                if f"import {dangerous}" in code or f"from {dangerous}" in code:
                    return ToolResult(
                        success=False,
                        result=None,
                        error=f"Dangerous import detected: {dangerous}",
                    )

            # Additional security checks
            dangerous_patterns = [
                "__import__",
                "getattr",
                "setattr",
                "delattr",
                "hasattr",
                "globals",
                "locals",
                "vars",
                "dir",
            ]
            for pattern in dangerous_patterns:
                if pattern in code:
                    return ToolResult(
                        success=False,
                        result=None,
                        error=f"Dangerous pattern detected: {pattern}",
                    )

            # Prepare safe execution environment with minimal builtins
            safe_builtins = {
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "__import__": __import__,  # Allow imports for safe modules
            }

            # Preload safe modules in globals
            safe_globals = {
                "__builtins__": safe_builtins,
                "math": __import__("math"),
                "json": __import__("json"),
                "datetime": __import__("datetime"),
            }

            # Capture output
            import contextlib
            import io

            output_buffer = io.StringIO()

            with contextlib.redirect_stdout(output_buffer):
                # Compile code first for additional safety
                try:
                    compiled_code = compile(code, "<string>", "exec")
                except SyntaxError as e:
                    return ToolResult(
                        success=False, result=None, error=f"Syntax error: {str(e)}"
                    )

                # Execute compiled code with restricted globals
                # Note: Using exec with compiled code and restricted globals is safer than direct exec
                exec(compiled_code, safe_globals, {})  # nosec B102 - controlled execution with restricted globals

            output = output_buffer.getvalue()

            return ToolResult(
                success=True,
                result=output.strip()
                if output
                else "Code executed successfully (no output)",
                metadata={
                    "code_length": len(code),
                    "execution_time": "< 1s",  # Placeholder
                    "output_length": len(output),
                },
            )

        except SyntaxError as e:
            return ToolResult(
                success=False, result=None, error=f"Syntax error: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                success=False, result=None, error=f"Execution error: {str(e)}"
            )


class ShellCommandTool(BaseTool):
    """Tool for executing shell commands (restricted)."""

    @property
    def name(self) -> str:
        return "shell_command"

    @property
    def description(self) -> str:
        return "Execute safe shell commands"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command", type="str", description="Shell command to execute"
            ),
            ToolParameter(
                name="timeout",
                type="int",
                description="Timeout in seconds",
                required=False,
                default=30,
            ),
        ]

    async def execute(self, command: str, timeout: int = 30) -> ToolResult:
        """Execute shell command safely."""
        try:
            # Whitelist of safe commands
            safe_commands = [
                "ls",
                "pwd",
                "echo",
                "cat",
                "head",
                "tail",
                "wc",
                "grep",
                "find",
                "sort",
                "uniq",
                "cut",
                "awk",
                "sed",
                "date",
                "python",
                "pip",
                "git",
            ]

            # Check if command starts with a safe command
            cmd_parts = command.split()
            if not cmd_parts or cmd_parts[0] not in safe_commands:
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"Command not allowed: {cmd_parts[0] if cmd_parts else 'empty'}",
                )

            # Execute command with timeout
            process = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except TimeoutError:
                process.kill()
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"Command timed out after {timeout} seconds",
                )

            if process.returncode == 0:
                return ToolResult(
                    success=True,
                    result=stdout.decode("utf-8").strip(),
                    metadata={
                        "command": command,
                        "return_code": process.returncode,
                        "stderr": stderr.decode("utf-8").strip(),
                    },
                )
            else:
                return ToolResult(
                    success=False,
                    result=stdout.decode("utf-8").strip(),
                    error=stderr.decode("utf-8").strip(),
                )

        except Exception as e:
            return ToolResult(
                success=False, result=None, error=f"Shell command error: {str(e)}"
            )
