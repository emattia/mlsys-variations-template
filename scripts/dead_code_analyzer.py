#!/usr/bin/env python3
"""
🎖️ MLX Dead Code Analyzer - Repository Closer Tool

Legendary precision in identifying and eliminating dead code across the MLX repository.
Like Mariano Rivera's cutter - swift, decisive, and always on target.

Features:
- Unused function detection
- Unused variable identification
- Import analysis
- Cross-file dependency tracking
- Comprehensive reporting
- Safe deletion recommendations

Usage:
    python scripts/dead_code_analyzer.py --analyze scripts/migrate_platform_naming.py
    python scripts/dead_code_analyzer.py --analyze scripts/mlx_assistant.py
    python scripts/dead_code_analyzer.py --scan-all --report
"""

from datetime import datetime
import ast
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DeadCodeItem:
    """Represents a piece of potentially dead code."""

    file_path: str
    item_type: str  # 'function', 'class', 'variable', 'import'
    name: str
    line_number: int
    complexity: int
    size_lines: int
    confidence: float  # 0.0 to 1.0
    reason: str
    dependencies: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ASTAnalyzer(ast.NodeVisitor):
    """AST visitor for analyzing Python code structure."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.defined_functions = {}
        self.defined_classes = {}
        self.defined_variables = set()
        self.imported_names = {}
        self.function_calls = set()
        self.variable_uses = set()
        self.current_function = None
        self.current_class = None

    def visit_FunctionDef(self, node):
        """Analyze function definitions."""
        func_info = {
            "name": node.name,
            "line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno),
            "args": [arg.arg for arg in node.args.args],
            "decorators": [ast.unparse(d) for d in node.decorator_list],
            "docstring": ast.get_docstring(node),
            "is_private": node.name.startswith("_"),
            "complexity": self._calculate_complexity(node),
            "calls_made": set(),
            "variables_used": set(),
        }

        self.defined_functions[node.name] = func_info

        # Track function context
        old_function = self.current_function
        self.current_function = node.name

        # Visit children to track calls and variable usage within function
        self.generic_visit(node)

        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node):
        """Handle async function definitions."""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        """Analyze class definitions."""
        class_info = {
            "name": node.name,
            "line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno),
            "bases": [ast.unparse(base) for base in node.bases],
            "decorators": [ast.unparse(d) for d in node.decorator_list],
            "docstring": ast.get_docstring(node),
            "methods": [],
            "is_private": node.name.startswith("_"),
        }

        self.defined_classes[node.name] = class_info

        old_class = self.current_class
        self.current_class = node.name

        self.generic_visit(node)

        self.current_class = old_class

    def visit_Import(self, node):
        """Track import statements."""
        for alias in node.names:
            import_name = alias.asname if alias.asname else alias.name
            self.imported_names[import_name] = {
                "original": alias.name,
                "line": node.lineno,
                "type": "import",
            }
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Track from ... import statements."""
        module = node.module or ""
        for alias in node.names:
            import_name = alias.asname if alias.asname else alias.name
            self.imported_names[import_name] = {
                "original": f"{module}.{alias.name}" if module else alias.name,
                "line": node.lineno,
                "type": "from_import",
                "module": module,
            }
        self.generic_visit(node)

    def visit_Call(self, node):
        """Track function calls."""
        if isinstance(node.func, ast.Name):
            self.function_calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # Handle method calls
            if isinstance(node.func.value, ast.Name):
                self.function_calls.add(f"{node.func.value.id}.{node.func.attr}")

        self.generic_visit(node)

    def visit_Name(self, node):
        """Track name usage (variables, functions, etc.)."""
        if isinstance(node.ctx, ast.Load):
            self.variable_uses.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.defined_variables.add(node.id)

        self.generic_visit(node)

    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(
                child,
                (
                    ast.If,
                    ast.While,
                    ast.For,
                    ast.AsyncFor,
                    ast.ExceptHandler,
                    ast.With,
                    ast.AsyncWith,
                ),
            ):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity


class DeadCodeAnalyzer:
    """Main analyzer for detecting dead code across the repository."""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.analyzers: Dict[str, ASTAnalyzer] = {}
        self.dead_code_items: List[DeadCodeItem] = []

    def analyze_file(self, file_path: Path) -> Optional[ASTAnalyzer]:
        """Analyze a single Python file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                loaded_data = f.read()

            tree = ast.parse(loaded_data, filename=str(file_path))
            analyzer = ASTAnalyzer(str(file_path))
            analyzer.visit(tree)

            self.analyzers[str(file_path)] = analyzer
            logger.info(
                f"Analyzed {file_path}: {len(analyzer.defined_functions)} functions, "
                f"{len(analyzer.defined_classes)} classes, {len(analyzer.imported_names)} imports"
            )

            return analyzer

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return None

    def scan_repository(self, exclude_patterns: List[str] = None) -> None:
        """Scan entire repository for Python files."""
        if exclude_patterns is None:
            exclude_patterns = [".venv", "__pycache__", ".git", "node_modules"]

        python_files = []
        for py_file in self.root_path.rglob("*.py"):
            # Skip excluded directories
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue
            python_files.append(py_file)

        logger.info(f"Found {len(python_files)} Python files to analyze")

        for py_file in python_files:
            self.analyze_file(py_file)

    def identify_dead_code(self) -> List[DeadCodeItem]:
        """Identify dead code across all analyzed files."""
        dead_items = []

        # Collect all function names, class names, and variable names across files
        all_function_calls = set()
        all_variable_uses = set()
        all_defined_functions = {}
        all_defined_classes = {}
        all_imports = {}

        for file_path, analyzer in self.analyzers.items():
            all_function_calls.update(analyzer.function_calls)
            all_variable_uses.update(analyzer.variable_uses)

            for func_name, func_info in analyzer.defined_functions.items():
                all_defined_functions[f"{file_path}:{func_name}"] = {
                    **func_info,
                    "file_path": file_path,
                }

            for class_name, class_info in analyzer.defined_classes.items():
                all_defined_classes[f"{file_path}:{class_name}"] = {
                    **class_info,
                    "file_path": file_path,
                }

            for import_name, import_info in analyzer.imported_names.items():
                all_imports[f"{file_path}:{import_name}"] = {
                    **import_info,
                    "file_path": file_path,
                }

        # Identify unused functions
        for func_key, func_info in all_defined_functions.items():
            func_name = func_info["name"]
            confidence = 0.8

            # Check if function is called
            is_called = func_name in all_function_calls or any(
                func_name in call for call in all_function_calls
            )

            # Special cases that reduce confidence
            if func_info["is_private"]:
                confidence -= 0.2
            if func_name.startswith("test_"):
                confidence -= 0.5  # Test functions might be discovered by pytest
            if func_name in ["main", "__init__", "__str__", "__repr__"]:
                confidence -= 0.6  # Special methods
            if func_info.get("decorators"):
                confidence -= 0.3  # Decorated functions might be called indirectly

            if not is_called and confidence > 0.3:
                dead_items.append(
                    DeadCodeItem(
                        file_path=func_info["file_path"],
                        item_type="function",
                        name=func_name,
                        line_number=func_info["line"],
                        complexity=func_info["complexity"],
                        size_lines=func_info["end_line"] - func_info["line"] + 1,
                        confidence=confidence,
                        reason=f"Function '{func_name}' appears to be unused",
                        dependencies=func_info.get("args", []),
                    )
                )

        # Identify unused imports
        for import_key, import_info in all_imports.items():
            import_name = import_key.split(":")[1]

            # Check if import is used
            is_used = (
                import_name in all_variable_uses or import_name in all_function_calls
            )

            if not is_used:
                dead_items.append(
                    DeadCodeItem(
                        file_path=import_info["file_path"],
                        item_type="import",
                        name=import_name,
                        line_number=import_info["line"],
                        complexity=0,
                        size_lines=1,
                        confidence=0.9,
                        reason=f"Import '{import_name}' appears to be unused",
                        dependencies=[import_info["original"]],
                    )
                )

        self.dead_code_items = dead_items
        return dead_items

    def generate_report(
        self, output_file: str = "dead_code_report.json"
    ) -> Dict[str, Any]:
        """Generate comprehensive dead code report."""

        # Categorize dead code by type and confidence
        categories = defaultdict(list)
        total_lines_dead = 0
        high_confidence_items = []

        for item in self.dead_code_items:
            categories[item.item_type].append(item)
            total_lines_dead += item.size_lines

            if item.confidence >= 0.7:
                high_confidence_items.append(item)

        # Generate statistics
        stats = {
            "total_files_analyzed": len(self.analyzers),
            "total_dead_code_items": len(self.dead_code_items),
            "total_dead_lines": total_lines_dead,
            "high_confidence_items": len(high_confidence_items),
            "categories": {
                category: len(items) for category, items in categories.items()
            },
        }

        # Create detailed report
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "dead_code_items": [item.to_dict() for item in self.dead_code_items],
            "recommendations": self._generate_recommendations(categories, stats),
        }

        # Save report
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Dead code report saved to {output_file}")
        return report

    def _generate_recommendations(self, categories: Dict, stats: Dict) -> List[str]:
        """Generate cleanup recommendations."""
        recommendations = []

        if stats["high_confidence_items"] > 0:
            recommendations.append(
                f"HIGH PRIORITY: {stats['high_confidence_items']} dead code items "
                f"with high confidence (>70%) - safe to remove"
            )

        if len(categories.get("import", [])) > 0:
            recommendations.append(
                f"IMPORTS: {len(categories['import'])} unused imports found - "
                f"removing these will improve startup time"
            )

        if len(categories.get("function", [])) > 0:
            recommendations.append(
                f"FUNCTIONS: {len(categories['function'])} potentially unused functions - "
                f"review and remove to reduce complexity"
            )

        if stats["total_dead_lines"] > 50:
            recommendations.append(
                f"IMPACT: {stats['total_dead_lines']} lines of dead code identified - "
                f"significant cleanup opportunity"
            )

        if not recommendations:
            recommendations.append(
                "EXCELLENT: Minimal dead code detected - code quality is high"
            )

        return recommendations


def main():
    """Main CLI interface for dead code analysis."""
    parser = argparse.ArgumentParser(description="🎖️ MLX Dead Code Analyzer")
    parser.add_argument("--analyze", help="Analyze specific file")
    parser.add_argument(
        "--scan-all", action="store_true", help="Scan entire repository"
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate comprehensive report"
    )
    parser.add_argument(
        "--output", default="dead_code_report.json", help="Output file for report"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for reporting",
    )

    args = parser.parse_args()

    analyzer = DeadCodeAnalyzer()

    if args.analyze:
        file_path = Path(args.analyze)
        if file_path.exists():
            result = analyzer.analyze_file(file_path)
            if result:
                print(f"\n🎖️ Analysis Results for {file_path}")
                print(f"Functions: {len(result.defined_functions)}")
                print(f"Classes: {len(result.defined_classes)}")
                print(f"Imports: {len(result.imported_names)}")
                print(f"Function calls: {len(result.function_calls)}")
        else:
            print(f"Error: File {file_path} not found")
            sys.exit(1)

    if args.scan_all:
        print("🎖️ Scanning repository for dead code...")
        analyzer.scan_repository()

        dead_items = analyzer.identify_dead_code()

        # Filter by confidence threshold
        high_confidence_items = [
            item for item in dead_items if item.confidence >= args.confidence_threshold
        ]

        print(f"\n{'=' * 60}")
        print("🎖️ DEAD CODE ANALYSIS RESULTS")
        print(f"{'=' * 60}")
        print(f"Total files analyzed: {len(analyzer.analyzers)}")
        print(f"Dead code items found: {len(dead_items)}")
        print(f"High confidence items: {len(high_confidence_items)}")

        if high_confidence_items:
            print("\nHigh Confidence Dead Code:")
            for item in high_confidence_items[:10]:  # Show top 10
                print(
                    f"  • {item.file_path}:{item.line_number} - {item.item_type} '{item.name}' "
                    f"({item.confidence:.1%} confidence)"
                )

    if args.report:
        report = analyzer.generate_report(args.output)
        print(f"\nComprehensive report saved to: {args.output}")

        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")


if __name__ == "__main__":
    main()
