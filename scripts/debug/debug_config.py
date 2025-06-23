#!/usr/bin/env python3
"""
Debug Configuration and Tools for MLSys Template
Provides debugging utilities, test runners, and diagnostics.
"""

import os
import sys
import logging
import pytest
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


@dataclass
class DebugConfig:
    """Debug configuration settings."""

    log_level: str = "DEBUG"
    test_mode: bool = True
    verbose: bool = True
    save_logs: bool = True
    log_file: Optional[str] = None
    test_filter: Optional[str] = None
    coverage_report: bool = True


class DebugLogger:
    """Enhanced logger for debugging."""

    def __init__(self, config: DebugConfig):
        self.config = config
        self.setup_logging()

    def setup_logging(self):
        """Setup enhanced logging for debugging."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            handlers=[],
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.log_level))
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)

        # File handler if specified
        handlers = [console_handler]
        if self.config.save_logs:
            log_file = (
                self.config.log_file
                or f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, self.config.log_level))
            file_handler.setFormatter(console_formatter)
            handlers.append(file_handler)

        # Update root logger
        root_logger = logging.getLogger()
        root_logger.handlers = handlers

        # Set specific loggers to debug level
        for logger_name in [
            "src.utils.rate_limiter",
            "src.utils.cache_manager",
            "src.utils.templates",
            "src.plugins.registry",
            "tests",
        ]:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)


class TestRunner:
    """Enhanced test runner with debugging capabilities."""

    def __init__(self, config: DebugConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run_single_test(self, test_path: str, **kwargs) -> int:
        """Run a single test with debugging."""
        args = [
            test_path,
            "-v",
            "--tb=long",
            "--capture=no" if self.config.verbose else "--capture=sys",
        ]

        if self.config.coverage_report:
            args.extend(
                ["--cov=src", "--cov-report=term-missing", "--cov-report=html:htmlcov"]
            )

        # Add any additional args
        args.extend(kwargs.get("extra_args", []))

        self.logger.info(f"Running test: {test_path}")
        self.logger.debug(f"Pytest args: {args}")

        return pytest.main(args)

    def run_failing_tests(self) -> Dict[str, Any]:
        """Run only the failing tests identified earlier."""
        failing_tests = [
            "tests/integration/test_ml_systems_integration.py::TestMLSystemsIntegration::test_llm_pipeline_with_all_components",
            "tests/integration/test_ml_systems_integration.py::TestMLSystemsIntegration::test_concurrent_usage_with_all_systems",
        ]

        results = {}
        for test in failing_tests:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"DEBUGGING TEST: {test}")
            self.logger.info(f"{'=' * 60}")

            result = self.run_single_test(
                test, extra_args=["--pdb-trace", "--maxfail=1"]
            )
            results[test] = result

            if result != 0:
                self.logger.error(f"Test failed: {test}")
            else:
                self.logger.info(f"Test passed: {test}")

        return results

    def run_unit_tests(self) -> int:
        """Run unit tests only."""
        return self.run_single_test(
            "tests/", extra_args=["-m", "not integration", "--maxfail=5"]
        )

    def run_integration_tests(self) -> int:
        """Run integration tests only."""
        return self.run_single_test("tests/integration/", extra_args=["--maxfail=3"])


class SystemDiagnostics:
    """System diagnostics and health checks."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def check_dependencies(self) -> Dict[str, Any]:
        """Check if all dependencies are properly installed."""
        deps_status = {}

        try:
            import pytest

            deps_status["pytest"] = {"installed": True, "version": pytest.__version__}
        except ImportError as e:
            deps_status["pytest"] = {"installed": False, "error": str(e)}

        try:
            import coverage

            deps_status["coverage"] = {
                "installed": True,
                "version": coverage.__version__,
            }
        except ImportError as e:
            deps_status["coverage"] = {"installed": False, "error": str(e)}

        # Check our modules
        try:
            from src.utils.rate_limiter import RateLimiter  # noqa: F401
            from src.utils.cache_manager import CacheManager  # noqa: F401
            from src.utils.templates import TemplateManager  # noqa: F401

            deps_status["core_modules"] = {"installed": True}
        except ImportError as e:
            deps_status["core_modules"] = {"installed": False, "error": str(e)}

        return deps_status

    def check_test_environment(self) -> Dict[str, Any]:
        """Check test environment setup."""
        env_status = {}

        # Check test directories
        test_dirs = ["tests/", "tests/unit/", "tests/integration/"]
        for test_dir in test_dirs:
            path = Path(test_dir)
            env_status[test_dir] = {
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "files": list(path.glob("*.py")) if path.exists() else [],
            }

        # Check source directories
        src_dirs = ["src/", "src/utils/", "src/models/", "src/data/"]
        for src_dir in src_dirs:
            path = Path(src_dir)
            env_status[src_dir] = {
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "files": list(path.glob("*.py")) if path.exists() else [],
            }

        return env_status

    def run_diagnostics(self) -> Dict[str, Any]:
        """Run all diagnostics."""
        self.logger.info("Running system diagnostics...")

        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "dependencies": self.check_dependencies(),
            "test_environment": self.check_test_environment(),
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "path": sys.path[:3],  # First 3 entries
        }

        return diagnostics


def create_debug_test_file():
    """Create a simple debug test file for troubleshooting."""
    debug_test_content = '''"""
Simple debug test file for troubleshooting.
"""
import pytest
from src.utils.rate_limiter import RateLimiter
from src.utils.cache_manager import CacheManager
from src.utils.templates import TemplateManager

def test_basic_imports():
    """Test that basic imports work."""
    assert RateLimiter is not None
    assert CacheManager is not None
    assert TemplateManager is not None

def test_simple_functionality():
    """Test simple functionality."""
    # Basic test that should always pass
    assert 1 + 1 == 2
    assert "hello" == "hello"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''

    with open("debug_test.py", "w") as f:
        f.write(debug_test_content)

    print("Created debug_test.py - run with: python -m pytest debug_test.py -v -s")


def main():
    """Main debug interface."""
    config = DebugConfig(
        log_level="DEBUG", verbose=True, save_logs=True, coverage_report=True
    )

    # Setup logging
    DebugLogger(config)
    logger = logging.getLogger(__name__)

    logger.info("Starting MLSys Debug Session")
    logger.info("=" * 50)

    # Run diagnostics
    diagnostics = SystemDiagnostics()
    diag_results = diagnostics.run_diagnostics()

    logger.info("System Diagnostics:")
    for key, value in diag_results.items():
        if key not in ["dependencies", "test_environment"]:
            logger.info(f"  {key}: {value}")

    # Check dependencies
    logger.info("\\nDependency Status:")
    for dep, status in diag_results["dependencies"].items():
        status_str = "✅" if status["installed"] else "❌"
        logger.info(f"  {dep}: {status_str}")
        if not status["installed"]:
            logger.error(f"    Error: {status.get('error', 'Unknown')}")

    # Run tests
    test_runner = TestRunner(config)

    print("\\n" + "=" * 60)
    print("DEBUG OPTIONS:")
    print("1. Run failing tests only")
    print("2. Run unit tests")
    print("3. Run integration tests")
    print("4. Create debug test file")
    print("5. Run all tests")
    print("=" * 60)

    choice = input("Enter choice (1-5): ").strip()

    if choice == "1":
        logger.info("Running failing tests...")
        results = test_runner.run_failing_tests()
        print(f"\\nResults: {results}")
    elif choice == "2":
        logger.info("Running unit tests...")
        result = test_runner.run_unit_tests()
        print(f"\\nUnit tests result: {result}")
    elif choice == "3":
        logger.info("Running integration tests...")
        result = test_runner.run_integration_tests()
        print(f"\\nIntegration tests result: {result}")
    elif choice == "4":
        create_debug_test_file()
        logger.info("Created debug_test.py")
    elif choice == "5":
        logger.info("Running all tests...")
        result = test_runner.run_single_test("tests/")
        print(f"\\nAll tests result: {result}")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
