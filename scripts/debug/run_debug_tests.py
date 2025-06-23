#!/usr/bin/env python3
"""
Comprehensive test runner with debugging capabilities.
Handles different test scenarios and provides detailed feedback.
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Enhanced test runner with debugging and reporting."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}

    def run_command(self, cmd: list[str], description: str) -> dict[str, Any]:
        """Run a command and capture results."""
        logger.info(f"Running: {description}")
        if self.verbose:
            logger.info(f"Command: {' '.join(cmd)}")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            duration = time.time() - start_time

            success = result.returncode == 0

            return {
                "success": success,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "description": description,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Command timed out after 5 minutes",
                "duration": time.time() - start_time,
                "description": description,
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "duration": time.time() - start_time,
                "description": description,
            }

    def run_fixed_tests(self) -> dict[str, Any]:
        """Run our fixed test files."""
        logger.info("=" * 60)
        logger.info("RUNNING FIXED TESTS")
        logger.info("=" * 60)

        results = {}

        # Run our fixed integration tests
        results["test_fix"] = self.run_command(
            ["python", "-m", "pytest", "test_fix.py", "-v", "-s"],
            "Fixed Integration Tests",
        )

        # Run debug tests if they exist
        if Path("debug_test.py").exists():
            results["debug_test"] = self.run_command(
                ["python", "-m", "pytest", "debug_test.py", "-v", "-s"], "Debug Tests"
            )

        return results

    def run_unit_tests(self) -> dict[str, Any]:
        """Run unit tests only."""
        logger.info("=" * 60)
        logger.info("RUNNING UNIT TESTS")
        logger.info("=" * 60)

        return {
            "unit_tests": self.run_command(
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/",
                    "-v",
                    "-m",
                    "not integration",
                    "--tb=short",
                ],
                "Unit Tests Only",
            )
        }

    def run_integration_tests(self) -> dict[str, Any]:
        """Run integration tests."""
        logger.info("=" * 60)
        logger.info("RUNNING INTEGRATION TESTS")
        logger.info("=" * 60)

        return {
            "integration_tests": self.run_command(
                ["python", "-m", "pytest", "tests/integration/", "-v", "--tb=short"],
                "Integration Tests",
            )
        }

    def run_specific_failing_tests(self) -> dict[str, Any]:
        """Run the specific tests that were failing."""
        logger.info("=" * 60)
        logger.info("RUNNING SPECIFIC FAILING TESTS")
        logger.info("=" * 60)

        failing_tests = [
            "tests/integration/test_ml_systems_integration.py::TestMLSystemsIntegration::test_llm_pipeline_with_all_components",
            "tests/integration/test_ml_systems_integration.py::TestMLSystemsIntegration::test_concurrent_usage_with_all_systems",
        ]

        results = {}
        for i, test in enumerate(failing_tests):
            test_name = f"failing_test_{i + 1}"
            results[test_name] = self.run_command(
                ["python", "-m", "pytest", test, "-v", "-s", "--tb=long"],
                f"Failing Test {i + 1}: {test.split('::')[-1]}",
            )

        return results

    def run_quick_smoke_test(self) -> dict[str, Any]:
        """Run a quick smoke test to verify basic functionality."""
        logger.info("=" * 60)
        logger.info("RUNNING QUICK SMOKE TEST")
        logger.info("=" * 60)

        return {
            "smoke_test": self.run_command(
                ["python", "-m", "pytest", "tests/", "-v", "--maxfail=3", "-x"],
                "Quick Smoke Test",
            )
        }

    def run_coverage_report(self) -> dict[str, Any]:
        """Generate coverage report."""
        logger.info("=" * 60)
        logger.info("GENERATING COVERAGE REPORT")
        logger.info("=" * 60)

        return {
            "coverage": self.run_command(
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/",
                    "--cov=src",
                    "--cov-report=term-missing",
                    "--cov-report=html",
                ],
                "Coverage Report",
            )
        }

    def print_summary(self, all_results: dict[str, dict[str, Any]]):
        """Print a summary of all test results."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)

        total_suites = 0
        passed_suites = 0

        for suite_name, suite_results in all_results.items():
            if not isinstance(suite_results, dict):
                continue

            logger.info(f"\n{suite_name.upper()} RESULTS:")
            logger.info("-" * 40)

            for test_name, result in suite_results.items():
                if not isinstance(result, dict):
                    continue

                total_suites += 1
                status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
                duration = f"{result.get('duration', 0):.2f}s"

                logger.info(f"  {test_name}: {status} ({duration})")

                if result.get("success", False):
                    passed_suites += 1
                else:
                    # Show error details for failed tests
                    stderr = result.get("stderr", "")
                    stdout = result.get("stdout", "")

                    if stderr:
                        logger.error(f"    Error: {stderr[:200]}...")
                    if "FAILED" in stdout:
                        # Extract failed test info
                        lines = stdout.split("\n")
                        for line in lines:
                            if "FAILED" in line and "::" in line:
                                logger.error(f"    Failed: {line.strip()}")

        if total_suites > 0:
            logger.info("\nOVERALL SUMMARY:")
            logger.info(f"  Test Suites: {passed_suites}/{total_suites} passed")
            logger.info(f"  Success Rate: {passed_suites / total_suites * 100:.1f}%")

            if passed_suites == total_suites:
                logger.info("🎉 ALL TESTS PASSED!")
            else:
                logger.warning(
                    f"⚠️  {total_suites - passed_suites} test suite(s) failed"
                )
        else:
            logger.warning("No test suites found or processed.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced test runner with debugging")
    parser.add_argument(
        "--mode",
        choices=["all", "fixed", "unit", "integration", "failing", "smoke", "coverage"],
        default="fixed",
        help="Test mode to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose)
    all_results = {}

    if args.mode == "all":
        logger.info("Running all test categories...")
        all_results.update(runner.run_fixed_tests())
        all_results.update(runner.run_unit_tests())
        all_results.update(runner.run_integration_tests())
        all_results.update(runner.run_coverage_report())
    elif args.mode == "fixed":
        all_results.update(runner.run_fixed_tests())
    elif args.mode == "unit":
        all_results.update(runner.run_unit_tests())
    elif args.mode == "integration":
        all_results.update(runner.run_integration_tests())
    elif args.mode == "failing":
        all_results.update(runner.run_specific_failing_tests())
    elif args.mode == "smoke":
        all_results.update(runner.run_quick_smoke_test())
    elif args.mode == "coverage":
        all_results.update(runner.run_coverage_report())

    runner.print_summary(all_results)

    # Exit with appropriate code
    all_passed = all(
        result["success"]
        for suite_results in all_results.values()
        for result in suite_results.values()
    )

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
