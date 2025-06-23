#!/usr/bin/env python3
"""
Smoke test script for the mlsys forking procedure.

This script provides a quick way to test the forking procedure locally
without running the full test suite. It's useful for development and debugging.

Usage:
    python scripts/test_forking_smoke.py [project-name]

Example:
    python scripts/test_forking_smoke.py my-test-project
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


def run_command(cmd: list, cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out: {' '.join(cmd)}")
        sys.exit(1)


def copy_project(source: Path, dest: Path) -> None:
    """Copy the project to a temporary directory."""
    print(f"📂 Copying project from {source} to {dest}")

    shutil.copytree(
        source,
        dest,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".venv",
            ".bootstrap_venv",
            "htmlcov",
            "logs",
            "*.egg-info",
            ".coverage*",
            ".mypy_cache",
            ".ruff_cache",
        ),
    )

    # Ensure mlsys is executable
    mlsys_path = dest / "mlsys"
    mlsys_path.chmod(0o755)


def verify_original_structure(project_dir: Path) -> bool:
    """Verify the original project structure exists."""
    print("🔍 Verifying original structure...")

    checks = [
        project_dir / "mlsys",
        project_dir / "src" / "analysis_template",
        project_dir / "pyproject.toml",
        project_dir / "README.md",
    ]

    for path in checks:
        if not path.exists():
            print(f"❌ Missing: {path}")
            return False

    print("✅ Original structure verified")
    return True


def run_transformation(project_dir: Path, project_name: str) -> bool:
    """Run the mlsys transformation."""
    print(f"🎭 Running transformation: {project_name}")

    mlsys_path = project_dir / "mlsys"
    result = run_command([str(mlsys_path), project_name], project_dir, timeout=120)

    if result.returncode != 0:
        print("❌ Transformation failed:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False

    print("✅ Transformation completed")
    return True


def verify_transformation(project_dir: Path, project_name: str) -> dict[str, bool]:
    """Verify the transformation was successful."""
    print("🔍 Verifying transformation results...")

    results = {
        "directory_renamed": False,
        "old_directory_removed": False,
        "pyproject_updated": False,
        "make_help_works": False,
    }

    # Check directory renaming
    snake_case = project_name.replace("-", "_")
    new_src = project_dir / "src" / snake_case
    old_src = project_dir / "src" / "analysis_template"

    results["directory_renamed"] = new_src.exists()
    results["old_directory_removed"] = not old_src.exists()

    # Check pyproject.toml updates
    pyproject_path = project_dir / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)

            kebab_case = project_name.replace("_", "-")

            # Check project name
            if "project" in data and data["project"].get("name") == kebab_case:
                results["pyproject_updated"] = True
            elif (
                "tool" in data
                and "poetry" in data["tool"]
                and data["tool"]["poetry"].get("name") == kebab_case
            ):
                results["pyproject_updated"] = True

        except Exception as e:
            print(f"Warning: Could not parse pyproject.toml: {e}")
            # Fall back to text-based verification
            loaded_data = pyproject_path.read_text()
            kebab_case = project_name.replace("_", "-")
            if f'name = "{kebab_case}"' in loaded_data:
                results["pyproject_updated"] = True
                print("✅ Verified project name update via text search")

    # Check make help works
    result = run_command(["make", "help"], project_dir, timeout=30)
    results["make_help_works"] = result.returncode == 0

    return results


def print_results(results: dict[str, bool]) -> bool:
    """Print the verification results."""
    print("\n📊 Test Results:")
    print("=" * 50)

    all_passed = True
    for test, passed in results.items():
        status = "✅" if passed else "❌"
        test_name = test.replace("_", " ").title()
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    print("=" * 50)

    if all_passed:
        print("🎉 All tests passed! The forking procedure is working correctly.")
    else:
        print("💥 Some tests failed. The forking procedure needs attention.")

    return all_passed


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Smoke test for mlsys forking procedure"
    )
    parser.add_argument(
        "project_name",
        nargs="?",
        default="smoke-test-project",
        help="Name for the test project (default: smoke-test-project)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary directory for debugging",
    )

    args = parser.parse_args()

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print("🧪 MLOps Template Forking Smoke Test")
    print("=" * 50)
    print(f"Project: {args.project_name}")
    print(f"Source: {project_root}")
    print()

    success = False
    temp_dir = None

    try:
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="mlsys_smoke_test_"))
        test_project_dir = temp_dir / "test_project"

        print(f"🏗️  Working directory: {test_project_dir}")
        print()

        # Step 1: Copy project
        copy_project(project_root, test_project_dir)

        # Step 2: Verify original structure
        if not verify_original_structure(test_project_dir):
            sys.exit(1)

        # Step 3: Run transformation
        if not run_transformation(test_project_dir, args.project_name):
            sys.exit(1)

        # Step 4: Verify transformation
        results = verify_transformation(test_project_dir, args.project_name)
        success = print_results(results)

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if temp_dir and temp_dir.exists():
            if args.keep_temp:
                print(f"\n📁 Temporary directory preserved: {temp_dir}")
            else:
                try:
                    shutil.rmtree(temp_dir)
                    print("\n🧹 Cleaned up temporary directory")
                except Exception as e:
                    print(f"\n⚠️  Could not clean up {temp_dir}: {e}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
