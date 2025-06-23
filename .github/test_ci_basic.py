#!/usr/bin/env python3
"""
Basic CI test to verify environment setup works correctly.
This test can be run in GitHub Actions to verify everything is working.
"""

import subprocess
import sys
from pathlib import Path

# Add project root to Python path to enable src imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_python_version():
    """Test that Python version is correct."""
    assert sys.version_info >= (3, 10), f"Python version {sys.version} is too old"
    print(f"✓ Python version: {sys.version}")


def test_project_structure():
    """Test that project structure is correct."""
    required_dirs = ["src", "tests", "workflows", ".github"]
    required_files = ["pyproject.toml", "Makefile", "README.md"]

    project_root = Path.cwd()

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Required directory {dir_name} not found"
        print(f"✓ Directory exists: {dir_name}")

    for file_name in required_files:
        file_path = project_root / file_name
        assert file_path.exists(), f"Required file {file_name} not found"
        print(f"✓ File exists: {file_name}")


def test_dependencies():
    """Test that key dependencies are importable."""
    try:
        import polars

        print(f"✓ polars: {polars.__version__}")
    except ImportError as e:
        print(f"✗ polars import failed: {e}")
        raise

    try:
        import sklearn

        print(f"✓ scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        raise

    try:
        import fastapi

        print(f"✓ fastapi: {fastapi.__version__}")
    except ImportError as e:
        print(f"✗ fastapi import failed: {e}")
        raise


def test_src_importable():
    """Test that src modules are importable."""
    try:
        from src.platform.utils.common import get_project_root

        project_root = get_project_root()
        assert project_root.exists(), "Project root not found"
        print(f"✓ src.utils.common importable, project root: {project_root}")
    except ImportError as e:
        print(f"✗ src.utils.common import failed: {e}")
        raise


def test_makefile_commands():
    """Test that Makefile commands work."""
    try:
        # Test help command
        result = subprocess.run(
            ["make", "help"], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, "make help failed"
        print("✓ make help works")

        # Test lint command
        result = subprocess.run(
            ["make", "lint"], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, f"make lint failed: {result.stderr}"
        print("✓ make lint works")

    except subprocess.TimeoutExpired:
        print("✗ Makefile command timed out")
        raise
    except Exception as e:
        print(f"✗ Makefile test failed: {e}")
        raise


if __name__ == "__main__":
    print("Running basic CI tests...")

    test_python_version()
    test_project_structure()
    test_dependencies()
    test_src_importable()
    test_makefile_commands()

    print("\n✅ All basic CI tests passed!")
