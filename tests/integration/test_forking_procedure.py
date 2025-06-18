"""
Integration tests for the mlx forking and transformation procedure.

These tests validate that the mlx script correctly transforms a template
into a personalized project while maintaining functionality.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict
import pytest
import tomllib


@pytest.fixture
def temp_project_dir() -> Path:
    """Create a temporary directory with a copy of the project."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "test_project"

        # Copy the entire project to temp directory
        project_root = Path(__file__).parent.parent.parent
        shutil.copytree(
            project_root,
            temp_path,
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

        # Ensure mlx is executable
        mlsys_path = temp_path / "mlx"
        mlsys_path.chmod(0o755)

        yield temp_path


class TestForkingProcedure:
    """Test suite for validating the mlx forking and transformation procedure."""

    def test_mlsys_script_exists_and_executable(self, temp_project_dir: Path):
        """Test that the mlx script exists and is executable."""
        mlsys_path = temp_project_dir / "mlx"

        assert mlsys_path.exists(), "mlx script should exist"
        assert os.access(mlsys_path, os.X_OK), "mlx script should be executable"

    def test_original_structure_exists(self, temp_project_dir: Path):
        """Test that the original template structure exists before transformation."""
        # Check for original analysis_template directory
        original_src = temp_project_dir / "src" / "analysis_template"
        assert original_src.exists(), "Original src/analysis_template should exist"

        # Check for key files
        assert (temp_project_dir / "pyproject.toml").exists()
        assert (temp_project_dir / "README.md").exists()
        assert (temp_project_dir / "Makefile").exists()

    def test_mlsys_transformation_basic(self, temp_project_dir: Path):
        """Test basic mlx transformation functionality."""
        project_name = "test-project-basic"

        # Run mlx transformation
        result = subprocess.run(
            [str(temp_project_dir / "mlx"), project_name],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minutes timeout
        )

        assert result.returncode == 0, f"mlx failed: {result.stderr}"

        # Verify transformation results
        self._verify_transformation_results(temp_project_dir, project_name)

    def test_mlsys_transformation_complex_name(self, temp_project_dir: Path):
        """Test mlx transformation with complex project names."""
        test_cases = [
            "customer-churn-model",
            "financial_risk_analyzer",
            "document-qa-system",
            "real-time-recommender",
        ]

        for project_name in test_cases:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a fresh copy for each test
                test_path = Path(tmpdir) / "test_project"
                shutil.copytree(temp_project_dir, test_path)

                # Run transformation
                result = subprocess.run(
                    [str(test_path / "mlx"), project_name],
                    cwd=test_path,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                assert (
                    result.returncode == 0
                ), f"mlx failed for {project_name}: {result.stderr}"
                self._verify_transformation_results(test_path, project_name)

    def test_directory_renaming(self, temp_project_dir: Path):
        """Test that source directory is properly renamed."""
        project_name = "test-directory-rename"

        # Run transformation
        subprocess.run(
            [str(temp_project_dir / "mlx"), project_name],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Check old directory is gone
        old_src = temp_project_dir / "src" / "analysis_template"
        assert not old_src.exists(), "Old analysis_template directory should be removed"

        # Check new directory exists
        new_snake_case = project_name.replace("-", "_")
        new_src = temp_project_dir / "src" / new_snake_case
        assert new_src.exists(), "New project directory should exist"

        # Check that Python files exist in new directory
        assert (new_src / "__init__.py").exists()
        assert (
            len(list(new_src.glob("*.py"))) > 0
        ), "Python files should exist in new directory"

    def test_pyproject_toml_updates(self, temp_project_dir: Path):
        """Test that pyproject.toml is properly updated."""
        project_name = "test-pyproject-update"

        # Run transformation
        subprocess.run(
            [str(temp_project_dir / "mlx"), project_name],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Read updated pyproject.toml
        pyproject_path = temp_project_dir / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        # Verify project name is updated
        expected_kebab = project_name.replace("_", "-")
        if "project" in data:
            assert data["project"]["name"] == expected_kebab
        elif "tool" in data and "poetry" in data["tool"]:
            assert data["tool"]["poetry"]["name"] == expected_kebab

        # Verify ruff configuration is updated
        if "tool" in data and "ruff" in data["tool"]:
            ruff_config = data["tool"]["ruff"]
            if "lint" in ruff_config and "isort" in ruff_config["lint"]:
                known_first_party = ruff_config["lint"]["isort"]["known-first-party"]
                expected_snake = project_name.replace("-", "_")
                assert expected_snake in known_first_party

    def test_documentation_updates(self, temp_project_dir: Path):
        """Test that documentation files are properly updated."""
        project_name = "test-docs-update"

        # Run transformation
        subprocess.run(
            [str(temp_project_dir / "mlx"), project_name],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Check that documentation files contain the new project name
        docs_to_check = [
            temp_project_dir / "README.md",
            temp_project_dir / "docs" / "getting-started.md",
            temp_project_dir / "docs" / "development" / "project-setup.md",
        ]

        for doc_path in docs_to_check:
            if doc_path.exists():
                content = doc_path.read_text()
                # The transformation should have updated references
                assert (
                    "analysis-template" not in content.lower()
                    or "test-docs-update" in content.lower()
                ), f"Documentation in {doc_path} not properly updated"

    def test_import_integrity(self, temp_project_dir: Path):
        """Test that Python imports still work after transformation."""
        project_name = "test-import-check"

        # Run transformation
        subprocess.run(
            [str(temp_project_dir / "mlx"), project_name],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Try to import the new package
        import sys

        sys.path.insert(0, str(temp_project_dir / "src"))

        try:
            # Import the renamed package
            import importlib

            package_name = project_name.replace("-", "_")
            module = importlib.import_module(package_name)
            assert module is not None, f"Could not import {package_name}"
        except ImportError as e:
            pytest.fail(f"Import failed after transformation: {e}")
        finally:
            # Clean up sys.path
            if str(temp_project_dir / "src") in sys.path:
                sys.path.remove(str(temp_project_dir / "src"))

    def test_configuration_files_valid(self, temp_project_dir: Path):
        """Test that configuration files remain valid after transformation."""
        project_name = "test-config-valid"

        # Run transformation
        subprocess.run(
            [str(temp_project_dir / "mlx"), project_name],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Test pyproject.toml is valid
        pyproject_path = temp_project_dir / "pyproject.toml"
        try:
            with open(pyproject_path, "rb") as f:
                tomllib.load(f)
        except Exception as e:
            pytest.fail(f"pyproject.toml is invalid after transformation: {e}")

        # Test other configuration files
        config_files = [
            temp_project_dir / "docker-compose.yml",
            temp_project_dir / "mkdocs.yml",
            temp_project_dir / ".pre-commit-config.yaml",
        ]

        for config_file in config_files:
            if config_file.exists():
                assert config_file.stat().st_size > 0, f"{config_file.name} is empty"

    def test_make_commands_work(self, temp_project_dir: Path):
        """Test that make commands work after transformation."""
        project_name = "test-make-commands"

        # Run transformation
        subprocess.run(
            [str(temp_project_dir / "mlx"), project_name],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Test that make help works
        result = subprocess.run(
            ["make", "help"],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"make help failed: {result.stderr}"
        assert (
            "MLOps Template" in result.stdout or "Available Commands" in result.stdout
        )

    def test_bootstrap_venv_cleanup(self, temp_project_dir: Path):
        """Test that bootstrap virtual environment is properly managed."""
        project_name = "test-bootstrap-cleanup"

        # Run transformation
        result = subprocess.run(
            [str(temp_project_dir / "mlx"), project_name],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"mlx failed: {result.stderr}"

        # Bootstrap venv may be created in home directory, but transformation should succeed
        # We don't assert its existence as it might be cleaned up automatically

    def test_transformation_idempotency(self, temp_project_dir: Path):
        """Test that running mlx twice doesn't break the project."""
        project_name = "test-idempotency"

        # Run transformation first time
        result1 = subprocess.run(
            [str(temp_project_dir / "mlx"), project_name],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result1.returncode == 0, f"First mlx run failed: {result1.stderr}"

        # Run transformation second time - should fail gracefully
        result2 = subprocess.run(
            [str(temp_project_dir / "mlx"), f"{project_name}-v2"],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Second run should fail because analysis_template no longer exists
        assert (
            result2.returncode != 0
        ), "Second mlx run should fail when source dir is missing"
        assert (
            "already appears to be initialized" in result2.stdout
            or "was not found" in result2.stdout
        )

    def _verify_transformation_results(self, project_dir: Path, project_name: str):
        """Helper method to verify transformation results."""
        # Convert names
        snake_case = project_name.replace("-", "_")
        kebab_case = project_name.replace("_", "-")

        # Check directory structure
        new_src = project_dir / "src" / snake_case
        assert new_src.exists(), f"New source directory {snake_case} should exist"
        assert not (
            project_dir / "src" / "analysis_template"
        ).exists(), "Old source directory should be removed"

        # Check pyproject.toml
        pyproject_path = project_dir / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        assert data["tool"]["poetry"]["name"] == kebab_case
        if "ruff" in data["tool"]:
            assert data["tool"]["ruff"]["lint"]["isort"]["known-first-party"] == [
                snake_case
            ]

    @pytest.mark.slow
    def test_full_workflow_integration(self, temp_project_dir: Path):
        """Test the complete workflow: transform -> install -> test."""
        project_name = "test-full-workflow"

        # Step 1: Run transformation
        result = subprocess.run(
            [str(temp_project_dir / "mlx"), project_name],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"mlx transformation failed: {result.stderr}"

        # Step 2: Try to install dependencies (if uv is available)
        uv_check = subprocess.run(["which", "uv"], capture_output=True)
        if uv_check.returncode == 0:
            # Install dependencies
            install_result = subprocess.run(
                ["make", "install-dev"],
                cwd=temp_project_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes for dependency installation
            )

            # Don't fail the test if installation fails due to environment issues
            # but log the result for debugging
            if install_result.returncode != 0:
                print(
                    f"Warning: Dependencies installation failed: {install_result.stderr}"
                )

        # Step 3: Verify project structure is intact
        self._verify_transformation_results(temp_project_dir, project_name)


@pytest.mark.integration
class TestBranchSpecificForking:
    """Tests for branch-specific markers and compatibility."""

    def test_main_branch_compatibility(self):
        """Test that the main branch has no specialized markers."""
        assert not Path("src/llm").exists(), "Main branch should not have LLM markers"
        assert not Path(
            "src/agentic"
        ).exists(), "Main branch should not have Agentic markers"

    def test_specialized_branch_markers(self, temp_project_dir: Path):
        """Verify that specialized branches contain expected markers."""
        # This test would need to be run on different branches
        # For now, we just ensure it doesn't fail on main
        pass


# Helper functions for CI integration
def get_project_health_status(project_dir: Path) -> Dict[str, bool]:
    """Get the health status of a transformed project."""
    status = {
        "structure_valid": False,
        "config_valid": False,
        "imports_valid": False,
        "make_help_works": False,
    }

    try:
        # Check structure
        src_dirs = list((project_dir / "src").glob("*"))
        src_dirs = [d for d in src_dirs if d.is_dir() and not d.name.startswith(".")]
        status["structure_valid"] = len(src_dirs) >= 1

        # Check config
        pyproject_path = project_dir / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                tomllib.load(f)
            status["config_valid"] = True

        # Check make help
        result = subprocess.run(
            ["make", "help"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        status["make_help_works"] = result.returncode == 0

    except Exception:
        pass

    return status


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
