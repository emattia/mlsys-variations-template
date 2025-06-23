"""
Integration tests for the mlx forking and transformation procedure.
These tests validate that the mlx script correctly transforms a template
into a personalized project while maintaining functionality.
"""

import tempfile
import shutil
import subprocess
import os
from pathlib import Path
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

    @pytest.mark.integration
    def test_mlsys_script_exists_and_executable(self, temp_project_dir: Path):
        """Test that the mlx script exists and is executable."""
        mlsys_path = temp_project_dir / "mlx"
        assert mlsys_path.exists(), "mlx script should exist"
        assert os.access(mlsys_path, os.X_OK), "mlx script should be executable"

    @pytest.mark.integration
    def test_original_structure_exists(self, temp_project_dir: Path):
        """Test that the original template structure exists before transformation."""
        # Check for original analysis_template directory
        original_src = temp_project_dir / "src" / "analysis_template"
        assert original_src.exists(), "Original src/analysis_template should exist"

        # Check for key files
        assert (temp_project_dir / "pyproject.toml").exists()
        assert (temp_project_dir / "README.md").exists()
        assert (temp_project_dir / "Makefile").exists()

    @pytest.mark.integration
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

    @pytest.mark.integration
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
                assert result.returncode == 0, (
                    f"mlx failed for {project_name}: {result.stderr}"
                )
                self._verify_transformation_results(test_path, project_name)

    @pytest.mark.integration
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
        assert len(list(new_src.glob("*.py"))) > 0, (
            "Python files should exist in new directory"
        )

    @pytest.mark.integration
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

    @pytest.mark.integration
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
                loaded_data = doc_path.read_text()
                # The transformation should have updated references
                assert (
                    "analysis-template" not in loaded_data.lower()
                    or "test-docs-update" in loaded_data.lower()
                ), f"Documentation in {doc_path} not properly updated"

    @pytest.mark.integration
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

        # Try to import the new module
        new_snake_case = project_name.replace("-", "_")
        import_test_script = f"""
import sys
sys.path.insert(0, "{temp_project_dir}")
try:
    import src.{new_snake_case}
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {{e}}")
    sys.exit(1)
"""

        result = subprocess.run(
            ["python", "-c", import_test_script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Import test failed: {result.stderr}"
        assert "Import successful" in result.stdout

    @pytest.mark.integration
    def test_configuration_file_updates(self, temp_project_dir: Path):
        """Test that configuration files are properly updated."""
        project_name = "test-config-update"

        # Run transformation
        subprocess.run(
            [str(temp_project_dir / "mlx"), project_name],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Check various config files
        config_files_to_check = [
            temp_project_dir / "conf" / "config.yaml",
            temp_project_dir / "conf" / "api" / "development.yaml",
        ]

        for config_file in config_files_to_check:
            if config_file.exists():
                content = config_file.read_text()
                # Should not contain old template references
                assert "analysis_template" not in content.lower(), (
                    f"Config file {config_file} still contains analysis_template references"
                )

    @pytest.mark.integration
    def test_docker_configuration_updates(self, temp_project_dir: Path):
        """Test that Docker configuration is properly updated."""
        project_name = "test-docker-update"

        # Run transformation
        subprocess.run(
            [str(temp_project_dir / "mlx"), project_name],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Check Docker files
        docker_files = [
            temp_project_dir / "Dockerfile",
            temp_project_dir / "docker-compose.yml",
        ]

        for docker_file in docker_files:
            if docker_file.exists():
                content = docker_file.read_text()
                # Should have project-specific references
                expected_snake = project_name.replace("-", "_")
                # At minimum, should not contain old template name
                assert "analysis_template" not in content, (
                    f"Docker file {docker_file} still contains analysis_template"
                )
                assert expected_snake in content, (
                    f"Docker file {docker_file} does not contain {expected_snake}"
                )

    @pytest.mark.integration
    def test_makefile_updates(self, temp_project_dir: Path):
        """Test that Makefile is properly updated."""
        project_name = "test-makefile-update"

        # Run transformation
        subprocess.run(
            [str(temp_project_dir / "mlx"), project_name],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Check Makefile
        makefile_path = temp_project_dir / "Makefile"
        if makefile_path.exists():
            content = makefile_path.read_text()
            # Should not contain old template references
            assert "analysis_template" not in content, (
                "Makefile still contains analysis_template references"
            )

    @pytest.mark.integration
    def test_github_workflows_updates(self, temp_project_dir: Path):
        """Test that GitHub workflows are properly updated."""
        project_name = "test-github-update"

        # Run transformation
        subprocess.run(
            [str(temp_project_dir / "mlx"), project_name],
            cwd=temp_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Check GitHub workflow files
        workflows_dir = temp_project_dir / ".github" / "workflows"
        if workflows_dir.exists():
            for workflow_file in workflows_dir.glob("*.yml"):
                content = workflow_file.read_text()
                # Should not contain old template references
                assert "analysis_template" not in content, (
                    f"Workflow {workflow_file} still contains analysis_template"
                )

    def _verify_transformation_results(self, project_dir: Path, project_name: str):
        """Helper method to verify transformation results."""
        # Convert project name to expected formats
        expected_snake = project_name.replace("-", "_")
        expected_kebab = project_name.replace("_", "-")

        # Check that source directory was renamed
        old_src = project_dir / "src" / "analysis_template"
        new_src = project_dir / "src" / expected_snake

        assert not old_src.exists(), "Old analysis_template directory should be removed"
        assert new_src.exists(), "New project directory should exist"

        # Check that __init__.py exists in new directory
        assert (new_src / "__init__.py").exists(), (
            "__init__.py should exist in new directory"
        )

        # Check pyproject.toml was updated
        pyproject_path = project_dir / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)

            # Check project name in various possible locations
            name_found = False
            if "project" in data and "name" in data["project"]:
                name_found = data["project"]["name"] == expected_kebab
            elif (
                "tool" in data
                and "poetry" in data["tool"]
                and "name" in data["tool"]["poetry"]
            ):
                name_found = data["tool"]["poetry"]["name"] == expected_kebab

            assert name_found, "Project name not updated correctly in pyproject.toml"
