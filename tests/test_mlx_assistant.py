"""Tests for MLX Assistant functionality.

These tests focus on *core* behaviours that can safely run inside the CI
environment without requiring network access or heavy external processes.
They purposefully exercise a subset of the original (now-commented) test plan
while keeping runtime low.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from scripts.mlx_assistant import MLXAssistant, app

# ---------------------------------------------------------------------------
# CLI runner fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runner() -> CliRunner:  # type: ignore[valid-type]
    """Shared Typer CLI runner."""

    return CliRunner()


# ---------------------------------------------------------------------------
# MLXAssistant core behaviour
# ---------------------------------------------------------------------------


def test_mlx_assistant_initialization() -> None:
    """Assistant should populate key attributes on construction."""

    assistant = MLXAssistant()
    assert assistant.project_root is not None
    assert isinstance(assistant.frameworks, dict)
    assert isinstance(assistant.project_state, dict)
    assert len(assistant.frameworks) >= 4  # golden_repos, security, plugins, glossary


def test_framework_discovery() -> None:
    """Discovery should return expected framework keys & structure."""

    assistant = MLXAssistant()
    frameworks = assistant._discover_frameworks()

    expected = {"golden_repos", "security", "plugins", "glossary"}
    assert expected.issubset(frameworks.keys())

    # Validate schema for a single entry
    sample = frameworks["golden_repos"]
    for field in ("name", "description", "script", "commands", "icon", "status"):
        assert field in sample


def test_project_state_analysis() -> None:
    """State analysis should detect MLX projects and recommend actions."""

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Create a *mock* (partial) MLX project
        (tmp_path / "mlx.config.json").write_text('{"platform": {"name": "test"}}')
        (tmp_path / "mlx-components").mkdir()
        (tmp_path / "plugins").mkdir()

        assistant = MLXAssistant()
        assistant.project_root = tmp_path  # Override cwd-based root
        state = assistant._analyze_project_state()

        assert state["is_mlx_project"] is True
        assert state["has_components"] is True
        assert "recommendations" in state


# ---------------------------------------------------------------------------
# CLI level tests
# ---------------------------------------------------------------------------


def test_cli_help(runner: CliRunner) -> None:  # type: ignore[valid-type]
    """The root --help flag should succeed and mention the assistant."""

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "MLX Assistant" in result.stdout
    assert (
        "intelligent companion" in result.stdout or "intelligent guide" in result.stdout
    )


def test_cli_version_flag(runner: CliRunner) -> None:  # type: ignore[valid-type]
    """The --version flag should print the version string."""

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "MLX Assistant v1.0.0" in result.stdout


def test_quick_start_command(runner: CliRunner) -> None:  # type: ignore[valid-type]
    """quick-start command should display a panel with basic steps."""

    result = runner.invoke(app, ["quick-start"])
    assert result.exit_code == 0
    assert "Quick Start" in result.stdout
    assert "Step 1" in result.stdout


def test_doctor_command(runner: CliRunner) -> None:  # type: ignore[valid-type]
    """doctor command should execute and print a Health Check table."""

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Health Check" in result.stdout


def test_analyze_command(runner: CliRunner) -> None:  # type: ignore[valid-type]
    """analyze command should finish and print Project Analysis."""

    # Patch *time.sleep* to avoid the intentional 2-second delay.
    with patch("time.sleep"):
        result = runner.invoke(app, ["analyze"])
    assert result.exit_code == 0
    assert "Project Analysis" in result.stdout


# ---------------------------------------------------------------------------
# Framework sub-command integration (mocked subprocesses)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subcmd", [["golden-repos", "list"], ["security", "scan"], ["plugins", "list"]]
)
@patch("subprocess.run")
def test_framework_commands(mock_run: Mock, runner: CliRunner, subcmd):  # type: ignore[valid-type]
    """Framework commands should invoke subprocess & exit properly."""

    mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
    result = runner.invoke(app, subcmd)
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Interactive mode (smoke test, no real prompt interaction)
# ---------------------------------------------------------------------------


@patch("rich.prompt.Prompt.ask", return_value="exit")
@patch("scripts.mlx_assistant.start_interactive_mode", return_value=None)
def test_interactive_mode_start(_mock_interactive, _mock_prompt, runner: CliRunner):  # type: ignore[valid-type]
    """Passing --interactive should trigger the interactive entry-point."""

    # We only smoke-test that the command exits cleanly.
    result = runner.invoke(app, ["--interactive"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Error handling paths
# ---------------------------------------------------------------------------


@patch("subprocess.run")
def test_framework_command_failure(mock_run: Mock, runner: CliRunner):  # type: ignore[valid-type]
    """CLI should surface subprocess errors gracefully."""

    mock_run.return_value = Mock(returncode=1, stdout="", stderr="boom")
    result = runner.invoke(app, ["golden-repos", "create", "invalid-spec"])
    # Non-zero exit or error keyword expected.
    assert result.exit_code != 0 or "Error" in result.stdout
