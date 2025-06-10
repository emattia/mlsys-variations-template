from __future__ import annotations

"""Command-line interface for the Analysis Template."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import tomllib
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# --- Globals & Setup -----------------------------------------------------------------

console = Console()
app = typer.Typer(
    add_completion=False,
    help="A self-configuring bootstrap and utility CLI for the Analysis Template.",
)

# --- Bootstrap Wizard --------------------------------------------------------------


def _get_git_remote_url() -> str | None:
    """Attempt to get the git remote URL for 'origin'."""
    try:
        # Check if we are in a git repository
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            check=True,
            capture_output=True,
        )

        # Get remote URL
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        )
        url = result.stdout.strip()
        # Convert ssh URL to https URL
        if url.startswith("git@"):
            url = url.replace(":", "/").replace("git@", "https://")
        if url.endswith(".git"):
            url = url[:-4]
        return url
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _is_bootstrapped(pyproject_path: Path) -> bool:
    """Check if the project has already been bootstrapped."""
    if not pyproject_path.exists():
        return False
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    return "atem" not in data.get("project", {}).get("scripts", {})


def _update_pyproject_toml(
    project_name: str, package_slug: str, cli_command: str, docs_base_url: str
) -> None:
    """Update pyproject.toml with new project settings."""
    console.print("📝 Updating [bold cyan]pyproject.toml[/bold cyan]...")
    pyproject_path = Path("pyproject.toml")
    with pyproject_path.open() as f:
        content = f.read()

    # Replace project name
    content = content.replace(
        'name = "mlsys-variations-template"', f'name = "{project_name}"'
    )

    # Replace tool.project_meta
    content = content.replace(
        'package_slug = "analysis_template"', f'package_slug = "{package_slug}"'
    )
    content = content.replace('cli_command = "atem"', f'cli_command = "{cli_command}"')
    content = content.replace(
        'docs_base_url = "https://github.com/yourusername/analysis-template/tree/main/docs"',
        f'docs_base_url = "{docs_base_url}"',
    )

    # Replace project.scripts
    content = content.replace(
        'atem = "analysis_template.cli:app"',
        f'{cli_command} = "{package_slug}.cli:app"',
    )

    # Replace hatch build target
    content = content.replace(
        'packages = ["src/analysis_template"]', f'packages = ["src/{package_slug}"]'
    )

    # Replace ruff isort config
    content = content.replace(
        'known-first-party = ["analysis_template"]',
        f'known-first-party = ["{package_slug}"]',
    )

    with pyproject_path.open("w") as f:
        f.write(content)


def _update_init_py(project_name: str) -> None:
    """Update the __init__.py file with the new project name for version lookup."""
    console.print("📝 Updating [bold cyan]__init__.py[/bold cyan]...")
    init_path = Path("src/analysis_template/__init__.py")
    with init_path.open() as f:
        content = f.read()

    content = content.replace(
        '__version__: str = metadata.version("mlsys-variations-template")',
        f'__version__: str = metadata.version("{project_name}")',
    )

    with init_path.open("w") as f:
        f.write(content)


def _rename_package_directory(package_slug: str) -> None:
    """Rename the source package directory."""
    console.print(
        f"Renaming [bold cyan]src/analysis_template[/bold cyan] to [bold cyan]src/{package_slug}[/bold cyan]..."
    )
    shutil.move("src/analysis_template", f"src/{package_slug}")


def _install_dependencies(cli_command: str) -> None:
    """Create venv and install dependencies."""
    console.print("🛠️  Preparing environment and installing dependencies...")
    venv_dir = Path(".venv")
    try:
        # 1. Create venv if it doesn't exist
        if not venv_dir.exists():
            console.print(
                f"🐍 Creating virtual environment in [bold cyan]{venv_dir}[/bold cyan]..."
            )
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

        # Determine python executable path based on OS
        if sys.platform == "win32":
            python_executable = venv_dir / "Scripts" / "python.exe"
        else:
            python_executable = venv_dir / "bin" / "python"

        # 2. Install dependencies using the venv's python
        console.print(
            f"📦 Installing dependencies into [bold cyan]{venv_dir}[/bold cyan]..."
        )
        subprocess.run(
            [str(python_executable), "-m", "pip", "install", "-e", ".[dev]"],
            check=True,
        )

        # 3. Install pre-commit hooks
        console.print("🤝 Running [bold cyan]pre-commit install[/bold cyan]...")
        pre_commit_executable = venv_dir / "bin" / "pre-commit"
        subprocess.run([str(pre_commit_executable), "install"], check=True)

        console.print(
            "\n[bold green]🚀 Bootstrap complete![/bold green]\n"
            f"Your project is ready. To get started, activate the virtual environment and run the CLI:\n\n"
            f"  [cyan]source {venv_dir / 'bin' / 'activate'}[/cyan]\n"
            f"  [cyan]{cli_command} --help[/cyan]\n"
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        console.print(f"[bold red]Error during dependency installation:[/bold red] {e}")
        console.print("Please try running the installation steps manually.")


@app.command()
def init(
    force: bool = typer.Option(
        False,
        "--force",
        help="Force re-initialization even if the project seems bootstrapped.",
    ),
):
    """Initialize this project by setting key metadata."""
    pyproject_path = Path("pyproject.toml")
    if _is_bootstrapped(pyproject_path) and not force:
        console.print(
            "[bold yellow]⚠️ Project already initialized.[/bold yellow] Use --force to re-run."
        )
        raise typer.Exit()

    console.rule("[bold magenta]Project Bootstrap Wizard[/bold magenta]")
    console.print(
        "This wizard will configure your project by renaming the core package and setting up the CLI."
    )

    repo_name = Path.cwd().name
    project_name = Prompt.ask("Enter PyPI project name", default=repo_name)
    package_slug = Prompt.ask(
        "Enter Python package name (slug)", default=project_name.replace("-", "_")
    )
    cli_command = Prompt.ask(
        "Enter CLI command name", default=package_slug.split("_")[0]
    )

    # Auto-detect docs URL from git remote
    default_docs_url = f"https://github.com/your-org/{repo_name}/tree/main/docs"
    git_remote_url = _get_git_remote_url()
    if git_remote_url:
        default_docs_url = f"{git_remote_url}/tree/main/docs"

    docs_base_url = Prompt.ask(
        "Confirm docs base URL",
        default=default_docs_url,
    )

    _update_init_py(project_name)
    _update_pyproject_toml(project_name, package_slug, cli_command, docs_base_url)
    _rename_package_directory(package_slug)
    _install_dependencies(cli_command)


# --- Utility Commands (placeholders) -----------------------------------------------


@app.command()
def doctor():
    """Run a basic system health check (stub)."""
    console.print(
        Panel("🩺 Health checks will be available after bootstrap.", title="doctor")
    )


@app.command()
def open(topic: str | None = typer.Argument(None)):
    """Open the documentation website (stub)."""
    del topic
    console.print(
        Panel("📖 Docs opener will be available after bootstrap.", title="open")
    )


if __name__ == "__main__":
    app()
