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


def _is_bootstrapped(pyproject_path: Path) -> bool:
    """Check if the project has already been bootstrapped."""
    if not pyproject_path.exists():
        return False
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    return "atem" not in data.get("project", {}).get("scripts", {})


def _update_pyproject_toml(
    package_slug: str, cli_command: str, docs_base_url: str
) -> None:
    """Update pyproject.toml with new project settings."""
    console.print("📝 Updating [bold cyan]pyproject.toml[/bold cyan]...")
    pyproject_path = Path("pyproject.toml")
    with pyproject_path.open() as f:
        content = f.read()

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


def _rename_package_directory(package_slug: str) -> None:
    """Rename the source package directory."""
    console.print(
        f"Renaming [bold cyan]src/analysis_template[/bold cyan] to [bold cyan]src/{package_slug}[/bold cyan]..."
    )
    shutil.move("src/analysis_template", f"src/{package_slug}")


def _install_dependencies(cli_command: str) -> None:
    """Create venv and install dependencies."""
    console.print("🛠️  Installing dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".[dev]"], check=True
        )
        console.print("✅ Dependencies installed.")
        console.print("🤝 Running [bold cyan]pre-commit install[/bold cyan]...")
        subprocess.run(["pre-commit", "install"], check=True)
        console.print(
            "\n[bold green]🚀 Bootstrap complete![/bold green]\n"
            f"Your project is ready. To get started, activate the virtual environment and run the CLI:\n\n"
            f"  [cyan]source .venv/bin/activate[/cyan]\n"
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
    package_slug = Prompt.ask(
        "Enter Python package name (slug)", default=repo_name.replace("-", "_")
    )
    cli_command = Prompt.ask(
        "Enter CLI command name", default=package_slug.split("_")[0]
    )
    docs_base_url = Prompt.ask(
        "Enter docs base URL",
        default=f"https://github.com/your-org/{repo_name}/tree/main/docs",
    )

    _update_pyproject_toml(package_slug, cli_command, docs_base_url)
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
