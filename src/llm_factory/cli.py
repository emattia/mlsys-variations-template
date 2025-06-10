from __future__ import annotations

"""Operational CLI for the llm-factory project."""

import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional

import tomllib
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# --- Globals & Setup -----------------------------------------------------------------

console = Console()


def _load_project_meta() -> dict:
    """Load project metadata from pyproject.toml."""
    try:
        pyproject_path = Path("pyproject.toml")
        if not pyproject_path.exists():
            return {}
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)

        project_tool_meta = data.get("tool", {}).get("project_meta", {})
        project_meta = data.get("project", {})
        # Combine them, giving tool.project_meta precedence
        return {**project_meta, **project_tool_meta}

    except FileNotFoundError:
        return {}


meta = _load_project_meta()
# The project title is a placeholder that will be replaced by the bootstrap script.
project_title = "Llm Factory"

app = typer.Typer(
    add_completion=False,
    help=f"CLI for {project_title}.",
    no_args_is_help=True,
)

# --- Commands ----------------------------------------------------------------------


@app.command()
def doctor():
    """Run a system health check."""
    console.rule(f"[bold green]{project_title} Health Check[/bold green]")
    ok_count = 0
    total_checks = 2

    if ".venv" in sys.prefix or "py" in sys.prefix:
        console.print("✅ Virtual environment is active.")
        ok_count += 1
    else:
        console.print("[bold red]❌ Virtual environment is not active.[/bold red]")

    if meta.get("docs_base_url"):
        console.print("✅ Documentation URL is configured.")
        ok_count += 1
    else:
        console.print("[bold red]❌ Documentation URL is not configured.[/bold red]")

    console.print(f"\\n[bold]Summary: {ok_count}/{total_checks} checks passed.[/bold]")


@app.command()
def docs(
    serve: bool = typer.Option(
        False, "--serve", "-s", help="Serve docs locally for development."
    ),
    build: bool = typer.Option(
        False, "--build", "-b", help="Build the static documentation site."
    ),
):
    """Build, serve, or open the project documentation URL."""
    venv_bin_dir = Path(sys.prefix) / "bin"
    mkdocs_executable = venv_bin_dir / "mkdocs"

    if not mkdocs_executable.exists():
        console.print("[bold red]Error: 'mkdocs' not found.[/bold red]")
        console.print(
            "Please ensure dev dependencies are installed with `pip install -e '.[dev]'`"
        )
        raise typer.Exit(1)

    if serve:
        console.print(
            "🚀 Serving documentation at [cyan]http://localhost:8000[/cyan] (Press Ctrl+C to stop)"
        )
        webbrowser.open("http://localhost:8000")
        subprocess.run([str(mkdocs_executable), "serve"])
    elif build:
        console.print("🏗️ Building documentation site...")
        subprocess.run([str(mkdocs_executable), "build"])
        console.print("✅ Site built in the [cyan]site/[/cyan] directory.")
    else:
        docs_url = meta.get("docs_base_url")
        if docs_url:
            console.print(f"🌐 Opening production docs URL: [cyan]{docs_url}[/cyan]")
            webbrowser.open(docs_url)
        else:
            console.print("[bold yellow]Production docs URL not set.[/bold yellow]")
            console.print("You can serve docs locally with the `--serve` flag.")


if __name__ == "__main__":
    app()
