from __future__ import annotations

"""One-time bootstrap wizard for the Analysis Template."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import tomllib
import typer
from rich.console import Console
from rich.prompt import Prompt

# --- Globals & Setup -----------------------------------------------------------------

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)

# --- Helper Functions ----------------------------------------------------------------


def _get_git_remote_url() -> str | None:
    """Attempt to get the git remote URL for 'origin'."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        )
        url = result.stdout.strip()
        if url.startswith("git@"):
            url = url.replace(":", "/").replace("git@", "https://")
        if url.endswith(".git"):
            url = url[:-4]
        return url
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _get_current_git_branch() -> str:
    """Get the current active git branch, falling back to 'main'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "main"


def _is_valid_python_identifier(s: str) -> bool:
    """Check if a string is a valid Python identifier."""
    return s.isidentifier()


# --- Core Logic Functions ------------------------------------------------------------


def _update_pyproject_toml(
    project_name: str, package_slug: str, cli_command: str, docs_base_url: str
) -> None:
    """Update pyproject.toml with new project settings."""
    console.print("📝 Updating [bold cyan]pyproject.toml[/bold cyan]...")
    pyproject_path = Path("pyproject.toml")
    with pyproject_path.open() as f:
        content = f.read()

    replacements = {
        'name = "mlsys-variations-template"': f'name = "{project_name}"',
        'package_slug = "analysis_template"': f'package_slug = "{package_slug}"',
        'cli_command = "mlsys"': f'cli_command = "{cli_command}"',
        'docs_base_url = "https://github.com/yourusername/analysis-template"': f'docs_base_url = "{docs_base_url}"',
        'mlsys = "analysis_template.cli:app"': f'{cli_command} = "{package_slug}.cli:app"',
        'packages = ["src/analysis_template"]': f'packages = ["src/{package_slug}"]',
        'known-first-party = ["analysis_template"]': f'known-first-party = ["{package_slug}"]',
    }
    for old, new in replacements.items():
        content = content.replace(old, new)
    with pyproject_path.open("w") as f:
        f.write(content)


def _update_final_cli(project_name: str) -> None:
    """Update placeholders in the final cli.py script."""
    console.print("📝 Configuring final CLI script...")
    cli_path = Path("src/analysis_template/cli.py")
    with cli_path.open() as f:
        content = f.read()

    project_title = project_name.replace("-", " ").replace("_", " ").title()
    content = content.replace("<<PROJECT_NAME>>", project_name)
    content = content.replace("<<PROJECT_TITLE>>", project_title)

    with cli_path.open("w") as f:
        f.write(content)


def _rename_package(package_slug: str) -> None:
    """Rename the source package directory."""
    console.print(f"Renaming package to [bold cyan]src/{package_slug}[/bold cyan]...")
    if Path(f"src/{package_slug}").exists():
        shutil.rmtree(f"src/{package_slug}")
    shutil.move("src/analysis_template", f"src/{package_slug}")


def _install_and_cleanup(package_slug: str) -> None:
    """Install final dependencies and remove bootstrap artifacts."""
    console.print("📦 Installing project dependencies...")
    venv_dir = Path(".venv")
    python_executable = venv_dir / (
        "bin/python" if sys.platform != "win32" else "Scripts/python.exe"
    )

    try:
        subprocess.run(
            [str(python_executable), "-m", "pip", "install", "-e", ".[dev]"], check=True
        )

        # Cleanup
        console.print("🗑️  Cleaning up bootstrap artifacts...")
        (Path(f"src/{package_slug}") / "bootstrap.py").unlink()
        Path("mlsys").unlink()

        console.print("\n[bold green]🚀 Bootstrap complete![/bold green]")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        console.print(f"[bold red]Error during final installation:[/bold red] {e}")


# --- Main Command -------------------------------------------------------------------


def _run_wizard():
    """The main wizard logic, callable from __main__."""
    console.rule("[bold magenta]Project Bootstrap Wizard[/bold magenta]")

    repo_name = Path.cwd().name
    project_name = Prompt.ask("Enter PyPI project name", default=repo_name)

    package_slug = ""
    while not package_slug:
        slug_prompt = Prompt.ask(
            "Enter Python package name (slug)", default=project_name.replace("-", "_")
        )
        if _is_valid_python_identifier(slug_prompt):
            package_slug = slug_prompt
        else:
            console.print(
                "[bold red]Invalid Python identifier.[/bold red] Please use only letters, numbers, and underscores."
            )

    cli_command = Prompt.ask(
        "Enter CLI command name", default=package_slug.split("_")[0]
    )

    repo_url = _get_git_remote_url() or f"https://github.com/your-org/{project_name}"
    current_branch = _get_current_git_branch()
    default_docs_url = f"{repo_url}/tree/{current_branch}"

    docs_base_url = Prompt.ask("Confirm project git URL", default=default_docs_url)

    _update_pyproject_toml(project_name, package_slug, cli_command, docs_base_url)
    _update_final_cli(project_name)
    _rename_package(package_slug)
    _install_and_cleanup(package_slug)


@app.command(hidden=True)
def init():
    """Initialize this project by setting key metadata."""
    _run_wizard()


if __name__ == "__main__":
    # This allows the script to be called directly with `python bootstrap.py`
    # and also handle arguments if passed via Typer in the future.
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] != "init"):
        _run_wizard()
    else:
        app()
