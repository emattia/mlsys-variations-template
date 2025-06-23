#!/usr/bin/env python3
"""
MLOps CLI - Unified Command Line Interface
Provides unified access to MLOps platform functionality
"""

import subprocess
import re
import typer
from pathlib import Path

app = typer.Typer(help="Unified CLI for the MLOps template.")

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
TEMPLATES_DIR = Path(__file__).parent / "templates"
PLUGIN_TEMPLATE = TEMPLATES_DIR / "plugin_skeleton.py.jinja"
DEFAULT_CATEGORY = "general"


def _pascal_case(name: str) -> str:
    """Convert a string to PascalCase."""
    return "".join(word.capitalize() for word in re.split(r"[\-_ ]+", name))


# ---------------------------------------------------------------------
# Repo sub-commands
# ---------------------------------------------------------------------
repo_app = typer.Typer(help="Repository insights & git helpers")


@repo_app.command("info")
def repo_info():
    """Show high-level repository information (branch, remotes, latest commit)."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        remote = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        typer.secho(f"[git error] {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Branch : {branch}")
    typer.echo(f"Commit : {commit}")
    typer.echo(f"Remote : {remote}")


@repo_app.command("branches")
def repo_branches(remote: bool = typer.Option(False, help="Include remote branches")):
    """List local (and optionally remote) branches."""
    cmd = ["git", "branch"] + (["-a"] if remote else [])
    branches = subprocess.check_output(cmd, text=True)
    typer.echo(branches)


# Add repo_app to main
app.add_typer(repo_app, name="repo")

# ---------------------------------------------------------------------
# Plugin sub-commands
# ---------------------------------------------------------------------
plugin_app = typer.Typer(help="Plugin management utilities")


@plugin_app.command("list")
def list_plugins(
    category: str | None = typer.Option(
        None, "--category", "-c", help="Filter by category."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show full plugin info."
    ),
):
    """List registered plugins."""
    try:
        from src.plugins import list_plugins as _lp

        plugins = _lp(category=category, with_info=verbose)
        typer.echo(plugins)
    except ImportError:
        typer.secho("Plugins module not available", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@plugin_app.command("info")
def plugin_info(name: str):
    """Show detailed information about a single plugin."""
    try:
        from src.plugins import get_registry

        registry = get_registry()
        info = registry.get_plugin_info(name)
        typer.echo(info)
    except ImportError:
        typer.secho("Plugins module not available", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except KeyError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)


@plugin_app.command("add")
def plugin_add(
    name: str = typer.Argument(..., help="Unique plugin name (snake/kebab case)."),
    category: str = typer.Option(
        DEFAULT_CATEGORY, "--category", "-c", help="Plugin category."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite if file exists."
    ),
):
    """Scaffold a new plugin inside src/plugins/ using the skeleton template."""
    project_root = Path(__file__).resolve().parents[1]
    dest_path = project_root / "src" / "plugins" / f"{name}.py"
    test_dest = project_root / "tests" / "plugins" / f"test_{name}.py"

    if dest_path.exists() and not force:
        typer.secho(
            f"[!] {dest_path} already exists. Use --force to overwrite.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    if not PLUGIN_TEMPLATE.exists():
        typer.secho(f"Template not found: {PLUGIN_TEMPLATE}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    class_name = _pascal_case(name)
    template_text = PLUGIN_TEMPLATE.read_text()
    rendered = (
        template_text.replace("{{ plugin_name }}", name)
        .replace("{{ class_name }}", class_name)
        .replace("{{ category }}", category)
    )
    dest_path.write_text(rendered)
    typer.secho(f"[+] Created plugin at {dest_path}", fg=typer.colors.GREEN)

    # Write test skeleton
    test_template_path = TEMPLATES_DIR / "plugin_test_skeleton.py.jinja"
    if test_template_path.exists():
        test_template = test_template_path.read_text()
        test_rendered = test_template.replace("{{ plugin_name }}", name)
        test_dest.parent.mkdir(parents=True, exist_ok=True)
        test_dest.write_text(test_rendered)
        typer.secho(f"[+] Created test at {test_dest}", fg=typer.colors.GREEN)


app.add_typer(plugin_app, name="plugin")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    app()
