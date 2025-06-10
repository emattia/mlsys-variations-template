import re
import shutil
import subprocess
from pathlib import Path

from rich.console import Console

console = Console()


def _update_pyproject_toml(
    project_name: str, package_slug: str, cli_command: str, docs_base_url: str
) -> None:
    """Update pyproject.toml with new project settings."""
    console.print("📝 Updating [bold cyan]pyproject.toml[/bold cyan]...")
    pyproject_path = Path("pyproject.toml")
    with pyproject_path.open() as f:
        content = f.read()

    # --- Robustly update [project.scripts] ---
    # This regex finds the [project.scripts] section and replaces the line(s) after it.
    new_script_entry = f'{cli_command} = "{package_slug}.cli:app"'
    content = re.sub(
        r"(\[project\.scripts\]\s*\n)[^\s\[\]]+.*",
        r"\\1" + new_script_entry,
        content,
        flags=re.DOTALL,
    )

    # --- Update other fields ---
    replacements = {
        'name = "mlsys-variations-template"': f'name = "{project_name}"',
        'package_slug = "analysis_template"': f'package_slug = "{package_slug}"',
        'cli_command = "mlsys"': f'cli_command = "{cli_command}"',
        'docs_base_url = "https://github.com/yourusername/analysis-template"': f'docs_base_url = "{docs_base_url}"',
        'packages = ["src/analysis_template"]': f'packages = ["src/{package_slug}"]',
        'known-first-party = ["analysis_template"]': f'known-first-party = ["{package_slug}"]',
    }
    for old, new in replacements.items():
        content = content.replace(old, new)

    with pyproject_path.open("w") as f:
        f.write(content)
