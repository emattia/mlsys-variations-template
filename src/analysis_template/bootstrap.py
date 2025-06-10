import re
import shutil
import subprocess
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

console = Console()


def get_git_remote_url():
    try:
        return (
            subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        return None


def get_git_current_branch():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        return "main"


def is_valid_package_name(name):
    return re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name) is not None


def bootstrap():
    """Initializes the project."""
    console.print("Welcome to the project bootstrap wizard!")

    project_name = Prompt.ask("Project name", default="analysis")

    while True:
        package_name = Prompt.ask(
            "Python package name", default=project_name.replace("-", "_")
        )
        if is_valid_package_name(package_name):
            break
        console.print(
            "[red]Invalid package name. Please use a valid Python identifier (letters, numbers, and underscores, not starting with a number).[/red]"
        )

    cli_name = Prompt.ask("Command-line tool name", default=package_name)

    # Docs config
    remote_url = get_git_remote_url()
    if remote_url:
        default_url = f"https://{remote_url.split('/')[-2]}.github.io/{remote_url.split('/')[-1].replace('.git', '')}"
        docs_base_url = Prompt.ask(
            "Documentation base URL",
            default=default_url,
        )
    else:
        docs_base_url = Prompt.ask("Documentation base URL")

    console.print("Configuring project...")

    # Define paths
    src_dir = Path("src")
    old_package_dir = src_dir / "analysis_template"
    new_package_dir = src_dir / package_name
    cli_template_path = old_package_dir / "cli_template.py"
    final_cli_path = new_package_dir / "cli.py"
    pyproject_path = Path("pyproject.toml")
    mkdocs_path = Path("mkdocs.yml")

    # Read template/config files
    cli_template = cli_template_path.read_text()
    pyproject_toml = pyproject_path.read_text()
    mkdocs_yml = mkdocs_path.read_text()

    # Perform substitutions
    cli_code = cli_template.replace("{{project_name}}", project_name)

    # Update pyproject.toml robustly
    pyproject_toml = pyproject_toml.replace(
        'name = "analysis-template"', f'name = "{project_name}"'
    )
    pyproject_toml = pyproject_toml.replace(
        'include = "analysis_template"', f'include = "{package_name}"'
    )
    pyproject_toml = pyproject_toml.replace(
        'known_first_party = ["analysis_template"]',
        f'known_first_party = ["{package_name}"]',
    )

    # Update scripts entry
    new_script_entry = f'{cli_name} = "{package_name}.cli:app"'
    pyproject_toml = re.sub(
        r'atem = "analysis_template\.bootstrap:bootstrap"',
        new_script_entry,
        pyproject_toml,
    )

    # Update mkdocs.yml
    mkdocs_yml = mkdocs_yml.replace(
        "site_url: https://placeholder.com", f"site_url: {docs_base_url}"
    )

    # Create new package directory and files
    new_package_dir.mkdir(parents=True, exist_ok=True)
    (new_package_dir / "__init__.py").touch()
    final_cli_path.write_text(cli_code)

    # Write updated config files
    pyproject_path.write_text(pyproject_toml)
    mkdocs_path.write_text(mkdocs_yml)

    # Final cleanup
    console.print("Cleaning up bootstrap files...")
    shutil.rmtree(old_package_dir)

    console.print("\n[green]Project initialized successfully![/green]")
    console.print("\nTo get started, install the project in editable mode:")
    console.print("  [cyan]pip install -e .[/cyan]")
    console.print("\nThen run your new CLI:")
    console.print(f"  [cyan]{cli_name} --help[/cyan]")
    console.print(f"  [cyan]{cli_name} hello[/cyan]")
    console.print(f"  [cyan]{cli_name} docs --serve[/cyan]")


if __name__ == "__main__":
    bootstrap()
