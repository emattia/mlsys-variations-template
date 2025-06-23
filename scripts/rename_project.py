from pathlib import Path

import toml
import typer

app = typer.Typer()

OLD_NAME = "mlsys-variations-base"
OLD_CLI_NAME = "mlsys-variations-base"

# Files to perform replacement in
FILES_TO_REPLACE = [
    "pyproject.toml",
    "src/config/models.py",
    "tests/unit/test_config.py",
    "tests/test_utils_common.py",
]


def replace_in_file(file_path: Path, old: str, new: str):
    """Replaces all occurrences of old with new in a file."""
    try:
        loaded_data = file_path.read_text()
        if old in loaded_data:
            _newloaded_data = loaded_data.replace(old, new)
            file_path.write_text(_newloaded_data)
            typer.echo(f"Updated {file_path}")
    except Exception as e:
        typer.echo(f"Error updating {file_path}: {e}", err=True)


def update_toml_cli_name(file_path: Path, new_name: str):
    """Updates the CLI entry point name in pyproject.toml."""
    try:
        _data = toml.load(file_path)
        if "project" in _data and "scripts" in _data["project"]:
            if OLD_CLI_NAME in _data["project"]["scripts"]:
                _data["project"]["scripts"][new_name] = _data["project"]["scripts"].pop(
                    OLD_CLI_NAME
                )
                with open(file_path, "w") as f:
                    toml.dump(_data, f)
                typer.echo(f"Updated CLI entry point in {file_path} to '{new_name}'")
    except Exception as e:
        typer.echo(f"Error updating {file_path}: {e}", err=True)


@app.command()
def rename(new_name: str):
    """
    Renames the project and updates the CLI entry point.
    """
    project_root = Path(__file__).parent.parent
    typer.echo(f"Project root: {project_root}")
    typer.echo(f"Renaming project to '{new_name}'...")

    for file_path_str in FILES_TO_REPLACE:
        file_path = project_root / file_path_str
        if file_path.exists():
            replace_in_file(file_path, OLD_NAME, new_name)
        else:
            typer.echo(f"Warning: {file_path} not found. Skipping.", err=True)

    # Update the CLI name in pyproject.toml
    toml_path = project_root / "pyproject.toml"
    if toml_path.exists():
        update_toml_cli_name(toml_path, new_name)

    # Ask for project properties
    data_source = typer.prompt(
        "What is your primary data source? (e.g., Snowflake, BigQuery, local)"
    )

    # Store the properties (for now, just echo them)
    # In a real implementation, you might save this to a config file.
    typer.echo("\\nProject properties captured:")
    typer.echo(f"  - Primary Data Source: {data_source}")

    typer.echo("\\nProject renaming complete!")
    typer.echo("To use the new CLI name, you must reinstall the package.")
    typer.echo("Please update the lock file and reinstall dependencies by running:")
    typer.echo("\\n  make install-dev\\n")


if __name__ == "__main__":
    app()
