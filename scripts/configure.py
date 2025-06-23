"""Configuration script for MLOps template."""

from pathlib import Path

import typer
import yaml

app = typer.Typer(help="Configure MLOps template settings")


def create_config_file(config_path: Path, config_data: dict) -> None:
    """Create configuration file."""
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        typer.echo(f"Configuration saved to {config_path}")
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(1) from None


@app.command()
def init(
    project_name: str = typer.Option(..., "--name", "-n", help="Project name"),
    config_dir: Path | None = typer.Option(
        None, "--config-dir", "-c", help="Configuration directory"
    ),
) -> None:
    """Initialize project configuration."""
    if config_dir is None:
        config_dir = Path("conf")

    config_path = config_dir / "config.yaml"

    if config_path.exists():
        if not typer.confirm(
            f"Configuration file {config_path} already exists. Overwrite?"
        ):
            typer.echo("Configuration cancelled.")
            return

    config_data = {
        "project": {
            "name": project_name,
            "version": "1.0.0",
        },
        "paths": {
            "data": "data",
            "models": "models",
            "logs": "logs",
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    }

    try:
        create_config_file(config_path, config_data)
        typer.echo(f"✅ Project '{project_name}' configured successfully!")
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(1) from None


@app.command()
def update(
    config_file: Path = typer.Option(
        "conf/config.yaml", "--config", "-c", help="Configuration file path"
    ),
    key: str = typer.Option(..., "--key", "-k", help="Configuration key to update"),
    value: str = typer.Option(..., "--value", "-v", help="New value"),
) -> None:
    """Update configuration value."""
    if not config_file.exists():
        typer.echo(f"Configuration file {config_file} not found.", err=True)
        raise typer.Exit(1) from None

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Simple key update (could be enhanced for nested keys)
        keys = key.split(".")
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        typer.echo(f"✅ Updated {key} = {value}")

    except Exception as e:
        typer.echo(f"Failed to update configuration: {e}", err=True)
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
