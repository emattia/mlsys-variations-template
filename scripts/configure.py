from pathlib import Path

import toml
import typer

app = typer.Typer()


@app.command()
def project_name(name: str):
    """
    Sets the project name in pyproject.toml.
    """
    try:
        pyproject_path = Path("pyproject.toml")
        if not pyproject_path.exists():
            typer.echo("pyproject.toml not found!", err=True)
            raise typer.Exit(1)

        data = toml.load(pyproject_path)
        data["project"]["name"] = name

        with open(pyproject_path, "w") as f:
            toml.dump(data, f)

        typer.echo(f"Project name updated to: {name}")

    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def package_name(name: str):
    """
    Sets the package name in pyproject.toml for Ruff's isort configuration.
    """
    try:
        pyproject_path = Path("pyproject.toml")
        if not pyproject_path.exists():
            typer.echo("pyproject.toml not found!", err=True)
            raise typer.Exit(1)

        data = toml.load(pyproject_path)
        if (
            "tool" in data
            and "ruff" in data["tool"]
            and "lint" in data["tool"]["ruff"]
            and "isort" in data["tool"]["ruff"]["lint"]
        ):
            data["tool"]["ruff"]["lint"]["isort"]["known-first-party"] = [name]
        else:
            typer.echo(
                "Could not find [tool.ruff.lint.isort] in pyproject.toml", err=True
            )
            raise typer.Exit(1)

        with open(pyproject_path, "w") as f:
            toml.dump(data, f)

        typer.echo(f"Package name for isort updated to: {name}")

    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
