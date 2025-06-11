import typer
import subprocess
from rich import print

app = typer.Typer()


@app.command()
def docs(
    serve: bool = typer.Option(
        False, "--serve", "-s", help="Serve the documentation after building."
    ),
    port: int = typer.Option(
        8000, "--port", "-p", help="Port to serve documentation on."
    ),
):
    """Builds and optionally serves the documentation."""
    print("Building documentation...")
    subprocess.run(["mkdocs", "build"], check=True)
    print("[green]Documentation built successfully.[/green]")

    if serve:
        print(f"Serving documentation at http://127.0.0.1:{port}")
        subprocess.run(["mkdocs", "serve", f"--dev-addr=127.0.0.1:{port}"], check=True)


@app.command()
def hello():
    """A simple example command."""
    print("Hello from your new project: llm!")


if __name__ == "__main__":
    app()
