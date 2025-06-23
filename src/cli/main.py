import typer

from . import agent, ai, demo, fine_tuning, project, rag

app = typer.Typer()

app.add_typer(project.app, name="project")
app.add_typer(demo.app, name="demo")
app.add_typer(rag.app, name="rag")
app.add_typer(agent.app, name="agent")
app.add_typer(ai.app, name="ai")
app.add_typer(fine_tuning.app, name="fine-tuning")


@app.command()
def main():
    """
    Main entry point for the CLI.
    """
    typer.echo("~~~ MLOps Variations CLI ~~~")


if __name__ == "__main__":
    app()
