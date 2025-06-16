"""CLI commands for agent operations."""

import typer

from src.config import ConfigManager
from src.plugins import ExecutionContext, get_plugin

app = typer.Typer()


def get_context() -> ExecutionContext:
    """Helper to create an execution context."""
    config_manager = ConfigManager()
    config = config_manager.get_config()
    return ExecutionContext(run_id="agent_cli", config=config)


@app.command()
def run(
    task: str = typer.Argument(..., help="The task for the agent to perform."),
    agent_name: str = typer.Option(
        "react_agent", "--agent", "-a", help="The name of the agent plugin to use."
    ),
):
    """Runs an agent to accomplish a given task."""
    typer.echo(f"Running agent '{agent_name}' on task: '{task}'")
    context = get_context()

    # Initialize agent plugin
    agent_plugin = get_plugin(
        name=agent_name,
        category="agent",
        config=context.config.agent.dict(),
    )
    agent_plugin.initialize(context)

    # Run the agent
    result = agent_plugin.run(task)

    typer.echo("\\n" + "=" * 20)
    typer.echo("✅ Agent Finished")
    typer.echo("=" * 20)
    typer.echo(result)
    typer.echo("=" * 20 + "\\n")


if __name__ == "__main__":
    app()
