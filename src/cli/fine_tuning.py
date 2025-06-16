"""CLI commands for fine-tuning operations."""

from pathlib import Path

import typer

from src.config import ConfigManager
from src.plugins import ExecutionContext, get_plugin

app = typer.Typer()


def get_context() -> ExecutionContext:
    """Helper to create an execution context."""
    config_manager = ConfigManager()
    config = config_manager.get_config()
    return ExecutionContext(run_id="ft_cli", config=config)


@app.command(name="create-job")
def create_job(
    dataset_path: Path = typer.Option(
        ..., "--dataset", "-d", help="Path to the training dataset file."
    ),
    model: str = typer.Option(
        "gpt-3.5-turbo", "--model", "-m", help="The base model to fine-tune."
    ),
):
    """Creates a new fine-tuning job."""
    typer.echo(
        f"Creating fine-tuning job for model '{model}' with dataset '{dataset_path}'"
    )
    context = get_context()

    ft_plugin = get_plugin(
        name="openai_fine_tuning",
        category="fine_tuning_pipeline",
        config=context.config.fine_tuning.dict(),
    )
    ft_plugin.initialize(context)
    result = ft_plugin.run(dataset_path=str(dataset_path), model=model)

    typer.echo("✅ Job created successfully!")
    typer.echo(f"   Job ID: {result['job_id']}")
    typer.echo(f"   Status: {result['status']}")


@app.command(name="list-jobs")
def list_jobs(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of jobs to list."),
):
    """Lists recent fine-tuning jobs."""
    context = get_context()
    ft_plugin = get_plugin(
        name="openai_fine_tuning",
        category="fine_tuning_pipeline",
        config=context.config.fine_tuning.dict(),
    )
    ft_plugin.initialize(context)
    jobs = ft_plugin.list_jobs(limit=limit)

    if not jobs:
        typer.echo("No fine-tuning jobs found.")
        return

    for job in jobs:
        typer.echo(f"- ID: {job.id}, Model: {job.model}, Status: {job.status}")


@app.command(name="get-status")
def get_status(job_id: str = typer.Argument(..., help="The ID of the job to check.")):
    """Gets the status of a specific fine-tuning job."""
    context = get_context()
    ft_plugin = get_plugin(
        name="openai_fine_tuning",
        category="fine_tuning_pipeline",
        config=context.config.fine_tuning.dict(),
    )
    ft_plugin.initialize(context)
    status = ft_plugin.get_status(job_id=job_id)

    typer.echo("Job Status:")
    for key, value in status.items():
        typer.echo(f"  - {key}: {value}")


if __name__ == "__main__":
    app()
