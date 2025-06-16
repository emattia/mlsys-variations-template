import typer
from demo_comprehensive import MLOpsPlatformDemo

app = typer.Typer()


@app.command()
def run(
    component: str = typer.Argument(
        "all",
        help="Specific component to demonstrate (all, data, models, api, docker, workflows, plugins)",
    ),
):
    """
    Runs the comprehensive MLOps platform demonstration.
    """
    demo = MLOpsPlatformDemo()

    if component == "all":
        demo.run_comprehensive_demo()
    elif component == "data":
        demo.demonstrate_data_workflows()
    elif component == "models":
        demo.demonstrate_model_workflows()
    elif component == "api":
        demo.demonstrate_api_service()
    elif component == "docker":
        demo.demonstrate_docker_integration()
    elif component == "workflows":
        demo.demonstrate_ml_workflows()
    elif component == "plugins":
        demo.demonstrate_plugin_system()
    else:
        typer.echo(f"Invalid component: {component}", err=True)


if __name__ == "__main__":
    app()
