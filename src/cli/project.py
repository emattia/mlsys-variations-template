import json
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def create_cursor_rules(
    data_source: str = typer.Option(
        "local", help="Primary data source (e.g., snowflake, bigquery, local)"
    ),
    cloud_provider: str = typer.Option(
        "aws", help="Cloud provider (e.g., aws, gcp, azure)"
    ),
):
    """
    Generates a .cursor-rules.json file with dynamic rules.
    """
    project_root = Path(__file__).parent.parent.parent
    base_rules_path = project_root / "templates" / "base-cursor-rules.json"
    output_path = project_root / ".cursor-rules.json"

    # Load base rules
    with open(base_rules_path) as f:
        rules = json.load(f)

    # Add dynamic rules based on properties
    if data_source.lower() == "snowflake":
        rules["rules"].append(
            {
                "name": "Snowflake Data Source",
                "description": "Context for Snowflake data source.",
                "include": ["src/plugins/data_sources.py"],
            }
        )

    if cloud_provider.lower() == "aws":
        rules["rules"].append(
            {
                "name": "AWS Integration",
                "description": "Context for AWS services and integration.",
                "include": [
                    "src/config/models.py"
                ],  # Assuming AWS settings are in config
            }
        )

    # Write the final rules file
    with open(output_path, "w") as f:
        json.dump(rules, f, indent=2)

    typer.echo(f"Successfully created .cursor-rules.json at {output_path}")


if __name__ == "__main__":
    app()
