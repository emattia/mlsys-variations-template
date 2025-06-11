# MLOps Template: Production-Ready ML Platform

A streamlined, modern data science project template with best practices for reproducibility, code quality, and collaboration. This template is designed to be forked and customized for new projects with a single command.

## Quickstart: From Zero to Project in 60 Seconds

This template includes a self-configuring bootstrap wizard. To get started:

1.  **Clone this repository:**
    ```bash
    git clone <base-repo-url>
    cd <repo-name>
    ```

2.  **Run the bootstrap wizard:**
    This command will guide you through setting up your project, including renaming the package, configuring the CLI, and installing all dependencies into a virtual environment.
    ```bash
    uv run python mlsys init
    ```

3.  **Activate your new environment and start working:**
    The wizard will create a `.venv` directory. Activate it to use your new, personalized CLI.
    ```bash
    source .venv/bin/activate
    ```
    You can now use your custom CLI command (e.g., `lmf --help`).

## Features

*   **One-Command Setup**: `python3 mlsys init` handles everything from dependency installation to project naming.
*   **Modern Python with `uv`**: Uses `uv` for high-speed dependency management and `pyproject.toml` for configuration.
*   **Integrated Code Quality**: Comes with `ruff`, `mypy`, and `pre-commit` hooks for formatting, linting, and type-checking.
*   **Automated Docs**: Documentation with `mkdocs` and Material for MkDocs is ready to go. Use `<your-cli> docs --serve` to preview locally.
*   **Comprehensive Testing**: A `tests/` directory is set up with `pytest` for unit and integration tests.

## Core Philosophy

This template is built on the idea that a good project should be easy to set up, easy to maintain, and easy for new contributors to join. By automating the initial configuration, it lets you focus on the data science, not the boilerplate.
