# Project Initialization

This template comes with a built-in command-line tool, `mlsys`, designed to make starting a new project simple and fast.

## Quickstart

To turn this template into your own project, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/analysis-template.git
    cd analysis-template
    ```

2.  **Run the Initialization Script:**
    Execute the `mlsys` script and provide your new project's name. The name should be in "kebab-case".

    ```bash
    ./mlsys init "my-awesome-project"
    ```

## What the Script Does

The `init` command automates all the necessary boilerplate changes to configure the template for your project:

1.  **Creates a Bootstrap Environment:** It safely creates a temporary virtual environment (`.bootstrap_venv`) to run itself without affecting your global Python packages.
2.  **Renames the Source Directory:** It renames the core `src/analysis_template` directory to match your project name (e.g., `src/my_awesome_project`).
3.  **Updates `pyproject.toml`:** It sets the `[project.name]` and updates the Ruff linter's `known-first-party` setting to match your new package name.
4.  **Updates Documentation:** It performs a search-and-replace across all documentation files to replace placeholder names like "Analysis Template" with your new project's title.

After the script finishes, your project is ready to go. You can install the dependencies and run the test suite to confirm everything is working correctly.
