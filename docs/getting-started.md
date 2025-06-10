# Getting Started

This guide will help you get started with the Analysis Template.

## Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Python package installer)
- Git

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/analysis-template.git
cd analysis-template
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
uv pip install -e ".[dev]"
```

4. Set up pre-commit hooks:

```bash
pre-commit install
```

## Environment Configuration

1. Copy the example environment file:

```bash
cp .env-example .env
```

2. Edit `.env` with your specific configuration values.

## Project Structure

The template follows a structured organization:

- **data/**: All data files
- **src/**: Source code modules
- **notebooks/**: Jupyter notebooks
- **workflows/**: Data processing and model training workflows
- **tests/**: Test suite
- **reports/**: Generated reports and visualizations
- **models/**: Trained models and artifacts
- **documentation/**: Project documentation
- **endpoints/**: API endpoints and services

## First Steps

Here are some suggested first steps to get familiar with the template:

1. **Explore the example notebook**: Open `notebooks/example_analysis.ipynb` to see a complete example of a data analysis workflow.

2. **Run the tests**: Execute `pytest` to run the test suite and verify that everything is working correctly.

3. **Examine the workflow example**: Look at `workflows/data_processing.py` to understand how data processing workflows are structured.

4. **Review the source code**: Check out `src/data_utils.py` to see how reusable functions are organized.

## Adding Your Own Data

1. Place raw data files in the `data/raw/` directory **or ingest directly from an external source via a plugin**.  For example, to pull a Snowflake table into `data/raw/` you can run:

```bash
uv pip install snowflake-connector-python
python -m workflows.data_ingest \
    data.source=snowflake \
    data.snowflake.account=<ACCOUNT> \
    data.snowflake.database=<DB> \
    data.snowflake.query='SELECT * FROM SALES'
```

2. Create data processing workflows in the `workflows/` directory (see `workflows/data_processing.py` for an example).  These workflows should read from `data/raw/` **only** and write their outputs to `data/processed/`.

3. Keep large external datasets out of Git.  Instead reference them via environment variables (e.g. an S3 URI) or a plugin-specific config file under `conf/data/`.

## Creating Notebooks

1. Create new notebooks in the `notebooks/` directory.
2. Follow the structure of the example notebook.
3. Document your notebooks using NBDoc (see `documentation/nbdoc_guide.md`).

## Developing Source Code

1. Put **pure, importable** functions and classes in `src/`.  These should not perform direct I/O – leave that to the workflows.
2. Orchestrate your library code inside `workflows/` scripts (or DAGs) which *do* perform I/O and move artefacts between `data/` ↔ `models/` ↔ `reports/`.
3. When adding new functionality:
   - Write unit tests under `tests/unit/` for every public function.
   - Add or update a smoke test under `workflows/tests/` that executes the full workflow against a small fixture dataset (see `tests/fixtures/`).
4. If the code needs to be surfaced as an API, create an endpoint inside `endpoints/` and add an integration test under `tests/integration/`.

## Running Tests

Run the test suite to ensure your code is working correctly:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src

# Run workflow tests
pytest workflows/tests/
```

## Building Documentation

```bash
# 1. Turn notebooks → markdown (nbdoc)
python -m nbdoc build notebooks/ -o docs/generated/notebooks

# 2. Generate API docs from docstrings (pdoc)
python -m pdoc --html --output-dir docs/generated/api src

# 3. Build & serve the MkDocs site with live reload
mkdocs serve  # http://127.0.0.1:8000

# 4. Or run everything in one step
make docs  # output goes to the local 'site/' directory
```

A GitHub Actions workflow (`.github/workflows/docs.yml`) publishes the built site to **GitHub Pages** on every push to `main`.

## Next Steps

Once you're familiar with the template, you can:

- Customize the project structure to fit your needs
- Add your own data processing workflows
- Develop custom models and analysis pipelines
- Set up continuous integration with GitHub Actions
- Deploy models as APIs using the endpoints framework
