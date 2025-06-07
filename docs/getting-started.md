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

1. Place raw data files in the `data/raw/` directory.
2. Create data processing workflows in the `workflows/` directory.
3. Process the data and save the results to `data/processed/`.

## Creating Notebooks

1. Create new notebooks in the `notebooks/` directory.
2. Follow the structure of the example notebook.
3. Document your notebooks using NBDoc (see `documentation/nbdoc_guide.md`).

## Developing Source Code

1. Add reusable functions and classes to the `src/` directory.
2. Organize code into modules by functionality.
3. Write tests for your code in the `tests/` directory.

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

Generate documentation for your project:

```bash
# Generate notebook documentation
python -m nbdoc build notebooks/ -o documentation/generated/notebooks

# Generate API documentation
python -m pdoc --html --output-dir documentation/generated/api src

# Build MkDocs site
mkdocs build
```

## Next Steps

Once you're familiar with the template, you can:

- Customize the project structure to fit your needs
- Add your own data processing workflows
- Develop custom models and analysis pipelines
- Set up continuous integration with GitHub Actions
- Deploy models as APIs using the endpoints framework
