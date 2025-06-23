# Project Structure

> "Design is not just what it looks like and feels like. Design is how it works." — Steve Jobs

This page explains the organization and structure of the Debug Toml Test.

## Overview

The template follows a structured organization designed to promote best practices in data science projects:

```
├── .env/               # Environment variables directory
├── .github/            # GitHub configuration (Actions workflows)
├── .projen/            # Projen configuration and generated files
├── conf/               # Hydra & project configuration files
├── data/               # Data files
│   ├── raw/            # Original, immutable data
│   ├── processed/      # Cleaned, transformed data
│   ├── interim/        # Intermediate data
│   └── external/       # Data from external sources
├── docs/               # Project documentation (MkDocs)
├── endpoints/          # API endpoints and services
├── mlx-components/     # MLX component registry and extracted components
├── models/             # Trained models and model artifacts
│   ├── trained/        # Saved model files
│   └── evaluation/     # Model evaluation results
├── notebooks/          # Jupyter notebooks
├── plugins/            # Plugin system and custom plugins
├── reports/            # Generated analysis reports
│   ├── figures/        # Generated graphics and figures
│   └── tables/         # Generated tables
├── src/                # Source code for use in this project
├── templates/          # Project templates and scaffolding
├── tests/              # Test suite
└── workflows/          # Data and model workflows
    └── tests/          # Workflow tests
```

## Key Directories

### .env

The `.env/` directory contains a Python virtual environment for the project. This directory is created when setting up the development environment and contains the Python interpreter and installed packages for the project.

### .projen/

The `.projen/` directory contains projen-generated configuration files and build artifacts. This directory is managed automatically by projen and should not be manually edited.

### mlx-components/

The `mlx-components/` directory contains the MLX component registry and extracted reusable components:

- **registry.json**: Component metadata and registry
- **extracted/**: Extracted components from source code
- **templates/**: Component templates for scaffolding

This directory is created when running `mlx extract` and is used by the MLX component system.

### plugins/

The `plugins/` directory contains the plugin system implementation and custom plugins:

- **core/**: Core plugin infrastructure
- **custom/**: Project-specific plugins
- **registry/**: Plugin registry and metadata
- **examples/**: Example plugin implementations

### templates/

The `templates/` directory contains project templates, scaffolding files, and code generation templates:

- **project/**: Project structure templates
- **component/**: Component templates
- **config/**: Configuration file templates
- **docs/**: Documentation templates

### data/

The `data/` directory contains all data files used in the project. It is organized into subdirectories:

- **raw/**: Original, immutable data dumps. This data should never be modified.
- **processed/**: Cleaned and processed data, ready for analysis.
- **interim/**: Intermediate data that has been transformed but is not yet in its final form.
- **external/**: Data from external sources, such as third-party datasets.

See [data/README.md](https://github.com/yourusername/debug-toml-test/blob/main/data/README.md) for more details.

### src/

The `src/` directory contains the core source code for the project, organized into reusable modules and packages:

- **data/**: Data loading, processing, and feature engineering
- **models/**: Model definition, training, and evaluation
- **visualization/**: Plotting and visualization utilities
- **utils/**: General utility functions
- **config/**: Configuration management
- **io/**: Input/output operations

See [src/README.md](https://github.com/yourusername/debug-toml-test/blob/main/src/README.md) for more details.

### notebooks/

The `notebooks/` directory contains Jupyter notebooks for exploratory data analysis, visualization, model development, and reporting:

- **00_exploration**: Initial data exploration
- **01_preprocessing**: Data cleaning and feature engineering
- **02_modeling**: Model development and training
- **03_evaluation**: Model evaluation and comparison
- **04_reporting**: Final results and visualizations

See [notebooks/README.md](https://github.com/yourusername/debug-toml-test/blob/main/notebooks/README.md) for more details.

### workflows/

The `workflows/` directory contains data processing and model training workflows that can be executed as standalone scripts or imported as modules:

- **data_processing.py**: Example workflow for data processing
- **model_training/**: Model training workflows
- **evaluation/**: Model evaluation workflows
- **deployment/**: Model deployment workflows
- **utils/**: Utility functions for workflows
- **config/**: Configuration files for workflows
- **tests/**: Tests for workflows

See [workflows/README.md](https://github.com/yourusername/debug-toml-test/blob/main/workflows/README.md) for more details.

### models/

The `models/` directory contains trained models, model artifacts, **and their evaluation results**:

- **trained/**: Saved model files (weights, checkpoints, tokenizer files)
- **evaluation/**: Metrics, plots (ROC, PR curves), feature-importance charts, and the JSON summaries generated by `workflows/model_evaluation.py`

Why evaluation lives here and not in `reports/`:

> A model and its quantitative evaluation form a single immutable artefact. Keeping them side-by-side guarantees lineage and avoids the "orphan ROC curve" problem when you have many model versions in flight.

If you need to publish an executive report, export the relevant charts from here into `reports/`.

### Hugging Face (or other) model caches

Large third-party models are downloaded into a **global cache** (default: `~/.cache/huggingface`).  To pin that cache inside the project (e.g. for reproducible CI) set the path in your config or override at runtime:

```bash
python -m workflows.model_training \
    model.name=bert-base-uncased \
    paths.hf_cache=models/trained/hf_cache
```

and in code:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained(cfg.model.name, cache_dir=cfg.paths.hf_cache)
```

### reports/

The `reports/` directory contains generated reports, figures, tables, and other outputs from analyses:

- **figures/**: Generated plots, charts, and visualizations
- **tables/**: Generated data tables and summaries
- **documents/**: Generated reports and documents
- **presentations/**: Slides and presentation materials

See [reports/README.md](https://github.com/yourusername/debug-toml-test/blob/main/reports/README.md) for more details.

### tests/

The `tests/` directory contains the test suite for the project, ensuring code quality and correctness:

- **unit/**: Tests for individual functions and classes
- **integration/**: Tests for interactions between components
- **functional/**: Tests for end-to-end functionality
- **fixtures/**: Test data and fixtures
- **conftest.py**: Shared pytest fixtures and configuration

See [tests/README.md](https://github.com/yourusername/debug-toml-test/blob/main/tests/README.md) for more details.

### docs/

`docs/` is the **single source-of-truth for human-written documentation** and the root used by MkDocs when building the site.  All guides (including this page), development references, security docs, etc. live here.  Auto-generated assets from NBDoc and pdoc are written to `docs/generated/` and are *not* committed to Git.

> Legacy note: an old `documentation/` folder has been removed; if you still have it locally you can delete it.

### endpoints/

The `endpoints/` directory contains API endpoints, services, and interfaces for exposing your data science models and workflows:

- **api/**: REST API implementations
- **batch/**: Batch processing endpoints
- **streaming/**: Streaming data endpoints
- **schemas/**: Data validation schemas
- **utils/**: Utility functions for endpoints

See [endpoints/README.md](https://github.com/yourusername/debug-toml-test/blob/main/endpoints/README.md) for more details.

## Configuration Files

- **pyproject.toml**: Python project configuration, including dependencies and tool settings
- **.env-example**: Example environment variables
- **.gitignore**: Files and directories to ignore in Git
- **.pre-commit-config.yaml**: Pre-commit hook configuration
- **mkdocs.yml**: MkDocs configuration for documentation site
- **nbdoc_config.json**: NBDoc configuration for notebook documentation

## Customizing the Structure

The template structure is designed to be flexible and can be customized to fit your specific needs:

1. **Add new directories**: Create new directories for specific aspects of your project
2. **Remove unused directories**: Remove directories that are not relevant to your project
3. **Rename directories**: Rename directories to better reflect your project's terminology
4. **Reorganize subdirectories**: Change the organization of subdirectories to match your workflow

When customizing the structure, make sure to update the documentation to reflect your changes.
