# Project Structure

This page explains the organization and structure of the Analysis Template.

## Overview

The template follows a structured organization designed to promote best practices in data science projects:

```
├── .github/            # GitHub configuration (Actions workflows)
├── data/               # Data files
│   ├── raw/            # Original, immutable data
│   ├── processed/      # Cleaned, transformed data
│   ├── interim/        # Intermediate data
│   └── external/       # Data from external sources
├── documentation/      # Project documentation
├── endpoints/          # API endpoints and services
├── models/             # Trained models and model artifacts
│   ├── trained/        # Saved model files
│   └── evaluation/     # Model evaluation results
├── notebooks/          # Jupyter notebooks
├── reports/            # Generated analysis reports
│   ├── figures/        # Generated graphics and figures
│   └── tables/         # Generated tables
├── src/                # Source code for use in this project
├── tests/              # Test suite
└── workflows/          # Data and model workflows
    └── tests/          # Workflow tests
```

## Key Directories

### data/

The `data/` directory contains all data files used in the project. It is organized into subdirectories:

- **raw/**: Original, immutable data dumps. This data should never be modified.
- **processed/**: Cleaned and processed data, ready for analysis.
- **interim/**: Intermediate data that has been transformed but is not yet in its final form.
- **external/**: Data from external sources, such as third-party datasets.

See [data/README.md](https://github.com/yourusername/analysis-template/blob/main/data/README.md) for more details.

### src/

The `src/` directory contains the core source code for the project, organized into reusable modules and packages:

- **data/**: Data loading, processing, and feature engineering
- **models/**: Model definition, training, and evaluation
- **visualization/**: Plotting and visualization utilities
- **utils/**: General utility functions
- **config/**: Configuration management
- **io/**: Input/output operations

See [src/README.md](https://github.com/yourusername/analysis-template/blob/main/src/README.md) for more details.

### notebooks/

The `notebooks/` directory contains Jupyter notebooks for exploratory data analysis, visualization, model development, and reporting:

- **00_exploration**: Initial data exploration
- **01_preprocessing**: Data cleaning and feature engineering
- **02_modeling**: Model development and training
- **03_evaluation**: Model evaluation and comparison
- **04_reporting**: Final results and visualizations

See [notebooks/README.md](https://github.com/yourusername/analysis-template/blob/main/notebooks/README.md) for more details.

### workflows/

The `workflows/` directory contains data processing and model training workflows that can be executed as standalone scripts or imported as modules:

- **data_processing.py**: Example workflow for data processing
- **model_training/**: Model training workflows
- **evaluation/**: Model evaluation workflows
- **deployment/**: Model deployment workflows
- **utils/**: Utility functions for workflows
- **config/**: Configuration files for workflows
- **tests/**: Tests for workflows

See [workflows/README.md](https://github.com/yourusername/analysis-template/blob/main/workflows/README.md) for more details.

### models/

The `models/` directory contains trained models, model artifacts, and evaluation results:

- **trained/**: Saved model files
- **evaluation/**: Model evaluation results
- **metadata/**: Model metadata and documentation
- **registry/**: Model registry information

See [models/README.md](https://github.com/yourusername/analysis-template/blob/main/models/README.md) for more details.

### reports/

The `reports/` directory contains generated reports, figures, tables, and other outputs from analyses:

- **figures/**: Generated plots, charts, and visualizations
- **tables/**: Generated data tables and summaries
- **documents/**: Generated reports and documents
- **presentations/**: Slides and presentation materials

See [reports/README.md](https://github.com/yourusername/analysis-template/blob/main/reports/README.md) for more details.

### tests/

The `tests/` directory contains the test suite for the project, ensuring code quality and correctness:

- **unit/**: Tests for individual functions and classes
- **integration/**: Tests for interactions between components
- **functional/**: Tests for end-to-end functionality
- **fixtures/**: Test data and fixtures
- **conftest.py**: Shared pytest fixtures and configuration

See [tests/README.md](https://github.com/yourusername/analysis-template/blob/main/tests/README.md) for more details.

### documentation/

The `documentation/` directory contains project documentation, guides, and references:

- **nbdoc_guide.md**: Guide for using NBDoc to document Jupyter notebooks
- **generated/**: Auto-generated documentation (not committed to Git)
  - **notebooks/**: Documentation generated from notebooks
  - **api/**: API documentation generated from source code

See [documentation/README.md](https://github.com/yourusername/analysis-template/blob/main/documentation/README.md) for more details.

### endpoints/

The `endpoints/` directory contains API endpoints, services, and interfaces for exposing your data science models and workflows:

- **api/**: REST API implementations
- **batch/**: Batch processing endpoints
- **streaming/**: Streaming data endpoints
- **schemas/**: Data validation schemas
- **utils/**: Utility functions for endpoints

See [endpoints/README.md](https://github.com/yourusername/analysis-template/blob/main/endpoints/README.md) for more details.

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
