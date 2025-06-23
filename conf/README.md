# MLOps Configuration System Guide

> **📁 Single Configuration Directory** - All configuration management consolidated here

This directory contains the **complete hierarchical configuration system** for the MLOps platform, powered by [Hydra](https://hydra.cc/) and [Pydantic](https://pydantic-docs.helpmanual.io/). All configuration types are now unified in this single location.

## 📋 **Configuration Architecture**

```
conf/
├── 📋 Base Configuration
│   ├── config.yaml              # Main configuration entry point
│   └── ml_systems.yaml          # ML system configurations
│
├── 🤖 ML Configuration
│   ├── model/                   # Model-specific configs
│   ├── data/                    # Data processing configs
│   ├── training/                # Training configurations
│   └── ml/                      # ML workflow settings
│
├── 🚀 Deployment Configuration
│   ├── api/                     # API server configs
│   ├── paths/                   # Path configurations
│   └── logging/                 # Logging settings
│
├── 🧠 AI/LLM Configuration
│   ├── prompts/                 # Prompt templates (moved from config/)
│   │   └── prompt_templates.yaml
│   └── llm/                     # LLM model configurations
│
└── 📝 Documentation
    └── README.md                # This file
```

### **✅ Unified Configuration Benefits**
- **Single source of truth** for all configuration
- **Consistent Hydra-based** management
- **Pydantic validation** across all config types
- **Environment-aware** configuration for all components
- **Version controlled** prompt templates alongside system configs

---

## 🚀 Guide for Data Scientists & ML Engineers

As a Data Scientist, your primary interaction with the config system will be for running experiments and training models.

### 1. How to Run an Experiment

The easiest way to run an experiment is by using the pre-defined workflows and overriding parameters from the command line.

**Example: Train a model with a different number of estimators.**

The default number of estimators for the Random Forest model is 100 (defined in `conf/model/random_forest.yaml`). You can override this directly:

```bash
# Run the training workflow with 150 estimators
python -m workflows.model_training model.parameters.n_estimators=150
```

Hydra automatically finds the `model_training` workflow and injects this new parameter.

### 2. How to Switch Models

The default model is `random_forest`. You can switch to a different model (like `xgboost`, once defined) by changing the `model` default.

**Example: Train with an XGBoost model.**

First, you would create a `conf/model/xgboost.yaml` file. Then, you can run the training workflow like this:

```bash
# Tell Hydra to use the 'xgboost.yaml' file for the 'model' config group
python -m workflows.model_training model=xgboost
```

### 3. Creating a New Experiment File

For more complex experiments, it's best to create a dedicated experiment file.

**Step 1: Create the experiment file.**

Create a new file, e.g., `conf/experiments/exp_001.yaml`:

```yaml
# conf/experiments/exp_001.yaml
defaults:
  - override /model: random_forest
  - _self_

# Override any parameters for this specific experiment
model:
  parameters:
    n_estimators: 250
    max_depth: 20

ml:
  test_size: 0.25
```

**Step 2: Run the experiment.**

Now, run the training workflow by pointing to your experiment config:

```bash
# The '-cn' flag tells Hydra to use your experiment file as the main config
python -m workflows.model_training --config-name experiments/exp_001
```

### 4. Working with Prompt Templates

**Example: Use a custom prompt for model analysis.**

The prompt templates are now in `conf/prompts/prompt_templates.yaml`. You can reference them in your workflows:

```python
# In your Python code
from src.config import load_config

config = load_config()
prompt = config.prompts.v1.classification_analysis.template.format(
    dataset_name="customer_churn",
    model_type="random_forest",
    metrics=model_metrics
)
```

---

## 🔧 Guide for Platform Engineers

As a Platform Engineer, you will configure the system for different environments and manage the operational settings.

### 1. How to Configure a New Environment

The system is designed for multiple environments (dev, staging, prod). The active environment is determined by the `ENVIRONMENT` shell variable.

**Example: Running the API in production mode.**

The main `config.yaml` defaults the API to `development`. The `api` configuration group allows switching between `development.yaml` and `production.yaml`.

```bash
# Set the environment variable to 'production'
export ENVIRONMENT=production

# Now, when you run the API, it will use the settings from 'conf/api/production.yaml'
# which has more workers, enables security features, etc.
uvicorn src.api.app:create_app --factory
```

### 2. How to Manage Secrets

**Never commit secrets to Git.** This system is designed to read secrets from environment variables.

**Step 1: Use environment variables.**

In your production environment, set shell variables for secrets:

```bash
export API_PORT=8080
export REDIS_URL="redis://prod-redis:6379"
# etc.
```

**Step 2: Reference them in config.**

The configuration files (e.g., `conf/api/production.yaml`) use Hydra's interpolation syntax to read these variables:

```yaml
# from conf/api/production.yaml
port: ${oc.env:API_PORT,8000} # Uses API_PORT if set, otherwise defaults to 8000
...
caching:
  redis_url: ${oc.env:REDIS_URL,"redis://localhost:6379"}
```

Use the `.env-example` file as a template for which environment variables are needed.

### 3. Configuring the API Server

You can change API settings like the number of workers or request timeouts by editing the relevant environment file (e.g., `conf/api/production.yaml`).

```yaml
# conf/api/production.yaml
workers: ${oc.env:API_WORKERS,4} # Set via shell variable or default to 4
timeout: 30
```

---

## 🏗️ Configuration System Architecture

This section contains the technical details of the configuration system.

### Directory Structure
```
conf/
├── config.yaml              # Main configuration entry point
│
├── model/                   # Model-specific configs (e.g., random_forest.yaml)
├── data/                    # Data processing configs
├── training/                # Training configs
├── api/                     # API server configs (development.yaml, production.yaml)
├── prompts/                 # LLM prompt templates (moved from config/)
└── logging/                 # Logging configurations
```

### How It Works

1.  **Main Entry Point**: `config.yaml` is the primary config file. It defines the global defaults and the overall structure.
2.  **Config Groups**: The `defaults` list in `config.yaml` sets up "config groups" (e.g., `model`, `api`). For example, `model: random_forest` tells Hydra to look for a `random_forest.yaml` file inside the `conf/model/` directory.
3.  **Composition**: Hydra reads the main file, finds the default files for each group, and composes them into a single configuration object.
4.  **Validation**: This composed configuration is then validated against the Pydantic models in `src/config/models.py` to ensure type safety.
5.  **Overrides**: Any command-line arguments (e.g., `model.parameters.n_estimators=150`) are applied as the final step, overriding any values from the YAML files.

### Pydantic Models (`src/config/models.py`)

The configuration is validated by a set of Pydantic models. The main model is `src.config.models.Config`, which is composed of smaller models like `MLConfig`, `APIConfig`, and `ModelConfig`. These models ensure that all required fields are present and have the correct data types.

### Configuration Manager (`src/config/manager.py`)

The `ConfigManager` class is the Python interface to the configuration system. It handles:
- Initializing Hydra.
- Loading and composing the YAML files.
- Validating the configuration with Pydantic.
- Providing the validated `Config` object to the rest of the application.

---

## 🔄 **Migration from config/ Directory**

**Note**: The separate `config/` directory has been consolidated into `conf/` for consistency:
- ✅ `config/prompt_templates.yaml` → `conf/prompts/prompt_templates.yaml`
- ✅ All configuration now uses consistent Hydra-based management
- ✅ Single source of truth for all project configuration
