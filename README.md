# MLOps Template → MLX Foundation

A production-ready ML systems template evolving into **MLX**: the first AI-enhanced composable ML platform. This repository serves as both a standalone template and the foundation for component-based ML development.

**Current Highlights**

- 🚀 **FastAPI micro-service** with interactive docs & production hardening
- 🐳 **Docker & Compose** for reproducible packaging
- 🔄 **CI/CD via GitHub Actions** with security scanning
- 🎯 **Advanced Configuration** with Hydra + Pydantic validation
- 🧩 **Plugin Architecture** for rapid extensibility
- 🔒 **Enterprise Security** with rate limiting & vulnerability scanning
- 📊 **Intelligent Caching** with cost optimization
- 🤖 **LLM Integration** with prompt management & versioning

## Quick Start
```bash
# Clone and rename the template
git clone <repo-url> my-project && cd my-project
./mlsys my_project_name          # transforms packages & configs

# Run quality gate + start the API
make all-checks && make run-api
```
Browse http://localhost:8000/docs for interactive API docs.

## 🎯 MLX Evolution Path

This template is evolving into **MLX**: an AI-enhanced composable ML platform. Current features map to future MLX components:

| Current Feature | Future MLX Component | Enhanced Capability |
|----------------|---------------------|-------------------|
| Plugin System | Component Framework | AI-guided composition |
| Branch Specializations | Domain Components | Smart recommendations |
| Configuration System | Config Management | Intelligent validation |
| CI/CD Workflows | Testing Infrastructure | AI test selection |
| Quality Stack | Code Quality Components | Automated optimization |

## Choose a Specialization
Template branches layer domain-specific tooling on the same foundation:

| Branch | Domain Focus | Key Technologies |
|--------|--------------|-----------------|
| main | General ML / analytics | scikit-learn, data workflows |
| agentic-ai-system | Multi-agent LLM apps | LangChain, AutoGen |
| llm-finetuning-system | LLM fine-tuning & serving | 🤗 Transformers, LoRA |
| chalk-feature-engineering | Real-time feature store | Chalk, streaming |

Clone directly into your specialization:
```bash
git clone -b agentic-ai-system <repo-url> my-agent-app
```
Full details in [`branching_strategy.md`](BRANCHING_STRATEGY.md).

## 🏗️ Advanced Architecture

### Production-Grade Configuration
- **Hierarchical configs** with environment inheritance
- **Type-safe validation** using Pydantic models
- **Secret management** with environment variable resolution
- **Experiment tracking** with A/B testing support
- **Runtime configuration** with dynamic overrides

```python
# Type-safe configuration with validation
from src.config import load_config

config = load_config(overrides={
    "model": {"model_type": "xgboost"},
    "api": {"workers": 4, "enable_cors": true}
})
```

### Plugin System
Extensible architecture for rapid capability addition:
```python
# Add new functionality via plugins
from src.plugins import PluginRegistry

registry = PluginRegistry()
llm_provider = registry.get_plugin("llm_providers", "openai")
result = llm_provider.generate(prompt, config.llm)
```

### Intelligent Features
- **Cost-optimized caching** (70% LLM cost reduction)
- **Multi-service rate limiting** with budget enforcement  
- **Template versioning** with A/B testing capabilities
- **Automated prompt optimization** with performance tracking

## Requirements
* Python 3.9+
* [uv](https://github.com/astral-sh/uv) package manager
* Docker (optional but recommended)

## Everyday Commands
```bash
make install-dev   # deps + pre-commit hooks
make lint          # ruff + mypy + security checks
make test          # pytest with coverage
make run-api       # uvicorn with auto-reload
make build-docker  # optimized multi-stage container
make docs-serve    # local documentation server
```
Run `make help` to list all targets.

## Configuration System

Advanced Hydra-based configuration with type safety:

```bash
# Environment-specific configuration
export ENVIRONMENT=production
python script.py

# Dynamic overrides
python script.py model.type=xgboost api.workers=8

# Experiment configuration
python script.py --config-name=experiments/hyperparameter_tuning
```

### Configuration Structure
```
conf/
├── config.yaml              # Main configuration
├── model/                   # Model-specific configs
│   ├── random_forest.yaml
│   ├── xgboost.yaml
│   └── neural_network.yaml
├── api/                     # API configurations
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
├── experiments/             # Experiment configurations
└── secrets.yaml.example     # Secret management template
```

See [`config/README.md`](config/README.md) for comprehensive configuration guide.

## 🧩 Plugin Architecture

Extend functionality without modifying core code:

```python
# Create custom plugins
from src.plugins.base import BasePlugin

class CustomMLPlugin(BasePlugin):
    def process(self, data, context):
        # Your custom ML logic
        return enhanced_data

# Register and use
registry.register_plugin("ml_processors", "custom", CustomMLPlugin())
```

## Documentation
Comprehensive guides in [`docs/`](docs/):
- **[Getting Started](docs/getting-started.md)** - Setup & first steps
- **[Configuration Guide](config/README.md)** - Advanced configuration
- **[Plugin Development](docs/development/)** - Extending functionality
- **[API Reference](docs/api/)** - Code documentation

Serve locally: `make docs-serve` → http://127.0.0.1:8001

## Contributing
- `make all-checks` must pass before opening PRs
- Pre-commit hooks enforce style and security
- Add tests for new functionality
- Update documentation for new features

## 🚀 Migration to MLX

This repository is preparing for migration to the MLX composable platform. The migration will:

1. **Extract reusable components** from current architecture
2. **Enable AI-enhanced testing** with smart permutation selection  
3. **Implement intelligent composition** with compatibility prediction
4. **Scale component ecosystem** with community contributions

Stay tuned for MLX CLI: `npx create-mlx@latest my-project`

## License
MIT
