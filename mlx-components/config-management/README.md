# Configuration Management Components

Production-ready configuration management with Hydra, Pydantic validation, and environment handling.

## Quick Start

Add configuration management capabilities to your MLX project:

```bash
# Add via MLX CLI
./mlx add config-management

# Or via uv package manager
uv pip install mlx-config-management
```

## Components

## Description
Hydra + Pydantic configuration system with multi-environment support and secret management

## Dependencies

### Python Dependencies
```bash
uv add hydra-core>=1.3.0
uv add pydantic>=2.0.0
```

### System Dependencies


### Environment Variables
Required: REDIS_URL, API_WORKERS, API_PORT, MODEL_DIRECTORY
Secrets: api_key_header

## Configuration
The component will be installed with the following merge strategies:
- conf/config.yaml: replace
- conf/ml_systems.yaml: replace
- conf/training/classification.yaml: replace
- conf/paths/default.yaml: replace
- conf/ml/default.yaml: replace
- conf/model/random_forest.yaml: replace
- conf/model/default.yaml: replace
- conf/prompts/prompt_templates.yaml: replace
- conf/api/production.yaml: enhance
- conf/api/development.yaml: merge
- conf/data/default.yaml: replace
- conf/logging/default.yaml: replace

## Monitoring
Health checks: src/config/models.py, src/config/manager.py
Endpoints:
