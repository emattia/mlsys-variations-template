# ADR-005: Hydra + Pydantic Configuration Management

## Status
Accepted (2025-06-23)

## Context

ML systems require sophisticated configuration management to handle:

1. **Hierarchical Configs**: Environment-specific overrides (dev/staging/prod)
2. **Type Safety**: Validate configuration values at startup
3. **Composition**: Combine configurations from multiple sources
4. **Experimentation**: Easy parameter sweeps and hyperparameter tuning
5. **Documentation**: Self-documenting configuration schemas
6. **CLI Integration**: Command-line override capabilities

Existing solutions had limitations:
- **Plain YAML/JSON**: No validation, no type safety, error-prone
- **Environment Variables**: Flat namespace, type conversion issues
- **Argparse**: Not suitable for complex nested configurations
- **Custom Solutions**: High maintenance, lacking features

The system needed to support complex use cases like:
```yaml
# Base configuration
ml_system:
  data_source:
    type: snowflake
    config:
      account: ${env:SNOWFLAKE_ACCOUNT}
      warehouse: COMPUTE_WH

  model:
    type: xgboost
    params:
      max_depth: 6
      learning_rate: 0.1
```

## Decision

We adopt **Hydra + Pydantic** for configuration management:

- **Hydra**: Handles composition, overrides, and CLI integration
- **Pydantic**: Provides type validation and documentation
- **OmegaConf**: Underlying configuration representation (used by Hydra)

### Architecture

```python
# Configuration schemas with Pydantic
class DataSourceConfig(BaseModel):
    type: str
    config: Dict[str, Any]

    @validator('type')
    def validate_type(cls, v):
        allowed_types = ['snowflake', 'postgres', 's3']
        if v not in allowed_types:
            raise ValueError(f'Invalid type: {v}. Must be one of {allowed_types}')
        return v

class MLSystemConfig(BaseModel):
    data_source: DataSourceConfig
    model: ModelConfig
    training: TrainingConfig

# Hydra application
@hydra.main(version_base=None, config_path="conf", config_name="ml_systems")
def main(cfg: DictConfig) -> None:
    # Validate with Pydantic
    validated_config = MLSystemConfig(**cfg)

    # Use type-safe configuration
    run_ml_pipeline(validated_config)
```

### Directory Structure
```
conf/
├── ml_systems.yaml          # Main config file
├── data_source/
│   ├── snowflake.yaml      # Data source configs
│   └── postgres.yaml
├── model/
│   ├── xgboost.yaml        # Model configs
│   └── pytorch.yaml
├── env/
│   ├── development.yaml    # Environment overrides
│   ├── staging.yaml
│   └── production.yaml
└── experiment/
    └── hyperparameter_sweep.yaml
```

## Alternatives Considered

### 1. Pure Pydantic with YAML
- **Pros**: Type safety, good validation, simple
- **Cons**: No composition, no CLI overrides, manual environment handling
- **Rejected**: Insufficient for complex ML experimentation needs

### 2. Pure Hydra with OmegaConf
- **Pros**: Excellent composition, CLI integration, flexible
- **Cons**: Runtime type errors, no schema documentation
- **Rejected**: Type safety critical for production ML systems

### 3. Dynaconf
- **Pros**: Multiple format support, environment-aware
- **Cons**: Limited composition, no type validation, smaller ecosystem
- **Rejected**: Feature gaps for ML use cases

### 4. Gin-Config (Google)
- **Pros**: Powerful function configuration, good for research
- **Cons**: Limited Python ecosystem, steep learning curve
- **Rejected**: Too specialized, limited adoption

### 5. MLflow Projects
- **Pros**: ML-specific, integrated with MLflow ecosystem
- **Cons**: Tied to MLflow, limited composition capabilities
- **Rejected**: Vendor lock-in concerns

## Consequences

### Positive
1. **Type Safety**: Pydantic catches configuration errors at startup
2. **Composition**: Hydra enables powerful configuration inheritance
3. **CLI Integration**: Easy parameter overrides from command line
4. **Documentation**: Pydantic models serve as configuration documentation
5. **Experimentation**: Built-in support for hyperparameter sweeps
6. **Environment Management**: Clean separation of environment-specific configs
7. **IDE Support**: Type hints provide excellent IDE autocompletion
8. **Validation**: Rich validation with custom validators and constraints

### Negative
1. **Learning Curve**: Developers must learn both Hydra and Pydantic
2. **Complexity**: More complex than simple YAML configuration
3. **Dependencies**: Additional dependencies on Hydra and Pydantic
4. **Debugging**: Configuration composition can be hard to debug
5. **Performance**: Validation overhead on application startup

### Implementation Patterns

#### Configuration Composition
```yaml
# conf/ml_systems.yaml
defaults:
  - data_source: snowflake
  - model: xgboost
  - env: development
  - _self_

# Override with: python main.py model=pytorch env=production
```

#### Type-Safe Plugin Configuration
```python
class PluginConfig(BaseModel):
    name: str
    type: PluginType
    config: Dict[str, Any]

    @validator('config')
    def validate_plugin_config(cls, v, values):
        plugin_type = values.get('type')
        # Validate config against plugin-specific schema
        return validate_plugin_config(plugin_type, v)
```

#### Environment-Specific Overrides
```yaml
# conf/env/production.yaml
# Override values for production
data_source:
  config:
    warehouse: PROD_WH
    pool_size: 20

monitoring:
  enabled: true
  level: INFO
```

#### Experiment Configuration
```yaml
# conf/experiment/hyperparameter_sweep.yaml
# +experiment=hyperparameter_sweep
defaults:
  - override /model: xgboost

model:
  params:
    max_depth: choice([3, 6, 9])
    learning_rate: range(0.01, 0.3, 0.01)

hydra:
  sweeper:
    _target_: hydra._internal.core.plugins.basic_sweeper.BasicSweeper
    max_jobs: 100
```

### Migration Strategy
1. **Phase 1**: Convert existing YAML configs to Hydra structure
2. **Phase 2**: Add Pydantic validation schemas
3. **Phase 3**: Implement plugin-specific configuration validation
4. **Phase 4**: Add experiment and sweep configurations

### Best Practices
1. **Schema Documentation**: Use Pydantic docstrings for configuration help
2. **Reasonable Defaults**: Provide sensible defaults for all optional fields
3. **Environment Variables**: Use Hydra resolvers for secrets and environment-specific values
4. **Validation**: Implement custom validators for complex business rules
5. **Testing**: Test configuration schemas with valid and invalid inputs
