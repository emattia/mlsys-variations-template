# Configuration Management

This directory contains the hierarchical configuration system that powers the MLOps platform. It provides flexible, environment-aware configuration management for both platform operators and ML engineers.

## 🎯 **Purpose**

The configuration system enables:

- **🔧 Environment Management**: Different configs for development, staging, production
- **📊 Experiment Configuration**: Parameterized ML experiments and workflows
- **🚀 API Configuration**: Production API server settings
- **🔒 Security Management**: Secure handling of secrets and credentials
- **🧩 Modular Configuration**: Composable config components
- **📝 Type Safety**: Pydantic-based validation and type checking

## 🏗️ **Configuration Architecture**

```
conf/
├── 📋 Base Configuration
│   ├── config.yaml              # Main configuration file
│   └── defaults.yaml            # Default values
│
├── 🤖 ML Configuration
│   ├── model/                   # Model-specific configs
│   │   ├── random_forest.yaml   # Random Forest settings
│   │   ├── xgboost.yaml         # XGBoost settings
│   │   └── neural_network.yaml  # Neural network configs
│   │
│   ├── data/                    # Data processing configs
│   │   ├── default.yaml         # Standard data processing
│   │   ├── preprocessing.yaml   # Feature engineering
│   │   └── validation.yaml      # Data quality rules
│   │
│   └── training/                # Training configurations
│       ├── classification.yaml  # Classification settings
│       ├── regression.yaml      # Regression settings
│       └── hyperparameters.yaml # Hyperparameter ranges
│
├── 🚀 Deployment Configuration
│   ├── api/                     # API server configs
│   │   ├── development.yaml     # Development settings
│   │   ├── staging.yaml         # Staging environment
│   │   └── production.yaml      # Production settings
│   │
│   └── docker/                  # Container configurations
│       ├── base.yaml            # Base container config
│       └── monitoring.yaml      # Monitoring stack
│
└── 🔒 Environment Configuration
    ├── local.yaml               # Local development
    ├── ci.yaml                  # CI/CD pipeline
    └── secrets.yaml.example     # Secret template (not committed)
```

---

## 🚀 **Quick Start**

### **Basic Configuration Usage**

```python
# Load default configuration
from src.config import load_config

config = load_config()
print(f"Model type: {config.model.model_type}")
print(f"API port: {config.api.port}")
```

### **Environment-Specific Configuration**

```bash
# Development environment
export ENVIRONMENT=development
python -m workflows.model_training

# Production environment
export ENVIRONMENT=production
python -m workflows.model_training

# Override specific values
export MODEL_TYPE=xgboost
export API_PORT=9000
python -m workflows.model_training
```

### **Custom Configuration Override**

```python
# Override configuration programmatically
from src.config import load_config

config = load_config(overrides={
    "model": {"model_type": "xgboost"},
    "ml": {"test_size": 0.3},
    "api": {"port": 9000}
})
```

---

## 📋 **Configuration Files**

### **Main Configuration (config.yaml)**

```yaml
# conf/config.yaml
defaults:
  - model: random_forest
  - data: default
  - training: classification
  - api: development
  - _self_

# Global settings
project_name: "mlops-template"
version: "1.0.0"
environment: ${oc.env:ENVIRONMENT,development}

# ML Configuration
ml:
  problem_type: "classification"
  target_column: "target"
  test_size: 0.2
  validation_size: 0.1
  cv_folds: 5
  random_seed: 42
  hyperparameter_search: false
  early_stopping: true

# Paths Configuration
paths:
  data_root: "data"
  model_root: "models"
  reports_root: "reports"
  logs_root: "logs"

# Logging Configuration
logging:
  level: ${oc.env:LOG_LEVEL,INFO}
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: null  # Set to file path to log to file

# Feature Engineering
features:
  scaling_method: "standard"  # standard, minmax, robust
  categorical_encoding: "one_hot"  # one_hot, target, ordinal
  feature_selection: "mutual_info"  # mutual_info, chi2, f_classif
  handle_missing: "median"  # median, mean, mode, drop
```

### **Model Configuration (model/random_forest.yaml)**

```yaml
# conf/model/random_forest.yaml
model_type: "random_forest"
algorithm: "sklearn"

# Model parameters
parameters:
  n_estimators: 100
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  max_features: "sqrt"
  bootstrap: true
  random_state: 42
  n_jobs: -1

# Hyperparameter search ranges
hyperparameter_ranges:
  n_estimators:
    type: "int"
    low: 50
    high: 300
    step: 50
  max_depth:
    type: "int"
    low: 3
    high: 20
    step: 1
  min_samples_split:
    type: "int"
    low: 2
    high: 20
    step: 1
  min_samples_leaf:
    type: "int"
    low: 1
    high: 10
    step: 1

# Training configuration
training:
  early_stopping: false
  validation_metric: "accuracy"
  save_best_only: true
```

### **API Configuration (api/production.yaml)**

```yaml
# conf/api/production.yaml
api:
  host: "0.0.0.0"
  port: ${oc.env:API_PORT,8000}
  workers: ${oc.env:API_WORKERS,4}
  timeout: 30
  max_request_size: 10485760  # 10MB

security:
  enable_cors: true
  cors_origins:
    - "https://your-frontend.com"
    - "https://admin.your-company.com"
  api_key_header: "X-API-Key"
  rate_limit_enabled: true
  rate_limits:
    predict: "100/minute"
    batch_predict: "10/minute"
    model_management: "20/minute"

models:
  auto_load: true
  model_directory: ${oc.env:MODEL_DIRECTORY,"models/trained"}
  max_models: 5
  model_timeout: 300
  lazy_loading: true

monitoring:
  enable_metrics: true
  metrics_endpoint: "/api/v1/metrics"
  health_endpoint: "/api/v1/health"
  log_requests: true
  prometheus_port: 9090

caching:
  enabled: true
  backend: "redis"
  redis_url: ${oc.env:REDIS_URL,"redis://localhost:6379"}
  default_ttl: 300
```

---

## 🔧 **Configuration Schema**

### **Pydantic Configuration Models**

```python
# src/config/models.py - Configuration validation
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum

class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTICLASS = "multiclass"

class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LOGISTIC_REGRESSION = "logistic_regression"
    NEURAL_NETWORK = "neural_network"

class MLConfig(BaseModel):
    """ML training configuration."""
    problem_type: ProblemType
    target_column: str
    test_size: float = Field(ge=0.1, le=0.5)
    validation_size: float = Field(ge=0.1, le=0.3)
    cv_folds: int = Field(ge=2, le=10)
    random_seed: int = Field(default=42)
    hyperparameter_search: bool = Field(default=False)
    early_stopping: bool = Field(default=True)

class ModelConfig(BaseModel):
    """Model configuration."""
    model_type: ModelType
    algorithm: str
    parameters: Dict[str, Any]
    hyperparameter_ranges: Optional[Dict[str, Any]] = None

class APIConfig(BaseModel):
    """API server configuration."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(ge=1024, le=65535)
    workers: int = Field(ge=1, le=32)
    timeout: int = Field(ge=10, le=300)
    max_request_size: int = Field(ge=1024)

class SecurityConfig(BaseModel):
    """Security configuration."""
    enable_cors: bool = Field(default=True)
    cors_origins: List[str] = Field(default_factory=list)
    api_key_header: str = Field(default="X-API-Key")
    rate_limit_enabled: bool = Field(default=True)
    rate_limits: Dict[str, str] = Field(default_factory=dict)

class Config(BaseModel):
    """Main configuration model."""
    project_name: str
    version: str
    environment: str
    ml: MLConfig
    model: ModelConfig
    api: APIConfig
    security: SecurityConfig

    @validator('environment')
    def validate_environment(cls, v):
        allowed = ['development', 'staging', 'production', 'test']
        if v not in allowed:
            raise ValueError(f'Environment must be one of {allowed}')
        return v
```

### **Configuration Loading**

```python
# src/config/manager.py - Configuration manager
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os
from typing import Optional, Dict, Any

from .models import Config

class ConfigManager:
    """Configuration manager with Hydra integration."""

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = config_dir or str(Path(__file__).parent.parent.parent / "conf")
        self._config = None

    def load_config(
        self,
        config_name: str = "config",
        overrides: Optional[List[str]] = None,
        validate: bool = True
    ) -> Config:
        """Load configuration with optional overrides."""

        # Initialize Hydra
        with initialize_config_dir(config_dir=self.config_dir, version_base="1.1"):
            # Compose configuration
            cfg = compose(config_name=config_name, overrides=overrides or [])

            # Convert to dictionary
            config_dict = OmegaConf.to_container(cfg, resolve=True)

            if validate:
                # Validate using Pydantic
                validated_config = Config(**config_dict)
                return validated_config
            else:
                return config_dict

    def get_config(self) -> Config:
        """Get cached configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def reload_config(self, **kwargs) -> Config:
        """Reload configuration with new parameters."""
        self._config = self.load_config(**kwargs)
        return self._config

# Global configuration manager instance
config_manager = ConfigManager()

def load_config(**kwargs) -> Config:
    """Load configuration using global manager."""
    return config_manager.load_config(**kwargs)
```

---

## 🔒 **Environment & Security**

### **Environment Variables**

```bash
# .env - Environment configuration
# Core settings
ENVIRONMENT=development
LOG_LEVEL=INFO
PYTHONPATH=/path/to/project

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/mlops
REDIS_URL=redis://localhost:6379

# Model Configuration
MODEL_DIRECTORY=models/trained
MODEL_TYPE=random_forest

# Security
API_KEY=your-secret-api-key
JWT_SECRET=your-jwt-secret

# Cloud Configuration (if using cloud services)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-west-2

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### **Secrets Management**

```yaml
# conf/secrets.yaml.example - Template for secrets
# DO NOT COMMIT ACTUAL SECRETS

database:
  url: "${oc.env:DATABASE_URL}"
  username: "${oc.env:DB_USERNAME}"
  password: "${oc.env:DB_PASSWORD}"

api:
  secret_key: "${oc.env:API_SECRET_KEY}"
  jwt_secret: "${oc.env:JWT_SECRET}"

cloud:
  aws:
    access_key_id: "${oc.env:AWS_ACCESS_KEY_ID}"
    secret_access_key: "${oc.env:AWS_SECRET_ACCESS_KEY}"

  gcp:
    service_account_path: "${oc.env:GOOGLE_APPLICATION_CREDENTIALS}"

monitoring:
  webhook_url: "${oc.env:SLACK_WEBHOOK_URL}"
  alert_email: "${oc.env:ALERT_EMAIL}"
```

---

## 🧪 **Configuration for Experiments**

### **Experiment Configuration**

```yaml
# conf/experiments/experiment_001.yaml
defaults:
  - base_config
  - model: xgboost
  - _self_

# Experiment metadata
experiment:
  name: "xgboost_hyperparameter_tuning"
  description: "Optimize XGBoost hyperparameters for classification"
  author: "data-scientist@company.com"
  tags: ["xgboost", "hyperparameter_tuning", "classification"]

# Override ML configuration for this experiment
ml:
  hyperparameter_search: true
  cv_folds: 10
  test_size: 0.25

# Model-specific overrides
model:
  hyperparameter_ranges:
    n_estimators:
      low: 100
      high: 500
      step: 100
    max_depth:
      low: 6
      high: 15
      step: 1
    learning_rate:
      low: 0.01
      high: 0.3
      step: 0.01

# Experiment tracking
tracking:
  experiment_name: "xgboost_optimization"
  run_name: "${experiment.name}_${now:%Y%m%d_%H%M%S}"
  log_model: true
  log_artifacts: true
```

### **Using Experiment Configurations**

```python
# Run experiment with specific configuration
from src.config import load_config
from workflows.model_training import train_and_evaluate_model

# Load experiment configuration
config = load_config(
    config_name="experiments/experiment_001",
    overrides=["ml.cv_folds=5"]  # Additional overrides
)

# Run training with experiment config
result = train_and_evaluate_model(
    data_path="data/processed/features.parquet",
    config=config
)
```

---

## 🔄 **Dynamic Configuration**

### **Runtime Configuration Updates**

```python
# Example: Dynamic model configuration
from src.config import ConfigManager

config_manager = ConfigManager()

# Load base configuration
config = config_manager.load_config()

# Update configuration based on data characteristics
if dataset_size > 100000:
    # Use more powerful model for large datasets
    config = config_manager.load_config(overrides=[
        "model=xgboost",
        "model.parameters.n_estimators=500",
        "ml.cv_folds=3"  # Reduce CV folds for speed
    ])

# Use updated configuration
model = train_model(data, config)
```

### **A/B Testing Configuration**

```python
# Example: A/B testing with different configurations
import random
from src.config import load_config

def get_experiment_config(user_id: str):
    """Get configuration based on A/B testing group."""

    # Simple hash-based assignment
    group = hash(user_id) % 2

    if group == 0:
        # Control group - Random Forest
        return load_config(overrides=[
            "model=random_forest",
            "experiment.group=control"
        ])
    else:
        # Treatment group - XGBoost
        return load_config(overrides=[
            "model=xgboost",
            "experiment.group=treatment"
        ])

# Usage
config = get_experiment_config("user_123")
```

---

## 🛠️ **Development Tools**

### **Configuration Validation**

```python
# Validate configuration files
from src.config import load_config
from pathlib import Path

def validate_all_configs():
    """Validate all configuration files."""
    config_dir = Path("conf")

    # Test main configurations
    configs_to_test = [
        "config",
        "api/development",
        "api/production",
        "model/random_forest",
        "model/xgboost"
    ]

    for config_name in configs_to_test:
        try:
            config = load_config(config_name=config_name)
            print(f"✅ {config_name}: Valid")
        except Exception as e:
            print(f"❌ {config_name}: {str(e)}")

if __name__ == "__main__":
    validate_all_configs()
```

### **Configuration Documentation Generator**

```python
# Generate configuration documentation
from src.config.models import Config
import json

def generate_config_docs():
    """Generate configuration schema documentation."""

    # Get Pydantic schema
    schema = Config.schema()

    # Save as JSON schema
    with open("docs/config_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # Generate markdown documentation
    with open("docs/configuration_guide.md", "w") as f:
        f.write("# Configuration Schema\n\n")
        f.write("## Properties\n\n")

        for prop, details in schema["properties"].items():
            f.write(f"### {prop}\n")
            f.write(f"- **Type**: {details.get('type', 'unknown')}\n")
            f.write(f"- **Description**: {details.get('description', 'No description')}\n")
            if 'default' in details:
                f.write(f"- **Default**: {details['default']}\n")
            f.write("\n")

if __name__ == "__main__":
    generate_config_docs()
```

---

## 📚 **Best Practices**

### **Configuration Organization**

1. **🏗️ Use composition**: Break configurations into reusable components
2. **🔒 Never commit secrets**: Use environment variables for sensitive data
3. **✅ Validate everything**: Use Pydantic models for type safety
4. **📝 Document thoroughly**: Include descriptions for all configuration options
5. **🧪 Test configurations**: Validate all config files in CI/CD
6. **🔄 Use defaults wisely**: Provide sensible defaults for optional settings

### **Environment Management**

1. **📁 Separate environments**: Different configs for dev/staging/prod
2. **🔧 Override patterns**: Use environment variables for environment-specific values
3. **🔒 Secure secrets**: Never hardcode secrets in configuration files
4. **📊 Monitor changes**: Track configuration changes in production
5. **🚀 Graceful degradation**: Handle missing configuration gracefully

---

## 🤝 **Contributing**

1. **Add new configuration schemas** to `src/config/models.py`
2. **Create environment-specific configs** in appropriate subdirectories
3. **Validate all changes** with configuration tests
4. **Update documentation** when adding new configuration options
5. **Test with different environments** before submitting PRs

---

## 📚 **Additional Resources**

- **[Hydra Documentation](https://hydra.cc/)** - Configuration management framework
- **[Pydantic Documentation](https://pydantic-docs.helpmanual.io/)** - Data validation
- **[Environment Variables Guide](../docs/environment_setup.md)** - Environment configuration
- **[Security Best Practices](../docs/security.md)** - Secure configuration management
