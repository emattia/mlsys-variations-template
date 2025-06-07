# MLOps Core Source Code

This directory contains the production-ready ML core components that power the MLOps platform. It provides reusable, tested, and documented modules for building robust machine learning systems.

## 🎯 **Purpose**

The `src/` directory is designed for **ML engineers and data scientists** who need to:

- **Build scalable ML pipelines** with reusable components
- **Deploy models to production** via FastAPI endpoints
- **Implement custom ML algorithms** using the plugin system
- **Process data reliably** with validation and error handling
- **Manage configurations** across different environments

## 🏗️ **Architecture Overview**

```
src/
├── 🚀 Production API
│   └── api/                    # FastAPI model serving
│       ├── app.py              # Application factory & middleware
│       ├── routes.py           # REST API endpoints
│       ├── service.py          # Model management service
│       └── models.py           # Pydantic validation models
│
├── 🤖 ML Core Components
│   ├── data/                   # Data processing & validation
│   │   ├── loading.py          # Data ingestion utilities
│   │   ├── processing.py       # Feature engineering
│   │   └── validation.py       # Data quality checks
│   │
│   ├── models/                 # Model lifecycle management
│   │   ├── training.py         # Training utilities
│   │   ├── evaluation.py       # Model evaluation metrics
│   │   └── inference.py        # Prediction interface
│   │
│   └── plugins/                # Extensible ML components
│       ├── base.py             # Abstract base classes
│       └── registry.py         # Plugin registration system
│
├── ⚙️ Infrastructure
│   ├── config/                 # Configuration management
│   │   ├── manager.py          # Config loading & validation
│   │   └── models.py           # Pydantic config schemas
│   │
│   └── utils/                  # Common utilities
│       └── common.py           # Shared helper functions
│
└── 📊 Analysis Utilities
    └── tabular_data_utils.py   # Tabular data processing
```

---

## 🚀 **API Development**

### **FastAPI Model Serving**

The API module provides production-ready model serving capabilities:

```python
# Example: Adding a custom endpoint
from fastapi import APIRouter, Depends
from src.api.service import ModelService
from src.api.models import PredictionRequest, PredictionResponse

router = APIRouter()

@router.post("/api/v1/custom-predict")
async def custom_predict(
    request: PredictionRequest,
    service: ModelService = Depends(get_model_service)
):
    # Custom prediction logic
    result = service.predict(request.features, request.model_name)
    return PredictionResponse(prediction=result)
```

### **Model Service Integration**

```python
# Example: Loading a custom model
from src.api.service import ModelService

service = ModelService(config_manager)

# Load model from file
success = service.load_model("my_model", "models/trained/my_model.pkl")

# Make predictions
predictions, probabilities, timing = service.predict(
    features=[1.0, 2.0, 3.0],
    model_name="my_model",
    return_probabilities=True
)
```

---

## 🤖 **ML Development**

### **Data Processing Pipeline**

```python
# Example: Data processing workflow
from src.data.loading import load_data
from src.data.processing import engineer_features
from src.data.validation import validate_data_quality

# Load and validate data
df = load_data("data/raw/dataset.csv")
validation_report = validate_data_quality(df)

# Engineer features
features_df = engineer_features(df, config={
    "target_column": "target",
    "categorical_encoding": "one_hot",
    "scaling_method": "standard"
})
```

### **Model Training & Evaluation**

```python
# Example: Training a model
from src.models.training import train_model, hyperparameter_tuning
from src.models.evaluation import evaluate_classification_model

# Train with hyperparameter optimization
model, tuning_results = hyperparameter_tuning(
    X_train, y_train,
    model_type="random_forest",
    param_grid={"n_estimators": [100, 200], "max_depth": [10, 20]}
)

# Comprehensive evaluation
metrics = evaluate_classification_model(
    y_true=y_test,
    y_pred=model.predict(X_test),
    y_proba=model.predict_proba(X_test)
)
```

### **Plugin System**

Create custom ML components using the plugin architecture:

```python
# Example: Custom model trainer plugin
from src.plugins.base import ModelTrainer, ComponentResult, ComponentStatus
from src.plugins import register_plugin

@register_plugin(
    name="custom_trainer",
    category="model_training",
    description="Custom neural network trainer",
    dependencies=["torch", "sklearn"],
    version="1.0.0"
)
class CustomModelTrainer(ModelTrainer):
    def execute(self, context: ExecutionContext) -> ComponentResult:
        # Custom training logic
        model = self._train_custom_model(context.input_data)

        return ComponentResult(
            status=ComponentStatus.SUCCESS,
            component_name=self.name,
            execution_time=time.time() - start_time,
            output_data={"model_path": str(model_path)},
            metrics={"accuracy": 0.95}
        )
```

---

## ⚙️ **Configuration Management**

### **Using Configurations**

```python
# Example: Loading and using configurations
from src.config import ConfigManager, load_config

# Load default configuration
config = load_config()

# Access nested configuration
model_config = config.model
api_config = config.api

# Override configurations programmatically
config_overrides = {
    "model": {"model_type": "xgboost"},
    "ml": {"test_size": 0.3}
}
config = load_config(overrides=config_overrides)
```

### **Environment-Specific Configs**

```yaml
# conf/config.yaml
defaults:
  - model: random_forest
  - data: default
  - _self_

# Environment-specific overrides
environment: ${oc.env:ENVIRONMENT,development}

api:
  host: ${oc.env:API_HOST,0.0.0.0}
  port: ${oc.env:API_PORT,8000}
  workers: ${oc.env:API_WORKERS,4}
```

---

## 🧪 **Development Practices**

### **Code Standards**

All source code follows these standards:

1. **Type Annotations**: Full type hints for better IDE support and error detection
2. **Documentation**: Comprehensive docstrings using Google/NumPy style
3. **Error Handling**: Proper exception handling with logging
4. **Testing**: Unit tests with >80% coverage requirement
5. **Logging**: Structured logging for debugging and monitoring

### **Example Module Template**

```python
"""
Module: Custom ML Component

This module provides functionality for [specific purpose].
Used in production ML pipelines for [use case].
"""

import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)

class MLComponent:
    """
    Production ML component for [specific task].

    This class provides [functionality description] and is designed
    for use in production ML pipelines.

    Attributes:
        config: Component configuration
        model: Trained model instance

    Example:
        >>> component = MLComponent(config=config)
        >>> result = component.process(data)
        >>> print(f"Processed {len(result)} samples")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the ML component.

        Args:
            config: Configuration dictionary with required parameters

        Raises:
            ValueError: If required configuration is missing
        """
        self.config = config
        self._validate_config()
        logger.info(f"Initialized {self.__class__.__name__}")

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process input data through the ML component.

        Args:
            data: Input DataFrame to process

        Returns:
            Processed DataFrame with transformations applied

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If processing fails
        """
        try:
            logger.debug(f"Processing {len(data)} samples")

            # Validate input
            self._validate_input(data)

            # Process data
            result = self._apply_transformations(data)

            logger.info(f"Successfully processed {len(result)} samples")
            return result

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise RuntimeError(f"Failed to process data: {str(e)}") from e

    def _validate_config(self) -> None:
        """Validate component configuration."""
        required_keys = ["param1", "param2"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config parameter: {key}")

    def _validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data format and quality."""
        if data.empty:
            raise ValueError("Input data cannot be empty")

        # Add specific validation logic
```

### **Testing Your Components**

```python
# tests/test_ml_component.py
import pytest
import pandas as pd
from src.your_module import MLComponent

class TestMLComponent:
    """Test suite for MLComponent."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "param1": "value1",
            "param2": 42
        }

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6]
        })

    def test_initialization(self, sample_config):
        """Test component initialization."""
        component = MLComponent(sample_config)
        assert component.config == sample_config

    def test_process_valid_data(self, sample_config, sample_data):
        """Test processing with valid data."""
        component = MLComponent(sample_config)
        result = component.process(sample_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_process_invalid_data(self, sample_config):
        """Test processing with invalid data."""
        component = MLComponent(sample_config)

        with pytest.raises(ValueError):
            component.process(pd.DataFrame())
```

---

## 📦 **Package Import Guide**

### **Standard Import Patterns**

```python
# Data processing
from src.data.loading import load_data, save_data
from src.data.processing import engineer_features, scale_features
from src.data.validation import validate_data_quality

# Model operations
from src.models.training import train_model, hyperparameter_tuning
from src.models.evaluation import evaluate_classification_model
from src.models.inference import batch_predict

# Configuration
from src.config import ConfigManager, load_config

# API components
from src.api.service import ModelService
from src.api.models import PredictionRequest, PredictionResponse

# Utilities
from src.utils.common import setup_logging, get_project_root
```

### **Plugin System Usage**

```python
# Register and use plugins
from src.plugins import register_plugin, get_plugin

# Get registered plugin
trainer = get_plugin("sklearn_trainer")

# Use in workflows
result = trainer.execute(context)
```

---

## 🔧 **Integration Examples**

### **Notebook Integration**

```python
# notebooks/example_analysis.ipynb
import sys
sys.path.append('..')

from src.data.loading import load_data
from src.models.training import train_model
from src.config import load_config

# Load configuration and data
config = load_config()
data = load_data("../data/raw/dataset.csv")

# Train model
model = train_model(data, config)
```

### **Workflow Integration**

```python
# workflows/custom_pipeline.py
from src.data.processing import engineer_features
from src.models.training import train_model
from src.api.service import ModelService

def run_pipeline(data_path: str, config: dict):
    """Custom ML pipeline using src components."""

    # Process data
    features = engineer_features(data_path, config)

    # Train model
    model = train_model(features, config)

    # Deploy to API
    service = ModelService(config_manager)
    service.load_model("custom_model", model)
```

---

## 🚀 **Production Deployment**

### **API Server Integration**

The source code integrates seamlessly with the production API server:

```bash
# Start development server
make serve-dev

# API endpoints automatically available:
# http://localhost:8000/docs - Interactive API documentation
# http://localhost:8000/api/v1/health - Health check
# http://localhost:8000/api/v1/predict - Model predictions
```

### **Container Deployment**

Source code is packaged in optimized Docker containers:

```dockerfile
# Multi-stage build automatically includes src/
FROM python:3.11-slim as builder
COPY src/ /app/src/
# ... build process

FROM python:3.11-slim as production
COPY --from=builder /app/src /app/src
# ... production setup
```

---

## 📚 **Additional Resources**

- **[API Documentation](../README.md#api-development)** - FastAPI integration guide
- **[Configuration Guide](../conf/README.md)** - Configuration management
- **[Testing Guide](../tests/README.md)** - Testing best practices
- **[Workflow Guide](../workflows/README.md)** - ML workflow development
- **[Plugin Development](plugins/README.md)** - Custom plugin creation

---

## 🤝 **Contributing to Source Code**

1. **Follow coding standards**: Type hints, docstrings, error handling
2. **Write comprehensive tests**: Aim for >80% coverage
3. **Add proper logging**: Use structured logging for debugging
4. **Update documentation**: Keep README and docstrings current
5. **Run quality checks**: `make all-checks` before committing
