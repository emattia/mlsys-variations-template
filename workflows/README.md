# ML Engineering Workflows

This directory contains production-ready ML workflows for data scientists and ML engineers. These workflows provide automated, configurable, and scalable pipelines for the complete ML lifecycle.

## 🎯 **Purpose**

The workflows directory enables **data scientists and ML engineers** to:

- **🔄 Automate ML pipelines** from data ingestion to model deployment
- **📊 Run reproducible experiments** with configuration management
- **🚀 Deploy models to production** via automated workflows
- **📈 Track experiment results** and model performance
- **🔧 Integrate with CI/CD** for continuous model deployment
- **🧪 Validate model quality** before production releases

## 🏗️ **Workflow Architecture**

```
workflows/
├── 📊 Data Engineering
│   ├── data_ingestion.py          # Raw data loading & validation
│   ├── feature_engineering.py     # Feature creation & transformation
│   └── data_validation.py         # Data quality assurance
│
├── 🤖 Model Engineering
│   ├── model_training.py          # Training with plugin system
│   ├── model_evaluation.py        # Comprehensive evaluation
│   ├── hyperparameter_tuning.py   # Automated optimization
│   └── model_validation.py        # Model quality gates
│
├── 🚀 Deployment Engineering
│   ├── batch_inference.py         # Batch prediction workflows
│   ├── model_deployment.py        # API deployment automation
│   └── model_monitoring.py        # Production monitoring
│
├── 🧪 Quality Assurance
│   └── tests/                     # Workflow testing
│       ├── test_workflow.py       # Integration tests
│       └── fixtures/              # Test data & configurations
│
└── 📋 Configuration
    ├── configs/                   # Workflow-specific configs
    └── templates/                 # Pipeline templates
```

---

## 🚀 **Quick Start Guide**

### **1. Data Science Exploration**

```bash
# Start with data exploration
python -m workflows.data_ingestion \
    --input-path data/raw/dataset.csv \
    --output-path data/interim/validated.parquet

# Feature engineering
python -m workflows.feature_engineering \
    --config-path conf/feature_config.yaml \
    --target-column target
```

### **2. Model Training & Experimentation**

```bash
# Basic model training
python -m workflows.model_training data/processed/features.parquet

# Advanced training with hyperparameter search
python -m workflows.model_training \
    --data-path data/processed/features.parquet \
    --model-type random_forest \
    --hyperparameter-search \
    --cv-folds 5 \
    --run-id experiment_001
```

### **3. Model Evaluation & Validation**

```bash
# Comprehensive model evaluation
python -m workflows.model_evaluation \
    --model-path models/trained/rf_model.pkl \
    --test-data data/processed/test.parquet \
    --generate-plots

# Model validation for production
python -m workflows.model_validation \
    --model-path models/trained/rf_model.pkl \
    --validation-data data/validation/holdout.parquet \
    --performance-threshold 0.85
```

### **4. Production Deployment**

```bash
# Deploy model to API
python -m workflows.model_deployment \
    --model-path models/trained/rf_model.pkl \
    --deployment-target api \
    --environment production

# Run batch inference
python -m workflows.batch_inference \
    --model-path models/trained/rf_model.pkl \
    --input-data data/new/batch_data.csv \
    --output-path predictions/batch_results.csv
```

---

## 📊 **Data Engineering Workflows**

### **Data Ingestion Pipeline**

Automated data loading with validation and quality checks:

```python
# workflows/data_ingestion.py - Example usage
from workflows.data_ingestion import ingest_data

result = ingest_data(
    source_path="data/raw/sales_data.csv",
    target_path="data/interim/sales_validated.parquet",
    validation_rules={
        "required_columns": ["date", "amount", "customer_id"],
        "date_format": "%Y-%m-%d",
        "null_threshold": 0.05
    }
)

print(f"Ingested {result['rows_processed']} rows with {result['quality_score']:.2f} quality score")
```

**Key Features:**
- **📊 Multi-format support**: CSV, Parquet, JSON, databases
- **✅ Automatic validation**: Schema checking, data quality metrics
- **🔄 Incremental loading**: Support for streaming/batch updates
- **📝 Audit logging**: Complete data lineage tracking

### **Feature Engineering Pipeline**

Scalable feature creation with configuration-driven transformations:

```python
# workflows/feature_engineering.py - Example usage
from workflows.feature_engineering import engineer_features

features = engineer_features(
    input_data="data/interim/sales_validated.parquet",
    config={
        "target_column": "revenue",
        "categorical_features": ["category", "region"],
        "numerical_features": ["price", "quantity"],
        "time_features": ["date"],
        "transformations": {
            "scaling": "standard",
            "encoding": "target_encoding",
            "feature_selection": "mutual_info"
        }
    }
)
```

**Key Features:**
- **🎛️ Configurable transformations**: Scaling, encoding, selection
- **⏰ Time-series features**: Lags, rolling windows, seasonality
- **🎯 Target encoding**: Advanced categorical encoding
- **📈 Feature selection**: Automated relevance scoring

---

## 🤖 **Model Engineering Workflows**

### **Training Pipeline with Plugin System**

The training workflow uses a plugin-based architecture for extensibility:

```python
# workflows/model_training.py - Advanced example
from workflows.model_training import train_and_evaluate_model

result = train_and_evaluate_model(
    data_path="data/processed/features.parquet",
    config_overrides={
        "model": {
            "model_type": "random_forest",
            "problem_type": "classification",
            "target_column": "churn"
        },
        "ml": {
            "test_size": 0.2,
            "cv_folds": 5,
            "hyperparameter_search": True,
            "random_seed": 42
        }
    },
    run_id="churn_model_v1"
)

# Access results
print(f"CV Score: {result.metrics['cv_score_mean']:.3f} ± {result.metrics['cv_score_std']:.3f}")
print(f"Model saved to: {result.artifacts['model']}")
```

**Supported Algorithms:**
- **🌳 Tree-based**: Random Forest, XGBoost, LightGBM
- **📏 Linear**: Logistic Regression, Linear Regression, Ridge/Lasso
- **🧠 Neural Networks**: MLPClassifier, custom PyTorch models
- **🔌 Custom Plugins**: Extensible plugin system

### **Hyperparameter Optimization**

Automated hyperparameter tuning with smart search strategies:

```python
# Example: Bayesian optimization for XGBoost
python -m workflows.model_training \
    --model-type xgboost \
    --hyperparameter-search \
    --search-strategy bayesian \
    --search-iterations 50 \
    --cv-folds 5
```

**Search Strategies:**
- **🎯 Grid Search**: Exhaustive parameter exploration
- **🎲 Random Search**: Efficient random sampling
- **🧠 Bayesian Optimization**: Smart parameter search
- **🏃 Early Stopping**: Automatic convergence detection

### **Model Evaluation & Validation**

Comprehensive model assessment with production-ready metrics:

```python
# workflows/model_evaluation.py - Example usage
from workflows.model_evaluation import evaluate_model

evaluation_results = evaluate_model(
    model_path="models/trained/churn_model.pkl",
    test_data="data/processed/test.parquet",
    config={
        "metrics": ["accuracy", "precision", "recall", "f1", "auc"],
        "generate_plots": True,
        "plot_types": ["confusion_matrix", "roc_curve", "feature_importance"],
        "output_dir": "reports/model_evaluation"
    }
)

# Results include comprehensive metrics
print(f"Model Performance: {evaluation_results['metrics']}")
print(f"Plots saved to: {evaluation_results['plots_dir']}")
```

**Evaluation Features:**
- **📊 Multiple Metrics**: Classification, regression, ranking metrics
- **📈 Visualization**: ROC curves, confusion matrices, SHAP plots
- **🎯 Business Metrics**: Custom business KPIs and thresholds
- **📋 Model Cards**: Automated model documentation

---

## 🚀 **Production Deployment Workflows**

### **API Deployment Pipeline**

Automated model deployment to production APIs:

```python
# workflows/model_deployment.py - Example usage
from workflows.model_deployment import deploy_model

deployment_result = deploy_model(
    model_path="models/trained/churn_model.pkl",
    deployment_config={
        "target": "fastapi",
        "environment": "production",
        "scaling": {
            "min_replicas": 2,
            "max_replicas": 10,
            "cpu_threshold": 70
        },
        "monitoring": {
            "enable_metrics": True,
            "alert_thresholds": {
                "latency_p95": 100,  # milliseconds
                "error_rate": 0.01   # 1%
            }
        }
    }
)

print(f"Model deployed to: {deployment_result['endpoint_url']}")
print(f"Health check: {deployment_result['health_endpoint']}")
```

### **Batch Inference Pipeline**

Scalable batch prediction processing:

```python
# workflows/batch_inference.py - Example usage
from workflows.batch_inference import run_batch_inference

batch_results = run_batch_inference(
    model_path="models/trained/churn_model.pkl",
    input_data="data/new_customers/batch_20241207.csv",
    output_path="predictions/churn_predictions_20241207.csv",
    config={
        "batch_size": 10000,
        "include_probabilities": True,
        "output_format": "csv",
        "parallel_workers": 4
    }
)

print(f"Processed {batch_results['num_predictions']} predictions")
print(f"Processing time: {batch_results['processing_time']:.2f} seconds")
```

---

## 🔧 **Configuration Management**

### **Workflow Configuration Files**

Workflows use hierarchical configuration for reproducibility:

```yaml
# workflows/configs/training_config.yaml
defaults:
  - model: random_forest
  - data: default
  - _self_

# Experiment tracking
experiment:
  name: "churn_prediction_v2"
  tags: ["churn", "production", "random_forest"]

# Model configuration
model:
  model_type: "random_forest"
  problem_type: "classification"
  target_column: "churn"
  feature_columns: null  # Auto-detect

# ML pipeline configuration
ml:
  test_size: 0.2
  validation_size: 0.1
  cv_folds: 5
  random_seed: 42
  hyperparameter_search: true
  early_stopping: true

# Training configuration
training:
  max_iter: 1000
  convergence_threshold: 1e-6
  checkpoint_frequency: 100

# Evaluation configuration
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "auc"]
  generate_reports: true
  plot_feature_importance: true
```

### **Environment-Specific Configurations**

```yaml
# workflows/configs/production.yaml
# Override for production environment
defaults:
  - base_config
  - _self_

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30

logging:
  level: "INFO"
  format: "json"

monitoring:
  enable_metrics: true
  metrics_port: 9090
```

---

## 🧪 **Testing & Validation**

### **Workflow Testing**

Comprehensive testing ensures workflow reliability:

```python
# tests/test_workflows.py - Example test
import pytest
from workflows.model_training import train_and_evaluate_model

class TestModelTraining:
    """Test suite for model training workflow."""

    @pytest.fixture
    def sample_config(self):
        return {
            "model": {"model_type": "random_forest"},
            "ml": {"test_size": 0.2, "cv_folds": 3}
        }

    def test_training_workflow_success(self, sample_data, sample_config):
        """Test successful training workflow execution."""
        result = train_and_evaluate_model(
            data_path=sample_data,
            config_overrides=sample_config
        )

        assert result.is_success()
        assert "cv_score_mean" in result.metrics
        assert result.metrics["cv_score_mean"] > 0.5

    def test_training_with_hyperparameter_search(self, sample_data):
        """Test training with hyperparameter optimization."""
        config = {
            "ml": {"hyperparameter_search": True, "cv_folds": 2}
        }

        result = train_and_evaluate_model(
            data_path=sample_data,
            config_overrides=config
        )

        assert result.is_success()
        assert "best_params" in result.metadata
```

### **Integration Testing**

```bash
# Run workflow integration tests
pytest workflows/tests/ -v

# Test specific workflow
pytest workflows/tests/test_training_workflow.py -v

# Test with real data
pytest workflows/tests/ --use-real-data -v
```

---

## 📊 **Monitoring & Observability**

### **Workflow Metrics**

Built-in metrics collection for production monitoring:

```python
# Example: Custom workflow metrics
from workflows.monitoring import WorkflowMetrics

metrics = WorkflowMetrics("model_training")

with metrics.timer("data_loading"):
    data = load_data(path)

with metrics.timer("model_training"):
    model = train_model(data)

metrics.log_metric("training_accuracy", accuracy)
metrics.log_metric("dataset_size", len(data))

# Metrics automatically exported to Prometheus
```

### **Workflow Logging**

Structured logging for debugging and audit trails:

```python
# Example: Workflow logging
import logging
from workflows.utils import setup_workflow_logging

logger = setup_workflow_logging("model_training")

logger.info("Starting model training workflow", extra={
    "run_id": run_id,
    "data_path": data_path,
    "model_type": config.model.model_type
})

logger.warning("Low data quality detected", extra={
    "quality_score": 0.7,
    "missing_percentage": 0.15
})
```

---

## 🚀 **Production Best Practices**

### **Workflow Orchestration**

Integration with workflow orchestration tools:

```python
# Example: Airflow DAG
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from workflows.model_training import train_and_evaluate_model

def training_task(**context):
    result = train_and_evaluate_model(
        data_path=context['params']['data_path'],
        config_overrides=context['params']['config']
    )
    return result.is_success()

dag = DAG('ml_training_pipeline', schedule_interval='@daily')

train_op = PythonOperator(
    task_id='train_model',
    python_callable=training_task,
    params={'data_path': 'data/daily/features.parquet'},
    dag=dag
)
```

### **Continuous Integration**

Automated workflow validation in CI/CD:

```yaml
# .github/workflows/ml-workflows.yml
name: ML Workflows

on: [push, pull_request]

jobs:
  test-workflows:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.11

      - name: Install dependencies
        run: make install

      - name: Test workflows
        run: |
          pytest workflows/tests/ -v
          make workflow-test

      - name: Validate model training
        run: |
          python -m workflows.model_training \
            --data-path tests/fixtures/sample_data.csv \
            --config-path tests/fixtures/test_config.yaml
```

---

## 📚 **Workflow Templates**

### **Custom Workflow Template**

Create new workflows using the template pattern:

```python
# workflows/custom_workflow.py - Template
"""
Custom ML Workflow Template

This template provides a starting point for creating new ML workflows.
Copy and modify this template for your specific use case.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import ConfigManager, load_config
from src.plugins.base import ComponentResult, ComponentStatus
from src.utils.common import setup_logging, create_run_id

logger = logging.getLogger(__name__)

def custom_workflow(
    input_path: str,
    output_path: str,
    config: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None
) -> ComponentResult:
    """
    Custom workflow template.

    Args:
        input_path: Path to input data
        output_path: Path for output artifacts
        config: Workflow configuration
        run_id: Unique run identifier

    Returns:
        ComponentResult with workflow execution results
    """
    start_time = time.time()

    # Setup
    if run_id is None:
        run_id = create_run_id("custom_workflow")

    if config is None:
        config = load_config()

    setup_logging(config)
    logger.info(f"Starting custom workflow with run ID: {run_id}")

    try:
        # Step 1: Data loading
        logger.info("Loading input data")
        # Your data loading logic here

        # Step 2: Processing
        logger.info("Processing data")
        # Your processing logic here

        # Step 3: Output generation
        logger.info("Generating outputs")
        # Your output generation logic here

        execution_time = time.time() - start_time

        return ComponentResult(
            status=ComponentStatus.SUCCESS,
            component_name="custom_workflow",
            execution_time=execution_time,
            output_data={"output_path": output_path},
            artifacts={"results": Path(output_path)},
            metrics={"processing_time": execution_time}
        )

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Workflow failed: {str(e)}")

        return ComponentResult(
            status=ComponentStatus.FAILED,
            component_name="custom_workflow",
            execution_time=execution_time,
            error_message=str(e)
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Custom ML Workflow")
    parser.add_argument("input_path", help="Path to input data")
    parser.add_argument("output_path", help="Path for output artifacts")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--run-id", help="Run identifier")

    args = parser.parse_args()

    config = None
    if args.config:
        config = load_config(config_path=args.config)

    result = custom_workflow(
        input_path=args.input_path,
        output_path=args.output_path,
        config=config,
        run_id=args.run_id
    )

    if result.is_success():
        print(f"✅ Workflow completed successfully in {result.execution_time:.2f}s")
    else:
        print(f"❌ Workflow failed: {result.error_message}")
        exit(1)
```

---

## 🤝 **Contributing New Workflows**

### **Workflow Development Guidelines**

1. **📋 Follow template structure**: Use the provided template as starting point
2. **⚙️ Add configuration support**: Make workflows configurable via YAML
3. **📝 Include comprehensive logging**: Add structured logging throughout
4. **🧪 Write tests**: Include unit and integration tests
5. **📚 Document usage**: Add docstrings and usage examples
6. **🔧 Add CLI interface**: Support command-line execution
7. **📊 Include metrics**: Add relevant performance metrics

### **Workflow Submission Process**

1. Create workflow in `workflows/` directory
2. Add configuration schema to `workflows/configs/`
3. Write comprehensive tests in `workflows/tests/`
4. Update this README with usage examples
5. Run quality checks: `make workflow-test`
6. Submit pull request with workflow documentation

---

## 📚 **Additional Resources**

- **[Configuration Guide](../conf/README.md)** - Workflow configuration management
- **[API Integration](../src/README.md#api-development)** - API deployment workflows
- **[Testing Guide](../tests/README.md)** - Workflow testing best practices
- **[Plugin Development](../src/plugins/README.md)** - Custom plugin creation
- **[Monitoring Guide](../docs/monitoring.md)** - Production monitoring setup
