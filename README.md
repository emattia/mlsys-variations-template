# MLOps Template: Production-Ready ML Platform

A comprehensive MLOps repository template that provides a production-ready foundation for machine learning projects. This template supports both **platform maintainers** who need to set up ML infrastructure and **data scientists/ML engineers** who need to build, train, and deploy ML models.

## 🎯 **For Platform Maintainers**

If you're setting up ML infrastructure for your organization, this template provides:

### **Production-Ready Components**
- **🚀 FastAPI Model Serving**: REST API with automatic validation, error handling, and OpenAPI docs
- **🐳 Docker Deployment**: Multi-stage builds, security hardening, and orchestration with Docker Compose
- **⚙️ CI/CD Pipelines**: Automated testing, security scanning, and multi-environment deployments
- **📊 Monitoring Stack**: Prometheus metrics and Grafana dashboards (optional)
- **🔧 Configuration Management**: Pydantic-based config with Hydra integration
- **🧩 Plugin Architecture**: Extensible system for custom ML components

### **Quick Platform Setup**

1. **Fork this repository** for your organization
2. **Configure CI/CD secrets** in GitHub Actions:
   ```bash
   # Required secrets for deployment
   DOCKER_REGISTRY_URL
   DOCKER_REGISTRY_USERNAME
   DOCKER_REGISTRY_PASSWORD
   STAGING_DEPLOY_KEY
   PRODUCTION_DEPLOY_KEY
   ```
3. **Customize configuration** in `conf/` directory
4. **Deploy the platform**:
   ```bash
   # Production deployment
   make docker-prod

   # Development environment
   make docker-dev

   # Full monitoring stack
   make monitoring-up
   ```

### **Platform Features**

| Feature | Description | Status |
|---------|-------------|--------|
| **API Gateway** | FastAPI with auto-validation | ✅ Ready |
| **Model Registry** | Automatic model discovery & loading | ✅ Ready |
| **Containerization** | Multi-stage Docker builds | ✅ Ready |
| **CI/CD** | GitHub Actions with security scanning | ✅ Ready |
| **Monitoring** | Prometheus + Grafana stack | ✅ Ready |
| **Security** | Trivy scanning, non-root containers | ✅ Ready |
| **Documentation** | Auto-generated API docs | ✅ Ready |

---

## 🏗️ System Architecture Overview

This repository is structured as a cohesive MLOps platform, where each directory plays a specific role in the lifecycle of a machine learning project. The system is built around a centralized configuration system that drives behavior across  components, from data processing to model deployment.

- The `conf/` directory defines the *what*—the declarative configuration for different environments, models, and experiments using YAML files.
- The `src/` directory contains the *how*—the Python source code that implements the logic, including the `src/config/` module, which reads, validates, and makes the configuration from `conf/` available to the rest of the application.

This separation of configuration from code is a core principle of the template, enabling reproducible experiments and seamless deployments across environments.

## Directory Structure

Here is a breakdown of the key directories and how they function as part of the broader system:

- **`conf/`**: **Configuration Hub**. Contains all YAML configuration files for the project. It uses a hierarchical structure (e.g., `conf/model`, `conf/api`) managed by Hydra. This is where you define parameters for experiments, data processing, and deployment environments. The `README.md` within this directory provides an in-depth guide.

- **`src/`**: **Core Application Logic**. The main Python source code for the platform.
    - **`src/api/`**: Implements the FastAPI production server for model serving.
    - **`src/config/`**: **Configuration Implementation**. This is the Python-based implementation of the configuration system. It uses Pydantic to validate the YAML files from `conf/` and provides a type-safe `Config` object to the rest of the application.
    - **`src/data/`**: Scripts for data ingestion, processing, and validation. These scripts are driven by settings in `conf/data/`.
    - **`src/models/`**: Code for training, evaluating, and running inference with ML models. This module consumes model parameters from `conf/model/`.
    - **`src/plugins/`**: An extensible plugin system for adding custom components like new model types or data processors.
    - **`src/utils/`**: Shared utilities used across the application.

- **`data/`**: **Dataset Storage**. The default location for storing datasets. It's typically divided into subdirectories like `raw/`, `processed/`, and `interim/`. `conf/paths.py` often defines the paths used to access this data.

- **`models/`**: **Trained Model Artifacts**. This directory stores the serialized outputs of model training (e.g., `.pkl` or `.onnx` files), along with versioning and metadata. It is the destination for trained models managed by the logic in `src/models/`.

- **`workflows/`**: **Orchestration Scripts**. Contains high-level Python scripts that orchestrate end-to-end ML workflows, such as `model_training.py` or `batch_inference.py`. These scripts load the configuration and tie together components from `src/` to execute a complete process.

- **`notebooks/`**: **Research and Exploration**. For Jupyter notebooks used in data analysis, experimentation, and visualization. A safe space for iterative, non-production work.

- **`tests/`**: **Quality Assurance**. Contains the entire test suite for the project, including unit, integration, and functional tests, ensuring code reliability.

- **`reports/`**: **Generated Outputs**. Stores outputs from model evaluation and analysis, such as performance metrics (`.json`), plots (`.png`), and HTML reports.

- **`docs/`**: **Project Documentation**. Contains user guides, architecture diagrams, and other generated documentation.

- **`.github/`**: **CI/CD Pipelines**. Defines the GitHub Actions workflows for continuous integration, testing, and deployment.

---

## 🚀 **Getting Started**

### **Prerequisites**

- **Python 3.10+**
- **[uv](https://github.com/astral-sh/uv)** (Python package manager)
- **Docker** (for containerization)
- **Make** (for workflow automation)

### **Installation**

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd mlops-template

# 2. Install dependencies and setup environment
make install

# 3. Setup development tools
make setup-dev

# 4. Verify installation
make all-checks
```

### **Environment Configuration**

```bash
# Copy and customize environment variables
cp .env-example .env
# Edit .env with your settings
```

---

## 📊 **Common Workflows**

### **For Data Scientists**

#### **1. Data Exploration & Analysis**
```bash
# Start Jupyter environment
make jupyter

# Process raw data
python -m workflows.data_ingestion data/raw/dataset.csv

# Run feature engineering
python -m workflows.feature_engineering
```

#### **2. Model Training & Experimentation**
```bash
# Train with default configuration
python -m workflows.model_training data/processed/features.parquet

# Train with custom parameters
python -m workflows.model_training \
    --model-type random_forest \
    --problem-type classification \
    --target-column target

# Hyperparameter tuning
python -m workflows.model_training --hyperparameter-search
```

#### **3. Model Evaluation**
```bash
# Evaluate trained model
python -m workflows.model_evaluation \
    --model-path models/trained/model.pkl \
    --data-path data/processed/test.parquet

# Generate comprehensive reports
python -m workflows.model_evaluation --generate-plots
```

#### **4. Batch Inference**
```bash
# Run batch predictions
python -m workflows.batch_inference \
    --model-path models/trained/model.pkl \
    --input-path data/raw/new_data.csv \
    --output-path predictions/batch_results.csv
```

### **For Platform Operations**

#### **1. API Deployment**
```bash
# Development server
make serve-dev

# Production deployment with Docker
make docker-build
make docker-prod

# Health check
curl http://localhost:8000/api/v1/health
```

#### **2. Monitoring & Observability**
```bash
# Start monitoring stack
make monitoring-up

# View metrics: http://localhost:3000 (Grafana)
# Prometheus: http://localhost:9090
```

#### **3. CI/CD Operations**
```bash
# Run full quality checks
make all-checks

# Security scanning
make security-scan

# Build multi-platform containers
make docker-build-multi
```

---

## 🛠️ **Development Tools**

### **Available Make Commands**

| Command | Description |
|---------|-------------|
| `make install` | Setup environment and install dependencies |
| `make all-checks` | Run linting, formatting, and tests |
| `make test` | Run all tests |
| `make serve-dev` | Start development API server |
| `make serve-prod` | Start production API server |
| `make docker-build` | Build Docker container |
| `make docker-dev` | Run development environment |
| `make jupyter` | Start Jupyter Lab |
| `make docs` | Generate documentation |
| `make clean` | Clean temporary files |

### **API Development**

```bash
# Start development server with hot reload
make serve-dev

# API Documentation: http://localhost:8000/docs
# Health Check: http://localhost:8000/api/v1/health
# Model Management: http://localhost:8000/api/v1/models
```

### **Testing**

```bash
# Run all tests
make test

# Run specific test categories
make unit-test          # Unit tests only
make integration-test   # Integration tests (requires API server)
make workflow-test      # Workflow tests

# Test with coverage
pytest --cov=src --cov-report=html
```

---

## 🔧 **Configuration**

The template uses a hierarchical configuration system:

### **Configuration Files**
- `conf/config.yaml` - Main configuration
- `conf/model/` - Model-specific configs
- `conf/data/` - Data processing configs
- `.env` - Environment variables

### **Example Configuration**
```yaml
# conf/config.yaml
model:
  model_type: "random_forest"
  problem_type: "classification"
  target_column: "target"

ml:
  test_size: 0.2
  random_seed: 42
  cv_folds: 5

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
```

---

## 🔒 **Security & Production**

### **Security Features**
- **Container Security**: Non-root user, minimal base image
- **Dependency Scanning**: Automated vulnerability checks
- **Secret Management**: Environment-based configuration
- **API Security**: Input validation, CORS protection

### **Production Readiness**
- **Health Checks**: Comprehensive monitoring endpoints
- **Graceful Shutdown**: Proper signal handling
- **Resource Limits**: Configurable memory/CPU limits
- **Multi-stage Builds**: Optimized container sizes
- **Monitoring**: Prometheus metrics integration

---

## 📈 **Monitoring & Observability**

### **Built-in Metrics**
- API response times and error rates
- Model prediction latency
- Resource utilization
- Custom business metrics

### **Dashboards**
```bash
# Start monitoring stack
make monitoring-up

# Access dashboards
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
```

---

## 🤝 **Contributing**

### **For Platform Maintainers**
1. Fork the repository for your organization
2. Customize configuration in `conf/`
3. Update CI/CD secrets and variables
4. Deploy and configure monitoring

### **For ML Engineers**
1. Create feature branches from `main`
2. Add your experiments in `notebooks/`
3. Implement models in `src/models/`
4. Add workflows in `workflows/`
5. Ensure tests pass: `make all-checks`
6. Submit pull requests

### **Code Quality Standards**
- **Linting**: Ruff for code formatting and linting
- **Type Checking**: MyPy for static type analysis
- **Testing**: Pytest with >80% coverage requirement
- **Documentation**: Comprehensive docstrings and README updates

---

## 📚 **Documentation**

- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs
- **[Configuration Guide](conf/README.md)** - Configuration management
- **[Development Guide](src/README.md)** - Code structure and standards
- **[Workflow Guide](workflows/README.md)** - ML workflow documentation
- **[Deployment Guide](docs/deployment.md)** - Production deployment

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **FastAPI** for modern Python API framework
- **Pydantic** for data validation and settings management
- **Docker** for containerization
- **GitHub Actions** for CI/CD automation
- **Cookiecutter Data Science** for project structure inspiration
