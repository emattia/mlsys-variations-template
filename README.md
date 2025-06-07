# MLOps Template: Production-Ready ML Platform

A comprehensive MLOps repository template that provides a production-ready foundation for machine learning projects. This template supports both **platform maintainers** who need to set up ML infrastructure and **data scientists/ML engineers** who need to build, train, and deploy ML models.

[![CI/CD Pipeline](https://github.com/your-org/mlops-template/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/mlops-template/actions/workflows/ci.yml)
[![Code Quality](https://github.com/your-org/mlops-template/actions/workflows/cd.yml/badge.svg)](https://github.com/your-org/mlops-template/actions/workflows/cd.yml)
[![Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen)](https://github.com/your-org/mlops-template)

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

## 🔬 **For Data Scientists & ML Engineers**

If you're building ML models and experiments, this template provides:

### **Complete ML Workflow**
- **📊 Data Processing**: Polars-based ETL with validation pipelines
- **🤖 Model Training**: Plugin-based training with multiple algorithms
- **📈 Experiment Tracking**: Configuration-driven experimentation
- **🔍 Model Evaluation**: Comprehensive metrics and visualization
- **🚀 Model Deployment**: One-command API deployment
- **📝 Research Notebooks**: Jupyter environment with plotting utilities

### **Quick Start for ML Work**

1. **Clone and setup environment**:
   ```bash
   git clone <your-org-repo>
   cd mlops-template
   make install
   ```

2. **Start development environment**:
   ```bash
   # Launch Jupyter for exploration
   make jupyter

   # Start API server for testing
   make serve-dev
   ```

3. **Run your first experiment**:
   ```bash
   # Train a model
   python -m workflows.model_training data/raw/your_data.csv

   # Evaluate results
   python -m workflows.model_evaluation

   # Deploy to API
   curl -X POST "http://localhost:8000/api/v1/predict" \
        -H "Content-Type: application/json" \
        -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
   ```

---

## 🏗️ **Project Structure**

```
mlops-template/
├── 🐳 Deployment & Infrastructure
│   ├── Dockerfile                 # Multi-stage container builds
│   ├── docker-compose.yml         # Development & production orchestration
│   ├── .github/workflows/         # CI/CD pipelines
│   └── conf/                      # Configuration management
│
├── 🚀 Production API
│   └── src/api/                   # FastAPI application
│       ├── app.py                 # Application factory
│       ├── routes.py              # API endpoints
│       ├── service.py             # Model service layer
│       └── models.py              # Pydantic validation models
│
├── 🤖 ML Core
│   └── src/
│       ├── data/                  # Data processing & validation
│       ├── models/                # Training, evaluation & inference
│       ├── plugins/               # Extensible ML components
│       ├── config/                # Configuration management
│       └── utils/                 # Common utilities
│
├── 🔬 Research & Development
│   ├── notebooks/                 # Jupyter notebooks
│   ├── workflows/                 # Training & evaluation scripts
│   ├── data/                      # Dataset storage (raw/processed)
│   ├── models/                    # Trained model artifacts
│   └── reports/                   # Analysis outputs
│
└── 🧪 Quality Assurance
    ├── tests/                     # Comprehensive test suite
    ├── .pre-commit-config.yaml    # Code quality automation
    └── Makefile                   # Development workflow
```

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
