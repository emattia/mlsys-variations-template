# MLOps Template Makefile
# Production-ready machine learning operations template

.PHONY: help install install-dev clean lint format type-check test unit-test integration-test
.PHONY: all-checks all-checks-strict pre-commit run-api docker-build docker-run docker-compose
.PHONY: demo-comprehensive demo-data demo-models demo-api demo-workflows demo-plugins

# Python and virtual environment settings
PYTHON := python3
VENV := .venv
VENV_ACTIVATE := $(VENV)/bin/activate
PIP := $(VENV)/bin/pip

# Docker settings
DOCKER_IMAGE := mlops-template
DOCKER_TAG := latest
DOCKER_REGISTRY := # Set your registry here

# Default target
help:
	@echo "MLOps Template - Available Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup & Environment:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make clean         Clean up generated files"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with ruff"
	@echo "  make type-check    Run type checking with mypy"
	@echo "  make test          Run all tests"
	@echo "  make unit-test     Run unit tests only"
	@echo "  make integration-test  Run integration tests only"
	@echo "  make all-checks    Run comprehensive checks (lint, format, test)"
	@echo "  make all-checks-strict  Run all checks including type checking"
	@echo "  make pre-commit    Setup pre-commit hooks"
	@echo ""
	@echo "Development & Demo:"
	@echo "  make demo-comprehensive  Run complete platform demonstration"
	@echo "  make demo-data     Demonstrate data workflows"
	@echo "  make demo-models   Demonstrate model workflows"
	@echo "  make demo-api      Demonstrate API endpoints"
	@echo "  make demo-workflows Demonstrate ML workflows"
	@echo "  make demo-plugins  Demonstrate plugin system"
	@echo ""
	@echo "API & Services:"
	@echo "  make run-api       Start FastAPI development server"
	@echo "  make run-api-prod  Start production API server"
	@echo ""
	@echo "Docker & Deployment:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo "  make docker-compose Start services with docker-compose"
	@echo "  make docker-push   Push image to registry"
	@echo "  make docker-clean  Clean Docker images and containers"

# Environment setup
install:
	@echo "Installing production dependencies..."
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements-dev.txt

clean:
	@echo "Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

# Code quality and testing
lint:
	@echo "Running linting checks..."
	@. $(VENV_ACTIVATE); ruff check .

format:
	@echo "Formatting code..."
	@. $(VENV_ACTIVATE); ruff format .
	@. $(VENV_ACTIVATE); ruff check --fix .

type-check:
	@echo "Running type checks..."
	@. $(VENV_ACTIVATE); mypy .

test:
	@echo "Running all tests..."
	@. $(VENV_ACTIVATE); python -m pytest tests/ -v --cov=src --cov-report=term-missing

unit-test:
	@echo "Running unit tests..."
	@. $(VENV_ACTIVATE); python -m pytest tests/ -v --cov=src --cov-report=term-missing -m "not integration"

integration-test:
	@echo "Running integration tests..."
	@. $(VENV_ACTIVATE); python -m pytest tests/integration/ -v

all-checks: lint format unit-test
	@echo "All development checks completed successfully!"

all-checks-strict: lint format type-check test
	@echo "All strict checks completed successfully!"

pre-commit:
	@echo "Setting up pre-commit hooks..."
	@. $(VENV_ACTIVATE); pre-commit install

# Demonstration targets
demo-comprehensive:
	@echo "Running comprehensive platform demonstration..."
	@. $(VENV_ACTIVATE); python demo_comprehensive.py

demo-data:
	@echo "Demonstrating data workflows..."
	@. $(VENV_ACTIVATE); python demo_comprehensive.py --component data

demo-models:
	@echo "Demonstrating model workflows..."
	@. $(VENV_ACTIVATE); python demo_comprehensive.py --component models

demo-api:
	@echo "Demonstrating API endpoints..."
	@. $(VENV_ACTIVATE); python demo_comprehensive.py --component api

demo-workflows:
	@echo "Demonstrating ML workflows..."
	@. $(VENV_ACTIVATE); python demo_comprehensive.py --component workflows

demo-plugins:
	@echo "Demonstrating plugin system..."
	@. $(VENV_ACTIVATE); python demo_comprehensive.py --component plugins

# API and service management
run-api:
	@echo "Starting FastAPI development server..."
	@. $(VENV_ACTIVATE); cd src && python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

run-api-prod:
	@echo "Starting production API server..."
	@. $(VENV_ACTIVATE); cd src && python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4

# Docker targets
docker-build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 --name mlops-container $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-compose:
	@echo "Starting services with docker-compose..."
	docker-compose up -d

docker-push:
	@echo "Pushing Docker image to registry..."
	@if [ -z "$(DOCKER_REGISTRY)" ]; then \
		echo "Error: DOCKER_REGISTRY not set. Set it in the Makefile or environment."; \
		exit 1; \
	fi
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-clean:
	@echo "Cleaning Docker images and containers..."
	-docker stop mlops-container
	-docker rm mlops-container
	-docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG)

# Development workflows
watch-tests:
	@echo "Watching for changes and running tests..."
	@. $(VENV_ACTIVATE); ptw --runner "python -m pytest tests/ -v"

serve-docs:
	@echo "Serving documentation..."
	@. $(VENV_ACTIVATE); mkdocs serve

build-docs:
	@echo "Building documentation..."
	@. $(VENV_ACTIVATE); mkdocs build
