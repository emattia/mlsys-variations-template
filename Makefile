# Makefile for mlsys-variations-template
# This file automates common development tasks.

.PHONY: help install test lint format check-types clean

# Variables
PYTHON = python3
VENV_DIR = .venv
VENV_ACTIVATE = $(VENV_DIR)/bin/activate

# Default target
help:
	@echo "Makefile for mlsys-variations-template"
	@echo ""
	@echo "Usage:"
	@echo "  make install        Install dependencies into a virtual environment"
	@echo "  make test           Run all tests (unit and workflow)"
	@echo "  make unit-test      Run unit tests with pytest"
	@echo "  make workflow-test  Run workflow tests"
	@echo "  make lint           Check code style with ruff"
	@echo "  make format         Format code with ruff"
	@echo "  make check-types    Run static type checking with mypy"
	@echo "  make all-checks     Run all checks (lint, format, types, tests)"
	@echo "  make clean          Remove temporary files"
	@echo ""
	@echo "Phase 1 Features:"
	@echo "  make demo-phase1    Run Phase 1 demonstration"
	@echo "  make config-demo    Demonstrate configuration system"
	@echo "  make test-config    Run configuration tests"
	@echo "  make test-plugins   Run plugin architecture tests"
	@echo "  make setup-hydra    Create default Hydra config files"
	@echo ""
	@echo "Phase 2 Features:"
	@echo "  make api-serve      Start FastAPI development server"
	@echo "  make api-serve-prod Start FastAPI production server"
	@echo "  make test-api       Run API unit tests"
	@echo "  make test-integration Run integration tests"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run Docker container"
	@echo "  make docker-dev     Start development with Docker Compose"
	@echo "  make docker-prod    Start production with Docker Compose"
	@echo "  make demo-phase2    Run Phase 2 demonstration"

# Environment setup
install:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "Virtual environment created at $(VENV_DIR)"; \
	fi
	@. $(VENV_ACTIVATE); \
	uv pip install -e ".[dev]"
	@echo "Dependencies installed."

# Testing
unit-test:
	@echo "Running unit tests..."
	@. $(VENV_ACTIVATE); pytest -m "not integration"

workflow-test:
	@echo "Running workflow tests..."
	@. $(VENV_ACTIVATE); pytest workflows/tests/

test: unit-test workflow-test
	@echo "All tests passed."

# Code quality
lint:
	@echo "Linting with ruff..."
	@. $(VENV_ACTIVATE); ruff check .

format:
	@echo "Formatting with ruff..."
	@. $(VENV_ACTIVATE); ruff format .

check-types:
	@echo "Type checking with mypy..."
	@. $(VENV_ACTIVATE); mypy src/

all-checks: lint format test
	@echo "All checks passed."

# Full checks including type checking (more strict)
all-checks-strict: lint format check-types test
	@echo "All strict checks passed."

# Phase 1 Features
demo-phase1:
	@echo "Running Phase 1 demonstration..."
	@. $(VENV_ACTIVATE); python demo_phase1.py

config-demo:
	@echo "Demonstrating configuration system..."
	@. $(VENV_ACTIVATE); python -c "from src.config import AppConfig; print('✅ Config system working!'); c = AppConfig(app_name='demo'); print(f'App: {c.app_name}, Env: {c.environment}')"

test-config:
	@echo "Running configuration tests..."
	@. $(VENV_ACTIVATE); pytest tests/unit/test_config.py -v

test-plugins:
	@echo "Running plugin architecture tests..."
	@. $(VENV_ACTIVATE); pytest tests/unit/test_plugins.py -v

setup-hydra:
	@echo "Setting up Hydra configuration files..."
	@. $(VENV_ACTIVATE); python -c "from src.config.manager import get_config_manager; get_config_manager().create_default_config_files(); print('✅ Hydra config files created in ./conf/')"

# Phase 2 Features - FastAPI & Containerization
api-serve:
	@echo "Starting FastAPI development server..."
	@. $(VENV_ACTIVATE); uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

api-serve-prod:
	@echo "Starting FastAPI production server..."
	@. $(VENV_ACTIVATE); uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4

test-api:
	@echo "Running API unit tests..."
	@. $(VENV_ACTIVATE); pytest tests/unit/test_api.py -v

test-integration:
	@echo "Running integration tests..."
	@. $(VENV_ACTIVATE); pytest tests/integration/ -v --timeout=60

test-api-performance:
	@echo "Running API performance tests..."
	@. $(VENV_ACTIVATE); pytest tests/integration/test_api_integration.py::TestAPIPerformance -v -m slow

docker-build:
	@echo "Building Docker image..."
	docker build -t mlops-template:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -d -p 8000:8000 --name mlops-api mlops-template:latest

docker-dev:
	@echo "Starting development environment with Docker Compose..."
	docker-compose --profile dev up --build

docker-prod:
	@echo "Starting production environment with Docker Compose..."
	docker-compose up --build

docker-monitoring:
	@echo "Starting with monitoring stack..."
	docker-compose --profile monitoring up --build

docker-stop:
	@echo "Stopping Docker containers..."
	docker-compose down

docker-clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f

demo-phase2:
	@echo "Running Phase 2 demonstration..."
	@. $(VENV_ACTIVATE); python demo_phase2.py

# Cleanup
clean:
	@echo "Cleaning up..."
	@rm -rf `find . -name __pycache__`
	@rm -f `find . -type f -name '*.py[co]'`
	@rm -f .coverage
	@rm -rf .pytest_cache
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info
	@rm -rf conf/
	@rm -rf outputs/
