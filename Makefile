# MLOps Template Makefile
# Production-ready machine learning operations template

.PHONY: help install install-dev install-uv clean lint format type-check test unit-test integration-test
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
	@echo "🚀 MLOps Template - Available Commands"
	@echo "======================================"
	@echo ""
	@echo "🆕 First time? Run: make setup-complete"
	@echo "🔍 Check your setup: make verify-setup"
	@echo ""
	@echo "Setup & Environment:"
	@echo "  make setup          Complete development environment setup"
	@echo "  make setup-basic    Basic setup (dependencies only)"
	@echo "  make setup-complete Complete setup + Docker build (recommended for first-time)"
	@echo "  make verify-setup   Verify development environment setup"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make clean         Clean up generated files"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with ruff"
	@echo "  make type-check    Run type checking with mypy"
	@echo "  make complexity-check  Run complexity checks (radon/xenon)"
	@echo "  make security-check    Run security checks (bandit)"
	@echo "  make trivy-fs-scan     Run Trivy filesystem scan (HIGH/CRITICAL only)"
	@echo "  make trivy-fs-scan-all Run comprehensive Trivy filesystem scan"
	@echo "  make trivy-image-scan  Run Trivy Docker image scan (requires IMAGE=)"
	@echo "  make security-scan-local    Run all local security scans"
	@echo "  make security-scan-comprehensive  Run comprehensive security scans"
	@echo "  make quality-checks    Run complexity and security checks"
	@echo "  make test          Run all tests"
	@echo "  make unit-test     Run unit tests only"
	@echo "  make integration-test  Run integration tests only"
	@echo "  make all-checks    Run comprehensive checks (lint, format, quality, test)"
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
	@echo "  make cli ARGS=\"<command>\"   Run unified CLI (e.g., make cli ARGS=\"repo info\")"
	@echo "  make plugin-add NAME=<name> [CATEGORY=<cat>]   Scaffold new plugin"
	@echo "  make plugins-list           List registered plugins"
	@echo "  make repo-info              Show git repo info"
	@echo ""
	@echo "Forking Procedure Testing:"
	@echo "  make test-forking-smoke     Quick smoke test for forking procedure"
	@echo "  make test-forking-full      Comprehensive forking procedure tests"
	@echo "  make test-forking-custom NAME=<name>  Test with custom project name"
	@echo "  make validate-forking       Validate forking procedure health"

# Environment setup
setup: install-uv install-trivy install-dev pre-commit
	@echo ""
	@echo "🎉 Complete development environment setup completed!"
	@echo ""
	@echo "✅ Tools installed:"
	@echo "   - uv package manager"
	@echo "   - Trivy security scanner"
	@echo "   - Development dependencies"
	@echo "   - Pre-commit hooks"
	@echo ""
	@echo "🚀 You're ready to start developing!"
	@echo "   Run 'make all-checks' to verify everything works"

setup-basic: install-uv install-dev
	@echo ""
	@echo "✅ Basic development setup completed!"
	@echo "   Run 'make setup' for complete environment with security tools"

setup-complete: setup docker-build
	@echo ""
	@echo "🚀 Complete setup with Docker image build finished!"
	@echo ""
	@echo "🧪 Test your setup:"
	@echo "   make all-checks              # Run all quality checks"
	@echo "   make trivy-fs-scan          # Test security scanning"
	@echo "   make trivy-image-scan IMAGE=$(DOCKER_IMAGE):$(DOCKER_TAG)  # Test Docker image scanning"

install-uv:
	@echo "Installing uv package manager..."
	@command -v uv >/dev/null 2>&1 && { echo "uv already installed"; exit 0; } || true
	curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "uv installed! You may need to restart your shell or run: source ~/.bashrc"

install-trivy:
	@echo "Installing Trivy security scanner..."
	@command -v trivy >/dev/null 2>&1 && { echo "Trivy already installed"; exit 0; } || true
	@if command -v brew >/dev/null 2>&1; then \
		echo "Installing Trivy via Homebrew..."; \
		brew install trivy; \
	elif command -v apt-get >/dev/null 2>&1; then \
		echo "Installing Trivy via apt..."; \
		sudo apt-get update && sudo apt-get install -y wget apt-transport-https gnupg lsb-release; \
		wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -; \
		echo "deb https://aquasecurity.github.io/trivy-repo/deb $$(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list; \
		sudo apt-get update && sudo apt-get install -y trivy; \
	else \
		echo "Installing Trivy via curl..."; \
		curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin; \
	fi
	@echo "Trivy installed successfully!"

install:
	@echo "Installing production dependencies with uv..."
	@command -v uv >/dev/null 2>&1 || { echo "uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv venv $(VENV)
	uv pip install -e .

install-dev:
	@echo "Installing development dependencies with uv..."
	@command -v uv >/dev/null 2>&1 || { echo "uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv venv $(VENV)
	uv pip install -e ".[dev]"

clean:
	@echo "Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf htmlcov
	rm -rf logs
	@echo "Removing local test coverage files..."
	find . -type f -name ".coverage.*" -delete

# Code quality and testing
lint:
	@echo "Running linting checks..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; ruff check .

format:
	@echo "Formatting code..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; ruff format .
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; ruff check --fix .

type-check:
	@echo "Running type checks..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; mypy .

complexity-check:
	@echo "Running complexity checks..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; radon cc src/ --min B

security-check:
	@echo "Running security checks..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; bandit -r src/ -f json || true

trivy-fs-scan:
	@echo "Running Trivy filesystem scan..."
	@command -v trivy >/dev/null 2>&1 || { echo "Trivy not found. Install with 'make install-trivy'"; exit 1; }
	trivy fs --scanners vuln,secret,misconfig --severity HIGH,CRITICAL .

trivy-fs-scan-all:
	@echo "Running comprehensive Trivy filesystem scan..."
	@command -v trivy >/dev/null 2>&1 || { echo "Trivy not found. Install with 'make install-trivy'"; exit 1; }
	trivy fs --scanners vuln,secret,misconfig .

trivy-image-scan:
	@echo "Running Trivy Docker image scan..."
	@command -v trivy >/dev/null 2>&1 || { echo "Trivy not found. Install with 'make install-trivy'"; exit 1; }
	@if [ -z "$(IMAGE)" ]; then \
		echo "Usage: make trivy-image-scan IMAGE=your-image:tag"; \
		echo "Example: make trivy-image-scan IMAGE=$(DOCKER_IMAGE):$(DOCKER_TAG)"; \
		exit 1; \
	fi
	trivy image --severity HIGH,CRITICAL $(IMAGE)

trivy-image-scan-all:
	@echo "Running comprehensive Trivy Docker image scan..."
	@command -v trivy >/dev/null 2>&1 || { echo "Trivy not found. Install with 'make install-trivy'"; exit 1; }
	@if [ -z "$(IMAGE)" ]; then \
		echo "Usage: make trivy-image-scan-all IMAGE=your-image:tag"; \
		echo "Example: make trivy-image-scan-all IMAGE=$(DOCKER_IMAGE):$(DOCKER_TAG)"; \
		exit 1; \
	fi
	trivy image $(IMAGE)

security-scan-local: security-check trivy-fs-scan
	@echo "Local security scans completed!"

security-scan-comprehensive: security-check trivy-fs-scan-all
	@echo "Comprehensive security scans completed!"

quality-checks: complexity-check security-scan-local
	@echo "Quality checks completed!"

test:
	@echo "Running all tests..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; python -m pytest tests/ -v --cov=src --cov-report=term-missing

unit-test:
	@echo "Running unit tests..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; python -m pytest tests/ -v --cov=src --cov-report=term-missing -m "not integration"

integration-test:
	@echo "Running integration tests..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; python -m pytest tests/integration/ -v

all-checks: lint format quality-checks unit-test
	@echo "All development checks completed successfully!"

all-checks-strict: lint format type-check quality-checks test
	@echo "All strict checks completed successfully!"

verify-setup:
	@echo "🔍 Verifying development environment setup..."
	@echo ""
	@echo "Checking tools..."
	@command -v uv >/dev/null 2>&1 && echo "✅ uv package manager" || echo "❌ uv package manager (run 'make setup')"
	@command -v trivy >/dev/null 2>&1 && echo "✅ Trivy security scanner" || echo "❌ Trivy security scanner (run 'make setup')"
	@command -v docker >/dev/null 2>&1 && echo "✅ Docker" || echo "❌ Docker (install Docker Desktop)"
	@echo ""
	@echo "Checking Python environment..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then \
		echo "✅ Virtual environment"; \
		. $(VENV_ACTIVATE) && python -c "import pytest, ruff, bandit, radon" && echo "✅ Development dependencies" || echo "❌ Development dependencies (run 'make setup')"; \
	else \
		echo "❌ Virtual environment (run 'make setup')"; \
	fi
	@echo ""
	@echo "Run 'make all-checks' to test everything works!"

pre-commit:
	@echo "Setting up pre-commit hooks..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; pre-commit install

# Demonstration targets
demo-comprehensive:
	@echo "Running comprehensive platform demonstration..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; python demo_comprehensive.py

demo-data:
	@echo "Demonstrating data workflows..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; python demo_comprehensive.py --component data

demo-models:
	@echo "Demonstrating model workflows..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; python demo_comprehensive.py --component models

demo-api:
	@echo "Demonstrating API endpoints..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; python demo_comprehensive.py --component api

demo-workflows:
	@echo "Demonstrating ML workflows..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; python demo_comprehensive.py --component workflows

demo-plugins:
	@echo "Demonstrating plugin system..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; python demo_comprehensive.py --component plugins

# API and service management
run-api:
	@echo "Starting FastAPI development server..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; cd src && python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

run-api-prod:
	@echo "Starting production API server..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; cd src && python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4

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

# CLI helpers
cli:
	@if [ -z "$(ARGS)" ]; then echo "Usage: make cli ARGS=\"<command>\""; exit 1; fi
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; python scripts/mlops_cli.py $(ARGS)

plugin-add:
	@if [ -z "$(NAME)" ]; then echo "Usage: make plugin-add NAME=<name> [CATEGORY=<cat>]"; exit 1; fi
	@$(MAKE) cli ARGS="plugin add $(NAME) $(if $(CATEGORY),--category $(CATEGORY),)"

plugins-list:
	@$(MAKE) cli ARGS="plugin list --verbose"

repo-info:
	@$(MAKE) cli ARGS="repo info"

# Forking procedure testing
test-forking-smoke:
	@echo "Running forking procedure smoke test..."
	@python3 scripts/test_forking_smoke.py

test-forking-full:
	@echo "Running comprehensive forking procedure tests..."
	@if [ -f "$(VENV_ACTIVATE)" ]; then . $(VENV_ACTIVATE); fi; python -m pytest tests/integration/test_forking_procedure.py -v

test-forking-custom:
	@if [ -z "$(NAME)" ]; then echo "Usage: make test-forking-custom NAME=<project-name>"; exit 1; fi
	@echo "Testing forking procedure with custom name: $(NAME)"
	@python3 scripts/test_forking_smoke.py $(NAME)

validate-forking:
	@echo "Validating forking procedure health..."
	@python3 scripts/test_forking_smoke.py --keep-temp
	@echo "Run 'make test-forking-full' for comprehensive testing"
