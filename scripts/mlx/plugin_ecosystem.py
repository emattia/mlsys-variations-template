#!/usr/bin/env python3
"""Plugin Ecosystem Development Framework

This module provides comprehensive plugin development and management capabilities:
- Plugin template generation
- Plugin validation and compatibility testing
- Plugin discovery and installation
- Plugin development workflow automation
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import yaml
from enum import Enum

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of MLX plugins."""

    ML_FRAMEWORK = "ml_framework"
    DATA_PROCESSOR = "data_processor"
    MODEL_PROVIDER = "model_provider"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    SECURITY = "security"
    UTILITY = "utility"


class PluginStatus(Enum):
    """Plugin development status."""

    TEMPLATE = "template"
    DEVELOPMENT = "development"
    TESTING = "testing"
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"


@dataclass
class PluginSpec:
    """Plugin specification for template generation."""

    name: str
    plugin_type: PluginType
    description: str
    author: str = "MLX Team"
    version: str = "0.1.0"
    python_version: str = ">=3.9"
    dependencies: List[str] = field(default_factory=list)
    dev_dependencies: List[str] = field(default_factory=list)
    entry_points: Dict[str, str] = field(default_factory=dict)
    interfaces: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    test_frameworks: List[str] = field(default_factory=lambda: ["pytest"])


@dataclass
class PluginValidationResult:
    """Result of plugin validation."""

    plugin_name: str
    validation_timestamp: float
    overall_status: str
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    compatibility_matrix: Dict[str, bool] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class PluginEcosystemManager:
    """Manager for plugin ecosystem development and validation."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        self.plugins_dir = self.workspace_dir / "plugins"
        self.templates_dir = self.workspace_dir / "scripts" / "mlx" / "plugin_templates"
        self.registry_file = (
            self.workspace_dir / "mlx-components" / "plugin_registry.json"
        )

        # Create necessary directories
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Load plugin templates and validation rules
        self.templates = self._load_plugin_templates()
        self.validation_rules = self._load_validation_rules()

    def _load_plugin_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load plugin templates for different plugin types."""
        return {
            PluginType.ML_FRAMEWORK.value: {
                "base_class": "MLFrameworkPlugin",
                "required_methods": ["train", "predict", "evaluate"],
                "optional_methods": [
                    "preprocess",
                    "postprocess",
                    "save_model",
                    "load_model",
                ],
                "dependencies": ["scikit-learn", "numpy", "pandas"],
                "config_schema": {
                    "model_params": {"type": "dict", "default": {}},
                    "training_params": {"type": "dict", "default": {}},
                    "preprocessing": {"type": "dict", "default": {}},
                },
            },
            PluginType.DATA_PROCESSOR.value: {
                "base_class": "DataProcessorPlugin",
                "required_methods": ["process", "validate"],
                "optional_methods": ["preprocess", "postprocess", "transform"],
                "dependencies": ["pandas", "pydantic"],
                "config_schema": {
                    "input_format": {"type": "str", "default": "csv"},
                    "output_format": {"type": "str", "default": "parquet"},
                    "validation_rules": {"type": "dict", "default": {}},
                },
            },
            PluginType.MODEL_PROVIDER.value: {
                "base_class": "ModelProviderPlugin",
                "required_methods": ["get_model", "list_models"],
                "optional_methods": ["upload_model", "delete_model", "model_info"],
                "dependencies": ["requests", "pydantic"],
                "config_schema": {
                    "api_endpoint": {"type": "str", "required": True},
                    "api_key": {"type": "str", "required": True},
                    "timeout": {"type": "int", "default": 30},
                },
            },
            PluginType.DEPLOYMENT.value: {
                "base_class": "DeploymentPlugin",
                "required_methods": ["deploy", "undeploy", "status"],
                "optional_methods": ["scale", "update", "logs"],
                "dependencies": ["docker", "kubernetes"],
                "config_schema": {
                    "platform": {"type": "str", "required": True},
                    "namespace": {"type": "str", "default": "default"},
                    "replicas": {"type": "int", "default": 1},
                },
            },
            PluginType.MONITORING.value: {
                "base_class": "MonitoringPlugin",
                "required_methods": [
                    "start_monitoring",
                    "stop_monitoring",
                    "get_metrics",
                ],
                "optional_methods": ["alert", "dashboard", "export_metrics"],
                "dependencies": ["prometheus-client", "grafana-api"],
                "config_schema": {
                    "metrics_endpoint": {"type": "str", "required": True},
                    "collection_interval": {"type": "int", "default": 60},
                    "retention_days": {"type": "int", "default": 30},
                },
            },
        }

    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for plugin development."""
        return {
            "naming_convention": {
                "pattern": r"^mlx-plugin-[a-z][a-z0-9-]*[a-z0-9]$",
                "description": "Plugin names must start with 'mlx-plugin-' and use kebab-case",
            },
            "required_files": [
                "pyproject.toml",
                "README.md",
                "src/{plugin_package}/__init__.py",
                "src/{plugin_package}/plugin.py",
                "tests/test_plugin.py",
            ],
            "code_quality": {
                "min_test_coverage": 80,
                "max_complexity": 10,
                "security_scan": True,
            },
            "compatibility": {
                "python_versions": ["3.9", "3.10", "3.11", "3.12"],
                "mlx_versions": [">=1.0.0"],
                "required_interfaces": ["BasePlugin"],
            },
            "performance": {
                "max_import_time": 2.0,  # seconds
                "max_memory_usage": 100,  # MB
                "max_initialization_time": 5.0,  # seconds
            },
        }

    def create_plugin_template(self, spec: PluginSpec) -> Path:
        """Create a new plugin from template."""
        plugin_name = spec.name
        if not plugin_name.startswith("mlx-plugin-"):
            plugin_name = f"mlx-plugin-{plugin_name}"

        plugin_dir = self.plugins_dir / plugin_name

        if plugin_dir.exists():
            raise ValueError(f"Plugin directory already exists: {plugin_dir}")

        logger.info(f"Creating plugin template: {plugin_name}")

        # Create plugin directory structure
        self._create_plugin_structure(plugin_dir, spec)

        # Generate plugin files from templates
        self._generate_plugin_files(plugin_dir, spec)

        # Create development environment
        self._setup_plugin_dev_environment(plugin_dir, spec)

        logger.info(f"Plugin template created at: {plugin_dir}")
        return plugin_dir

    def _create_plugin_structure(self, plugin_dir: Path, spec: PluginSpec):
        """Create plugin directory structure."""
        package_name = spec.name.replace("-", "_")

        # Create directories
        dirs = [
            Path("src") / package_name,
            Path("tests"),
            Path("docs"),
            Path("examples"),
            Path("conf"),
            Path(".github") / "workflows",
        ]

        for dir_path in dirs:
            (plugin_dir / dir_path).mkdir(parents=True, exist_ok=True)

    def _generate_plugin_files(self, plugin_dir: Path, spec: PluginSpec):
        """Generate plugin files from templates."""
        package_name = spec.name.replace("-", "_")
        src_dir = plugin_dir / "src" / package_name

        # Generate pyproject.toml
        self._generate_pyproject_toml(plugin_dir, spec)

        # Generate main plugin module
        self._generate_plugin_module(src_dir, spec)

        # Generate __init__.py
        self._generate_init_file(src_dir, spec)

        # Generate tests
        self._generate_test_files(plugin_dir / "tests", spec)

        # Generate documentation
        self._generate_documentation(plugin_dir, spec)

        # Generate CI/CD workflows
        self._generate_workflows(plugin_dir / ".github" / "workflows", spec)

        # Generate configuration
        self._generate_config_files(plugin_dir / "conf", spec)

    def _generate_pyproject_toml(self, plugin_dir: Path, spec: PluginSpec):
        """Generate pyproject.toml file."""
        spec.name.replace("-", "_")

        config = {
            "project": {
                "name": spec.name,
                "version": spec.version,
                "description": spec.description,
                "authors": [{"name": spec.author}],
                "requires-python": spec.python_version,
                "dependencies": spec.dependencies,
                "optional-dependencies": {
                    "dev": spec.dev_dependencies
                    + [
                        "pytest>=7.0.0",
                        "pytest-cov>=4.0.0",
                        "ruff>=0.1.0",
                        "mypy>=1.0.0",
                    ]
                },
                "classifiers": [
                    "Development Status :: 3 - Alpha",
                    "Intended Audience :: Developers",
                    "License :: OSI Approved :: MIT License",
                    f"Programming Language :: Python :: {spec.python_version.split('>=')[1]}",
                ],
            },
            "build-system": {
                "requires": ["setuptools>=61.0"],
                "build-backend": "setuptools.build_meta",
            },
            "tool": {
                "pytest": {
                    "testpaths": ["tests"],
                    "addopts": "--cov=src --cov-report=term-missing",
                },
                "ruff": {"line-length": 88, "target-version": "py39"},
                "mypy": {"python_version": "3.9", "strict": True},
            },
        }

        # Add entry points if specified
        if spec.entry_points:
            config["project"]["entry-points"] = {"mlx.plugins": spec.entry_points}

        with open(plugin_dir / "pyproject.toml", "w") as f:
            import toml

            toml.dump(config, f)

    def _generate_plugin_module(self, src_dir: Path, spec: PluginSpec):
        """Generate main plugin module."""
        template_info = self.templates.get(spec.plugin_type.value, {})
        base_class = template_info.get("base_class", "BasePlugin")
        required_methods = template_info.get("required_methods", [])
        template_info.get("optional_methods", [])

        plugin_code = f'''"""
{spec.description}

This module implements the {spec.name} plugin for the MLX platform.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# MLX plugin base class (would be imported from mlx-core)
class BasePlugin:
    """Base plugin interface."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {{}}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        return True
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass


class {base_class.replace("Plugin", "")}Plugin(BasePlugin):
    """
    {spec.description}
    
    Plugin Type: {spec.plugin_type.value}
    Version: {spec.version}
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "{spec.name}"
        self.version = "{spec.version}"
        self.plugin_type = "{spec.plugin_type.value}"
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the {spec.plugin_type.value} plugin."""
        try:
            # Plugin-specific initialization logic
            self._setup_plugin()
            self._initialized = True
            self.logger.info(f"{{self.name}} plugin initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize {{self.name}} plugin: {{e}}")
            return False
    
    def _setup_plugin(self) -> None:
        """Setup plugin-specific components."""
        # Implement plugin-specific setup logic
        pass
'''

        # Add required methods
        for method in required_methods:
            if method == "train":
                plugin_code += '''
    def train(self, data: Any, **kwargs) -> Any:
        """Train a model with the provided _data."""
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")
        
        # Implement training logic
        self.logger.info("Training started")
        # TODO: Implement actual training logic
        return {"status": "completed", "model": None}
'''
            elif method == "predict":
                plugin_code += '''
    def predict(self, data: Any, model: Any = None, **kwargs) -> Any:
        """Make predictions with the model."""
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")
        
        # Implement prediction logic
        self.logger.info("Making predictions")
        # TODO: Implement actual prediction logic
        return {"predictions": [], "confidence": []}
'''
            elif method == "process":
                plugin_code += '''
    def process(self, data: Any, **kwargs) -> Any:
        """Process the input _data."""
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")
        
        # Implement data processing logic
        self.logger.info("Processing data")
        # TODO: Implement actual processing logic
        return data
'''
            elif method == "validate":
                plugin_code += '''
    def validate(self, data: Any, **kwargs) -> bool:
        """Validate input _data."""
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")
        
        # Implement validation logic
        self.logger.info("Validating data")
        # TODO: Implement actual validation logic
        return True
'''
            else:
                # Generic method template
                plugin_code += f'''
    def {method}(self, *args, **kwargs) -> Any:
        """
        {method.replace("_", " ").title()} functionality.
        
        TODO: Implement {method} logic for {spec.plugin_type.value} plugin.
        """
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")
        
        self.logger.info(f"Executing {method}")
        # TODO: Implement actual {method} logic
        return None
'''

        # Add cleanup method
        plugin_code += '''
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        if self._initialized:
            # Implement cleanup logic
            self._initialized = False
            self.logger.info(f"{self.name} plugin cleaned up")
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.name,
            "version": self.version,
            "type": self.plugin_type,
            "initialized": self._initialized,
            "config": self.config
        }
'''

        with open(src_dir / "plugin.py", "w") as f:
            f.write(plugin_code)

    def _generate_init_file(self, src_dir: Path, spec: PluginSpec):
        """Generate __init__.py file."""
        base_class = self.templates.get(spec.plugin_type.value, {}).get(
            "base_class", "BasePlugin"
        )
        class_name = base_class.replace("Plugin", "") + "Plugin"

        init_code = f'''"""
{spec.name} - {spec.description}

MLX Plugin for {spec.plugin_type.value} functionality.
"""

__version__ = "{spec.version}"
__author__ = "{spec.author}"

from .plugin import {class_name}

__all__ = ["{class_name}"]

# Plugin metadata for MLX discovery
PLUGIN_INFO = {{
    "name": "{spec.name}",
    "version": __version__,
    "type": "{spec.plugin_type.value}",
    "class": {class_name},
    "interfaces": {spec.interfaces},
    "entry_point": "{spec.name.replace("-", "_")}"
}}
'''

        with open(src_dir / "__init__.py", "w") as f:
            f.write(init_code)

    def _generate_test_files(self, tests_dir: Path, spec: PluginSpec):
        """Generate test files."""
        base_class = self.templates.get(spec.plugin_type.value, {}).get(
            "base_class", "BasePlugin"
        )
        class_name = base_class.replace("Plugin", "") + "Plugin"
        package_name = spec.name.replace("-", "_")

        test_code = f'''"""
Tests for {spec.name} plugin.
"""

import pytest
from unittest.mock import Mock, patch
from src.{package_name}.plugin import {class_name}


class Test{class_name}:
    """Test suite for {class_name}."""
    
    @pytest.fixture
    def plugin(self):
        """Create plugin instance for testing."""
        return {class_name}()
    
    @pytest.fixture
    def plugin_with_config(self):
        """Create plugin instance with test configuration."""
        config = {{
            "test_mode": True,
            "timeout": 30
        }}
        return {class_name}(config)
    
    def test_plugin_initialization(self, plugin):
        """Test plugin initialization."""
        assert plugin.name == "{spec.name}"
        assert plugin.version == "{spec.version}"
        assert plugin.plugin_type == "{spec.plugin_type.value}"
        assert not plugin._initialized
    
    def test_plugin_initialize_success(self, plugin):
        """Test successful plugin initialization."""
        result = plugin.initialize()
        assert result is True
        assert plugin._initialized is True
    
    def test_plugin_cleanup(self, plugin):
        """Test plugin cleanup."""
        plugin.initialize()
        assert plugin._initialized is True
        
        plugin.cleanup()
        assert plugin._initialized is False
    
    def test_plugin_info(self, plugin):
        """Test plugin info retrieval."""
        info = plugin.get_info()
        
        assert info["name"] == "{spec.name}"
        assert info["version"] == "{spec.version}"
        assert info["type"] == "{spec.plugin_type.value}"
        assert "initialized" in info
        assert "config" in info
    
    def test_plugin_with_config(self, plugin_with_config):
        """Test plugin with custom configuration."""
        assert plugin_with_config.config["test_mode"] is True
        assert plugin_with_config.config["timeout"] == 30
'''

        # Add method-specific tests
        required_methods = self.templates.get(spec.plugin_type.value, {}).get(
            "required_methods", []
        )
        for method in required_methods:
            test_code += f'''
    def test_{method}(self, plugin):
        """Test {method} method."""
        plugin.initialize()
        
        # TODO: Add specific test logic for {method}
        result = plugin.{method}(test_data="sample")
        assert result is not None
'''

        # Add performance tests
        test_code += '''
    @pytest.mark.performance
    def test_plugin_performance(self, plugin):
        """Test plugin performance metrics."""
        import time
        
        start_time = time.time()
        plugin.initialize()
        init_time = time.time() - start_time
        
        # Check initialization time
        assert init_time < 5.0, f"Initialization took too long: {init_time}s"
        
        # Check memory usage (simplified)
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 100, f"Memory usage too high: {memory_mb}MB"
'''

        with open(tests_dir / "test_plugin.py", "w") as f:
            f.write(test_code)

        # Create conftest.py for shared test fixtures
        conftest_code = '''"""
Shared test fixtures for plugin testing.
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "input": "test_input",
        "parameters": {"param1": "value1"},
        "metadata": {"source": "test"}
    }
'''

        with open(tests_dir / "conftest.py", "w") as f:
            f.write(conftest_code)

    def _generate_documentation(self, plugin_dir: Path, spec: PluginSpec):
        """Generate plugin documentation."""

        readme_content = f"""# {spec.name}

{spec.description}

## Installation

```bash
pip install {spec.name}
```

## Usage

```python
from {spec.name.replace("-", "_")} import {spec.name.replace("-", "_").title()}Plugin

# Initialize the plugin
plugin = {spec.name.replace("-", "_").title()}Plugin()

# Use the plugin
result = plugin.process(data)
```

## Configuration

The plugin can be configured using the following parameters:

```yaml
plugin:
  name: {spec.name}
  version: {spec.version}
  type: {spec.plugin_type.value}
  enabled: true
```

## API Reference

### Main Plugin Class

#### `{spec.name.replace("-", "_").title()}Plugin`

The main plugin class that implements the MLX plugin interface.

**Methods:**
"""

        # Add method documentation
        required_methods = self.templates.get(spec.plugin_type.value, {}).get(
            "required_methods", []
        )
        for method in required_methods:
            readme_content += f"""
#### `{method}()`

Description of {method} method.

**Parameters:**
- `data`: Input data to process
- `**kwargs`: Additional parameters

**Returns:**
- Processed result

"""

        readme_content += f"""
## Development

### Setup Development Environment

```bash
git clone <repository-url>
cd {spec.name}
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Quality

```bash
ruff check .
mypy .
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## Support

For support, please open an issue on the repository or contact {spec.author}.
"""

        with open(plugin_dir / "README.md", "w") as f:
            f.write(readme_content)

    def _generate_workflows(self, workflows_dir: Path, spec: PluginSpec):
        """Generate CI/CD workflows."""
        # Generate GitHub Actions workflow
        ci_workflow = """name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Lint with ruff
      run: |
        ruff check .
    
    - name: Type check with mypy
      run: |
        mypy src/
    
    - name: Test with pytest
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
"""

        with open(workflows_dir / "ci.yml", "w") as f:
            f.write(ci_workflow)

    def _generate_config_files(self, conf_dir: Path, spec: PluginSpec):
        """Generate configuration files."""
        # Generate default config
        default_config = {
            "plugin": {
                "name": spec.name,
                "version": spec.version,
                "type": spec.plugin_type.value,
                "enabled": True,
            },
            "settings": spec.config_schema,
        }

        with open(conf_dir / "config.yaml", "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

        # Generate development config
        dev_config = {
            **default_config,
            "development": {"debug": True, "log_level": "DEBUG", "test_mode": True},
        }

        with open(conf_dir / "development.yaml", "w") as f:
            yaml.dump(dev_config, f, default_flow_style=False)

    def _setup_plugin_dev_environment(self, plugin_dir: Path, spec: PluginSpec):
        """Setup development environment for the plugin."""
        # Create .gitignore
        gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions  
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# MLX specific
mlruns/
outputs/
artifacts/
*.pkl
*.joblib
"""

        with open(plugin_dir / ".gitignore", "w") as f:
            f.write(gitignore_content)

        # Create Makefile for common tasks
        makefile_content = f"""# Makefile for {spec.name}

.PHONY: install install-dev test lint format typecheck clean help

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {{FS = ":.*?## "}}; {{printf "  \\033[36m%-15s\\033[0m %s\\n", $$1, $$2}}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term

lint:  ## Run linting
	ruff check .

format:  ## Format code
	ruff format .

typecheck:  ## Run type checking
	mypy src/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	python -m build

publish:  ## Publish to PyPI (requires credentials)
	python -m twine upload dist/*

dev:  ## Start development mode
	python -m {spec.name.replace("-", "_")}

all: install-dev lint typecheck test  ## Run all checks
"""

        with open(plugin_dir / "Makefile", "w") as f:
            f.write(makefile_content)

    def validate_plugin(self, plugin_path: Path) -> PluginValidationResult:
        """Validate a plugin against MLX standards."""
        plugin_name = plugin_path.name
        logger.info(f"Validating plugin: {plugin_name}")

        result = PluginValidationResult(
            plugin_name=plugin_name,
            validation_timestamp=time.time(),
            overall_status="unknown",
        )

        # Run validation checks
        result.checks["structure"] = self._validate_plugin_structure(plugin_path)
        result.checks["code_quality"] = self._validate_code_quality(plugin_path)
        result.checks["tests"] = self._validate_tests(plugin_path)
        result.checks["documentation"] = self._validate_documentation(plugin_path)
        result.checks["security"] = self._validate_security(plugin_path)
        result.checks["performance"] = self._validate_performance(plugin_path)

        # Check compatibility
        result.compatibility_matrix = self._check_compatibility(plugin_path)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result.checks)

        # Determine overall status
        failed_checks = [
            name
            for name, check in result.checks.items()
            if check.get("status") == "failed"
        ]

        if not failed_checks:
            result.overall_status = "passed"
        elif len(failed_checks) <= 2:
            result.overall_status = "warning"
        else:
            result.overall_status = "failed"

        logger.info(f"Plugin validation completed: {result.overall_status}")
        return result

    def _validate_plugin_structure(self, plugin_path: Path) -> Dict[str, Any]:
        """Validate plugin directory structure."""

        # Check required files
        # Extract plugin name from directory name (remove mlx-plugin- prefix if present)
        plugin_name = plugin_path.name
        if plugin_name.startswith("mlx-plugin-"):
            plugin_name = plugin_name[11:]  # Remove "mlx-plugin-" prefix
        package_name = plugin_name.replace("-", "_")

        required_files = [
            "pyproject.toml",
            "README.md",
            f"src/{package_name}/__init__.py",
            f"src/{package_name}/plugin.py",
            "tests/test_plugin.py",
        ]

        for file_path in required_files:
            full_path = plugin_path / file_path
            if full_path.exists():
                result["details"][file_path] = "present"
            else:
                result["details"][file_path] = "missing"
                result["missing_files"].append(file_path)

        if result["missing_files"]:
            result["status"] = "failed"

        return result

    def _validate_code_quality(self, plugin_path: Path) -> Dict[str, Any]:
        """Validate code quality using ruff and mypy."""

        try:
            # Run ruff
            subprocess.run(
                ["ruff", "check", str(plugin_path)],
                capture_output=True,
                text=True,
                cwd=plugin_path,
            )
            result["tools"]["ruff"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            # Run mypy
            subprocess.run(
                ["mypy", "src/"], capture_output=True, text=True, cwd=plugin_path
            )
            result["tools"]["mypy"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            # Determine overall status
            if result.returncode == 0 and result.returncode == 0:
                result["status"] = "passed"
            else:
                result["status"] = "failed"

        except FileNotFoundError as e:
            result["status"] = "skipped"
            result["error"] = f"Tool not available: {e}"

        return result

    def _validate_tests(self, plugin_path: Path) -> Dict[str, Any]:
        """Validate test coverage and execution."""

        try:
            # Run tests with coverage
            subprocess.run(
                ["pytest", "--cov=src", "--cov-report=json", "--json-report"],
                capture_output=True,
                text=True,
                cwd=plugin_path,
            )

            result["test_results"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            # Try to read coverage report
            coverage_file = plugin_path / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    result["coverage"] = coverage_data.get("totals", {}).get(
                        "percent_covered", 0
                    )

            # Check if tests pass and coverage meets minimum
            min_coverage = self.validation_rules["code_quality"]["min_test_coverage"]
            if result.returncode == 0 and result["coverage"] >= min_coverage:
                result["status"] = "passed"
            else:
                result["status"] = "failed"

        except FileNotFoundError:
            result["status"] = "skipped"
            result["error"] = "pytest not available"

        return result

    def _validate_documentation(self, plugin_path: Path) -> Dict[str, Any]:
        """Validate plugin documentation."""

        # Check README.md
        readme_path = plugin_path / "README.md"
        if readme_path.exists():
            loaded_data = readme_path.read_text()
            result["files"]["README.md"] = {
                "exists": True,
                "length": len(loaded_data),
                "has_installation": "installation" in loaded_data.lower(),
                "has_usage": "usage" in loaded_data.lower()
                or "example" in loaded_data.lower(),
                "has_api": "api" in loaded_data.lower()
                or "method" in loaded_data.lower(),
            }
        else:
            result["files"]["README.md"] = {"exists": False}
            result["status"] = "failed"

        return result

    def _validate_security(self, plugin_path: Path) -> Dict[str, Any]:
        """Validate plugin security."""

        try:
            # Run bandit
            subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                capture_output=True,
                text=True,
                cwd=plugin_path,
            )

            result["tools"]["bandit"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            # Parse bandit results
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    high_severity = len(
                        [
                            r
                            for r in bandit_data.get("results", [])
                            if r.get("issue_severity") == "HIGH"
                        ]
                    )
                    result["high_severity_issues"] = high_severity

                    if high_severity == 0:
                        result["status"] = "passed"
                    else:
                        result["status"] = "failed"
                except json.JSONDecodeError:
                    result["status"] = "unknown"

        except FileNotFoundError:
            result["status"] = "skipped"
            result["error"] = "bandit not available"

        return result

    def _validate_performance(self, plugin_path: Path) -> Dict[str, Any]:
        """Validate plugin performance characteristics."""

        try:
            # Run performance tests
            subprocess.run(
                ["pytest", "-m", "performance", "--json-report"],
                capture_output=True,
                text=True,
                cwd=plugin_path,
            )

            result["test_results"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            if result.returncode == 0:
                result["status"] = "passed"
            else:
                result["status"] = "failed"

        except FileNotFoundError:
            result["status"] = "skipped"
            result["error"] = "pytest not available"

        return result

    def _check_compatibility(self, plugin_path: Path) -> Dict[str, bool]:
        """Check plugin compatibility with MLX platform."""
        compatibility = {}

        # Check Python version compatibility
        pyproject_file = plugin_path / "pyproject.toml"
        if pyproject_file.exists():
            try:
                import toml

                pyproject_data = toml.load(pyproject_file)
                requires_python = pyproject_data.get("project", {}).get(
                    "requires-python", ""
                )

                # Simple compatibility check
                compatibility["python_version"] = ">=3.9" in requires_python
                compatibility["dependencies"] = True  # Would check actual dependencies
                compatibility["interfaces"] = True  # Would check plugin interfaces

            except Exception:
                compatibility["python_version"] = False
                compatibility["dependencies"] = False
                compatibility["interfaces"] = False

        return compatibility

    def _generate_recommendations(self, checks: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Structure recommendations
        structure_check = checks.get("structure", {})
        if structure_check.get("missing_files"):
            recommendations.append(
                f"Add missing files: {', '.join(structure_check['missing_files'])}"
            )

        # Code quality recommendations
        code_quality = checks.get("code_quality", {})
        if code_quality.get("status") == "failed":
            recommendations.append("Fix code quality issues reported by ruff and mypy")

        # Test recommendations
        tests = checks.get("tests", {})
        if tests.get("coverage", 0) < 80:
            recommendations.append(
                f"Increase test coverage to at least 80% (current: {tests.get('coverage', 0):.1f}%)"
            )

        # Security recommendations
        security = checks.get("security", {})
        if security.get("high_severity_issues", 0) > 0:
            recommendations.append("Fix high-severity security issues found by bandit")

        # Documentation recommendations
        docs = checks.get("documentation", {})
        readme_info = docs.get("files", {}).get("README.md", {})
        if not readme_info.get("has_installation"):
            recommendations.append("Add installation instructions to README")
        if not readme_info.get("has_usage"):
            recommendations.append("Add usage examples to README")

        return recommendations


# CLI interface for plugin ecosystem management
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plugin Ecosystem Manager")
    parser.add_argument("command", choices=["create", "validate", "list", "info"])
    parser.add_argument("--name", help="Plugin name")
    parser.add_argument(
        "--type", choices=[t.value for t in PluginType], help="Plugin type"
    )
    parser.add_argument("--description", help="Plugin description")
    parser.add_argument("--workspace", default=".", help="Workspace directory")
    parser.add_argument("--plugin-path", help="Path to plugin for validation")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    manager = PluginEcosystemManager(Path(args.workspace))

    try:
        if args.command == "create":
            if not args.name or not args.type:
                print("--name and --type required for create command")
                exit(1)

            spec = PluginSpec(
                name=args.name,
                plugin_type=PluginType(args.type),
                description=args.description or f"MLX plugin for {args.type}",
            )

            plugin_dir = manager.create_plugin_template(spec)
            print(f"Plugin template created at: {plugin_dir}")

        elif args.command == "validate":
            if not args.plugin_path:
                print("--plugin-path required for validate command")
                exit(1)

            result = manager.validate_plugin(Path(args.plugin_path))
            print(
                json.dumps(
                    {
                        "plugin": result.plugin_name,
                        "status": result.overall_status,
                        "checks": result.checks,
                        "recommendations": result.recommendations,
                    },
                    indent=2,
                    default=str,
                )
            )

        elif args.command == "list":
            print("Available plugin types:")
            for plugin_type in PluginType:
                print(f"  {plugin_type.value}")

        elif args.command == "info":
            if not args.type:
                print("--type required for info command")
                exit(1)

            template_info = manager.templates.get(args.type, {})
            print(json.dumps(template_info, indent=2))

    except Exception as e:
        logger.error(f"Command failed: {e}")
        exit(1)
