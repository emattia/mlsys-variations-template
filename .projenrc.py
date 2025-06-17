#!/usr/bin/env python3
"""
MLX Foundation Project Configuration

Projen-driven configuration for the MLX composable ML platform foundation.
Manages dependencies, tasks, CI/CD, and development workflows.

IMPORTANT: This configuration maintains the existing repository structure
and uses UV (not Poetry) for package management.
"""

from projen.python import PythonProject
from projen import Project, ProjectType
from typing import List, Dict, Any
import os

class MLXProject(PythonProject):
    """
    Custom projen project type for MLX Foundation.
    
    Extends PythonProject with ML-specific configurations while preserving
    the existing repository structure:
    - src/ directory layout (api/, config/, plugins/, etc.)
    - Existing UV-based dependency management
    - Current tooling integration (Makefile, docker-compose, etc.)
    - Documentation organization in docs/
    
    Key Features:
    - AI/ML dependencies for compatibility prediction
    - Component extraction and management
    - Smart testing infrastructure  
    - Security scanning and hardening
    - Production-ready FastAPI setup
    """
    
    def __init__(self, **kwargs):
        # Core MLX project configuration - maintaining existing structure
        super().__init__(
            name="mlx-foundation",
            module_name="mlx_foundation",  # Updated to match existing src layout
            author_name="MLX Team",
            author_email="team@mlx.dev",
            version="0.1.0",
            description="AI-enhanced composable ML platform foundation",
            
            # Preserve existing directory structure
            sample=False,  # Don't generate sample files that might conflict
            
            # Core ML dependencies - matching existing requirements
            deps=[
                "fastapi>=0.110.0",
                "uvicorn[standard]>=0.30.0", 
                "pydantic>=2.5.0",
                "pydantic-settings>=2.1.0",
                "hydra-core>=1.3.2",
                "typer[all]>=0.9.0",
                "rich>=13.5.2",
                "click>=8.1.3",
                "httpx>=0.27.0",
                "jinja2>=3.1.2",
                "pyyaml>=6.0",
                "joblib>=1.3.2",
                
                # Scientific computing - aligned with existing
                "numpy>=1.26.4",
                "scikit-learn>=1.5.0", 
                "matplotlib>=3.9.0",
                "polars>=1.30.0",
                
                # AI/LLM integration  
                "openai>=1.0.0",
                "anthropic>=0.5.0",
                
                # Async support
                "asgi-lifespan>=2.1.0",
            ],
            
            # Development and testing dependencies - using UV not Poetry
            dev_deps=[
                "pytest>=7.4.3",  # Match existing version to avoid conflicts
                "pytest-cov>=5.0.0",
                "pytest-asyncio>=0.23.7",
                "ruff>=0.3.2",
                "mypy>=1.8.0",
                "bandit[toml]>=1.7.8",
                "radon>=5.1.0",
                
                # Documentation - matching existing mkdocs setup
                "mkdocs>=1.5.2",
                "mkdocs-material>=9.2.8",
                "mkdocstrings[python]>=0.25.1",
                "pymdown-extensions>=10.2.1",
                "nbdoc>=0.0.8",
            ],
            
            **kwargs
        )
        
        # Add MLX-specific configurations without disrupting existing structure
        self.mlx_components = []
        self.setup_ml_configuration()
        self.setup_component_framework()
        self.setup_ai_enhanced_testing()
        self.setup_development_tasks()
        self.setup_security_and_quality()
        self.setup_documentation_organization()
        
    def setup_ml_configuration(self):
        """Configure ML-specific project settings while preserving existing layout."""
        
        # Preserve existing pytest configuration
        if hasattr(self.pytest, 'add_testpath'):
            self.pytest.add_testpath("tests/unit")
            self.pytest.add_testpath("tests/integration") 
            self.pytest.add_testpath("tests/contracts")
        
        # Configure pytest for async ML operations
        if hasattr(self.pytest, 'add_option'):
            self.pytest.add_option("asyncio_mode = auto")
            self.pytest.add_option("testpaths = ['tests']")
            self.pytest.add_option("python_files = ['test_*.py', '*_test.py']")
            self.pytest.add_option("python_classes = ['Test*']")
            self.pytest.add_option("python_functions = ['test_*']")
        
        # Preserve existing .gitignore patterns while adding ML-specific ones
        ml_ignore_patterns = [
            "*.pkl", "*.joblib",  # Model files
            "data/raw/", "data/processed/",  # Data directories
            "models/artifacts/",  # Model artifacts (preserve existing models/ structure)
            "reports/generated/",  # Generated reports (preserve existing reports/ structure)
            "logs/",  # Log files
        ]
        
        for pattern in ml_ignore_patterns:
            self.add_git_ignore(pattern)
        
    def setup_component_framework(self):
        """Setup MLX component extraction and management."""
        
        # Component validation task
        self.add_task(
            "mlx:validate-components", 
            exec="python scripts/mlx/validate_components.py --check-existing",
            description="Validate MLX component integrity and compatibility"
        )
        
        # Component synthesis task - preserve existing structure
        self.add_task(
            "mlx:synth-components",
            exec="python scripts/mlx/synth_components.py --preserve-structure", 
            description="Synthesize components while preserving existing layout"
        )
        
    def setup_ai_enhanced_testing(self):
        """Setup AI-enhanced testing capabilities using existing test structure."""
        
        # Smart test selection - work with existing tests/
        self.add_task(
            "test:smart",
            exec="python -m pytest tests/ --collect-only | python scripts/mlx/smart_test_selector.py",
            description="Run AI-selected optimal test suite from existing tests"
        )
        
        # Compatibility testing - integrate with existing plugin tests
        self.add_task(
            "test:compatibility",
            exec="python scripts/mlx/compatibility_tester.py --test-plugins src/plugins/",
            description="Test component compatibility with existing plugins"
        )
        
        # Performance benchmarking - use existing structure
        self.add_task(
            "test:benchmark",
            exec="python -m pytest tests/ -m benchmark --benchmark-only",
            description="Run performance benchmarks for existing components"
        )
        
    def setup_development_tasks(self):
        """Setup development tasks and workflows with MLX enhancement."""
        
        # Standard MLX development tasks
        self.add_task("mlx:status", exec="python -c \"print('📊 MLX Project Status'); import json; config = json.load(open('mlx.config.json')); print(f'Components: {config[\\\"platform\\\"][\\\"components\\\"]}'); print('✅ Project structure preserved')\"")
        
        # Add Phase 2 MLX component extraction and management tasks
        self.add_task("mlx:extract-components", 
                     exec="python scripts/mlx/component_extractor.py",
                     description="Extract production-ready components from existing source code")
        
        self.add_task("mlx:add", 
                     exec="python scripts/mlx/component_injector.py",
                     description="Add a component to the current project",
                     args=["COMPONENT_NAME"])
        
        self.add_task("mlx:list", 
                     exec="python -c \"import sys; sys.path.append('scripts/mlx'); from component_injector import ComponentInjector; from pathlib import Path; injector = ComponentInjector(Path('.'), Path('mlx-components')); components = injector.list_available_components(); [print(f'📦 {c[\\\"name\\\"]}: {c[\\\"description\\\"]}') for c in components]\"",
                     description="List available MLX components")
        
        self.add_task("mlx:info",
                     exec="python -c \"import sys; sys.path.append('scripts/mlx'); from component_injector import ComponentInjector; from pathlib import Path; import json; injector = ComponentInjector(Path('.'), Path('mlx-components')); info = injector.get_component_info(sys.argv[1]); print(json.dumps(info, indent=2)) if info else print('Component not found')\"",
                     description="Get detailed information about a component",
                     args=["COMPONENT_NAME"])
        
        self.add_task("mlx:validate-registry",
                     exec="python -c \"import json; registry = json.load(open('mlx-components/registry.json')); print(f'✅ Registry valid: {len(registry[\\\"components\\\"])} components')\"",
                     description="Validate component registry")
        
        # API development - use existing FastAPI app
        self.add_task(
            "api:dev",
            exec="uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000",
            description="Start existing FastAPI development server with auto-reload"
        )
        
        self.add_task(
            "api:prod",
            exec="uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4",
            description="Start production FastAPI server"
        )
        
        # Documentation tasks - use existing mkdocs.yml
        self.add_task(
            "docs:serve",
            exec="mkdocs serve",
            description="Serve documentation locally using existing mkdocs config"
        )
        
        self.add_task(
            "docs:build",
            exec="mkdocs build",
            description="Build static documentation"
        )
        
        # Preserve existing Makefile integration
        self.add_task(
            "make:test",
            exec="make test",
            description="Run tests using existing Makefile"
        )
        
        self.add_task(
            "make:lint",
            exec="make lint",
            description="Run linting using existing Makefile"
        )
        
    def setup_security_and_quality(self):
        """Setup security scanning and code quality tasks."""
        
        # Code quality tasks - integrate with existing tools
        self.add_task(
            "lint",
            exec="ruff check src/ tests/",
            description="Run linting checks with ruff on existing structure"
        )
        
        self.add_task(
            "format",
            exec="ruff format src/ tests/ && ruff check --fix src/ tests/",
            description="Format code and fix linting issues"
        )
        
        self.add_task(
            "type-check",
            exec="mypy src/",
            description="Run type checking with mypy on source code"
        )
        
        # Security scanning - use existing .trivyignore
        self.add_task(
            "security:bandit",
            exec="bandit -r src/ -f json",
            description="Run security checks with bandit"
        )
        
        self.add_task(
            "security:trivy",
            exec="trivy fs --scanners vuln,secret,misconfig --severity HIGH,CRITICAL .",
            description="Run vulnerability scanning with trivy (uses existing .trivyignore)"
        )
        
        # Comprehensive quality check
        self.add_task(
            "quality:all",
            exec="projen lint && projen format && projen type-check && projen security:bandit",
            description="Run all quality checks"
        )
        
        # Pre-commit setup - integrate with existing .pre-commit-config.yaml
        self.add_task(
            "pre-commit:install",
            exec="pre-commit install",
            description="Install pre-commit hooks (uses existing config)"
        )
        
    def setup_documentation_organization(self):
        """Setup proper documentation organization."""
        
        # Task to move root documentation files
        self.add_task(
            "docs:organize",
            exec="python scripts/mlx/organize_docs.py",
            description="Organize documentation files into docs/ directory structure"
        )
        
        # Task to update documentation links
        self.add_task(
            "docs:update-links",
            exec="python scripts/mlx/update_doc_links.py", 
            description="Update documentation cross-references after reorganization"
        )
        
    def add_mlx_component(self, component_id: str, config: Dict[str, Any] = None):
        """
        Add MLX component to project with projen synthesis.
        
        Args:
            component_id: Unique identifier for the component
            config: Component-specific configuration
        """
        component = {
            "id": component_id,
            "config": config or {},
            "templates": [],
            "dependencies": [],
            "source_path": f"src/{component_id.replace('-', '_')}"  # Map to existing src structure
        }
        
        self.mlx_components.append(component)
        
        # Add component-specific tasks that respect existing structure
        self.add_task(
            f"mlx:update-{component_id}",
            exec=f"python scripts/mlx/update_component.py --component {component_id} --source src/",
            description=f"Update {component_id} component from existing source"
        )
        
        self.add_task(
            f"mlx:test-{component_id}",
            exec=f"python -m pytest tests/ -k {component_id}",
            description=f"Test {component_id} component using existing tests"
        )
        
        # Component will be synthesized during projen synth
        print(f"Added MLX component: {component_id} (mapped to existing structure)")
        
    def post_synthesize(self):
        """Post-synthesis hook for MLX-specific file generation."""
        super().post_synthesize()
        
        # Generate MLX-specific configuration files
        self._generate_mlx_config()
        self._generate_cursor_rules()
        self._generate_component_registry()
        self._create_documentation_organization_script()
        
    def _generate_mlx_config(self):
        """Generate MLX platform configuration."""
        mlx_config = {
            "platform": {
                "name": "mlx-foundation",
                "version": "0.1.0",
                "components": [comp["id"] for comp in self.mlx_components],
                "package_manager": "uv",  # Explicitly specify UV not Poetry
                "existing_structure_preserved": True
            },
            "ai": {
                "compatibility_prediction": True,
                "smart_test_selection": True,
                "optimization_suggestions": True
            },
            "development": {
                "auto_component_synthesis": True,
                "intelligent_cursor_rules": True,
                "ai_enhanced_ci_cd": True,
                "preserve_existing_tooling": True
            },
            "paths": {
                "source": "src/",
                "tests": "tests/",
                "docs": "docs/",
                "config": "conf/",
                "scripts": "scripts/"
            }
        }
        
        # Write MLX configuration
        import json
        with open("mlx.config.json", "w") as f:
            json.dump(mlx_config, f, indent=2)
            
    def _generate_cursor_rules(self):
        """Generate intelligent Cursor rules based on existing project structure."""
        
        cursor_rules = """# MLX Foundation - Auto-generated Cursor Rules

## Project Structure (Preserved from Original)
This is an MLX Foundation project using projen for configuration management while maintaining the original repository structure.

### Core Directories (Existing Structure Maintained)
- `src/api/`: FastAPI application components  
- `src/config/`: Configuration management
- `src/plugins/`: Plugin architecture and registry
- `src/cli/`: Command-line interface
- `src/utils/`: Utility modules
- `src/models/`: Model definitions
- `src/data/`: Data processing modules
- `tests/`: Multi-layered testing (unit/integration/contracts)
- `conf/`: Hydra configuration files
- `docs/`: Comprehensive documentation
- `scripts/`: Development and automation scripts
- `tools/`: Development tools and utilities

### Package Management
- **Primary**: UV (not Poetry) - extremely fast Python package installer
- **Lock File**: `uv.lock` contains exact dependency versions
- **Requirements**: Generated by projen but compatible with existing UV workflow

### MLX Components (Extracted from Existing Structure)
""" + "\n".join([f"- {comp['id']}: Mapped from {comp.get('source_path', 'src/')}" 
                 for comp in self.mlx_components])

        cursor_rules += """

### Development Patterns
- Use `projen` for all project configuration (preserves existing structure)
- Components are AI-enhanced with compatibility prediction
- Tests use smart selection for efficiency
- Configuration is hierarchical with Hydra + Pydantic (existing pattern)
- UV package management for all dependencies

### Common Commands (Preserved & Enhanced)
- `projen` - Synthesize project configuration
- `projen test:smart` - Run AI-selected tests from existing test suite
- `projen mlx:status` - Check project health 
- `projen api:dev` - Start existing FastAPI development server
- `make test` - Use existing Makefile (preserved)
- `uv pip install package` - Add dependencies with UV

### Existing Tooling Integration
- Makefile: Preserved existing tasks, enhanced with projen
- Docker: Existing Dockerfile and docker-compose.yml maintained
- CI/CD: GitHub Actions workflows preserved
- Pre-commit: Existing .pre-commit-config.yaml maintained
- MkDocs: Existing documentation system enhanced
"""
        
        # Ensure .cursor/rules directory exists
        os.makedirs(".cursor/rules", exist_ok=True)
        
        with open(".cursor/rules/mlx_foundation.md", "w") as f:
            f.write(cursor_rules)
            
    def _generate_component_registry(self):
        """Generate component registry for MLX platform."""
        
        # Ensure mlx-components directory exists (new for MLX)
        os.makedirs("mlx-components", exist_ok=True)
        
        registry = {
            "registry_version": "1.0.0",
            "source_structure_preserved": True,
            "package_manager": "uv",
            "components": {},
            "compatibility_matrix": {},
            "ai_metadata": {
                "last_updated": "auto-generated",
                "prediction_model_version": "1.0.0",
                "existing_source_mapped": True
            }
        }
        
        for component in self.mlx_components:
            registry["components"][component["id"]] = {
                "version": "1.0.0",
                "type": "mlx-component",
                "config": component["config"],
                "dependencies": component["dependencies"],
                "compatibility_score": 1.0,  # Will be updated by AI
                "templates": component["templates"],
                "source_path": component.get("source_path", "src/"),
                "extracted_from_existing": True
            }
        
        # Write component registry
        import json
        with open("mlx-components/registry.json", "w") as f:
            json.dump(registry, f, indent=2)
    
    def _create_documentation_organization_script(self):
        """Create script to organize documentation files."""
        
        # Ensure scripts/mlx directory exists
        os.makedirs("scripts/mlx", exist_ok=True)
        
        organize_script = '''#!/usr/bin/env python3
"""
Documentation Organization Script for MLX Foundation

Moves root-level documentation files to appropriate docs/ subdirectories
while maintaining proper cross-references and updating links.
"""

import os
import shutil
from pathlib import Path

def organize_documentation():
    """Move documentation files to proper locations."""
    
    # Define file moves
    moves = [
        ("ANALYSIS_AND_RECOMMENDATIONS.md", "docs/project-analysis.md"),
        ("BRANCHING_STRATEGY.md", "docs/development/branching-strategy.md"),
        # Add more as needed
    ]
    
    for source, destination in moves:
        if os.path.exists(source):
            # Ensure destination directory exists
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(source, destination)
            print(f"Moved {source} -> {destination}")
        else:
            print(f"Source file not found: {source}")

if __name__ == "__main__":
    organize_documentation()
'''
        
        with open("scripts/mlx/organize_docs.py", "w") as f:
            f.write(organize_script)
            
        # Make script executable
        os.chmod("scripts/mlx/organize_docs.py", 0o755)


# Initialize MLX Foundation project with existing structure preservation
project = MLXProject()

# Add initial MLX components based on existing src/ structure
project.add_mlx_component("api-serving", {
    "framework": "fastapi",
    "port": 8000,
    "workers": 4,
    "security_enabled": True,
    "existing_source": "src/api/"
})

project.add_mlx_component("config-management", {
    "framework": "hydra",
    "validation": "pydantic", 
    "hierarchical": True,
    "existing_source": "src/config/"
})

project.add_mlx_component("plugin-registry", {
    "discovery": "automatic",
    "hot_reload": True,
    "contract_testing": True,
    "existing_source": "src/plugins/"
})

# Synthesize the project while preserving existing structure
project.synth() 