"""
MLX Component Extraction Engine

Performs comprehensive analysis of existing ML system components
for production-ready extraction and template generation.

This is NOT simple file copying - it's sophisticated infrastructure analysis
for complex ML systems with databases, APIs, secrets, and deployment requirements.
"""

import ast
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
import re
import subprocess
import os
from collections import defaultdict


@dataclass
class ComponentDependency:
    """Represents a component dependency with environment requirements."""
    name: str
    type: str  # "package", "service", "api", "database"
    version: Optional[str] = None
    environment_variables: List[str] = None
    configuration_path: Optional[str] = None
    required_for: List[str] = None  # environments where this is required
    docker_image: Optional[str] = None

    def __post_init__(self):
        if self.environment_variables is None:
            self.environment_variables = []
        if self.required_for is None:
            self.required_for = ["development", "staging", "production"]


@dataclass 
class ComponentMetadata:
    """Complete component metadata for production deployment."""
    name: str
    description: str
    version: str
    component_type: str  # "infrastructure", "application", "utility"
    
    # File mappings
    source_files: List[str]
    config_files: List[str] 
    infrastructure_files: List[str]
    template_files: List[str]
    
    # Dependencies and requirements
    python_dependencies: List[str]
    system_dependencies: List[ComponentDependency]
    environment_variables: List[str]
    required_secrets: List[str]
    
    # Integration and compatibility
    injection_points: Dict[str, str]
    merge_strategies: Dict[str, str]
    compatibility_matrix: Dict[str, List[str]]
    conflicts_with: List[str]
    
    # Deployment and infrastructure
    docker_requirements: Optional[Dict[str, Any]]
    monitoring_endpoints: List[str]
    health_checks: List[str]

    def __post_init__(self):
        # Initialize lists if None
        for field in ['source_files', 'config_files', 'infrastructure_files', 'template_files',
                     'python_dependencies', 'system_dependencies', 'environment_variables', 
                     'required_secrets', 'monitoring_endpoints', 'health_checks', 'conflicts_with']:
            if getattr(self, field) is None:
                setattr(self, field, [])
        
        # Initialize dicts if None
        for field in ['injection_points', 'merge_strategies', 'compatibility_matrix']:
            if getattr(self, field) is None:
                setattr(self, field, {})


class ComponentAnalysisVisitor(ast.NodeVisitor):
    """AST visitor for detailed Python code analysis."""
    
    def __init__(self):
        self.imports = []
        self.endpoints = []
        self.env_vars = []
        self.external_calls = []
        self.classes = []
        self.functions = []
        self.decorators = []
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.classes.append({
            'name': node.name,
            'bases': [base.id if hasattr(base, 'id') else str(base) for base in node.bases],
            'decorators': [self._extract_decorator_name(dec) for dec in node.decorator_list]
        })
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        func_info = {
            'name': node.name,
            'decorators': [self._extract_decorator_name(dec) for dec in node.decorator_list],
            'is_endpoint': False
        }
        
        # Look for FastAPI route decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if hasattr(decorator.func, 'attr'):
                    if decorator.func.attr in ['get', 'post', 'put', 'delete', 'patch', 'options', 'head']:
                        self.endpoints.append(node.name)
                        func_info['is_endpoint'] = True
                        break
        
        self.functions.append(func_info)
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # Look for os.environ or os.getenv calls
        if isinstance(node.func, ast.Attribute):
            if (hasattr(node.func.value, 'id') and 
                node.func.value.id == 'os' and 
                node.func.attr in ['getenv', 'environ']):
                if node.args and isinstance(node.args[0], ast.Constant):
                    self.env_vars.append(node.args[0].value)
                elif node.args and isinstance(node.args[0], ast.Str):  # Python < 3.8 compatibility
                    self.env_vars.append(node.args[0].s)
        
        # Look for external API calls (requests, httpx, etc.)
        if isinstance(node.func, ast.Attribute):
            if hasattr(node.func.value, 'id'):
                if node.func.value.id in ['requests', 'httpx', 'aiohttp']:
                    self.external_calls.append({
                        'library': node.func.value.id,
                        'method': node.func.attr
                    })
        
        self.generic_visit(node)
    
    def _extract_decorator_name(self, decorator):
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        return str(decorator)


class ProductionComponentExtractor:
    """Extract production-ready components with full infrastructure analysis."""
    
    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analysis caches
        self.environment_vars = self._extract_environment_variables()
        self.docker_config = self._analyze_docker_setup()
        self.requirements = self._parse_requirements()
    
    def extract_all_components(self) -> Dict[str, ComponentMetadata]:
        """Extract all components with complete production metadata."""
        print("🔍 Starting comprehensive component extraction...")
        
        components = {}
        
        # Define component extraction mappings based on the implementation guide
        component_mappings = {
            "api-serving": {
                "source_paths": ["src/api"],
                "config_paths": ["conf/api", "conf/logging"], 
                "infrastructure_paths": ["docker", "scripts/api"],
                "type": "infrastructure",
                "description": "FastAPI-based production API server with security, monitoring, and scalability"
            },
            "config-management": {
                "source_paths": ["src/config"],
                "config_paths": ["conf"],
                "infrastructure_paths": [],
                "type": "infrastructure",
                "description": "Hydra + Pydantic configuration system with multi-environment support and secret management"
            },
            "plugin-registry": {
                "source_paths": ["src/plugins"],
                "config_paths": ["conf/plugins"],
                "infrastructure_paths": ["scripts/plugins"],
                "type": "application",
                "description": "Dynamic plugin discovery and loading system with security validation"
            },
            "data-processing": {
                "source_paths": ["src/data"],
                "config_paths": ["conf/data"],
                "infrastructure_paths": ["scripts/data"],
                "type": "application", 
                "description": "Data processing utilities and pipeline components"
            },
            "utilities": {
                "source_paths": ["src/utils"],
                "config_paths": [],
                "infrastructure_paths": [],
                "type": "utility",
                "description": "Common utility functions and helper modules"
            }
        }
        
        for component_name, mapping in component_mappings.items():
            print(f"📦 Extracting {component_name}...")
            metadata = self._extract_component(component_name, mapping)
            if metadata and metadata.source_files:  # Only include if has actual source files
                components[component_name] = metadata
                self._generate_component_template(metadata)
                print(f"✅ {component_name}: {len(metadata.source_files)} files, {len(metadata.python_dependencies)} dependencies")
            else:
                print(f"⚠️  {component_name}: No source files found, skipping")
        
        return components
    
    def _extract_component(self, name: str, mapping: Dict) -> Optional[ComponentMetadata]:
        """Extract a single component with full analysis."""
        
        # Analyze source code
        source_analysis = self._analyze_source_code(mapping["source_paths"])
        if not source_analysis["files"]:
            return None
            
        # Analyze configuration
        config_analysis = self._analyze_configuration(mapping["config_paths"])
        
        # Analyze infrastructure
        infra_analysis = self._analyze_infrastructure(mapping["infrastructure_paths"])
        
        # Extract dependencies
        dependencies = self._extract_component_dependencies(source_analysis, config_analysis)
        
        # Determine injection points and merge strategies
        injection_points = self._determine_injection_points(source_analysis, name)
        merge_strategies = self._determine_merge_strategies(config_analysis, name)
        
        # Build compatibility matrix
        compatibility = self._build_compatibility_matrix(name, dependencies)
        
        return ComponentMetadata(
            name=name,
            description=mapping.get("description", f"Production-ready {name} component"),
            version="1.0.0",
            component_type=mapping["type"],
            source_files=source_analysis["files"],
            config_files=config_analysis["files"],
            infrastructure_files=infra_analysis["files"],
            template_files=[],  # Will be generated
            python_dependencies=dependencies["python"],
            system_dependencies=dependencies["system"],
            environment_variables=dependencies["env_vars"],
            required_secrets=dependencies["secrets"],
            injection_points=injection_points,
            merge_strategies=merge_strategies,
            compatibility_matrix=compatibility["enhances"],
            conflicts_with=compatibility["conflicts"],
            docker_requirements=infra_analysis["docker"],
            monitoring_endpoints=source_analysis["endpoints"],
            health_checks=source_analysis["health_checks"]
        )
    
    def _analyze_source_code(self, source_paths: List[str]) -> Dict:
        """Comprehensive source code analysis using AST parsing."""
        analysis = {
            "files": [],
            "imports": set(),
            "endpoints": [],
            "health_checks": [],
            "env_var_usage": set(),
            "external_calls": [],
            "classes": [],
            "functions": []
        }
        
        for path_str in source_paths:
            path = self.source_dir / path_str
            if not path.exists():
                continue
                
            for py_file in path.rglob("*.py"):
                rel_path = str(py_file.relative_to(self.source_dir))
                analysis["files"].append(rel_path)
                
                # Parse AST for detailed analysis
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        tree = ast.parse(content)
                        visitor = ComponentAnalysisVisitor()
                        visitor.visit(tree)
                        
                        analysis["imports"].update(visitor.imports)
                        analysis["endpoints"].extend(visitor.endpoints)
                        analysis["env_var_usage"].update(visitor.env_vars)
                        analysis["external_calls"].extend(visitor.external_calls)
                        analysis["classes"].extend(visitor.classes)
                        analysis["functions"].extend(visitor.functions)
                        
                        # Look for health check patterns
                        if any(keyword in content.lower() for keyword in ['health', 'status', 'ping', 'ready']):
                            analysis["health_checks"].append(rel_path)
                        
                except (SyntaxError, UnicodeDecodeError, Exception) as e:
                    print(f"⚠️  Warning: Could not parse {py_file}: {e}")
                    continue
        
        # Convert sets back to lists for JSON serialization
        analysis["imports"] = list(analysis["imports"])
        analysis["env_var_usage"] = list(analysis["env_var_usage"])
        
        return analysis
    
    def _analyze_configuration(self, config_paths: List[str]) -> Dict:
        """Analyze Hydra configuration files and dependencies."""
        analysis = {
            "files": [],
            "env_vars": set(),
            "secrets": set(),
            "external_services": []
        }
        
        for path_str in config_paths:
            path = self.source_dir / path_str
            if not path.exists():
                continue
                
            for config_file in path.rglob("*.yaml"):
                rel_path = str(config_file.relative_to(self.source_dir))
                analysis["files"].append(rel_path)
                
                # Parse YAML for environment variables and secrets
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                        if config_data:
                            self._extract_config_dependencies(config_data, analysis)
                except yaml.YAMLError as e:
                    print(f"⚠️  Warning: Could not parse YAML {config_file}: {e}")
                    continue
        
        # Convert sets to lists
        analysis["env_vars"] = list(analysis["env_vars"])
        analysis["secrets"] = list(analysis["secrets"])
        
        return analysis
    
    def _extract_config_dependencies(self, config_data: Any, analysis: Dict):
        """Extract dependencies from configuration data."""
        if isinstance(config_data, dict):
            for key, value in config_data.items():
                if isinstance(value, str):
                    # Look for environment variable patterns: ${oc.env:VAR_NAME,default}
                    env_pattern = r'\$\{oc\.env:([^,}]+)'
                    matches = re.findall(env_pattern, value)
                    analysis["env_vars"].update(matches)
                    
                    # Look for secret patterns
                    if any(secret_word in key.lower() for secret_word in ['secret', 'key', 'token', 'password', 'credential']):
                        analysis["secrets"].add(key)
                        
                elif isinstance(value, (dict, list)):
                    self._extract_config_dependencies(value, analysis)
                    
        elif isinstance(config_data, list):
            for item in config_data:
                self._extract_config_dependencies(item, analysis)
    
    def _analyze_infrastructure(self, infra_paths: List[str]) -> Dict:
        """Analyze infrastructure files (Docker, scripts, etc.)."""
        analysis = {
            "files": [],
            "docker": {},
            "scripts": [],
            "monitoring": []
        }
        
        # Check for Docker configuration
        docker_files = ["Dockerfile", "docker-compose.yml", "docker/"]
        for docker_file in docker_files:
            docker_path = self.source_dir / docker_file
            if docker_path.exists():
                if docker_path.is_file():
                    analysis["files"].append(docker_file)
                    if docker_file == "Dockerfile":
                        analysis["docker"] = self._parse_dockerfile(docker_path)
                else:
                    # Directory - add all files
                    for f in docker_path.rglob("*"):
                        if f.is_file():
                            analysis["files"].append(str(f.relative_to(self.source_dir)))
        
        # Check infrastructure paths
        for path_str in infra_paths:
            path = self.source_dir / path_str
            if path.exists():
                for f in path.rglob("*"):
                    if f.is_file():
                        rel_path = str(f.relative_to(self.source_dir))
                        analysis["files"].append(rel_path)
                        
                        # Categorize files
                        if f.suffix in ['.sh', '.py'] and any(keyword in f.name.lower() for keyword in ['deploy', 'setup', 'install']):
                            analysis["scripts"].append(rel_path)
                        elif any(keyword in f.name.lower() for keyword in ['monitor', 'health', 'metrics']):
                            analysis["monitoring"].append(rel_path)
        
        return analysis
    
    def _parse_dockerfile(self, dockerfile_path: Path) -> Dict:
        """Parse Dockerfile for metadata."""
        docker_info = {
            "base_image": None,
            "ports": [],
            "volumes": [],
            "environment_files": []
        }
        
        try:
            with open(dockerfile_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('FROM '):
                        docker_info["base_image"] = line.split()[1]
                    elif line.startswith('EXPOSE '):
                        ports = line.split()[1:]
                        docker_info["ports"].extend(ports)
                    elif line.startswith('VOLUME '):
                        volume = line.split('[')[1].split(']')[0] if '[' in line else line.split()[1]
                        docker_info["volumes"].append(volume.strip('"'))
        except Exception as e:
            print(f"⚠️  Warning: Could not parse Dockerfile: {e}")
        
        return docker_info
    
    def _extract_component_dependencies(self, source_analysis: Dict, config_analysis: Dict) -> Dict:
        """Extract comprehensive component dependencies."""
        dependencies = {
            "python": [],
            "system": [],
            "env_vars": [],
            "secrets": []
        }
        
        # Extract Python dependencies from imports
        python_deps = set()
        for import_name in source_analysis["imports"]:
            # Map common imports to packages
            if import_name.startswith('fastapi'):
                python_deps.add("fastapi>=0.110.0")
            elif import_name.startswith('uvicorn'):
                python_deps.add("uvicorn[standard]>=0.30.0")
            elif import_name.startswith('pydantic'):
                python_deps.add("pydantic>=2.0.0")
            elif import_name.startswith('hydra'):
                python_deps.add("hydra-core>=1.3.0")
            elif import_name.startswith('redis'):
                python_deps.add("redis>=5.0.0")
                # Add system dependency
                dependencies["system"].append(ComponentDependency(
                    name="redis",
                    type="service",
                    environment_variables=["REDIS_URL"],
                    docker_image="redis:7-alpine"
                ))
            elif import_name.startswith('sqlalchemy'):
                python_deps.add("sqlalchemy>=2.0.0")
            elif import_name.startswith('psycopg'):
                python_deps.add("psycopg2-binary>=2.9.0")
                # Add database dependency
                dependencies["system"].append(ComponentDependency(
                    name="postgresql",
                    type="database", 
                    environment_variables=["DATABASE_URL", "DB_PASSWORD"],
                    required_for=["production"],
                    docker_image="postgres:15"
                ))
        
        dependencies["python"] = list(python_deps)
        dependencies["env_vars"] = list(set(source_analysis["env_var_usage"] + config_analysis["env_vars"]))
        dependencies["secrets"] = config_analysis["secrets"]
        
        return dependencies
    
    def _determine_injection_points(self, source_analysis: Dict, component_name: str) -> Dict:
        """Determine where and how to inject this component into other projects."""
        injection_points = {}
        
        # API serving specific injection points
        if component_name == "api-serving":
            for file_path in source_analysis["files"]:
                if "app.py" in file_path:
                    injection_points["app_creation"] = {
                        "file": file_path,
                        "function": "create_app",
                        "location": "after_initialization"
                    }
                    injection_points["route_registration"] = {
                        "file": file_path,
                        "location": "end_of_app_setup"
                    }
        
        # Config management injection points
        elif component_name == "config-management":
            injection_points["config_initialization"] = {
                "file": "src/config/manager.py",
                "class": "ConfigManager",
                "location": "class_definition"
            }
        
        return injection_points
    
    def _determine_merge_strategies(self, config_analysis: Dict, component_name: str) -> Dict:
        """Determine how to merge configurations when adding components."""
        strategies = {}
        
        for config_file in config_analysis["files"]:
            if "development" in config_file or "staging" in config_file:
                strategies[config_file] = "merge"
            elif "production" in config_file:
                strategies[config_file] = "enhance"
            else:
                strategies[config_file] = "replace"
        
        return strategies
    
    def _build_compatibility_matrix(self, component_name: str, dependencies: Dict) -> Dict:
        """Build compatibility matrix for component relationships."""
        compatibility = {
            "enhances": {},
            "conflicts": []
        }
        
        # Define known compatibility relationships
        compatibility_rules = {
            "api-serving": {
                "requires": ["config-management"],
                "enhances": ["plugin-registry", "monitoring"],
                "optional": ["caching", "rate-limiting", "authentication"]
            },
            "config-management": {
                "requires": [],
                "enhances": ["api-serving", "plugin-registry", "data-processing"],
                "optional": ["monitoring", "logging"]
            },
            "plugin-registry": {
                "requires": ["config-management"],
                "enhances": ["api-serving"],
                "optional": ["monitoring"]
            }
        }
        
        if component_name in compatibility_rules:
            compatibility["enhances"] = compatibility_rules[component_name]
        
        return compatibility
    
    def _generate_component_template(self, metadata: ComponentMetadata):
        """Generate component template files for injection."""
        component_dir = self.output_dir / metadata.name
        component_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate component metadata file
        metadata_file = component_dir / "component.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
        
        # Copy and templatize source files
        template_dir = component_dir / "templates"
        template_dir.mkdir(exist_ok=True)
        
        for source_file in metadata.source_files:
            source_path = self.source_dir / source_file
            if source_path.exists():
                self._templatize_file(source_path, template_dir, metadata)
        
        # Generate installation instructions
        self._generate_installation_guide(component_dir, metadata)
        
        print(f"📝 Generated templates for {metadata.name} in {component_dir}")
    
    def _templatize_file(self, source_path: Path, template_dir: Path, metadata: ComponentMetadata):
        """Convert source file to template with variable substitution."""
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Template variable substitutions
            template_content = content
            template_content = re.sub(r'mlx-platform-template', '{{PROJECT_NAME}}', template_content)
            template_content = re.sub(r'localhost:8000', '{{API_HOST}}:{{API_PORT}}', template_content)
            
            # Create template file
            template_file = template_dir / f"{source_path.name}.template"
            with open(template_file, 'w', encoding='utf-8') as f:
                f.write(template_content)
                
        except Exception as e:
            print(f"⚠️  Warning: Could not templatize {source_path}: {e}")
    
    def _generate_installation_guide(self, component_dir: Path, metadata: ComponentMetadata):
        """Generate installation guide for the component."""
        guide_content = f"""# {metadata.name} Component Installation Guide

## Description
{metadata.description}

## Dependencies

### Python Dependencies
```bash
{chr(10).join(f"uv add {dep}" for dep in metadata.python_dependencies)}
```

### System Dependencies
{chr(10).join(f"- {dep.name} ({dep.type})" for dep in metadata.system_dependencies)}

### Environment Variables
Required: {', '.join(metadata.environment_variables)}
Secrets: {', '.join(metadata.required_secrets)}

## Installation
```bash
./mlx add {metadata.name}
```

## Configuration
The component will be installed with the following merge strategies:
{chr(10).join(f"- {file}: {strategy}" for file, strategy in metadata.merge_strategies.items())}

## Monitoring
Health checks: {', '.join(metadata.health_checks)}
Endpoints: {', '.join(metadata.monitoring_endpoints)}
"""
        
        guide_file = component_dir / "README.md"
        with open(guide_file, 'w') as f:
            f.write(guide_content)
    
    def _extract_environment_variables(self) -> Dict:
        """Extract environment variables from .env files."""
        env_vars = {}
        env_files = [".env", ".env-example", ".env.development"]
        
        for env_file in env_files:
            env_path = self.source_dir / env_file
            if env_path.exists():
                try:
                    with open(env_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                env_vars[key] = value
                except Exception as e:
                    print(f"⚠️  Warning: Could not parse {env_file}: {e}")
        
        return env_vars
    
    def _analyze_docker_setup(self) -> Dict:
        """Analyze Docker configuration."""
        docker_info = {}
        
        # Check for docker-compose.yml
        compose_path = self.source_dir / "docker-compose.yml"
        if compose_path.exists():
            try:
                with open(compose_path, 'r') as f:
                    docker_info["compose"] = yaml.safe_load(f)
            except Exception as e:
                print(f"⚠️  Warning: Could not parse docker-compose.yml: {e}")
        
        return docker_info
    
    def _parse_requirements(self) -> List[str]:
        """Parse requirements.txt for dependency information."""
        requirements = []
        req_files = ["requirements.txt", "requirements-dev.txt"]
        
        for req_file in req_files:
            req_path = self.source_dir / req_file
            if req_path.exists():
                try:
                    with open(req_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                requirements.append(line)
                except Exception as e:
                    print(f"⚠️  Warning: Could not parse {req_file}: {e}")
        
        return requirements


def main():
    """Main extraction function."""
    source_dir = Path(".")
    output_dir = Path("mlx-components")
    
    print("🚀 MLX Component Extraction Engine v1.0.0")
    print("=" * 50)
    
    extractor = ProductionComponentExtractor(source_dir, output_dir)
    components = extractor.extract_all_components()
    
    if not components:
        print("❌ No components found to extract")
        return
    
    # Generate master registry
    registry = {
        "$schema": "https://mlx.dev/registry.schema.json",
        "version": "1.0.0",
        "metadata": {
            "generator": "MLX Component Extractor v1.0.0",
            "extracted_from": "mlx-platform-template",
            "extraction_date": "2024-12-17",
            "total_components": len(components)
        },
        "components": {name: asdict(meta) for name, meta in components.items()},
        "installation_order": ["config-management", "api-serving", "plugin-registry", "data-processing", "utilities"]
    }
    
    # Write registry
    registry_path = output_dir / "registry.json"
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2, default=str)
    
    print("=" * 50)
    print(f"✅ Extraction complete! Generated {len(components)} production-ready components:")
    for name, meta in components.items():
        print(f"   📦 {name}: {len(meta.source_files)} files, {len(meta.python_dependencies)} dependencies")
    
    print(f"\n📋 Registry saved to: {registry_path}")
    print(f"🎯 Components ready for: ./mlx add <component-name>")


if __name__ == "__main__":
    main() 