# Phase 2 Implementation Guide: MLX Component Extraction Framework

> **For Next Chat Agent** | **Phase 2**: Component Extraction  
> **Estimated Duration**: 4-6 hours | **Complexity**: HIGH | **Priority**: CRITICAL

## 🎯 **Mission: Build ML Platform Component System**

You are implementing **Phase 2** of the MLX Foundation - a **comprehensive ML platform component system**. This is NOT simply "shadcn for ML" - it's a sophisticated system for composing production-ready ML infrastructures with complex dependencies, secrets management, and deep system integration.

---

## 🏗️ **MLX vs Shadcn: Critical Differences**

### **Why MLX Components Are More Complex**

| Aspect | Shadcn UI | MLX ML Components |
|--------|-----------|-------------------|
| **Scope** | Single React component | Complete ML subsystem |
| **Files** | 1-3 component files | 10+ files across multiple directories |
| **Dependencies** | NPM packages | Databases, APIs, cloud services |
| **Configuration** | Component props | Multi-environment configs, secrets |
| **Infrastructure** | None | Docker, monitoring, CI/CD |
| **Integration** | Import/export | Deep configuration merging |
| **Secrets** | None | API keys, database credentials |
| **Runtime** | Browser only | Local dev + cloud deployment |

### **Example Component Complexity**

**Shadcn Button Component:**
```
components/ui/
└── button.tsx          # Single file, simple props
```

**MLX API-Serving Component:**
```
src/api/                 # Application code
├── app.py              # FastAPI application
├── middleware.py       # Security, rate limiting
├── models.py           # Request/response models
└── monitoring.py       # Health checks, metrics

conf/api/               # Configuration
├── development.yaml    # Dev environment
├── staging.yaml        # Staging environment 
└── production.yaml     # Production settings

scripts/api/            # Infrastructure
├── deploy.sh          # Deployment scripts
└── health_check.py    # Monitoring scripts

docker/                 # Containerization
├── api.Dockerfile     # API container
└── nginx.conf         # Load balancer config

.env-example           # Required environment variables
├── API_KEY=            # External service credentials
├── DATABASE_URL=       # Database connection
└── REDIS_URL=          # Cache connection
```

---

## 📋 **What Phase 1 Established (Context)**

### ✅ **MLX Foundation Architecture**
- **Fork-Based Model**: Teams fork MLX foundation, customize with projen
- **Single CLI Gateway**: `./mlx` handles all operations (NOT global tool)
- **Unified Configuration**: `conf/` directory with Hydra-based management
- **Projen-Driven**: Infrastructure as code with intelligent synthesis
- **Production-Ready**: Docker, CI/CD, monitoring, security built-in

### ✅ **Repository Structure Ready for Phase 2**
```
mlx-platform-template/           # ← MLX Foundation (template)
├── src/                            # ← Extract components FROM here
│   ├── api/                        # → api-serving component
│   ├── config/                     # → config-management component
│   ├── plugins/                    # → plugin-registry component
│   └── utils/                      # → Various utility components
├── conf/                           # ← Hydra configurations
├── mlx-components/                 # ← Component templates HERE
├── scripts/mlx/                    # ← MLX tooling
└── .projenrc.py                    # ← Projen configuration
```

---

## 🎯 **Your Phase 2 Objectives: Component Extraction Engine**

### **2.1 Intelligent Component Extraction** (CRITICAL)
Build `scripts/mlx/extract_component.py` that:
- **Analyzes existing `src/` code** using AST parsing and dependency analysis
- **Extracts complete component packages** (code + config + infrastructure)
- **Handles complex dependencies** (databases, APIs, external services)
- **Manages secrets and environment variables** across components
- **Creates mergeable templates** for intelligent component injection

### **2.2 Production-Ready Component Registry** (CRITICAL)  
Build `mlx-components/registry.json` that:
- **Catalogs infrastructure requirements** (databases, APIs, cloud services)
- **Defines environment variable dependencies** (API keys, credentials)
- **Specifies component compatibility matrix** (which components work together)
- **Handles configuration merging strategies** (replace, merge, enhance)
- **Supports multi-environment deployment** (dev, staging, production)

---

## 📁 **Component Extraction Targets (Production Systems)**

### **🚀 API-Serving Component** (`src/api/` → `mlx-components/api-serving/`)
**Complexity**: HIGH | **Dependencies**: Database, Redis, Monitoring

**Extract These Elements:**
```
Source Analysis:
├── src/api/app.py              # FastAPI application entry point
├── src/api/middleware.py       # Security, CORS, rate limiting
├── src/api/models.py          # Pydantic request/response models
├── src/api/auth.py            # Authentication & authorization
├── src/api/monitoring.py      # Health checks, metrics collection
└── src/api/routes/            # API route definitions

Configuration Analysis:
├── conf/api/development.yaml   # Development environment settings
├── conf/api/staging.yaml      # Staging environment configuration
├── conf/api/production.yaml   # Production settings (workers, timeouts)
└── conf/logging/api.yaml      # API-specific logging configuration

Infrastructure Analysis:
├── docker/api.Dockerfile      # Container configuration
├── docker-compose.yml         # Local development stack
├── .env-example              # Required environment variables
└── scripts/api/              # Deployment and maintenance scripts

Dependency Analysis:
├── Database: PostgreSQL/SQLite
├── Cache: Redis
├── Monitoring: Prometheus + Grafana  
├── Secrets: API keys, database credentials
└── External APIs: Rate limiting, authentication services
```

### **⚙️ Config-Management Component** (`src/config/` → `mlx-components/config-management/`)
**Complexity**: MEDIUM | **Dependencies**: Hydra, Pydantic, Environment Variables

**Extract These Elements:**
```
Source Analysis:
├── src/config/manager.py       # Configuration loading and validation
├── src/config/models.py        # Pydantic configuration models
├── src/config/secrets.py       # Secure secret handling
└── src/config/environments.py  # Environment-specific overrides

Configuration Analysis:
├── conf/config.yaml           # Main configuration entry point
├── conf/*/                    # All configuration directories
└── .env-example              # Environment variable template

Integration Points:
├── Environment variable injection
├── Secret management integration
├── Multi-environment support
└── Configuration validation pipeline
```

### **🔌 Plugin-Registry Component** (`src/plugins/` → `mlx-components/plugin-registry/`)
**Complexity**: HIGH | **Dependencies**: Dynamic Loading, Component Registry

**Extract These Elements:**
```
Source Analysis:
├── src/plugins/registry.py     # Plugin discovery and loading
├── src/plugins/interface.py    # Plugin interface contracts
├── src/plugins/loader.py       # Dynamic plugin loading
└── src/plugins/validators.py   # Plugin validation and security

Plugin System Features:
├── Dynamic plugin discovery
├── Plugin lifecycle management
├── Security validation
├── Dependency injection
└── Plugin communication protocols
```

---

## 🔧 **Implementation Strategy: Production-Grade Extraction**

### **Step 1: Multi-Dimensional Source Analysis**

```python
# scripts/mlx/extract_component.py
"""
MLX Component Extraction Engine

Performs comprehensive analysis of existing ML system components
for production-ready extraction and template generation.
"""

import ast
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
import re
import subprocess

@dataclass
class ComponentDependency:
    """Represents a component dependency with environment requirements."""
    name: str
    type: str  # "package", "service", "api", "database"
    version: Optional[str] = None
    environment_variables: List[str] = None
    configuration_path: Optional[str] = None
    required_for: List[str] = None  # environments where this is required

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
    docker_requirements: Optional[Dict[str, str]]
    monitoring_endpoints: List[str]
    health_checks: List[str]

class ProductionComponentExtractor:
    """Extract production-ready components with full infrastructure analysis."""
    
    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.environment_vars = self._extract_environment_variables()
        self.docker_config = self._analyze_docker_setup()
    
    def extract_all_components(self) -> Dict[str, ComponentMetadata]:
        """Extract all components with complete production metadata."""
        components = {}
        
        # Define component extraction mappings
        component_mappings = {
            "api-serving": {
                "source_paths": ["src/api"],
                "config_paths": ["conf/api", "conf/logging"], 
                "infrastructure_paths": ["docker", "scripts/api"],
                "type": "infrastructure"
            },
            "config-management": {
                "source_paths": ["src/config"],
                "config_paths": ["conf"],
                "infrastructure_paths": [],
                "type": "infrastructure"
            },
            "plugin-registry": {
                "source_paths": ["src/plugins"],
                "config_paths": ["conf/plugins"],
                "infrastructure_paths": ["scripts/plugins"],
                "type": "application" 
            }
        }
        
        for component_name, mapping in component_mappings.items():
            metadata = self._extract_component(component_name, mapping)
            if metadata:
                components[component_name] = metadata
                self._generate_component_template(metadata)
        
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
        dependencies = self._extract_component_dependencies(source_analysis)
        
        # Determine injection points and merge strategies
        injection_points = self._determine_injection_points(source_analysis, name)
        merge_strategies = self._determine_merge_strategies(config_analysis, name)
        
        # Build compatibility matrix
        compatibility = self._build_compatibility_matrix(name, dependencies)
        
        return ComponentMetadata(
            name=name,
            description=f"Production-ready {name} component",
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
            "imports": [],
            "endpoints": [],
            "health_checks": [],
            "env_var_usage": [],
            "external_calls": []
        }
        
        for path_str in source_paths:
            path = self.source_dir / path_str
            if not path.exists():
                continue
                
            for py_file in path.rglob("*.py"):
                analysis["files"].append(str(py_file.relative_to(self.source_dir)))
                
                # Parse AST for detailed analysis
                with open(py_file) as f:
                    try:
                        tree = ast.parse(f.read())
                        visitor = ComponentAnalysisVisitor()
                        visitor.visit(tree)
                        
                        analysis["imports"].extend(visitor.imports)
                        analysis["endpoints"].extend(visitor.endpoints)
                        analysis["env_var_usage"].extend(visitor.env_vars)
                        analysis["external_calls"].extend(visitor.external_calls)
                        
                    except SyntaxError:
                        continue  # Skip files with syntax errors
        
        return analysis
    
    def _analyze_configuration(self, config_paths: List[str]) -> Dict:
        """Analyze Hydra configuration files and dependencies."""
        analysis = {
            "files": [],
            "env_vars": [],
            "secrets": [],
            "external_services": []
        }
        
        for path_str in config_paths:
            path = self.source_dir / path_str
            if not path.exists():
                continue
                
            for config_file in path.rglob("*.yaml"):
                analysis["files"].append(str(config_file.relative_to(self.source_dir)))
                
                # Parse YAML for environment variables and secrets
                with open(config_file) as f:
                    try:
                        config_data = yaml.safe_load(f)
                        self._extract_config_dependencies(config_data, analysis)
                    except yaml.YAMLError:
                        continue
        
        return analysis
    
    def _extract_config_dependencies(self, config_data: Dict, analysis: Dict):
        """Extract dependencies from configuration data."""
        if not isinstance(config_data, dict):
            return
            
        for key, value in config_data.items():
            if isinstance(value, str):
                # Look for environment variable patterns: ${oc.env:VAR_NAME,default}
                env_pattern = r'\$\{oc\.env:([^,}]+)'
                matches = re.findall(env_pattern, value)
                analysis["env_vars"].extend(matches)
                
                # Look for secret patterns
                if any(secret_word in key.lower() for secret_word in ['secret', 'key', 'token', 'password']):
                    analysis["secrets"].append(key)
                    
            elif isinstance(value, dict):
                self._extract_config_dependencies(value, analysis)
    
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
                else:
                    analysis["files"].extend([
                        str(f.relative_to(self.source_dir)) 
                        for f in docker_path.rglob("*") if f.is_file()
                    ])
        
        return analysis
    
    def _generate_component_template(self, metadata: ComponentMetadata):
        """Generate component template files for injection."""
        component_dir = self.output_dir / metadata.name
        component_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate component metadata file
        metadata_file = component_dir / "component.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Copy and templatize source files
        for source_file in metadata.source_files:
            source_path = self.source_dir / source_file
            if source_path.exists():
                self._templatize_file(source_path, component_dir, metadata)
        
        # Generate installation instructions
        self._generate_installation_guide(component_dir, metadata)

class ComponentAnalysisVisitor(ast.NodeVisitor):
    """AST visitor for detailed Python code analysis."""
    
    def __init__(self):
        self.imports = []
        self.endpoints = []
        self.env_vars = []
        self.external_calls = []
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
    
    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)
    
    def visit_FunctionDef(self, node):
        # Look for FastAPI route decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if hasattr(decorator.func, 'attr'):
                    if decorator.func.attr in ['get', 'post', 'put', 'delete']:
                        self.endpoints.append(node.name)
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # Look for os.environ or os.getenv calls
        if isinstance(node.func, ast.Attribute):
            if (hasattr(node.func.value, 'id') and 
                node.func.value.id == 'os' and 
                node.func.attr in ['getenv', 'environ']):
                if node.args and isinstance(node.args[0], ast.Str):
                    self.env_vars.append(node.args[0].s)
        
        self.generic_visit(node)

# Usage
if __name__ == "__main__":
    extractor = ProductionComponentExtractor(
        source_dir=Path("src"),
        output_dir=Path("mlx-components")
    )
    
    components = extractor.extract_all_components()
    
    # Generate master registry
    registry = {
        "version": "1.0.0", 
        "schema": "https://mlx.dev/registry.schema.json",
        "components": {name: asdict(meta) for name, meta in components.items()}
    }
    
    with open("mlx-components/registry.json", 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ Extracted {len(components)} production-ready components")
```

### **Step 2: Production Registry with Infrastructure Awareness**

```json
// mlx-components/registry.json
{
  "$schema": "https://mlx.dev/registry.schema.json",
  "version": "1.0.0",
  "metadata": {
    "generator": "MLX Component Extractor v1.0.0",
    "extracted_from": "mlx-platform-template",
    "extraction_date": "2024-12-17",
    "total_components": 6
  },
  
  "components": {
    "api-serving": {
      "name": "Production API Server",
      "description": "FastAPI-based production API server with security, monitoring, and scalability",
      "version": "1.0.0",
      "type": "infrastructure",
      
      "files": [
        {
          "source": "mlx-components/api-serving/app.py.template",
          "target": "src/api/app.py",
          "merge_strategy": "enhance",
          "template_variables": ["{{PROJECT_NAME}}", "{{API_PREFIX}}"]
        },
        {
          "source": "mlx-components/api-serving/middleware.py",
          "target": "src/api/middleware.py", 
          "merge_strategy": "replace"
        },
        {
          "source": "mlx-components/api-serving/development.yaml",
          "target": "conf/api/development.yaml",
          "merge_strategy": "merge"
        }
      ],
      
      "dependencies": {
        "python": [
          "fastapi>=0.110.0",
          "uvicorn[standard]>=0.30.0",
          "pydantic>=2.0.0",
          "python-multipart>=0.0.6"
        ],
        "system": [
          {
            "name": "redis",
            "type": "service",
            "required_for": ["production", "staging"],
            "environment_variables": ["REDIS_URL"],
            "docker_image": "redis:7-alpine"
          },
          {
            "name": "postgresql", 
            "type": "database",
            "required_for": ["production"],
            "environment_variables": ["DATABASE_URL", "DB_PASSWORD"],
            "docker_image": "postgres:15"
          }
        ]
      },
      
      "environment_variables": {
        "required": [
          "API_PORT",
          "API_HOST", 
          "ENVIRONMENT"
        ],
        "optional": [
          "API_WORKERS",
          "API_TIMEOUT",
          "CORS_ORIGINS"
        ],
        "secrets": [
          "JWT_SECRET_KEY",
          "API_KEY",
          "DATABASE_URL"
        ]
      },
      
      "infrastructure": {
        "docker": {
          "base_image": "python:3.11-slim",
          "ports": ["8000"],
          "volumes": ["/app/logs", "/app/data"],
          "environment_files": [".env"]
        },
        "monitoring": {
          "health_endpoint": "/health",
          "metrics_endpoint": "/metrics", 
          "prometheus_config": "monitoring/api.yml"
        },
        "deployment": {
          "min_replicas": 1,
          "max_replicas": 10,
          "resource_limits": {
            "cpu": "1000m",
            "memory": "512Mi"
          }
        }
      },
      
      "compatibility": {
        "requires": ["config-management"],
        "enhances": ["plugin-registry", "monitoring"],
        "optional": ["caching", "rate-limiting", "authentication"],
        "conflicts": []
      },
      
      "injection_points": {
        "middleware_stack": {
          "file": "src/api/app.py",
          "function": "create_app",
          "location": "after_cors_setup"
        },
        "route_registration": {
          "file": "src/api/app.py", 
          "function": "register_routes",
          "location": "end_of_function"
        },
        "configuration_loading": {
          "file": "src/api/app.py",
          "class": "APIConfig",
          "location": "class_definition"
        }
      }
    },
    
    "config-management": {
      "name": "Configuration Management",
      "description": "Hydra + Pydantic configuration system with multi-environment support and secret management",
      "version": "1.0.0", 
      "type": "infrastructure",
      
      "files": [
        {
          "source": "mlx-components/config-management/manager.py.template",
          "target": "src/config/manager.py",
          "merge_strategy": "enhance"
        },
        {
          "source": "mlx-components/config-management/models.py.template", 
          "target": "src/config/models.py",
          "merge_strategy": "merge"
        },
        {
          "source": "mlx-components/config-management/config.yaml",
          "target": "conf/config.yaml",
          "merge_strategy": "merge"
        }
      ],
      
      "dependencies": {
        "python": [
          "hydra-core>=1.3.0",
          "pydantic>=2.0.0",
          "pydantic-settings>=2.0.0"
        ],
        "system": []
      },
      
      "environment_variables": {
        "required": ["ENVIRONMENT"],
        "optional": ["CONFIG_PATH", "LOG_LEVEL"],
        "secrets": ["SECRET_KEY", "ENCRYPTION_KEY"]
      },
      
      "compatibility": {
        "requires": [],
        "enhances": ["api-serving", "plugin-registry", "data-processing"],
        "optional": ["monitoring", "logging"],
        "conflicts": []
      }
    }
  },
  
  "installation_order": [
    "config-management",
    "api-serving", 
    "plugin-registry",
    "monitoring",
    "caching"
  ],
  
  "environment_templates": {
    "development": {
      "required_services": ["redis"],
      "optional_services": ["postgresql"],
      "environment_file": ".env.development"
    },
    "production": {
      "required_services": ["redis", "postgresql", "monitoring"],
      "optional_services": [],
      "environment_file": ".env.production"
    }
  }
}
```

---

## 🧪 **Phase 2 Success Criteria**

### **✅ Component Extraction Engine**
- [ ] **AST-based source analysis** extracting imports, endpoints, dependencies
- [ ] **Configuration dependency mapping** for environment variables and secrets
- [ ] **Infrastructure analysis** for Docker, monitoring, deployment requirements
- [ ] **Template generation** with variable substitution and merge strategies
- [ ] **Compatibility matrix generation** based on dependency analysis

### **✅ Production Registry**
- [ ] **Complete component catalog** with infrastructure requirements
- [ ] **Environment variable management** with required/optional/secrets classification
- [ ] **Service dependency tracking** (databases, APIs, external services)
- [ ] **Deployment configuration** with Docker and Kubernetes specifications
- [ ] **Installation order determination** based on dependency graph

### **✅ Projen Integration** 
- [ ] **MLX extraction tasks** integrated with projen (`./mlx extract`)
- [ ] **Registry validation** ensuring component integrity
- [ ] **Template testing** verifying component injection works correctly

### **✅ Zero Breaking Changes**
- [ ] **Existing functionality preserved** - all current features working
- [ ] **Source code unchanged** - extraction is non-destructive
- [ ] **Test suite passing** - no regressions introduced

---

## 🎯 **Key Implementation Insights**

### **1. This is NOT Simple File Copying**
MLX components are **complex system integrations** requiring:
- **Dependency analysis** across multiple languages/tools
- **Configuration merging** with conflict resolution
- **Infrastructure provisioning** with service dependencies
- **Secret management** with environment-specific handling

### **2. Focus on Production Readiness**
Every extracted component must be **production-deployable**:
- **Docker containers** with proper resource limits
- **Health checks** and monitoring endpoints  
- **Security configurations** with proper authentication
- **Scaling parameters** for cloud deployment

### **3. Template Intelligence Required**
Components need **smart templating**:
- **Variable substitution** for project-specific values
- **Conditional inclusion** based on environment
- **Merge conflict resolution** when components overlap
- **Upgrade path preservation** for component updates

---

## 🚀 **Phase 3 Preview: What You're Building Toward**

Your Phase 2 extraction engine enables Phase 3 capabilities:
- **AI-powered component recommendation** based on existing codebase analysis
- **Intelligent conflict resolution** when components have overlapping functionality  
- **Automated infrastructure provisioning** with cloud resource management
- **Component marketplace** with community-contributed components

**Your component extraction quality directly determines Phase 3 success!**

---

**Ready to build the future of ML platform composition? The foundation is solid, the architecture is clear, and the path is mapped. Let's build something remarkable! 🚀** 