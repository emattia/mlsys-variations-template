# Mlx Platform Glossary & Naming Conventions

> "The beginning of wisdom is the definition of terms." — Socrates

## 🎯 **Purpose**

This document establishes **reserved namespaces**, **naming conventions**, and **terminology standards** for the Mlx Platform ecosystem to ensure consistency, avoid conflicts, and facilitate discoverability.

---

## 📚 **Core Terminology**

### **Template vs Component vs Plugin**

| Term | Definition | Example | Scope |
|------|------------|---------|-------|
| **Template** | Base MLOps foundation repository that teams fork | `mlx-platform-template` | Repository structure |
| **Component** | Self-contained capability with templates and metadata | `api-serving`, `config-management` | Feature delivery |
| **Plugin** | Runtime-loaded extension for specific domains | `mlx-plugin-transformers` | Runtime extensibility |

### **Mlx Platform Concepts**

| Term | Definition | Usage |
|------|------------|-------|
| **MLX Foundation** | The base template repository | Fork → customize → deploy pattern |
| **Component Registry** | Catalog of available components with metadata | `mlx-components/registry.json` |
| **Plugin System** | Runtime extensibility framework | Dynamic loading, hot-reload |
| **Production Patterns** | Hardened architectural patterns | Security, monitoring, deployment |

---

## 🏷️ **Naming Conventions**

### **Reserved Namespaces**

#### **Package Prefixes**
```
mlx-*           # Official MLX packages (reserved)
mlx-plugin-*    # Plugin packages  
mlx-component-* # Component packages
mlx-template-*  # Template variations
```

#### **Directory Prefixes**
```
.mlx/           # MLX metadata (reserved)
mlx-*/          # MLX directories
scripts/mlx/    # MLX tooling scripts
```

#### **Configuration Keys**
```
mlx:            # MLX-specific config keys
mlx_*           # Environment variables
MLX_*           # Global constants
```

### **Component Naming**

#### **Component Names** (kebab-case)
```
✅ api-serving
✅ config-management  
✅ plugin-registry
✅ data-processing
✅ model-training

❌ ApiServing (PascalCase)
❌ api_serving (snake_case)
❌ apiServing (camelCase)
```

#### **Plugin Names** (kebab-case with domain prefix)
```
✅ mlx-plugin-transformers
✅ mlx-plugin-sklearn
✅ mlx-plugin-agentic
✅ mlx-plugin-streaming

❌ transformers-plugin (wrong order)
❌ mlx_plugin_transformers (snake_case)
```

#### **Template Files** (.template extension)
```
✅ component.py.template
✅ config.yaml.template
✅ Dockerfile.template

❌ component.py.tmpl
❌ component.py.jinja
```

### **Configuration Naming**

#### **Environment Variables**
```
# Mlx Platform variables
MLX_ENVIRONMENT=production
MLX_LOG_LEVEL=info
MLX_PLUGIN_PATH=/path/to/plugins

# Component-specific variables  
API_HOST=localhost
API_PORT=8000
DATABASE_URL=postgresql://...

# Plugin-specific variables
TRANSFORMERS_MODEL_PATH=/models
SKLEARN_CACHE_DIR=/cache
```

#### **Configuration Sections**
```yaml
# mlx.config.json - Platform configuration
{
  "mlx": {
    "version": "1.0.0",
    "components": {...},
    "plugins": {...}
  }
}

# conf/config.yaml - Application configuration  
mlx:
  environment: ${oc.env:MLX_ENVIRONMENT,development}
  log_level: ${oc.env:MLX_LOG_LEVEL,info}

api:
  host: ${oc.env:API_HOST,localhost}
  port: ${oc.env:API_PORT,8000}
```

---

## 🔧 **File & Directory Standards**

### **Project Structure Conventions**

```
project-name/                    # kebab-case project names
├── .mlx/                       # MLX metadata (reserved)
│   ├── components.json         # Installed components
│   ├── plugins.json           # Active plugins  
│   └── manifest.json          # Project manifest
├── mlx-components/            # Component registry
│   ├── registry.json         # Component catalog
│   └── {component-name}/      # Component templates
├── src/                       # Application source
├── conf/                      # Configuration hierarchy
├── tests/                     # Test suites
└── scripts/mlx/              # MLX tooling
```

### **Component Structure**

```
mlx-components/{component-name}/
├── templates/                 # Template files
│   ├── src/                  # Source code templates
│   ├── conf/                 # Configuration templates
│   └── tests/                # Test templates
├── metadata.json             # Component metadata
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── examples/                 # Usage examples
```

### **Plugin Structure**

```
mlx-plugin-{domain}/
├── src/mlx_plugin_{domain}/  # Python package (snake_case)
│   ├── __init__.py
│   ├── plugin.py            # Main plugin class
│   └── models.py            # Plugin-specific models
├── conf/                    # Plugin configuration
├── tests/                   # Plugin tests
├── pyproject.toml          # Package metadata
└── README.md               # Plugin documentation
```

---

## 🏗️ **Code Conventions**

### **Python Classes**

```python
# Component classes (PascalCase)
class ApiServingComponent(BaseComponent):
    pass

class ConfigManagementComponent(BaseComponent):
    pass

# Plugin classes (PascalCase with Plugin suffix)
class TransformersPlugin(BasePlugin):
    pass

class SklearnPlugin(BasePlugin):
    pass

# Utility classes (PascalCase)
class ComponentRegistry:
    pass

class PluginLoader:
    pass
```

### **Python Functions & Variables**

```python
# Functions (snake_case)
def load_component(name: str) -> Component:
    pass

def register_plugin(plugin: BasePlugin) -> None:
    pass

# Variables (snake_case)
component_registry = ComponentRegistry()
plugin_loader = PluginLoader()
api_config = load_config("api")

# Constants (SCREAMING_SNAKE_CASE)
DEFAULT_API_PORT = 8000
MAX_PLUGIN_LOAD_TIME = 30
COMPONENT_REGISTRY_PATH = "mlx-components/registry.json"
```

### **Configuration Keys**

```yaml
# YAML keys (snake_case)
api_server:
  host: localhost
  port: 8000
  workers: 4

model_training:
  batch_size: 32
  learning_rate: 0.001

plugin_settings:
  auto_load: true
  timeout: 30
```

---

## 📦 **Package & Distribution Standards**

### **Python Packages**

```toml
# pyproject.toml for components
[project]
name = "mlx-component-api-serving"
version = "1.0.0"
description = "Production-ready FastAPI component"
authors = [{name = "MLX Team", email = "team@mlx.com"}]

# pyproject.toml for plugins
[project]
name = "mlx-plugin-transformers"
version = "1.0.0"
description = "Hugging Face transformers plugin"
```

### **Docker Images**

```dockerfile
# Image naming convention
FROM mlx/foundation:1.0.0
FROM mlx/component-api-serving:1.0.0
FROM mlx/plugin-transformers:1.0.0

# Labels (kebab-case)
LABEL mlx.component="api-serving"
LABEL mlx.version="1.0.0"
LABEL mlx.compatibility=">=1.0.0"
```

### **GitHub Repositories**

```
# Repository naming
mlx-template-foundation        # Main template
mlx-component-api-serving     # Component repo
mlx-plugin-transformers       # Plugin repo

# Branch naming
main                          # Default branch
develop                       # Development branch
feature/component-extraction  # Feature branches
hotfix/security-patch        # Hotfix branches
```

---

## 🏪 **Marketplace & Discovery**

### **Component Categories**

```json
{
  "categories": {
    "infrastructure": ["api-serving", "config-management"],
    "data": ["data-processing", "data-validation"],
    "ml": ["model-training", "model-evaluation"],
    "deployment": ["docker-optimization", "kubernetes-deployment"],
    "monitoring": ["observability", "logging"],
    "security": ["authentication", "vulnerability-scanning"]
  }
}
```

### **Plugin Domains**

```json
{
  "domains": {
    "ml-frameworks": ["sklearn", "pytorch", "tensorflow"],
    "llm": ["transformers", "openai", "anthropic"],
    "data": ["pandas", "polars", "dask"],
    "deployment": ["kubernetes", "docker", "serverless"],
    "streaming": ["kafka", "pulsar", "kinesis"]
  }
}
```

### **Tags & Labels**

```yaml
# Component metadata
tags:
  - production-ready
  - fastapi
  - rest-api
  - microservice

labels:
  - api
  - web-service
  - production

# Plugin metadata
tags:
  - machine-learning
  - transformers
  - huggingface
  - nlp

domains:
  - llm
  - text-processing
```

---

## 🔄 **Versioning Strategy**

### **Semantic Versioning**

```
# Mlx Platform
1.0.0 - Initial release
1.1.0 - New components/features
1.0.1 - Bug fixes
2.0.0 - Breaking changes

# Components
api-serving:1.0.0
config-management:1.2.0
plugin-registry:1.0.0

# Plugins  
mlx-plugin-transformers:1.0.0
mlx-plugin-sklearn:2.1.0
```

### **Compatibility Matrix**

```json
{
  "compatibility": {
    "mlx-platform": ">=1.0.0,<2.0.0",
    "python": ">=3.9,<4.0",
    "api-serving": ">=1.0.0,<2.0.0"
  }
}
```

---

## 🚦 **Status & Lifecycle**

### **Component Status**

| Status | Definition | Badge |
|--------|------------|-------|
| `experimental` | Under development, breaking changes | 🧪 |
| `beta` | Feature complete, testing | 🚧 |
| `stable` | Production ready | ✅ |
| `deprecated` | Will be removed | ⚠️ |
| `archived` | No longer maintained | 📦 |

### **Plugin Status**

```json
{
  "status": "stable",
  "lifecycle": {
    "created": "2024-01-01",
    "stable_since": "2024-03-01",
    "last_updated": "2024-12-01"
  },
  "support": {
    "level": "community",
    "maintainer": "mlx-team"
  }
}
```

---

## 🎯 **Implementation Guidelines**

### **For Component Developers**

1. **Follow naming conventions** strictly
2. **Use reserved namespaces** appropriately  
3. **Include proper metadata** in all components
4. **Implement required interfaces** for compatibility
5. **Document configuration keys** thoroughly

### **For Plugin Developers**

1. **Prefix plugin names** with `mlx-plugin-`
2. **Use domain-specific naming** for clarity
3. **Implement BasePlugin interface** completely
4. **Include comprehensive tests** 
5. **Document plugin capabilities** clearly

### **For Template Users**

1. **Respect reserved namespaces** in customizations
2. **Follow project naming** conventions
3. **Use environment variables** properly
4. **Maintain metadata files** correctly
5. **Document customizations** thoroughly

---

## 🔍 **Validation & Enforcement**

### **Automated Checks**

```bash
# Naming convention validation
make validate-naming

# Reserved namespace checking  
make check-namespaces

# Component/plugin compliance
make validate-compliance

# Metadata verification
make verify-metadata
```

### **CI/CD Integration**

```yaml
# .github/workflows/validate.yml
- name: Validate Naming Conventions
  run: |
    python scripts/mlx/validate_naming.py
    python scripts/mlx/check_namespaces.py
    python scripts/mlx/verify_compliance.py
```

---

## 🤝 **Contributing**

When contributing to the MLX ecosystem:

1. **Review this glossary** before naming anything
2. **Check reserved namespaces** to avoid conflicts
3. **Follow established conventions** consistently
4. **Update documentation** when adding new terms
5. **Validate compliance** before submitting PRs

---

**This glossary is a living document that evolves with the mlx platform. When in doubt, consistency is key!** 