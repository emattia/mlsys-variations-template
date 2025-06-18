# 🤖 MLX Assistant - Complete User Guide

## Overview

The **MLX Assistant** is your intelligent companion for navigating the Mlx Platform Foundation. It unifies all Phase 3 frameworks under a single, beautifully designed CLI interface with rich styling and contextual guidance.

## 🚀 Quick Start

### Launch the Assistant
```bash
# View the main dashboard
mlx assistant # Start interactive mode
mlx assistant --interactive

# Quick health check
mlx assistant doctor
```

### Discover Available Frameworks
```bash
# Show all available frameworks
mlx frameworks

# Launch the unified assistant interface
mlx assistant ```

## 🛠️ Framework Operations

### 🏗️ Golden Repository Testing

Create and manage reference implementations for testing:

```bash
# List available repository specifications
mlx assistant golden-repos list

# Create a standard golden repository
mlx assistant golden-repos create standard

# Create minimal repository for basic testing
mlx assistant golden-repos create minimal

# Validate an existing repository
mlx assistant golden-repos validate standard
```

**Available Repository Types:**
- `minimal` - Basic MLOps template with essential components
- `standard` - Full-featured template with core components  
- `advanced` - Complex multi-component setup
- `plugin_heavy` - Multiple plugins for integration testing
- `performance` - Optimized for benchmarking

### 🔒 Security Hardening

Comprehensive security scanning and hardening:

```bash
# Run security scan with default enhanced level
mlx assistant security scan

# Run enterprise-level security scan
mlx assistant security scan --level enterprise

# Generate Software Bill of Materials (SBOM) 
mlx assistant security sbom

# Security scan with HTML report
mlx assistant security scan --output html
```

**Security Levels:**
- `basic` - Essential security checks
- `enhanced` - Comprehensive scanning (default)
- `enterprise` - Business-critical requirements
- `critical` - Maximum security validation

### 🧩 Plugin Ecosystem

Develop and manage MLX plugins:

```bash
# Create a new ML framework plugin
mlx assistant plugins create \
  --name my-ml-plugin \
  --type ml_framework \
  --description "My custom ML framework"

# List all available plugins
mlx assistant plugins list

# Validate a plugin
mlx assistant plugins validate plugins/mlx-plugin-my-plugin
```

**Plugin Types:**
- `ml_framework` - Machine learning frameworks
- `data_processor` - Data processing utilities
- `model_provider` - Model serving providers
- `deployment` - Deployment automation
- `monitoring` - System monitoring
- `security` - Security enhancements
- `utility` - General utilities

### 📚 Glossary & Standards

Access mlx platform documentation and standards:

```bash
# View the complete glossary
mlx assistant glossary view

# Search for specific terms
mlx assistant glossary search "component"
mlx assistant glossary search "plugin"
```

## 🎯 Interactive Mode

The interactive mode provides a guided experience for exploring the platform:

```bash
mlx assistant --interactive
```

**Interactive Commands:**
- `help` - Show available commands
- `status` - Display detailed project status  
- `analyze` - Analyze current project state
- `recommend` - Get intelligent recommendations
- `exit` - Exit interactive mode

## 🩺 Health & Diagnostics

### Platform Health Check
```bash
# Comprehensive platform health check
mlx assistant doctor

# Alternative access via main script
mlx doctor
```

The health check validates:
- ✅ mlx project structure
- ✅ Framework availability
- ✅ Dependencies status
- ✅ Configuration validity

### Project Analysis
```bash
# Analyze current project state
mlx assistant analyze
```

Provides insights on:
- Project type and configuration
- Available components
- Plugin ecosystem status
- Intelligent recommendations

## 🎨 Rich Interface Features

### Professional Styling
- **Consistent Color Scheme**: Cyan commands, green success, red errors
- **Rich Tables**: Professional layouts with proper spacing
- **Progress Indicators**: Visual feedback for long operations
- **Panel Borders**: Organized information display

### Icons & Visual Cues
- 🤖 Assistant operations
- 🏗️ Golden repositories
- 🔒 Security operations  
- 🧩 Plugin operations
- 📚 Documentation
- ✅ Success states
- ❌ Error states
- ⚠️ Warnings

## 🧠 Intelligent Features

### Context-Aware Recommendations

The assistant analyzes your project state and provides intelligent suggestions:

```bash
# If no mlx project detected:
🎯 Start with: mlx assistant quick-start to set up your mlx project

# If project exists but no components:
📦 Extract components: mlx assistant golden-repos create standard

# If no plugins available:
🧩 Create your first plugin: mlx assistant plugins create

# Always suggested:
🛡️ Run security scan: mlx assistant security scan
```

### Smart Workflows

The assistant guides you through complex operations:

1. **Project Setup**: From template to production-ready project
2. **Security Hardening**: Comprehensive security implementation
3. **Plugin Development**: From idea to validated plugin
4. **Component Management**: Extract, inject, and manage components

## 📋 Command Reference

### Main Commands
| Command | Description |
|---------|-------------|
| `mlx assistant` | Launch main dashboard |
| `mlx assistant --interactive` | Start guided mode |
| `mlx assistant doctor` | Health check |
| `mlx assistant quick-start` | New user guide |
| `mlx assistant analyze` | Project analysis |

### Framework Groups
| Group | Purpose | Example |
|-------|---------|---------|
| `golden-repos` | Repository testing | `mlx assistant golden-repos create standard` |
| `security` | Security hardening | `mlx assistant security scan --level enhanced` |
| `plugins` | Plugin management | `mlx assistant plugins create --name my-plugin` |
| `glossary` | Documentation | `mlx assistant glossary search "component"` |

### Utility Commands
| Command | Description |
|---------|-------------|
| `mlx frameworks` | Show framework status |
| `mlx doctor` | Quick health check |
| `mlx assistant --version` | Version information |

## 🔗 Integration with Existing Workflow

### Main mlx Script Integration
The assistant integrates seamlessly with existing `mlx` commands:

```bash
# Traditional workflow
./mlx create my-project
./mlx add api-serving

# Enhanced workflow with assistant
mlx assistant golden-repos create standard
mlx assistant security scan
mlx assistant plugins create --name custom-api
```

### Framework Script Compatibility
The assistant works alongside existing framework scripts:

```bash
# Direct framework access (still works)
python tests/golden_repos.py create --spec standard
python scripts/security/security_hardening.py scan

# Unified assistant access (recommended)
mlx assistant golden-repos create standard  
mlx assistant security scan
```

## 🚀 Best Practices

### 1. Start with Health Check
Always begin with a health check to understand your project state:
```bash
mlx assistant doctor
```

### 2. Use Interactive Mode for Exploration
When learning the platform, use interactive mode:
```bash
mlx assistant --interactive
```

### 3. Follow Intelligent Recommendations
The assistant provides contextual suggestions - follow them for optimal workflows.

### 4. Leverage Rich Output
The assistant's rich formatting makes it easy to understand complex operations and results.

### 5. Combine Framework Operations
Use the assistant to coordinate complex multi-framework operations:
```bash
# Complete project hardening workflow
mlx assistant golden-repos create advanced
mlx assistant security scan --level enterprise
mlx assistant plugins validate plugins/
```

## 🧪 Testing & Validation

### Golden Repository Testing
```bash
# Create test environments
mlx assistant golden-repos create minimal
mlx assistant golden-repos create standard
mlx assistant golden-repos create advanced

# Validate all repositories
mlx assistant golden-repos validate minimal
mlx assistant golden-repos validate standard
mlx assistant golden-repos validate advanced
```

### Security Validation
```bash
# Progressive security hardening
mlx assistant security scan --level basic
mlx assistant security scan --level enhanced  
mlx assistant security scan --level enterprise
mlx assistant security sbom
```

### Plugin Development Cycle
```bash
# Complete plugin development workflow
mlx assistant plugins create --name test-plugin --type utility
mlx assistant plugins validate plugins/mlx-plugin-test-plugin
mlx assistant security scan  # Validate plugin security
```

## 📈 Advanced Usage

### Scripting with the Assistant
The assistant can be used in automated workflows:

```bash
#!/bin/bash
# Automated project setup script

echo "Setting up mlx project with assistant..."

# Health check
mlx assistant doctor || exit 1

# Create golden repository
mlx assistant golden-repos create standard || exit 1

# Security scan
mlx assistant security scan --level enhanced || exit 1

echo "mlx project setup complete!"
```

### CI/CD Integration
Integrate assistant commands in your CI/CD pipeline:

```yaml
# .github/workflows/mlx-validation.yml
- name: MLX Health Check
  run: mlx assistant doctor

- name: Security Scan  
  run: mlx assistant security scan --level enterprise

- name: Plugin Validation
  run: |
    for plugin in plugins/mlx-plugin-*; do
      mlx assistant plugins validate "$plugin"
    done
```

## 🔮 Future Enhancements (Phase 4)

The assistant is designed to integrate future AI-enhanced features:

- **Natural Language Queries**: Ask questions in plain English
- **Smart Dependency Management**: AI-powered dependency updates
- **Automated Workflow Generation**: Generate workflows from project analysis
- **Intelligent Plugin Recommendations**: ML-based plugin suggestions
- **Performance Optimization**: AI-driven performance improvements

## 🆘 Troubleshooting

### Common Issues

**Assistant won't start:**
```bash
# Check Python environment
python --version
pip install typer rich

# Verify file permissions
chmod +x scripts/mlx_assistant.py
```

**Framework commands fail:**
```bash
# Run health check for diagnostics
mlx assistant doctor

# Check individual frameworks
python tests/golden_repos.py --help
python scripts/security/security_hardening.py --help
```

**Interactive mode issues:**
```bash
# Try non-interactive mode first
mlx assistant # Check terminal compatibility
echo $TERM
```

### Getting Help

1. **In-app help**: `mlx assistant --help`
2. **Interactive help**: Start interactive mode and type `help`
3. **Health diagnostics**: `mlx assistant doctor`
4. **Framework status**: `mlx frameworks`

---

The MLX Assistant represents the culmination of Phase 3 "Core Hardening" efforts, providing a production-ready, intelligent interface for the complete Mlx Platform Foundation. It transforms the fragmented CLI experience into a unified, professional, and user-friendly system that guides you through complex MLOps workflows with confidence. 