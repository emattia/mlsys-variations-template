# MLOps Template → MLX Foundation

> "The future of ML development is composable." — MLX Team

Production-ready ML systems template evolving into **MLX**: the first AI-enhanced composable ML platform. Build ML systems by combining battle-tested components rather than starting from monolithic templates.

## 🚀 Quick Links

- **[Getting Started](getting-started.md)** - Setup and configuration
- **[Configuration Guide](../config/README.md)** - Advanced Hydra + Pydantic system
- **[Branch Strategy](../BRANCHING_STRATEGY.md)** - Choose your specialization
- **[Project Structure](user-guide/project-structure.md)** - Architecture overview
- **[API Reference](api/data_utils.md)** - Code documentation
- **[Plugin Development](development/)** - Extend functionality

## 🎯 MLX Evolution

This template is the foundation for **MLX**, transforming ML development from template selection to component composition:

### Current Template Features → Future MLX Components

| Template Feature | MLX Component | AI Enhancement |
|-----------------|---------------|----------------|
| 🧩 Plugin System | Component Framework | Smart composition |
| 🎯 Configuration | Config Management | Intelligent validation |
| 🔄 CI/CD Workflows | Testing Infrastructure | AI test selection |
| 🔒 Security Stack | Security Components | Automated hardening |
| 📊 Caching System | Performance Components | Cost optimization |

## 🏗️ Advanced Features

### Production-Grade Configuration System
- **Hierarchical configurations** with environment inheritance
- **Type-safe validation** using Pydantic models
- **Secret management** with secure environment resolution
- **Experiment tracking** with A/B testing capabilities
- **Runtime overrides** for dynamic configuration

### Enterprise Security & Performance
- **Multi-service rate limiting** with cost enforcement
- **Intelligent caching** (70% LLM cost reduction)
- **Security scanning** with vulnerability management
- **Production monitoring** with health checks

### Plugin Architecture
- **Extensible design** without core modification
- **Registry system** for plugin discovery
- **Contract testing** for interface compliance
- **Hot-swappable** components for rapid iteration

## 🎭 Specializations

**This template now uses a plugin-based architecture instead of branches. See the plugin section below.**

## 🚀 Quick Setup

```bash
# 1. Clone the unified template
git clone <repo-url> my-ml-project

# 2. Transform to your project
./mlx my-awesome-project

# 3. Add domain capabilities via plugins
uv add mlx-plugin-agentic mlx-plugin-transformers

# 4. Configure environment
cp .env-example .env  # Edit as needed

# 5. Start development
make all-checks && make run-api
```

## 📁 Project Architecture

```
my-project/
├── src/                    # Core ML components
│   ├── api/               # FastAPI service + routes
│   ├── config/            # Configuration management
│   ├── data/              # Data processing pipelines
│   ├── models/            # Model training/inference
│   ├── plugins/           # Extensible components
│   └── utils/             # Shared utilities
├── conf/                   # Hydra configuration hierarchy
│   ├── config.yaml        # Main configuration
│   ├── model/             # Model configurations
│   ├── api/               # API configurations
│   └── experiments/       # Experiment configs
├── workflows/             # ML pipeline orchestration
├── tests/                 # Multi-layered test suite
│   ├── unit/              # Fast isolated tests
│   ├── integration/       # Component interaction
│   └── contracts/         # Plugin interface tests
├── docs/                  # Comprehensive documentation
├── data/                  # Data storage
│   ├── raw/               # Original datasets
│   └── processed/         # Cleaned/transformed
├── models/                # Trained model artifacts
└── reports/               # Analysis outputs
```

## 🛠️ Development Commands

```bash
# Environment setup
make install-dev          # Install deps + pre-commit hooks
make setup-env            # Configure environment

# Development workflow
make lint                 # Code quality (ruff + mypy + security)
make test                 # Full test suite with coverage
make test-unit            # Fast unit tests only
make test-integration     # Integration test suite

# Service management
make run-api              # Start FastAPI with auto-reload
make run-worker           # Start background worker
make run-pipeline         # Execute ML pipeline

# Production preparation
make build-docker         # Multi-stage optimized container
make deploy-staging       # Deploy to staging environment
make security-scan        # Security vulnerability assessment

# Documentation
make docs-serve           # Local documentation server
make docs-build           # Build static documentation
```

## ⚙️ Advanced Configuration Examples

### Environment-Specific Configuration
```bash
# Development with debug enabled
ENVIRONMENT=development python train_model.py

# Production with optimized settings
ENVIRONMENT=production python train_model.py

# Custom overrides
python train_model.py model.type=xgboost api.workers=8
```

### Experiment Configuration
```bash
# Run hyperparameter tuning experiment
python train_model.py --config-name=experiments/hp_tuning

# A/B test different model configurations
python train_model.py experiment.group=control
python train_model.py experiment.group=treatment
```

### Plugin System Usage
```python
# Extend functionality with custom plugins
from src.plugins import PluginRegistry

registry = PluginRegistry()

# Add custom data processor
registry.register_plugin("data_processors", "custom", CustomProcessor())

# Use LLM provider plugins
llm = registry.get_plugin("llm_providers", "openai")
result = llm.generate(prompt, context)
```

## 🧪 Testing Strategy

Multi-layered testing approach for reliability:

- **Unit Tests**: Fast, isolated component testing
- **Integration Tests**: Component interaction verification
- **Contract Tests**: Plugin interface compliance
- **End-to-End Tests**: Full workflow validation
- **Performance Tests**: Load and benchmark testing

## 🔮 Roadmap to MLX

### Phase 1: Component Extraction (Current)
- Extract reusable components from template branches
- Standardize plugin interfaces and contracts
- Build compatibility matrix foundation

### Phase 2: AI-Enhanced Testing
- Implement intelligent test selection with RL agents
- Add compatibility prediction using LLM models
- Deploy smart CI/CD with cost optimization

### Phase 3: Composable Platform
- Launch MLX CLI with component marketplace
- Enable community component contributions
- Provide AI-guided system composition

### Phase 4: Ecosystem Intelligence
- Self-improving component recommendations
- Automated performance optimization
- Predictive maintenance and upgrades

## 🤝 Contributing

1. **Install development dependencies**: `make install-dev`
2. **Run quality checks**: `make all-checks`
3. **Add comprehensive tests** for new functionality
4. **Update documentation** for new features
5. **Follow plugin architecture** for extensions
6. **Submit pull request** with clear description

## 📚 Documentation Structure

- **[Getting Started](getting-started.md)** - Initial setup and configuration
- **[Configuration Guide](../config/README.md)** - Comprehensive config system
- **[User Guide](user-guide/)** - Feature-specific documentation
- **[Development Guide](development/)** - Plugin and extension development
- **[API Reference](api/)** - Generated code documentation
- **[About](about/)** - Project background and philosophy

## 🔗 External Resources

- **[Hydra Documentation](https://hydra.cc/)** - Configuration framework
- **[FastAPI Documentation](https://fastapi.tiangolo.com/)** - API framework
- **[Pydantic Documentation](https://pydantic-docs.helpmanual.io/)** - Data validation
- **[MLX Roadmap](https://github.com/mlx-platform)** - Future platform development

## License

MIT License - see [LICENSE](../LICENSE)

---

*Building towards a future where ML systems are composed, not templated.*

## 🧩 Domain Specializations via Plugins

Instead of separate branches, this template uses **composable plugins** for domain-specific functionality:

| Plugin Package | Domain Focus | Key Technologies | Installation |
|---------------|--------------|-----------------|--------------|
| `mlx-plugin-sklearn` | General ML/Analytics | Scikit-learn, Pandas | `uv add mlx-plugin-sklearn` |
| `mlx-plugin-agentic` | Multi-agent AI | LangChain, AutoGen | `uv add mlx-plugin-agentic` |
| `mlx-plugin-transformers` | LLM training/serving | 🤗 Transformers, PyTorch | `uv add mlx-plugin-transformers` |
| `mlx-plugin-streaming` | Real-time features | Kafka, streaming | `uv add mlx-plugin-streaming` |

### Composable Architecture Benefits

✅ **Mix and Match**: Combine agentic AI with streaming features in one project
✅ **Single Codebase**: One template, multiple capabilities via plugins
✅ **Community Focus**: All contributions go to one main repository
✅ **Easy Updates**: Plugin updates don't require template rebasing

### Example: Multi-Domain Project
```bash
# Create project with multiple domain capabilities
git clone <repo-url> my-hybrid-project && cd my-hybrid-project
./mlx my_hybrid_project

# Add domain-specific plugins as needed
uv add mlx-plugin-agentic mlx-plugin-streaming mlx-plugin-transformers

# Configure in your project
python -c "
from src.plugins import PluginRegistry
registry = PluginRegistry()
registry.discover_plugins()  # Auto-discovers installed plugins
"
```

## Get Started

### Clone & Setup
```bash
# Clone the foundation template
git clone <repo-url> my-project && cd my-project

# Personalize your project
./mlx my_project_name

# Add plugins for your domain
uv add mlx-plugin-<domain>
```

### Quick Demo
```bash
# Run all quality checks
make all-checks

# Start the API server
make run-api
```

## 🧩 Plugin-Based Architecture

This template uses **composable plugins** instead of separate branches for domain specializations. The unified approach provides:

✅ **Single codebase** - easier maintenance and community focus
✅ **True composability** - mix and match capabilities
✅ **No branch divergence** - consistent base template
✅ **Plugin ecosystem** - extend capabilities as needed

See the [Domain Specializations via Plugins](#-domain-specializations-via-plugins) section below for available plugins.
