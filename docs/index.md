# MLOps Template Documentation

> "AI is the new electricity." — Andrew Ng

Production-ready machine learning platform template with specialized branches for different ML domains.

## Quick Links

- **[Getting Started](getting-started.md)** - Setup and configuration
- **[Branch Strategy](../branching_strategy.md)** - Choose your specialization
- **[Project Structure](user-guide/project-structure.md)** - Architecture overview
- **[API Reference](api/data_utils.md)** - Code documentation

## Template Features

- **FastAPI service** with auto-generated docs
- **Docker containerization** for deployment
- **Plugin system** for extensibility
- **Quality tools** (ruff, mypy, pytest, bandit)
- **CI/CD pipelines** via GitHub Actions
- **Configuration management** with Hydra

## Specializations

| Branch | Purpose | Technologies |
|--------|---------|-------------|
| `main` | General ML/Analytics | Scikit-learn, Pandas |
| `agentic-ai-system` | Multi-agent AI | LangChain, AutoGen |
| `llm-finetuning-system` | LLM training | Transformers, PyTorch |
| `chalk-feature-engineering` | Real-time features | Chalk, streaming |

## Quick Setup

```bash
# 1. Clone template
git clone <repo-url> my-project && cd my-project

# 2. Transform to your project
./mlsys my-awesome-project

# 3. Start development
make all-checks && make run-api
```

## Project Structure

```
├── src/                # Core ML components
│   ├── api/           # FastAPI service
│   ├── data/          # Data processing
│   ├── models/        # Model training/inference
│   └── plugins/       # Extensible components
├── workflows/         # ML pipelines
├── tests/             # Test suite
├── docs/              # Documentation
├── data/              # Data files
│   ├── raw/           # Original data
│   └── processed/     # Cleaned data
├── models/            # Trained models
└── reports/           # Analysis outputs
```

## Development Commands

```bash
make install-dev      # Install dependencies
make test             # Run tests
make lint             # Code quality checks
make run-api          # Start API service
make build-docker     # Build container
```

## Configuration

- **Environment**: Copy `.env-example` to `.env`
- **Configs**: Override in `conf/` directory
- **Plugins**: Extend functionality in `src/plugins/`

## Contributing

1. Install development dependencies: `make install-dev`
2. Run quality checks: `make all-checks`
3. Submit pull request

## License

MIT License - see [LICENSE](../LICENSE)
