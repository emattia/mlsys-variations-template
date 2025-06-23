# Getting Started

> "The journey of a thousand miles begins with a single step." — Lao Tzu

A five-minute path from clone to running API.

## 🚀 Installation

### Prerequisites
- Python 3.9+ (3.11+ recommended)
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Quick Start

```bash
# Clone the unified template
git clone <repo-url> my-ml-project
cd my-ml-project

# Personalize your project
./mlx my_project_name

# Add domain-specific plugins as needed
uv add mlx-plugin-agentic        # For AI agents
uv add mlx-plugin-transformers   # For LLM fine-tuning

# Set up environment
cp .env-example .env  # Edit with your configuration

# Validate installation
make all-checks

# Start the API
make run-api
```

### Domain-Specific Setup

Choose your plugins based on your ML domain:

**AI Agents & Multi-Agent Systems**
```bash
uv add mlx-plugin-agentic
# Configure API keys in .env file
```

**LLM Fine-tuning & Deployment**
```bash
uv add mlx-plugin-transformers
# Set up GPU environment and Hugging Face access
```

**Real-time Feature Engineering**
```bash
uv add mlx-plugin-streaming
# Configure streaming data sources
```

**Hybrid Systems**
```bash
# Compose multiple domains
uv add mlx-plugin-agentic mlx-plugin-transformers mlx-plugin-streaming
```

---

## 🎉 You're Ready!

**Congratulations!** You now have a production-ready ML system that includes:

- ✅ **Quality Assured**: Automated testing, linting, security scanning
- ✅ **Production Ready**: Docker containers, CI/CD pipelines, monitoring
- ✅ **Well Documented**: API docs, user guides, development guides
- ✅ **Extensible**: Plugin architecture, configuration management
- ✅ **Team Friendly**: Consistent patterns, automated quality checks

### Next Steps

1. **📊 Explore the Demo**: `make demo-comprehensive`
2. **📖 Read the Docs**: Browse `docs/` directory or visit the [documentation site](docs/index.md)
3. **🔌 Check the APIs**: Visit `http://localhost:8000/docs`
4. **🧪 Write Your First Test**: Add to `tests/` directory
5. **🚀 Deploy**: Use Docker or cloud deployment guides

### API Documentation

Visit `http://localhost:8000/docs` to see:
- Interactive API documentation
- Available endpoints from your installed plugins
- Request/response schemas
- Try-it-out functionality

**Happy Building!** 🚀
