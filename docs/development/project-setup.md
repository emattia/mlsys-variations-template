# MLOps Template Startup Guide

> *"Give me six hours to chop down a tree and I will spend the first four sharpening the axe."* - Abraham Lincoln

# Project Setup

This guide transforms you from template curious to production ready in under 10 minutes.

## 🎯 Architecture: One Template, Composable Capabilities

**Instead of separate branches, this template uses composable plugins:**

✅ **One codebase** - easier maintenance and updates
✅ **Mix & match** - combine AI agents with streaming features
✅ **Community focus** - all contributions benefit everyone
✅ **Plugin ecosystem** - extend capabilities as needed

## 🚀 The 3-Command Setup

### Command 1: Clone the Foundation Template

```bash
# Clone the unified template
git clone <repo-url> my-ml-project
cd my-ml-project
```

### Command 2: Transform with mlx

```bash
# 🎭 Transform template into your personalized project
./mlx your-project-name

# Examples:
./mlx customer-churn-predictor
./mlx financial-risk-model
./mlx document-qa-system
./mlx real-time-recommender
```

**⚡ What happens during transformation:**

1. **🔄 Intelligent Bootstrapping**
   - Creates isolated Python environment (`.bootstrap_venv`)
   - Installs required dependencies (typer, rich)
   - Handles dependency conflicts gracefully

2. **📝 Smart Renaming Engine**
   - Renames `src/analysis_template` → `src/your_project_name`
   - Updates all imports and references throughout codebase
   - Converts between naming conventions (snake_case, kebab-case, Title Case)

3. **⚙️ Configuration Alignment**
   - Updates `pyproject.toml` with new project name
   - Configures linting rules (ruff) for new package structure
   - Aligns Docker and CI/CD configurations

4. **📚 Documentation Refresh**
   - Personalizes README, docs, and API references
   - Updates example code and configuration snippets
   - Maintains documentation structure and formatting

5. **🧪 Validation & Verification**
   - Ensures all configuration files are valid
   - Verifies imports resolve correctly
   - Confirms project structure integrity

### Command 3: Add Domain Capabilities

```bash
# 🌟 Activate your personalized environment
source .venv/bin/activate

# 🎯 Add domain-specific plugins as needed
uv add mlx-plugin-agentic      # For AI agents
uv add mlx-plugin-transformers # For LLM fine-tuning
uv add mlx-plugin-streaming    # For real-time features

# 🚀 Run comprehensive checks
make all-checks

# 🎪 See what's possible (optional but recommended)
make demo-comprehensive
```

## Domain-Specific Setup

Choose your ML domain and install the corresponding plugins:

| Domain | Plugin Package | Key Technologies |
|--------|---------------|-----------------|
| 📊 **General ML/Analytics, Classification, Regression** | `mlx-plugin-sklearn` | Scikit-learn, Pandas, Jupyter |
| 🤖 **AI Agents, Multi-agent systems, Tool-calling AI** | `mlx-plugin-agentic` | LangChain, AutoGen, tool integration, safety |
| 🧠 **LLM Fine-tuning, Language model deployment** | `mlx-plugin-transformers` | 🤗 Transformers, LoRA/QLoRA, distributed training |
| 🏗️ **Real-time features, Data transformation pipelines** | `mlx-plugin-streaming` | Kafka, streaming, monitoring |

### Base Template Setup
```bash
# Clone and customize the base template
git clone <repo-url> my-ml-project
cd my-ml-project
./mlx my_ml_project  # Customizes project name and structure
```

### Add Domain Capabilities via Plugins

#### AI Agents & Multi-Agent Systems
```bash
# Install agentic AI capabilities
uv add mlx-plugin-agentic

# Plugins auto-register - restart your API to see new endpoints
make run-api
```

#### LLM Fine-tuning & Deployment
```bash
# Install transformer capabilities
uv add mlx-plugin-transformers

# Configure for your specific model
# Edit conf/model/transformers.yaml as needed
```

#### Real-time Feature Engineering
```bash
# Install streaming capabilities
uv add mlx-plugin-streaming

# Configure streaming sources
# Edit conf/data/streaming.yaml as needed
```

### Hybrid Systems: Compose Multiple Domains
```bash
# Install multiple domain plugins
uv add mlx-plugin-agentic mlx-plugin-transformers mlx-plugin-streaming

# The plugin system automatically discovers and integrates all capabilities
# Access via unified API endpoints and configuration
```

## ✅ Success Validation Checklist

After running the setup commands, verify your configuration:

```bash
# 🔍 Environment Check
make verify-setup
# Expected: ✅ All tools (uv, trivy, docker) available
# Expected: ✅ Virtual environment configured
# Expected: ✅ Development dependencies installed

# 🧪 Quality Gates
make all-checks
# Expected: ✅ Linting passes (ruff)
# Expected: ✅ Formatting consistent (ruff format)
# Expected: ✅ Security scans clean (bandit, trivy)
# Expected: ✅ Tests pass (pytest)

# 🚀 System Integration
make run-api &
curl http://localhost:8000/health
# Expected: {"status": "healthy", "timestamp": "..."}

# 🎯 Project Personalization
grep -r "your-project-name" src/
# Expected: Your project name appears in imports, configs

# 🧩 Plugin Discovery
python -c "
from src.plugins import PluginRegistry
registry = PluginRegistry()
registry.discover_plugins()
print('Discovered plugins:', registry.list_plugins())
"
```

## 🛠️ Advanced Configuration

### Plugin-Specific Setup

#### 🤖 Agentic AI Systems
```bash
# After installing mlx-plugin-agentic, configure AI providers
cp .env-example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your-key
# ANTHROPIC_API_KEY=your-key
# LANGSMITH_API_KEY=your-key

# Plugin automatically provides additional endpoints:
# /v1/agents/chat - Multi-agent conversations
# /v1/agents/tools - Tool calling capabilities
```

#### 🧠 LLM Fine-tuning Systems
```bash
# After installing mlx-plugin-transformers, configure GPU environment
nvidia-smi  # Verify GPU availability

# Set up model cache directory
mkdir -p ~/.cache/huggingface
export HF_HOME=~/.cache/huggingface

# Login to Hugging Face (for gated models)
huggingface-cli login

# Plugin provides training endpoints:
# /v1/models/train - Start fine-tuning jobs
# /v1/models/status - Monitor training progress
```

#### 🏗️ Feature Engineering Systems
```bash
# After installing mlx-plugin-streaming, configure data connections
export KAFKA_SERVERS="your-kafka-servers"
export REDIS_URL="your-redis-url"

# Plugin provides streaming endpoints:
# /v1/features/stream - Real-time feature ingestion
# /v1/features/query - Feature retrieval
```

### Development Workflow Setup

```bash
# 🔄 Set up pre-commit hooks for code quality
make pre-commit

# 📖 Start documentation server (optional)
make serve-docs &
# View at http://localhost:8000

# 🐳 Build Docker image (for deployment testing)
make docker-build

# 🔍 Run security scans
make security-scan-local
```

## 🚨 Troubleshooting

### Common Issues & Solutions

**Issue**: `mlx` command not found
```bash
# Solution: Ensure file is executable
chmod +x mlx
./mlx your-project-name
```

**Issue**: Dependencies not installing
```bash
# Solution: Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal
```

**Issue**: Plugin not discovered
```bash
# Solution: Verify plugin installation and restart
uv list | grep mlx-plugin
make run-api  # Restart API to discover new plugins
```

**Issue**: Configuration conflicts between plugins
```bash
# Solution: Check plugin compatibility
python -c "
from src.plugins import PluginRegistry
registry = PluginRegistry()
registry.validate_plugin_compatibility()
"
```

**Happy Building!** 🚀
