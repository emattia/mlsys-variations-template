# MLOps Template Startup Guide

> *"Give me six hours to chop down a tree and I will spend the first four sharpening the axe."* - Abraham Lincoln

# Project Setup

This guide transforms you from template curious to production ready in under 10 minutes.

## 🎯 Quick Decision Matrix

**Choose your path based on your ML system type:**

| If you're building... | Use this branch | Key capabilities |
|---------------------|----------------|-----------------|
| 🤖 **AI Agents, Multi-agent systems, Tool-calling AI** | `agentic-ai-system` | LangChain, AutoGen, tool integration, safety |
| 🧠 **LLM Fine-tuning, Language model deployment** | `llm-finetuning-system` | Transformers, LoRA/QLoRA, distributed training |
| 🏗️ **Real-time features, Data transformation pipelines** | `chalk-feature-engineering` | Chalk integration, streaming, monitoring |
| 📊 **Traditional ML, Data science, Experimentation** | `main` (general branch) | Scikit-learn, notebooks, analysis workflows |

## 🚀 The 3-Command Setup

### Command 1: Clone Your Chosen Specialization

```bash
# 🤖 Agentic AI Systems
git clone -b agentic-ai-system https://github.com/yourusername/debug-toml-test.git my-agent-project
cd my-agent-project

# 🧠 LLM Fine-tuning Systems
git clone -b llm-finetuning-system https://github.com/yourusername/debug-toml-test.git my-llm-project
cd my-llm-project

# 🏗️ Feature Engineering Systems
git clone -b chalk-feature-engineering https://github.com/yourusername/debug-toml-test.git my-feature-project
cd my-feature-project

# 📊 General ML/Data Science
git clone https://github.com/yourusername/debug-toml-test.git my-analysis-project
cd my-analysis-project
```

### Command 2: Transform with mlsys

```bash
# 🎭 Transform template into your personalized project
./mlsys your-project-name

# Examples:
./mlsys customer-churn-predictor
./mlsys financial-risk-model
./mlsys document-qa-system
./mlsys real-time-recommender
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

### Command 3: Activate & Validate

```bash
# 🌟 Activate your personalized environment
source .venv/bin/activate

# 🎯 Verify everything works perfectly
make verify-setup

# 🚀 Run comprehensive checks
make all-checks

# 🎪 See what's possible (optional but recommended)
make demo-comprehensive
```

## ✅ Success Validation Checklist

After running the three commands, verify your setup:

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
```

## 🛠️ Advanced Configuration

### Branch-Specific Setup

#### 🤖 Agentic AI Systems
```bash
# After basic setup, configure AI providers
cp .env-example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your-key
# ANTHROPIC_API_KEY=your-key
# LANGSMITH_API_KEY=your-key

# Install additional dependencies if needed
uv pip install langchain-experimental crewai autogen
```

#### 🧠 LLM Fine-tuning Systems
```bash
# Configure GPU environment
nvidia-smi  # Verify GPU availability

# Set up model cache directory
mkdir -p ~/.cache/huggingface
export HF_HOME=~/.cache/huggingface

# Login to Hugging Face (for gated models)
huggingface-cli login
```

#### 🏗️ Feature Engineering Systems
```bash
# Configure Chalk integration
cp conf/features/chalk_example.yaml conf/features/chalk.yaml
# Edit with your Chalk project details

# Set up data connections
export SNOWFLAKE_CONNECTION="your-connection-string"
export KAFKA_SERVERS="your-kafka-servers"
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

**Issue**: `mlsys` command not found
```bash
# Solution: Ensure file is executable
chmod +x mlsys
./mlsys your-project-name
```

**Issue**: Dependencies not installing
```bash
# Solution: Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal
```

**Issue**: Tests failing after transformation
```bash
# Solution: Check import paths
make lint  # Will show any import issues
# Fix imports manually or re-run mlsys
```

**Issue**: Docker build failing
```bash
# Solution: Check Docker is running
docker --version
make docker-build
```

### Branch-Specific Troubleshooting

#### 🤖 Agentic AI Issues
- **API Key Errors**: Verify `.env` file configuration
- **Tool Import Errors**: Run `uv pip install -e ".[agents]"`
- **Memory Issues**: Adjust agent concurrency in `conf/agentic/`

#### 🧠 LLM Fine-tuning Issues
- **CUDA Errors**: Verify GPU setup with `nvidia-smi`
- **Memory Errors**: Reduce batch size in training configs
- **Model Access**: Ensure Hugging Face authentication

#### 🏗️ Feature Engineering Issues
- **Chalk Connection**: Verify API keys and project configuration
- **Data Source Errors**: Check connection strings in `.env`
- **Feature Validation**: Run `make demo-data` to test pipelines

## 🎉 You're Ready!

**Congratulations!** You now have a production-ready ML system that includes:

- ✅ **Quality Assured**: Automated testing, linting, security scanning
- ✅ **Production Ready**: Docker containers, CI/CD pipelines, monitoring
- ✅ **Well Documented**: API docs, user guides, development guides
- ✅ **Extensible**: Plugin architecture, configuration management
- ✅ **Team Friendly**: Consistent patterns, automated quality checks

### Next Steps

1. **📊 Explore the Demo**: `make demo-comprehensive`
2. **📖 Read the Docs**: Browse `docs/` directory
3. **🔌 Check the APIs**: Visit `http://localhost:8000/docs`
4. **🧪 Write Your First Test**: Add to `tests/` directory
5. **🚀 Deploy**: Use Docker or cloud deployment guides

**Happy Building!** 🚀
