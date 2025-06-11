# Getting Started

## 🚀 From Template to Production: The Complete Journey

> *"Every expert was once a beginner. Every pro was once an amateur. Every icon was once an unknown."* - Robin Sharma

This MLOps template is designed to transform from a generic foundation into your specialized machine learning system in minutes, not hours. Whether you're building agentic AI systems, fine-tuning large language models, or creating advanced feature engineering pipelines, this template provides the production-ready foundation you need.

### The Three-Step Transformation

#### Step 1: Fork & Clone (Choose Your Path)

The template supports three specialized branches, each optimized for different ML system types:

```bash
# 🤖 For Agentic AI Systems (multi-agent, tool-calling, autonomous decision-making)
git clone -b agentic-ai-system https://github.com/yourusername/debug-toml-test.git my-agent-project

# 🧠 For LLM Fine-tuning Systems (training, evaluation, deployment of language models)
git clone -b llm-finetuning-system https://github.com/yourusername/debug-toml-test.git my-llm-project

# 🏗️ For Feature Engineering Systems (Chalk integration, real-time features, data transformation)
git clone -b chalk-feature-engineering https://github.com/yourusername/debug-toml-test.git my-feature-project

# 📊 For General ML/Data Science Projects (classic analysis, modeling, experimentation)
git clone https://github.com/yourusername/debug-toml-test.git my-analysis-project
```

#### Step 2: Bootstrap & Transform (One Command Setup)

The `mlsys` script is your project's digital alchemist—it transforms the generic template into your personalized project:

```bash
cd my-project-directory

# 🎭 Transform the template (this is where the magic happens)
./mlsys your-project-name
# Example: ./mlsys customer-churn-predictor
```

**What happens during transformation:**
- 🔄 **Intelligent Bootstrapping**: Creates isolated environment, installs dependencies
- 📝 **Smart Renaming**: Updates package names, imports, and configurations throughout the codebase
- 🎨 **Documentation Refresh**: Personalizes all documentation with your project name
- ⚙️ **Configuration Alignment**: Updates `pyproject.toml`, linting rules, and tool configurations
- 🧪 **Validation**: Ensures everything is properly configured and ready for development

#### Step 3: Activate & Accelerate (Start Building)

```bash
# 🌟 Activate your personalized environment
source .venv/bin/activate

# 🎯 Verify everything works
make all-checks

# 🚀 Start developing immediately
make demo-comprehensive  # See what's possible
make run-api            # Start the API server
make test               # Run the test suite
```

### The Template's Philosophy: "Stable Foundation, Infinite Possibilities"

```
🌱 SEED (Template)     →  🌳 TREE (Your Project)  →  🍎 FRUIT (Production System)
   Potential               Specialization            Value Creation
```

- **🏗️ Production-Ready Foundation**: Docker, CI/CD, testing, monitoring, security scanning
- **🔧 Zero-Configuration Start**: One command transforms template to working project
- **🎨 Specialization Support**: Branch-specific optimizations for different ML domains
- **📈 Scale-Ready Architecture**: Plugin system, configuration management, distributed computing support
- **👥 Team-Friendly**: Clear documentation, consistent patterns, automated quality checks

### Quick Verification Checklist

After running `./mlsys your-project-name`, verify your setup:

```bash
✅ make verify-setup     # Check tools and environment
✅ make all-checks       # Run quality checks
✅ make test            # Execute test suite
✅ make run-api         # Start development server
✅ make demo-*          # Run demonstrations
```

**🎉 Success Indicators:**
- All quality checks pass
- Tests execute successfully
- API server starts on `http://localhost:8000`
- Your project name appears throughout the codebase
- Documentation reflects your project specifics

### Next Steps Based on Your Branch

| Branch Type | Immediate Next Steps |
|-------------|---------------------|
| **🤖 Agentic AI** | Configure LLM providers → Define agent tools → Build multi-agent workflows |
| **🧠 LLM Fine-tuning** | Set up GPU environment → Prepare training data → Configure model parameters |
| **🏗️ Feature Engineering** | Connect to Chalk → Define feature schemas → Build transformation pipelines |
| **📊 General ML** | Load your data → Explore with notebooks → Build training workflows |

---

## Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Python package installer)
- Git

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/debug-toml-test.git
cd debug-toml-test
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
uv pip install -e ".[dev]"
```

4. Set up pre-commit hooks:

```bash
pre-commit install
```

## Environment Configuration

1. Copy the example environment file:

```bash
cp .env-example .env
```

2. Edit `.env` with your specific configuration values.

## Project Structure

The template follows a structured organization:

- **data/**: All data files
- **src/**: Source code modules
- **notebooks/**: Jupyter notebooks
- **workflows/**: Data processing and model training workflows
- **tests/**: Test suite
- **reports/**: Generated reports and visualizations
- **models/**: Trained models and artifacts
- **documentation/**: Project documentation
- **endpoints/**: API endpoints and services

## First Steps

Here are some suggested first steps to get familiar with the template:

1. **Explore the example notebook**: Open `notebooks/example_analysis.ipynb` to see a complete example of a data analysis workflow.

2. **Run the tests**: Execute `pytest` to run the test suite and verify that everything is working correctly.

3. **Examine the workflow example**: Look at `workflows/data_processing.py` to understand how data processing workflows are structured.

4. **Review the source code**: Check out `src/data_utils.py` to see how reusable functions are organized.

## Adding Your Own Data

1. Place raw data files in the `data/raw/` directory **or ingest directly from an external source via a plugin**.  For example, to pull a Snowflake table into `data/raw/` you can run:

```bash
uv pip install snowflake-connector-python
python -m workflows.data_ingest \
    data.source=snowflake \
    data.snowflake.account=<ACCOUNT> \
    data.snowflake.database=<DB> \
    data.snowflake.query='SELECT * FROM SALES'
```

2. Create data processing workflows in the `workflows/` directory (see `workflows/data_processing.py` for an example).  These workflows should read from `data/raw/` **only** and write their outputs to `data/processed/`.

3. Keep large external datasets out of Git.  Instead reference them via environment variables (e.g. an S3 URI) or a plugin-specific config file under `conf/data/`.

## Creating Notebooks

1. Create new notebooks in the `notebooks/` directory.
2. Follow the structure of the example notebook.
3. Document your notebooks using NBDoc (see `documentation/nbdoc_guide.md`).

## Developing Source Code

1. Put **pure, importable** functions and classes in `src/`.  These should not perform direct I/O – leave that to the workflows.
2. Orchestrate your library code inside `workflows/` scripts (or DAGs) which *do* perform I/O and move artefacts between `data/` ↔ `models/` ↔ `reports/`.
3. When adding new functionality:
   - Write unit tests under `tests/unit/` for every public function.
   - Add or update a smoke test under `workflows/tests/` that executes the full workflow against a small fixture dataset (see `tests/fixtures/`).
4. If the code needs to be surfaced as an API, create an endpoint inside `endpoints/` and add an integration test under `tests/integration/`.

## Running Tests

Run the test suite to ensure your code is working correctly:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src

# Run workflow tests
pytest workflows/tests/
```

## Building Documentation

```bash
# 1. Turn notebooks → markdown (nbdoc)
python -m nbdoc build notebooks/ -o docs/generated/notebooks

# 2. Generate API docs from docstrings (pdoc)
python -m pdoc --html --output-dir docs/generated/api src

# 3. Build & serve the MkDocs site with live reload
mkdocs serve  # http://127.0.0.1:8000

# 4. Or run everything in one step
make docs  # output goes to the local 'site/' directory
```

A GitHub Actions workflow (`.github/workflows/docs.yml`) publishes the built site to **GitHub Pages** on every push to `main`.

## Next Steps

Once you're familiar with the template, you can:

- Customize the project structure to fit your needs
- Add your own data processing workflows
- Develop custom models and analysis pipelines
- Set up continuous integration with GitHub Actions
- Deploy models as APIs using the endpoints framework
