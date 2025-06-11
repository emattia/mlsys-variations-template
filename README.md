# MLOps Template

A batteries-included template for shipping machine-learning systems to production.

**Highlights**

- FastAPI micro-service with interactive docs
- Docker & Docker-Compose for reproducible packaging
- CI/CD via GitHub Actions
- Opinionated quality stack: ruff, mypy, pytest, bandit, trivy
- Hydra + Pydantic configuration
- Plugin architecture for rapid extensibility

## Quick Start
```bash
# Clone and rename the template
git clone <repo-url> my-project && cd my-project
./mlsys my_project_name          # transforms packages & configs

# Run quality gate + start the API
make all-checks && make run-api
```
Browse http://localhost:8000/docs for interactive API docs.

## Choose a Specialization
Template branches layer domain-specific tooling on the same foundation:

| Branch | Domain Focus | Extras |
|--------|--------------|--------|
| main | General ML / analytics | scikit-learn, data workflows |
| agentic-ai-system | Multi-agent LLM apps | LangChain, AutoGen |
| llm-finetuning-system | LLM fine-tuning & serving | 🤗 Transformers, LoRA |
| chalk-feature-engineering | Real-time feature store | Chalk, streaming |

Clone directly into the specialization you need:
```bash
git clone -b agentic-ai-system <repo-url> my-agent-app
```
Full details live in `branching_strategy.md`.

## Requirements
* Python 3.9+
* [uv](https://github.com/astral-sh/uv) package manager
* Docker (optional but recommended)

## Everyday Commands
```bash
make install-dev   # deps + pre-commit hooks
make lint          # ruff + mypy
make test          # pytest
make run-api       # uvicorn src.api.main:app --reload
make build-docker  # multi-stage container
```
Run `make help` to list all targets.

## Configuration
1. `cp .env-example .env` and edit as needed.
2. Override any Hydra config in `conf/` or via CLI:
```bash
python script.py model.type=xgboost api.debug=false
```

## Documentation
Full guides live in `docs/` and can be served locally:
```bash
make docs-serve   # http://127.0.0.1:8001
```

## Contributing
`make all-checks` must pass before opening a PR. Pre-commit hooks enforce style and security checks.

## License
MIT
