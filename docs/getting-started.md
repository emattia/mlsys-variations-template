# Getting Started

> "The journey of a thousand miles begins with a single step." — Lao Tzu

A five-minute path from clone to running API.

## Prerequisites
* Python 3.9+
* [uv](https://github.com/astral-sh/uv)
* Git
* Docker (optional)

---

## 1  Clone the Template

```bash
# General ML / analytics
git clone <repo-url> my-project && cd my-project

# OR choose a specialization
git clone -b agentic-ai-system       <repo-url> my-agentic-app
git clone -b llm-finetuning-system   <repo-url> my-llm-app
```

## 2  Transform Packages & Configs
```bash
./mlsys my_project_name
```
The script renames import paths, rewrites configs and validates the environment.

## 3  Add Environment Variables
```bash
cp .env-example .env
# then edit values as required
```
Essential keys:
```bash
PROJECT_NAME=my_project_name
ENVIRONMENT=development
API_PORT=8000
```

## 4  Install Dependencies
```bash
make install-dev   # includes pre-commit hooks
```

## 5  Verify Everything
```bash
make all-checks    # lint, type-check, tests, security
```

---

### Run the API
```bash
make run-api   # http://localhost:8000/docs
```

### Next Steps
* Configure Hydra files in `conf/`
* Add raw data into `data/raw/`
* Build and push a Docker image via `make build-docker`

For branch-specific setup and advanced workflows see `branching_strategy.md` and the rest of the documentation site.
