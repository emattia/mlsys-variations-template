# Branching Strategy

Four specialized branches provide optimized templates for different ML domains while maintaining the same production-ready foundation.

## Branch Overview

| Branch | Purpose | Key Technologies |
|--------|---------|------------------|
| `main` | General ML/Analytics | Scikit-learn, Pandas, Jupyter |
| `agentic-ai-system` | Multi-agent AI systems | LangChain, AutoGen, OpenAI |
| `llm-finetuning-system` | LLM training/deployment | Transformers, PyTorch, LoRA |
| `chalk-feature-engineering` | Real-time features | Chalk, streaming pipelines |

## Quick Setup

### General ML (main)
```bash
git clone <repo-url> my-project
cd my-project
./mlsys my-project-name
```

### Agentic AI
```bash
git clone -b agentic-ai-system <repo-url> my-agent-project
cd my-agent-project
./mlsys my-agent-project
# Set API keys in .env
export OPENAI_API_KEY=your_key
export LANGSMITH_API_KEY=your_key
```

### LLM Fine-tuning
```bash
git clone -b llm-finetuning-system <repo-url> my-llm-project
cd my-llm-project
./mlsys my-llm-project
# Install GPU dependencies if available
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Feature Engineering (Chalk)
```bash
git clone -b chalk-feature-engineering <repo-url> my-features-project
cd my-features-project
./mlsys my-features-project
# Configure Chalk credentials in .env
```

## Branch-Specific Components

### Agentic AI (`agentic-ai-system`)

**Additional Dependencies:**
- LangChain + OpenAI integration
- AutoGen for multi-agent coordination
- LangSmith for monitoring

**New Modules:**
```
src/agents/
├── base.py              # Agent interface
├── reasoning.py         # Planning agents
├── tools.py            # Tool calling
├── memory.py           # Context management
└── coordination.py     # Multi-agent workflows
```

**Example Usage:**
```python
from src.agents import ReasoningAgent, ToolAgent

# Create multi-agent system
planner = ReasoningAgent("gpt-4")
executor = ToolAgent(tools=["data_analyzer", "model_trainer"])

# Execute coordinated workflow
result = planner.coordinate_with(executor).execute(task)
```

### LLM Fine-tuning (`llm-finetuning-system`)

**Additional Dependencies:**
- Transformers + PyTorch
- PEFT (LoRA/QLoRA)
- Accelerate for distributed training
- VLLM for inference

**New Modules:**
```
src/llm/
├── training.py          # Fine-tuning workflows
├── evaluation.py        # LLM evaluation
├── inference.py         # Optimized serving
└── quantization.py      # Model compression
```

**Example Usage:**
```python
from src.llm import LLMTrainer, LLMEvaluator

# Fine-tune model
trainer = LLMTrainer("microsoft/DialoGPT-medium")
model = trainer.fine_tune(dataset="data/conversations.json", method="lora")

# Evaluate performance
evaluator = LLMEvaluator()
metrics = evaluator.evaluate(model, benchmark="hellaswag")
```

### Feature Engineering (`chalk-feature-engineering`)

**Additional Dependencies:**
- Chalk feature store
- Apache Kafka
- Redis for caching

**New Modules:**
```
src/features/
├── stores.py           # Feature store integration
├── streaming.py        # Real-time processing
├── transformations.py  # Feature engineering
└── monitoring.py       # Feature drift detection
```

**Example Usage:**
```python
from src.features import ChalkFeatureStore, StreamProcessor

# Define real-time features
store = ChalkFeatureStore()
processor = StreamProcessor()

# Stream processing pipeline
features = processor.transform(raw_data).to_chalk(store)
```

## Creating Custom Branches

### 1. Fork from Base Branch
```bash
# Create new specialized branch
git checkout main
git checkout -b my-custom-branch

# Add specific dependencies
echo "custom-package>=1.0.0" >> requirements.txt

# Commit specialization
git add .
git commit -m "Add custom ML specialization"
git push -u origin my-custom-branch
```

### 2. Customize Structure
Add domain-specific modules:
```bash
mkdir -p src/my_domain
touch src/my_domain/__init__.py
touch src/my_domain/core.py
```

### 3. Update Configuration
```yaml
# conf/my_domain/default.yaml
my_domain:
  model_type: "custom"
  parameters:
    learning_rate: 0.001
    batch_size: 32
```

### 4. Test Specialization
```bash
# Verify setup works
./mlsys test-project
make all-checks
make test
```

## Merging Strategy

### Upstream Updates
```bash
# Get latest base template improvements
git fetch origin main
git merge origin/main

# Resolve conflicts, prioritizing branch-specific code
# Test thoroughly after merge
make all-checks
```

### Contributing Back
```bash
# Create feature branch from main
git checkout main
git checkout -b feature/improvement

# Make changes that benefit all branches
# Submit PR to main branch
```

## Best Practices

1. **Keep base functionality**: Don't remove core template features
2. **Add, don't replace**: Extend rather than modify base components
3. **Document changes**: Update README and docs for branch-specific features
4. **Test compatibility**: Ensure `./mlsys` transformation works correctly
5. **Maintain CI/CD**: Update GitHub Actions for branch-specific testing

## Migration Between Branches

### From General to Specialized
```bash
# Backup current work
git stash

# Switch to specialized branch
git fetch origin
git checkout specialized-branch-name

# Apply your changes
git stash pop

# Resolve conflicts and test
```

### From Specialized to General
```bash
# Remove branch-specific dependencies
uv pip uninstall specialized-packages

# Switch to main branch
git checkout main

# Reinstall base dependencies
make install-dev
```
