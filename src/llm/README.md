# LLM Workflows

This directory contains Large Language Model (LLM) specific workflows and processing that are distinct from traditional ML approaches.

## Purpose

The `src/llm/` directory is reserved for:

- LLM-specific model loading and inference
- Prompt engineering and template management
- LLM fine-tuning workflows
- RAG (Retrieval-Augmented Generation) implementations
- Vector store interactions for LLM applications
- LLM evaluation and benchmarking

## Structure

```
src/llm/
├── __init__.py          # Package initialization
├── README.md           # This file
├── models/             # LLM model loading and management
├── prompts/            # Prompt templates and engineering
├── evaluation/         # LLM-specific evaluation metrics
└── workflows/          # End-to-end LLM workflows
```

## Distinction from src/ml/

- **src/ml/**: Traditional ML (sklearn, tabular data, classification/regression)
- **src/llm/**: Large Language Models (transformers, generation, text processing)

This separation ensures clear boundaries between different AI/ML paradigms in the codebase.
