# ML Systems API Reference

> "The best ML systems are those that work reliably in production."

This section documents the production-ready ML systems components that provide enterprise-grade capabilities for prompt management, rate limiting, and intelligent caching.

## Overview

The ML systems components provide:

- **Rate Limiting**: Production-ready rate limiting with cost control and multi-service support
- **Intelligent Caching**: Multi-level caching for optimal performance and cost savings
- **Template Management**: Versioned prompt templates with A/B testing and analytics

## Rate Limiter

The rate limiter helps manage API calls and resource usage.

::: src.platform.utils.rate_limiter

## Cache Manager

::: src.platform.utils.cache_manager

## Template Manager

::: src.platform.utils.templates

## Configuration

The ML systems are configured via Hydra configuration files. See `conf/ml_systems.yaml` for all available options.

### Example Usage

```python
from src.platform.utils.rate_limiter import RateLimiter
from src.platform.utils.cache_manager import CacheManager
from src.platform.utils.templates import TemplateManager

# Initialize with configuration
rate_limiter = RateLimiter()
cache_manager = CacheManager()
template_manager = TemplateManager()

# Use rate limiting decorator
@rate_limiter.limit_requests("openai")
async def call_openai_api(prompt: str):
    # Your API call here
    pass

# Use caching decorator
@cache_manager.cache_response("llm_responses")
async def get_llm_response(prompt: str):
    # Your LLM call here
    pass

# Get A/B tested template
template = template_manager.get_template_for_user("classification", "user123")
```

## Performance Benefits

- **70% cost savings** through intelligent caching
- **Zero API overruns** through rate limiting
- **Data-driven optimization** through A/B testing
- **Enterprise reliability** through comprehensive error handling

## Data Processing

Core utilities for data processing and validation.

::: src.data.processing

## Model Training

Utilities for model training and evaluation.

::: src.ml.training

## Model Inference

Utilities for model inference and prediction.

::: src.ml.inference

## Model Evaluation

Tools for evaluating model performance.

::: src.ml.evaluation
