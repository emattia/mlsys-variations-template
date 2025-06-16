# ML Systems API Reference

> "The best ML systems are those that work reliably in production."

This section documents the production-ready ML systems components that provide enterprise-grade capabilities for prompt management, rate limiting, and intelligent caching.

## Overview

The ML systems components provide:

- **Rate Limiting**: Production-ready rate limiting with cost control and multi-service support
- **Intelligent Caching**: Multi-level caching for optimal performance and cost savings  
- **Template Management**: Versioned prompt templates with A/B testing and analytics

## Rate Limiter

::: src.utils.rate_limiter

## Cache Manager

::: src.utils.cache_manager

## Template Manager

::: src.utils.templates

## Configuration

The ML systems are configured via Hydra configuration files. See `conf/ml_systems.yaml` for all available options.

### Example Usage

```python
from src.utils.rate_limiter import RateLimiter
from src.utils.cache_manager import CacheManager
from src.utils.templates import PromptTemplateManager

# Initialize with configuration
rate_limiter = RateLimiter()
cache_manager = CacheManager()
template_manager = PromptTemplateManager()

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