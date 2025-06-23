# MLX Platform API Documentation

## Overview

The MLX Platform provides comprehensive APIs for building, deploying, and managing ML systems. This documentation covers all public APIs, including REST endpoints, Python SDK, plugin interfaces, and workflow APIs.

## API Categories

### Core Platform APIs
- **[Plugin Management API](plugin-management-api.md)** - Register, discover, and manage plugins
- **[Workflow Orchestration API](workflow-api.md)** - Create and execute ML workflows
- **[Configuration API](configuration-api.md)** - Hierarchical configuration management
- **[Observability API](observability-api.md)** - Metrics, tracing, and monitoring

### ML-Specific APIs
- **[Model Management API](model-management-api.md)** - Model training, evaluation, and deployment
- **[Data Processing API](data-processing-api.md)** - Data ingestion, transformation, and validation
- **[Feature Store API](feature-store-api.md)** - Feature engineering and serving
- **[Experiment Tracking API](experiment-tracking-api.md)** - ML experiment management

### Integration APIs
- **[External Service API](external-service-api.md)** - Third-party service integrations
- **[Authentication API](authentication-api.md)** - Security and access control
- **[Webhook API](webhook-api.md)** - Event notifications and callbacks

## API Standards

### Authentication
All APIs use JWT-based authentication with role-based access control (RBAC).

```bash
# Include authentication header
curl -H "Authorization: Bearer ${MLX_API_TOKEN}" \
     -H "Content-Type: application/json" \
     https://api.mlx.platform/v1/plugins
```

### Response Format
All APIs return standardized JSON responses:

```json
{
  "success": true,
  "data": { ... },
  "metadata": {
    "request_id": "req_123456",
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "v1",
    "execution_time_ms": 150
  },
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "has_next": true
  }
}
```

### Error Handling
Standardized error responses with troubleshooting information:

```json
{
  "success": false,
  "error": {
    "code": "PLUGIN_NOT_FOUND",
    "message": "Plugin 'databricks' not found",
    "details": {
      "plugin_name": "databricks",
      "available_plugins": ["snowflake", "mlflow"],
      "suggestions": ["Check plugin name spelling", "Ensure plugin is installed"]
    },
    "request_id": "req_123456",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Quick Start Examples

### Python SDK
```python
from mlx_platform import MLXClient

# Initialize client
client = MLXClient(
    api_token=os.getenv("MLX_API_TOKEN"),
    base_url="https://api.mlx.platform"
)

# List available plugins
plugins = client.plugins.list()

# Execute workflow
workflow = client.workflows.create({
    "name": "training_pipeline",
    "steps": [
        {"plugin": "snowflake", "operation": "load_data"},
        {"plugin": "databricks", "operation": "train_model"}
    ]
})

result = client.workflows.execute(workflow.id)
```

### REST API
```bash
# List plugins
curl https://api.mlx.platform/v1/plugins

# Create workflow
curl -X POST https://api.mlx.platform/v1/workflows \
  -d '{
    "name": "training_pipeline",
    "steps": [
      {"plugin": "snowflake", "operation": "load_data"},
      {"plugin": "databricks", "operation": "train_model"}
    ]
  }'
```

### CLI Interface
```bash
# Install MLX CLI
pip install mlx-cli

# Configure authentication
mlx auth login

# Manage plugins
mlx plugins list
mlx plugins install databricks

# Run workflows
mlx workflows create training_pipeline.yaml
mlx workflows run training_pipeline
```

## API Versioning

The MLX Platform uses semantic versioning for APIs:

- **Major version** (v1, v2): Breaking changes
- **Minor version** (v1.1, v1.2): New features, backward compatible
- **Patch version** (v1.1.1): Bug fixes

### Version Headers
```bash
# Specify API version
curl -H "Accept: application/vnd.mlx.v1+json" \
     https://api.mlx.platform/plugins
```

## Rate Limiting

APIs are rate limited to ensure fair usage:

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
X-RateLimit-Window: 3600
```

## Webhooks and Events

Subscribe to platform events for real-time notifications:

```python
# Register webhook
webhook = client.webhooks.create({
    "url": "https://your-app.com/webhooks/mlx",
    "events": ["workflow.completed", "model.deployed"],
    "secret": "webhook_secret_key"
})

# Webhook payload example
{
  "event": "workflow.completed",
  "data": {
    "workflow_id": "wf_123456",
    "status": "success",
    "execution_time": 1800,
    "artifacts": ["s3://bucket/model.pkl"]
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## SDK Reference

### Installation
```bash
pip install mlx-platform-sdk
```

### Configuration
```python
import mlx_platform

# Configure globally
mlx_platform.configure(
    api_token="your_token",
    base_url="https://api.mlx.platform",
    timeout=30,
    retries=3
)

# Or per-client
client = mlx_platform.MLXClient(
    api_token="your_token",
    region="us-west-2"
)
```

## Testing APIs

### Test Environment
Use the sandbox environment for development:

```python
client = MLXClient(
    api_token="test_token",
    base_url="https://sandbox.mlx.platform",
    environment="sandbox"
)
```

### Mock Responses
SDK includes mock capabilities for testing:

```python
from mlx_platform.testing import MockMLXClient

with MockMLXClient() as mock_client:
    mock_client.plugins.list.return_value = [
        {"name": "test_plugin", "version": "1.0.0"}
    ]

    # Your test code here
    plugins = mock_client.plugins.list()
    assert len(plugins) == 1
```

## Performance Guidelines

### Batch Operations
Use batch APIs for bulk operations:

```python
# Batch plugin operations
results = client.plugins.batch_execute([
    {"plugin": "snowflake", "operation": "query", "sql": "SELECT * FROM table1"},
    {"plugin": "snowflake", "operation": "query", "sql": "SELECT * FROM table2"}
])
```

### Async Operations
Use async APIs for long-running operations:

```python
# Start async workflow
workflow = client.workflows.start_async("training_pipeline")

# Poll for completion
while not workflow.is_complete():
    time.sleep(10)
    workflow.refresh()

print(f"Workflow completed with status: {workflow.status}")
```

### Caching
SDK includes intelligent caching:

```python
# Enable response caching
client = MLXClient(
    cache_enabled=True,
    cache_ttl=300  # 5 minutes
)

# First call hits API
plugins = client.plugins.list()

# Second call uses cache
plugins_cached = client.plugins.list()  # Returns cached result
```

## Support and Resources

- **[API Reference](./reference/)** - Complete API documentation
- **[SDK Examples](./examples/)** - Code samples and tutorials
- **[Postman Collection](./postman/)** - Ready-to-use API collection
- **[OpenAPI Specification](./openapi.yaml)** - Machine-readable API spec
- **Support**: api-support@mlx.platform
- **Status Page**: https://status.mlx.platform
