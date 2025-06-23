# Plugin Management API

## Overview

The Plugin Management API provides comprehensive capabilities for registering, discovering, configuring, and managing plugins in the MLX platform. This API supports both runtime plugin management and development-time plugin operations.

## Base URL
```
https://api.mlx.platform/v1/plugins
```

## Authentication
All endpoints require authentication via JWT token in the Authorization header.

## Endpoints

### List Plugins

**GET** `/plugins`

List all available plugins with filtering and pagination.

#### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `type` | string | No | Filter by plugin type (e.g., `data_source`, `ml_platform`) |
| `status` | string | No | Filter by status (`active`, `inactive`, `error`) |
| `category` | string | No | Filter by category |
| `page` | integer | No | Page number (default: 1) |
| `limit` | integer | No | Items per page (default: 20, max: 100) |
| `search` | string | No | Search plugin names and descriptions |

#### Example Request
```bash
curl -H "Authorization: Bearer $TOKEN" \
     "https://api.mlx.platform/v1/plugins?type=data_source&status=active"
```

#### Example Response
```json
{
  "success": true,
  "data": [
    {
      "id": "plugin_snowflake_001",
      "name": "snowflake",
      "type": "data_source",
      "version": "1.2.0",
      "status": "active",
      "description": "Snowflake cloud data warehouse integration",
      "capabilities": ["sql_queries", "snowpark_ml", "secure_views"],
      "conflicts_with": [],
      "integration_points": ["ml_platforms", "feature_stores"],
      "resource_requirements": {
        "min_memory_mb": 1024,
        "min_cpu_cores": 2,
        "gpu_required": false
      },
      "configuration": {
        "required_fields": ["account", "user", "password", "database"],
        "optional_fields": ["warehouse", "schema", "role"]
      },
      "health": {
        "status": "healthy",
        "last_check": "2024-01-15T10:30:00Z",
        "response_time_ms": 45
      },
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-15T09:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 1,
    "has_next": false
  }
}
```

### Get Plugin Details

**GET** `/plugins/{plugin_id}`

Get detailed information about a specific plugin.

#### Path Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `plugin_id` | string | Yes | Unique plugin identifier |

#### Example Request
```bash
curl -H "Authorization: Bearer $TOKEN" \
     "https://api.mlx.platform/v1/plugins/plugin_snowflake_001"
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "id": "plugin_snowflake_001",
    "name": "snowflake",
    "type": "data_source",
    "version": "1.2.0",
    "status": "active",
    "description": "Snowflake cloud data warehouse with Snowpark ML",
    "metadata": {
      "capabilities": ["sql_queries", "snowpark_ml", "time_travel"],
      "data_formats": ["sql_table", "parquet", "csv", "json"],
      "integration_points": ["ml_platforms", "feature_stores"],
      "conflicts_with": [],
      "requires": [],
      "optional_deps": ["dbt-plugin", "feast-plugin"],
      "supported_platforms": ["linux", "darwin", "win32"],
      "min_python_version": "3.9"
    },
    "configuration_schema": {
      "account": {
        "type": "string",
        "required": true,
        "description": "Snowflake account identifier"
      },
      "user": {
        "type": "string",
        "required": true,
        "description": "Username for authentication"
      },
      "password": {
        "type": "string",
        "required": true,
        "secret": true,
        "description": "Password for authentication"
      },
      "warehouse": {
        "type": "string",
        "default": "COMPUTE_WH",
        "description": "Compute warehouse to use"
      }
    },
    "performance_metrics": {
      "avg_execution_time_ms": 250,
      "success_rate": 0.998,
      "error_rate": 0.002,
      "throughput_ops_per_second": 45
    },
    "documentation_url": "https://docs.mlx.platform/plugins/snowflake",
    "repository_url": "https://github.com/mlx-platform/plugin-snowflake"
  }
}
```

### Register Plugin

**POST** `/plugins`

Register a new plugin in the platform.

#### Request Body
```json
{
  "name": "custom_data_source",
  "type": "data_source",
  "version": "1.0.0",
  "description": "Custom data source plugin",
  "class_path": "my_company.plugins.CustomDataSource",
  "configuration_schema": {
    "connection_string": {
      "type": "string",
      "required": true,
      "description": "Database connection string"
    }
  },
  "metadata": {
    "capabilities": ["sql_queries", "batch_processing"],
    "conflicts_with": [],
    "requires": [],
    "min_memory_mb": 512,
    "min_cpu_cores": 1
  }
}
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "id": "plugin_custom_001",
    "name": "custom_data_source",
    "status": "registered",
    "validation_results": {
      "schema_valid": true,
      "class_loadable": true,
      "dependencies_available": true,
      "conflicts_detected": false
    },
    "next_steps": [
      "Activate plugin to make it available for use",
      "Configure plugin with required parameters",
      "Test plugin functionality"
    ]
  }
}
```

### Activate Plugin

**POST** `/plugins/{plugin_id}/activate`

Activate a registered plugin to make it available for execution.

#### Request Body
```json
{
  "configuration": {
    "connection_string": "postgresql://user:pass@host:5432/db",
    "pool_size": 10,
    "timeout": 30
  },
  "auto_start": true
}
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "id": "plugin_custom_001",
    "status": "active",
    "activation_time": "2024-01-15T10:35:00Z",
    "health_check": {
      "status": "healthy",
      "response_time_ms": 120,
      "last_check": "2024-01-15T10:35:30Z"
    }
  }
}
```

### Deactivate Plugin

**POST** `/plugins/{plugin_id}/deactivate`

Deactivate an active plugin.

#### Request Body
```json
{
  "force": false,
  "graceful_shutdown_timeout": 30
}
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "id": "plugin_custom_001",
    "status": "inactive",
    "deactivation_time": "2024-01-15T10:40:00Z",
    "pending_operations": 0
  }
}
```

### Update Plugin Configuration

**PATCH** `/plugins/{plugin_id}/configuration`

Update plugin configuration without restarting.

#### Request Body
```json
{
  "configuration": {
    "pool_size": 15,
    "timeout": 45
  },
  "restart_required": false
}
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "id": "plugin_custom_001",
    "configuration_updated": true,
    "restart_required": false,
    "updated_fields": ["pool_size", "timeout"],
    "effective_time": "2024-01-15T10:45:00Z"
  }
}
```

### Plugin Health Check

**GET** `/plugins/{plugin_id}/health`

Get current health status of a plugin.

#### Example Response
```json
{
  "success": true,
  "data": {
    "plugin_id": "plugin_snowflake_001",
    "status": "healthy",
    "checks": {
      "connectivity": {
        "status": "pass",
        "response_time_ms": 45,
        "last_check": "2024-01-15T10:30:00Z"
      },
      "resource_usage": {
        "status": "pass",
        "memory_usage_mb": 512,
        "cpu_usage_percent": 15
      },
      "dependencies": {
        "status": "pass",
        "missing_dependencies": []
      }
    },
    "metrics": {
      "uptime_seconds": 86400,
      "total_requests": 1542,
      "successful_requests": 1539,
      "failed_requests": 3,
      "avg_response_time_ms": 234
    }
  }
}
```

### Plugin Metrics

**GET** `/plugins/{plugin_id}/metrics`

Get performance and usage metrics for a plugin.

#### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `start_time` | string | No | Start time for metrics (ISO 8601) |
| `end_time` | string | No | End time for metrics (ISO 8601) |
| `granularity` | string | No | Metrics granularity (`minute`, `hour`, `day`) |

#### Example Response
```json
{
  "success": true,
  "data": {
    "plugin_id": "plugin_snowflake_001",
    "time_range": {
      "start": "2024-01-15T09:00:00Z",
      "end": "2024-01-15T10:00:00Z"
    },
    "metrics": {
      "execution_count": 145,
      "success_rate": 0.993,
      "avg_execution_time_ms": 287,
      "p95_execution_time_ms": 650,
      "p99_execution_time_ms": 1200,
      "error_count": 1,
      "resource_usage": {
        "avg_memory_mb": 423,
        "peak_memory_mb": 678,
        "avg_cpu_percent": 12
      }
    },
    "timeseries": [
      {
        "timestamp": "2024-01-15T09:00:00Z",
        "execution_count": 24,
        "avg_execution_time_ms": 245,
        "success_rate": 1.0
      }
    ]
  }
}
```

### Discover Plugins

**GET** `/plugins/discover`

Discover plugins from various sources (file system, package registries, etc.).

#### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sources` | array | No | Discovery sources (`filesystem`, `pypi`, `git`) |
| `include_namespace` | string | No | Include namespace packages |
| `auto_register` | boolean | No | Automatically register discovered plugins |

#### Example Response
```json
{
  "success": true,
  "data": {
    "discovered_plugins": [
      {
        "name": "mlflow_integration",
        "source": "pypi",
        "version": "2.1.0",
        "type": "experiment_tracker",
        "compatible": true,
        "conflicts": [],
        "auto_registered": false
      }
    ],
    "discovery_summary": {
      "total_discovered": 1,
      "compatible_plugins": 1,
      "incompatible_plugins": 0,
      "auto_registered": 0
    }
  }
}
```

### Plugin Composition Validation

**POST** `/plugins/validate-composition`

Validate a composition of plugins for conflicts and compatibility.

#### Request Body
```json
{
  "plugins": [
    {
      "name": "snowflake",
      "version": "1.2.0"
    },
    {
      "name": "databricks",
      "version": "2.0.0"
    },
    {
      "name": "mlflow",
      "version": "1.8.0"
    }
  ]
}
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "composition_valid": true,
    "conflicts": [],
    "warnings": [
      {
        "type": "version_mismatch",
        "message": "MLflow version 1.8.0 is older than recommended 2.0.0+",
        "severity": "low",
        "plugins": ["mlflow"]
      }
    ],
    "synergies": [
      {
        "plugins": ["snowflake", "databricks"],
        "integration_points": ["data_sharing", "delta_lake"],
        "benefits": ["Zero-copy data sharing", "Unified governance"]
      }
    ],
    "recommendations": [
      "Consider upgrading MLflow to version 2.0.0+ for better Databricks integration"
    ]
  }
}
```

## Error Responses

### Common Error Codes
| Code | Description |
|------|-------------|
| `PLUGIN_NOT_FOUND` | Plugin does not exist |
| `PLUGIN_CONFLICT` | Plugin conflicts with active plugins |
| `INVALID_CONFIGURATION` | Plugin configuration is invalid |
| `PLUGIN_HEALTH_CHECK_FAILED` | Plugin health check failed |
| `INSUFFICIENT_RESOURCES` | Not enough resources to activate plugin |
| `DEPENDENCY_MISSING` | Required dependency not available |

### Example Error Response
```json
{
  "success": false,
  "error": {
    "code": "PLUGIN_CONFLICT",
    "message": "Plugin 'wandb' conflicts with active plugin 'mlflow'",
    "details": {
      "conflicting_plugin": "mlflow",
      "conflict_type": "singleton_type",
      "resolution_options": [
        "Deactivate 'mlflow' before activating 'wandb'",
        "Use 'mlflow' for experiment tracking instead"
      ]
    },
    "request_id": "req_123456"
  }
}
```

## SDK Examples

### Python SDK
```python
from mlx_platform import MLXClient

client = MLXClient(api_token="your_token")

# List plugins
plugins = client.plugins.list(type="data_source", status="active")

# Get plugin details
plugin = client.plugins.get("plugin_snowflake_001")

# Register new plugin
new_plugin = client.plugins.register({
    "name": "custom_source",
    "type": "data_source",
    "version": "1.0.0",
    "class_path": "my_package.CustomPlugin"
})

# Activate plugin
client.plugins.activate(new_plugin.id, {
    "configuration": {"connection_string": "..."}
})

# Health check
health = client.plugins.health_check(new_plugin.id)
print(f"Plugin health: {health.status}")
```

### Async Python SDK
```python
import asyncio
from mlx_platform.async_client import AsyncMLXClient

async def manage_plugins():
    async with AsyncMLXClient(api_token="your_token") as client:
        # Parallel plugin operations
        plugins, health = await asyncio.gather(
            client.plugins.list(),
            client.plugins.health_check("plugin_snowflake_001")
        )

        return plugins, health

plugins, health = asyncio.run(manage_plugins())
```
