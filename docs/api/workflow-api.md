# Workflow Orchestration API

## Overview

The Workflow API provides durable, fault-tolerant orchestration for ML pipelines using Temporal as the execution engine. It supports complex DAG execution, automatic retry logic, and distributed execution across multiple compute resources.

## Base URL
```
https://api.mlx.platform/v1/workflows
```

## Endpoints

### List Workflows

**GET** `/workflows`

List all workflow definitions with filtering and search capabilities.

#### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `status` | string | No | Filter by status (`active`, `paused`, `completed`, `failed`) |
| `created_by` | string | No | Filter by creator user ID |
| `tags` | array | No | Filter by workflow tags |
| `page` | integer | No | Page number (default: 1) |
| `limit` | integer | No | Items per page (default: 20, max: 100) |

#### Example Response
```json
{
  "success": true,
  "data": [
    {
      "id": "wf_training_pipeline_001",
      "name": "training_pipeline",
      "version": "1.2.0",
      "status": "active",
      "description": "End-to-end ML training pipeline",
      "created_by": "user_123",
      "created_at": "2024-01-15T10:00:00Z",
      "last_execution": {
        "execution_id": "exec_456",
        "status": "completed",
        "started_at": "2024-01-15T09:00:00Z",
        "completed_at": "2024-01-15T09:45:00Z",
        "duration_seconds": 2700
      },
      "metrics": {
        "total_executions": 45,
        "success_rate": 0.978,
        "avg_duration_seconds": 2650
      },
      "tags": ["ml", "training", "production"]
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

### Create Workflow

**POST** `/workflows`

Create a new workflow definition.

#### Request Body
```json
{
  "name": "customer_churn_pipeline",
  "version": "1.0.0",
  "description": "Customer churn prediction model training",
  "steps": [
    {
      "id": "data_extraction",
      "plugin": "snowflake",
      "operation": "extract_data",
      "config": {
        "sql": "SELECT * FROM customer_features WHERE created_at >= '2024-01-01'",
        "output_format": "parquet"
      },
      "retry_policy": {
        "max_attempts": 3,
        "backoff_coefficient": 2.0,
        "initial_interval": "1m"
      },
      "timeout": "30m"
    },
    {
      "id": "feature_engineering",
      "plugin": "databricks",
      "operation": "process_features",
      "depends_on": ["data_extraction"],
      "config": {
        "cluster_id": "cluster_001",
        "notebook_path": "/feature_engineering/churn_features"
      },
      "resources": {
        "min_workers": 2,
        "max_workers": 10,
        "node_type": "i3.xlarge"
      }
    },
    {
      "id": "model_training",
      "plugin": "databricks",
      "operation": "train_model",
      "depends_on": ["feature_engineering"],
      "config": {
        "algorithm": "xgboost",
        "hyperparameters": {
          "max_depth": 6,
          "learning_rate": 0.1,
          "n_estimators": 100
        }
      }
    },
    {
      "id": "model_validation",
      "plugin": "mlflow",
      "operation": "validate_model",
      "depends_on": ["model_training"],
      "config": {
        "validation_dataset": "holdout_set",
        "metrics": ["auc", "precision", "recall", "f1"]
      }
    }
  ],
  "schedule": {
    "type": "cron",
    "expression": "0 2 * * *",
    "timezone": "UTC"
  },
  "notifications": {
    "on_success": ["user_123@company.com"],
    "on_failure": ["ml-team@company.com", "oncall@company.com"]
  },
  "tags": ["churn", "ml", "scheduled"]
}
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "id": "wf_customer_churn_001",
    "name": "customer_churn_pipeline",
    "version": "1.0.0",
    "status": "created",
    "validation_results": {
      "steps_valid": true,
      "dependencies_resolved": true,
      "plugins_available": true,
      "schedule_valid": true
    },
    "estimated_cost": {
      "compute_hours": 4.5,
      "estimated_usd": 23.50
    },
    "next_steps": [
      "Activate workflow to enable scheduling",
      "Test workflow with manual execution",
      "Configure monitoring alerts"
    ]
  }
}
```

### Execute Workflow

**POST** `/workflows/{workflow_id}/execute`

Start a workflow execution with optional parameter overrides.

#### Request Body
```json
{
  "execution_name": "churn_pipeline_20240115",
  "parameters": {
    "data_start_date": "2024-01-01",
    "data_end_date": "2024-01-15",
    "model_version": "v2.1"
  },
  "step_overrides": {
    "model_training": {
      "config": {
        "hyperparameters": {
          "n_estimators": 200,
          "max_depth": 8
        }
      }
    }
  },
  "priority": "high",
  "async": true
}
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "execution_id": "exec_789",
    "workflow_id": "wf_customer_churn_001",
    "status": "running",
    "started_at": "2024-01-15T14:30:00Z",
    "estimated_completion": "2024-01-15T17:15:00Z",
    "current_step": "data_extraction",
    "progress": {
      "completed_steps": 0,
      "total_steps": 4,
      "percentage": 0
    },
    "execution_url": "https://temporal.mlx.platform/execution/exec_789"
  }
}
```

### Get Execution Status

**GET** `/workflows/executions/{execution_id}`

Get detailed status and progress for a workflow execution.

#### Example Response
```json
{
  "success": true,
  "data": {
    "execution_id": "exec_789",
    "workflow_id": "wf_customer_churn_001",
    "status": "running",
    "started_at": "2024-01-15T14:30:00Z",
    "updated_at": "2024-01-15T14:45:00Z",
    "current_step": "feature_engineering",
    "progress": {
      "completed_steps": 1,
      "total_steps": 4,
      "percentage": 25
    },
    "step_details": [
      {
        "id": "data_extraction",
        "status": "completed",
        "started_at": "2024-01-15T14:30:00Z",
        "completed_at": "2024-01-15T14:42:00Z",
        "duration_seconds": 720,
        "output": {
          "records_extracted": 1500000,
          "file_size_mb": 450,
          "output_path": "s3://bucket/data/2024-01-15/customer_features.parquet"
        }
      },
      {
        "id": "feature_engineering",
        "status": "running",
        "started_at": "2024-01-15T14:42:30Z",
        "progress": {
          "current_operation": "calculating_aggregations",
          "percentage": 35
        },
        "resource_usage": {
          "cluster_id": "cluster_001",
          "workers": 5,
          "cpu_usage": 75,
          "memory_usage": 68
        }
      },
      {
        "id": "model_training",
        "status": "pending",
        "depends_on": ["feature_engineering"]
      },
      {
        "id": "model_validation",
        "status": "pending",
        "depends_on": ["model_training"]
      }
    ],
    "resource_usage": {
      "total_compute_hours": 1.2,
      "cost_usd": 6.75,
      "peak_memory_gb": 128,
      "peak_cpu_cores": 40
    }
  }
}
```

### Stop Execution

**POST** `/workflows/executions/{execution_id}/stop`

Stop a running workflow execution with optional cleanup options.

#### Request Body
```json
{
  "reason": "Manual stop requested by user",
  "graceful": true,
  "cleanup_resources": true,
  "save_partial_results": true
}
```

### Get Execution Logs

**GET** `/workflows/executions/{execution_id}/logs`

Retrieve logs for workflow execution with filtering options.

#### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `step_id` | string | No | Filter logs for specific step |
| `level` | string | No | Log level filter (`DEBUG`, `INFO`, `WARN`, `ERROR`) |
| `start_time` | string | No | Start time for log range (ISO 8601) |
| `end_time` | string | No | End time for log range (ISO 8601) |
| `limit` | integer | No | Number of log entries (default: 1000, max: 10000) |

#### Example Response
```json
{
  "success": true,
  "data": {
    "execution_id": "exec_789",
    "logs": [
      {
        "timestamp": "2024-01-15T14:30:15Z",
        "level": "INFO",
        "step_id": "data_extraction",
        "plugin": "snowflake",
        "message": "Starting data extraction from customer_features table",
        "metadata": {
          "correlation_id": "corr_123",
          "trace_id": "trace_456"
        }
      },
      {
        "timestamp": "2024-01-15T14:35:22Z",
        "level": "INFO",
        "step_id": "data_extraction",
        "plugin": "snowflake",
        "message": "Extracted 1,500,000 records successfully",
        "metadata": {
          "records_count": 1500000,
          "query_duration_ms": 45000
        }
      }
    ],
    "pagination": {
      "total_entries": 234,
      "returned_entries": 50,
      "has_more": true
    }
  }
}
```

### Schedule Management

**POST** `/workflows/{workflow_id}/schedule`

Create or update workflow schedule.

#### Request Body
```json
{
  "enabled": true,
  "schedule": {
    "type": "cron",
    "expression": "0 2 * * *",
    "timezone": "UTC"
  },
  "parameters": {
    "data_lookback_days": 7
  },
  "overlap_policy": "skip",
  "max_concurrent_executions": 1
}
```

### Workflow Templates

**GET** `/workflows/templates`

List available workflow templates.

#### Example Response
```json
{
  "success": true,
  "data": [
    {
      "id": "template_ml_training",
      "name": "ML Training Pipeline",
      "description": "Standard ML model training workflow",
      "category": "machine_learning",
      "steps": ["data_extraction", "preprocessing", "training", "validation", "deployment"],
      "estimated_duration": "45m",
      "resource_requirements": {
        "min_compute_hours": 2,
        "min_memory_gb": 16
      },
      "compatible_plugins": ["snowflake", "databricks", "mlflow"]
    }
  ]
}
```

## Webhook Events

The Workflow API publishes events for workflow lifecycle changes:

### Event Types
- `workflow.execution.started`
- `workflow.execution.completed`
- `workflow.execution.failed`
- `workflow.step.completed`
- `workflow.step.failed`

### Example Webhook Payload
```json
{
  "event": "workflow.execution.completed",
  "data": {
    "execution_id": "exec_789",
    "workflow_id": "wf_customer_churn_001",
    "status": "completed",
    "duration_seconds": 2650,
    "cost_usd": 18.45,
    "artifacts": [
      "s3://bucket/models/churn_model_v1.2.pkl",
      "s3://bucket/reports/validation_report.html"
    ]
  },
  "timestamp": "2024-01-15T17:15:00Z"
}
```

## Error Handling

### Workflow-specific Error Codes
| Code | Description |
|------|-------------|
| `WORKFLOW_NOT_FOUND` | Workflow definition does not exist |
| `EXECUTION_NOT_FOUND` | Workflow execution does not exist |
| `STEP_DEPENDENCY_ERROR` | Step dependencies cannot be resolved |
| `PLUGIN_UNAVAILABLE` | Required plugin is not available |
| `RESOURCE_INSUFFICIENT` | Insufficient compute resources |
| `SCHEDULE_CONFLICT` | Schedule conflicts with existing workflows |

## Performance Guidelines

### Optimization Strategies
- Use step parallelization where possible
- Configure appropriate retry policies
- Monitor resource usage and adjust cluster sizes
- Use workflow templates for common patterns
- Implement proper cleanup procedures

### Monitoring
- Set up alerts for workflow failures
- Monitor execution duration trends
- Track resource utilization
- Set up cost monitoring and budgets
