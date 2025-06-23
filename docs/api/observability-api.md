# Observability API

## Overview

The Observability API provides comprehensive monitoring, metrics collection, tracing, and alerting capabilities for the MLX platform. Built on OpenTelemetry and Prometheus, it offers real-time insights into system performance and ML operation health.

## Base URL
```
https://api.mlx.platform/v1/observability
```

## Metrics API

### Query Metrics

**GET** `/metrics/query`

Query time-series metrics using PromQL syntax.

#### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | PromQL query expression |
| `time` | string | No | Evaluation timestamp (RFC3339) |
| `timeout` | string | No | Query timeout (default: 30s) |

#### Example Request
```bash
curl "https://api.mlx.platform/v1/observability/metrics/query?query=mlx_plugin_execution_duration_seconds"
```

#### Example Response
```json
{
  "success": true,
  "data": {
    "resultType": "vector",
    "result": [
      {
        "metric": {
          "__name__": "mlx_plugin_execution_duration_seconds",
          "plugin": "snowflake",
          "operation": "query",
          "status": "success"
        },
        "value": [1642248000, "2.45"]
      }
    ]
  }
}
```

### Query Range Metrics

**GET** `/metrics/query_range`

Query metrics over a time range.

#### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | PromQL query expression |
| `start` | string | Yes | Start timestamp (RFC3339) |
| `end` | string | Yes | End timestamp (RFC3339) |
| `step` | string | Yes | Query resolution step width |

#### Example Response
```json
{
  "success": true,
  "data": {
    "resultType": "matrix",
    "result": [
      {
        "metric": {
          "plugin": "databricks",
          "operation": "train_model"
        },
        "values": [
          [1642248000, "45.2"],
          [1642248060, "47.8"],
          [1642248120, "44.1"]
        ]
      }
    ]
  }
}
```

### Available Metrics

**GET** `/metrics/metadata`

List all available metrics with descriptions and types.

#### Example Response
```json
{
  "success": true,
  "data": {
    "mlx_plugin_execution_duration_seconds": {
      "type": "histogram",
      "help": "Time spent executing plugin operations",
      "unit": "seconds",
      "labels": ["plugin", "operation", "status"]
    },
    "mlx_model_prediction_latency_seconds": {
      "type": "histogram",
      "help": "Model prediction latency",
      "unit": "seconds",
      "labels": ["model_name", "model_version"]
    },
    "mlx_data_drift_score": {
      "type": "gauge",
      "help": "Data drift detection score",
      "unit": "ratio",
      "labels": ["dataset", "feature_group"]
    },
    "mlx_workflow_execution_total": {
      "type": "counter",
      "help": "Total workflow executions",
      "unit": "count",
      "labels": ["workflow_name", "status"]
    }
  }
}
```

## Tracing API

### Get Traces

**GET** `/traces`

Search and retrieve distributed traces.

#### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `service` | string | No | Filter by service name |
| `operation` | string | No | Filter by operation name |
| `start_time` | string | No | Start time for search (RFC3339) |
| `end_time` | string | No | End time for search (RFC3339) |
| `min_duration` | string | No | Minimum trace duration |
| `max_duration` | string | No | Maximum trace duration |
| `tags` | string | No | Tag filters (key:value format) |
| `limit` | integer | No | Maximum number of traces (default: 100) |

#### Example Response
```json
{
  "success": true,
  "data": {
    "traces": [
      {
        "trace_id": "abc123def456",
        "spans": [
          {
            "span_id": "span_001",
            "operation_name": "workflow.execute",
            "start_time": "2024-01-15T14:30:00Z",
            "end_time": "2024-01-15T14:35:00Z",
            "duration_ms": 300000,
            "tags": {
              "workflow.name": "training_pipeline",
              "workflow.version": "1.2.0"
            },
            "logs": [
              {
                "timestamp": "2024-01-15T14:30:15Z",
                "level": "info",
                "message": "Starting workflow execution"
              }
            ]
          }
        ],
        "processes": {
          "p1": {
            "service_name": "mlx-workflow-engine",
            "tags": {
              "hostname": "worker-01",
              "version": "2.1.0"
            }
          }
        }
      }
    ],
    "total": 1,
    "limit": 100
  }
}
```

### Get Trace Details

**GET** `/traces/{trace_id}`

Get detailed information for a specific trace.

#### Example Response
```json
{
  "success": true,
  "data": {
    "trace_id": "abc123def456",
    "duration_ms": 300000,
    "span_count": 15,
    "service_count": 4,
    "error_count": 0,
    "spans": [
      {
        "span_id": "span_001",
        "parent_span_id": null,
        "operation_name": "workflow.execute",
        "service_name": "mlx-workflow-engine",
        "start_time": "2024-01-15T14:30:00Z",
        "duration_ms": 300000,
        "tags": {
          "component": "workflow",
          "workflow.id": "wf_001",
          "execution.id": "exec_789"
        },
        "process": {
          "service_name": "mlx-workflow-engine",
          "hostname": "worker-01",
          "version": "2.1.0"
        }
      }
    ]
  }
}
```

## Alerting API

### List Alert Rules

**GET** `/alerts/rules`

List all configured alert rules.

#### Example Response
```json
{
  "success": true,
  "data": [
    {
      "id": "alert_high_error_rate",
      "name": "High Plugin Error Rate",
      "description": "Alert when plugin error rate exceeds 5%",
      "query": "rate(mlx_plugin_execution_total{status=\"error\"}[5m]) / rate(mlx_plugin_execution_total[5m]) > 0.05",
      "for": "2m",
      "severity": "warning",
      "enabled": true,
      "labels": {
        "team": "ml-platform",
        "category": "reliability"
      },
      "annotations": {
        "summary": "Plugin {{ $labels.plugin }} error rate is {{ $value | humanizePercentage }}",
        "description": "Error rate has been above 5% for more than 2 minutes"
      }
    }
  ]
}
```

### Create Alert Rule

**POST** `/alerts/rules`

Create a new alert rule.

#### Request Body
```json
{
  "name": "Model Prediction Latency High",
  "description": "Alert when model prediction latency is consistently high",
  "query": "histogram_quantile(0.95, mlx_model_prediction_latency_seconds) > 2.0",
  "for": "5m",
  "severity": "critical",
  "labels": {
    "team": "ml-ops",
    "category": "performance"
  },
  "annotations": {
    "summary": "Model prediction latency is high",
    "description": "95th percentile latency has been above 2 seconds for 5 minutes",
    "runbook": "https://docs.company.com/runbooks/model-latency"
  },
  "notification_channels": ["slack-ml-alerts", "pagerduty-oncall"]
}
```

### Get Active Alerts

**GET** `/alerts/active`

Get currently firing alerts.

#### Example Response
```json
{
  "success": true,
  "data": [
    {
      "fingerprint": "alert_123456",
      "rule_id": "alert_high_error_rate",
      "rule_name": "High Plugin Error Rate",
      "state": "firing",
      "started_at": "2024-01-15T14:45:00Z",
      "last_seen": "2024-01-15T14:50:00Z",
      "value": 0.087,
      "labels": {
        "plugin": "databricks",
        "severity": "warning"
      },
      "annotations": {
        "summary": "Plugin databricks error rate is 8.7%"
      }
    }
  ]
}
```

## Health Checks API

### System Health

**GET** `/health`

Get overall system health status.

#### Example Response
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-15T15:00:00Z",
    "components": {
      "plugins": {
        "status": "healthy",
        "active_count": 8,
        "error_count": 0,
        "details": {
          "snowflake": "healthy",
          "databricks": "healthy",
          "mlflow": "healthy"
        }
      },
      "workflows": {
        "status": "healthy",
        "running_count": 3,
        "failed_count": 0,
        "queue_depth": 2
      },
      "resources": {
        "status": "healthy",
        "cpu_usage": 45.2,
        "memory_usage": 62.1,
        "disk_usage": 28.5
      },
      "external_services": {
        "status": "degraded",
        "details": {
          "prometheus": "healthy",
          "jaeger": "healthy",
          "temporal": "degraded"
        }
      }
    }
  }
}
```

### Component Health

**GET** `/health/{component}`

Get health status for a specific component.

#### Example Response
```json
{
  "success": true,
  "data": {
    "component": "plugins",
    "status": "healthy",
    "last_check": "2024-01-15T15:00:00Z",
    "checks": [
      {
        "name": "plugin_availability",
        "status": "pass",
        "message": "All 8 plugins are responding",
        "response_time_ms": 25
      },
      {
        "name": "plugin_errors",
        "status": "pass",
        "message": "Error rate below threshold (0.2%)",
        "threshold": 0.05,
        "current_value": 0.002
      }
    ],
    "metrics": {
      "active_plugins": 8,
      "total_executions_last_hour": 1542,
      "avg_response_time_ms": 234,
      "error_rate": 0.002
    }
  }
}
```

## Dashboard API

### List Dashboards

**GET** `/dashboards`

List available monitoring dashboards.

#### Example Response
```json
{
  "success": true,
  "data": [
    {
      "id": "dashboard_platform_overview",
      "name": "MLX Platform Overview",
      "description": "High-level platform metrics and health",
      "category": "overview",
      "panels": [
        "plugin_execution_rate",
        "workflow_success_rate",
        "resource_utilization",
        "error_trends"
      ],
      "url": "https://grafana.mlx.platform/d/platform-overview"
    }
  ]
}
```

### Export Dashboard

**GET** `/dashboards/{dashboard_id}/export`

Export dashboard configuration.

#### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `format` | string | No | Export format (`json`, `yaml`) |

## Custom Metrics API

### Create Custom Metric

**POST** `/metrics/custom`

Create a custom metric definition.

#### Request Body
```json
{
  "name": "model_accuracy_score",
  "type": "gauge",
  "description": "Model accuracy score from validation",
  "unit": "ratio",
  "labels": ["model_name", "dataset", "validation_type"],
  "retention_days": 90
}
```

### Record Metric Value

**POST** `/metrics/custom/{metric_name}/record`

Record a value for a custom metric.

#### Request Body
```json
{
  "value": 0.924,
  "labels": {
    "model_name": "churn_prediction_v2",
    "dataset": "holdout_2024_01",
    "validation_type": "cross_validation"
  },
  "timestamp": "2024-01-15T15:30:00Z"
}
```

## SLA Monitoring

### Configure SLA

**POST** `/sla/configure`

Configure service level agreement monitoring.

#### Request Body
```json
{
  "name": "model_prediction_sla",
  "description": "Model prediction latency SLA",
  "objectives": [
    {
      "metric": "mlx_model_prediction_latency_seconds",
      "percentile": 95,
      "threshold": 1.0,
      "time_window": "24h"
    },
    {
      "metric": "mlx_model_prediction_success_rate",
      "threshold": 0.995,
      "time_window": "24h"
    }
  ],
  "notification_channels": ["sla-alerts"]
}
```

### Get SLA Status

**GET** `/sla/{sla_name}/status`

Get current SLA compliance status.

#### Example Response
```json
{
  "success": true,
  "data": {
    "sla_name": "model_prediction_sla",
    "status": "compliant",
    "objectives": [
      {
        "name": "latency_p95",
        "status": "compliant",
        "current_value": 0.85,
        "threshold": 1.0,
        "compliance_percentage": 98.2
      },
      {
        "name": "success_rate",
        "status": "compliant",
        "current_value": 0.997,
        "threshold": 0.995,
        "compliance_percentage": 99.8
      }
    ],
    "time_window": "24h",
    "last_updated": "2024-01-15T15:45:00Z"
  }
}
```

This observability API provides comprehensive monitoring capabilities essential for production ML systems, enabling teams to maintain high reliability and performance.
