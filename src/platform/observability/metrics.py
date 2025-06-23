"""Metrics collection and export for the MLX platform.

This module provides:
- Prometheus metrics collection with custom ML metrics
- Performance monitoring for ML operations
- Resource utilization tracking
- Model quality metrics and drift detection
- Business metrics for ML systems
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        Summary,
        generate_latest,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    # Fallback implementations for when prometheus_client is not available
    PROMETHEUS_AVAILABLE = False

    class _MetricStub:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    Counter = Histogram = Gauge = Summary = Info = _MetricStub

    def CollectorRegistry():
        return None

    def generate_latest(x):
        return b""

    def start_http_server(x):
        return None

    CONTENT_TYPE_LATEST = "text/plain"


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricDefinition:
    """Definition of a metric."""

    name: str
    metric_type: MetricType
    description: str
    labels: list[str]
    buckets: list[float] | None = None  # For histograms


class MetricsCollector:
    """Central metrics collection and management."""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        self.metrics: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self._setup_default_metrics()

    def _setup_default_metrics(self):
        """Set up default platform metrics."""
        # Request metrics
        self.register_metric(
            MetricDefinition(
                name="mlx_requests_total",
                metric_type=MetricType.COUNTER,
                description="Total number of requests",
                labels=["method", "endpoint", "status"],
            )
        )

        self.register_metric(
            MetricDefinition(
                name="mlx_request_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Request duration in seconds",
                labels=["method", "endpoint"],
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
            )
        )

        # System metrics
        self.register_metric(
            MetricDefinition(
                name="mlx_system_info",
                metric_type=MetricType.INFO,
                description="System information",
                labels=["version", "environment", "hostname"],
            )
        )

        self.register_metric(
            MetricDefinition(
                name="mlx_memory_usage_bytes",
                metric_type=MetricType.GAUGE,
                description="Memory usage in bytes",
                labels=["type"],
            )
        )

        self.register_metric(
            MetricDefinition(
                name="mlx_cpu_usage_percent",
                metric_type=MetricType.GAUGE,
                description="CPU usage percentage",
                labels=["core"],
            )
        )

        # Plugin metrics
        self.register_metric(
            MetricDefinition(
                name="mlx_plugin_executions_total",
                metric_type=MetricType.COUNTER,
                description="Total plugin executions",
                labels=["plugin_name", "status"],
            )
        )

        self.register_metric(
            MetricDefinition(
                name="mlx_plugin_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Plugin execution duration",
                labels=["plugin_name"],
                buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            )
        )

        # Workflow metrics
        self.register_metric(
            MetricDefinition(
                name="mlx_workflow_executions_total",
                metric_type=MetricType.COUNTER,
                description="Total workflow executions",
                labels=["workflow_name", "status"],
            )
        )

        self.register_metric(
            MetricDefinition(
                name="mlx_workflow_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Workflow execution duration",
                labels=["workflow_name"],
                buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0],
            )
        )

        self.register_metric(
            MetricDefinition(
                name="mlx_workflow_steps_total",
                metric_type=MetricType.GAUGE,
                description="Number of steps in workflow",
                labels=["workflow_name"],
            )
        )

    def register_metric(self, definition: MetricDefinition) -> Any:
        """Register a new metric."""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus client not available, using stub metrics")
            self.metrics[definition.name] = _MetricStub()
            return self.metrics[definition.name]

        metric_kwargs = {
            "name": definition.name,
            "documentation": definition.description,
            "labelnames": definition.labels,
            "registry": self.registry,
        }

        if definition.metric_type == MetricType.COUNTER:
            metric = Counter(**metric_kwargs)
        elif definition.metric_type == MetricType.GAUGE:
            metric = Gauge(**metric_kwargs)
        elif definition.metric_type == MetricType.HISTOGRAM:
            if definition.buckets:
                metric_kwargs["buckets"] = definition.buckets
            metric = Histogram(**metric_kwargs)
        elif definition.metric_type == MetricType.SUMMARY:
            metric = Summary(**metric_kwargs)
        elif definition.metric_type == MetricType.INFO:
            metric = Info(**metric_kwargs)
        else:
            raise ValueError(f"Unknown metric type: {definition.metric_type}")

        self.metrics[definition.name] = metric
        self.logger.debug(f"Registered metric: {definition.name}")
        return metric

    def get_metric(self, name: str) -> Any:
        """Get a registered metric by name."""
        return self.metrics.get(name)

    def increment(
        self, name: str, labels: dict[str, str] | None = None, value: float = 1
    ):
        """Increment a counter metric."""
        metric = self.get_metric(name)
        if metric:
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None):
        """Set a gauge metric value."""
        metric = self.get_metric(name)
        if metric:
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None):
        """Observe a value for histogram/summary metrics."""
        metric = self.get_metric(name)
        if metric:
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)

    def set_info(self, name: str, info: dict[str, str]):
        """Set info metric."""
        metric = self.get_metric(name)
        if metric:
            metric.info(info)

    @contextmanager
    def time_operation(self, metric_name: str, labels: dict[str, str] | None = None):
        """Time an operation and record duration."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.observe(metric_name, duration, labels)

    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry)
        return b"# Prometheus client not available\n"

    def start_metrics_server(self, port: int = 9090):
        """Start HTTP server for metrics export."""
        if PROMETHEUS_AVAILABLE:
            start_http_server(port, registry=self.registry)
            self.logger.info(f"Metrics server started on port {port}")
        else:
            self.logger.warning(
                "Cannot start metrics server - Prometheus client not available"
            )


class MLMetrics:
    """ML-specific metrics collection."""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self._setup_ml_metrics()

    def _setup_ml_metrics(self):
        """Set up ML-specific metrics."""
        # Model training metrics
        self.collector.register_metric(
            MetricDefinition(
                name="mlx_model_training_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Model training duration",
                labels=["model_type", "problem_type", "dataset"],
                buckets=[10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0],
            )
        )

        self.collector.register_metric(
            MetricDefinition(
                name="mlx_model_accuracy",
                metric_type=MetricType.GAUGE,
                description="Model accuracy score",
                labels=["model_id", "model_version", "dataset"],
            )
        )

        self.collector.register_metric(
            MetricDefinition(
                name="mlx_model_predictions_total",
                metric_type=MetricType.COUNTER,
                description="Total model predictions",
                labels=["model_id", "model_version"],
            )
        )

        self.collector.register_metric(
            MetricDefinition(
                name="mlx_model_prediction_latency_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Model prediction latency",
                labels=["model_id", "model_version"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            )
        )

        # Data quality metrics
        self.collector.register_metric(
            MetricDefinition(
                name="mlx_data_quality_score",
                metric_type=MetricType.GAUGE,
                description="Data quality score",
                labels=["dataset", "check_type"],
            )
        )

        self.collector.register_metric(
            MetricDefinition(
                name="mlx_data_drift_score",
                metric_type=MetricType.GAUGE,
                description="Data drift score",
                labels=["dataset", "feature"],
            )
        )

        self.collector.register_metric(
            MetricDefinition(
                name="mlx_model_drift_score",
                metric_type=MetricType.GAUGE,
                description="Model drift score",
                labels=["model_id", "model_version"],
            )
        )

        # Feature store metrics
        self.collector.register_metric(
            MetricDefinition(
                name="mlx_feature_requests_total",
                metric_type=MetricType.COUNTER,
                description="Total feature requests",
                labels=["feature_group", "feature_name"],
            )
        )

        self.collector.register_metric(
            MetricDefinition(
                name="mlx_feature_freshness_seconds",
                metric_type=MetricType.GAUGE,
                description="Feature freshness in seconds",
                labels=["feature_group", "feature_name"],
            )
        )

        # Cost metrics
        self.collector.register_metric(
            MetricDefinition(
                name="mlx_cost_total",
                metric_type=MetricType.COUNTER,
                description="Total cost",
                labels=["service", "cost_type"],
            )
        )

        self.collector.register_metric(
            MetricDefinition(
                name="mlx_llm_tokens_total",
                metric_type=MetricType.COUNTER,
                description="Total LLM tokens used",
                labels=["provider", "model", "token_type"],
            )
        )

    def record_training_metrics(
        self,
        model_type: str,
        problem_type: str,
        dataset: str,
        duration: float,
        accuracy: float,
        model_id: str,
        model_version: str = "latest",
    ):
        """Record model training metrics."""
        self.collector.observe(
            "mlx_model_training_duration_seconds",
            duration,
            {
                "model_type": model_type,
                "problem_type": problem_type,
                "dataset": dataset,
            },
        )

        self.collector.set_gauge(
            "mlx_model_accuracy",
            accuracy,
            {"model_id": model_id, "model_version": model_version, "dataset": dataset},
        )

    def record_prediction_metrics(
        self,
        model_id: str,
        model_version: str,
        latency: float,
        count: int = 1,
    ):
        """Record model prediction metrics."""
        labels = {"model_id": model_id, "model_version": model_version}

        self.collector.increment("mlx_model_predictions_total", labels, count)

        self.collector.observe("mlx_model_prediction_latency_seconds", latency, labels)

    def record_data_quality(
        self,
        dataset: str,
        check_type: str,
        quality_score: float,
    ):
        """Record data quality metrics."""
        self.collector.set_gauge(
            "mlx_data_quality_score",
            quality_score,
            {"dataset": dataset, "check_type": check_type},
        )

    def record_drift_metrics(
        self,
        dataset: str = "",
        feature: str = "",
        model_id: str = "",
        model_version: str = "",
        drift_score: float = 0.0,
        drift_type: str = "data",
    ):
        """Record drift detection metrics."""
        if drift_type == "data":
            self.collector.set_gauge(
                "mlx_data_drift_score",
                drift_score,
                {"dataset": dataset, "feature": feature},
            )
        elif drift_type == "model":
            self.collector.set_gauge(
                "mlx_model_drift_score",
                drift_score,
                {"model_id": model_id, "model_version": model_version},
            )

    def record_feature_metrics(
        self,
        feature_group: str,
        feature_name: str,
        freshness_seconds: float,
    ):
        """Record feature store metrics."""
        labels = {"feature_group": feature_group, "feature_name": feature_name}

        self.collector.increment("mlx_feature_requests_total", labels)
        self.collector.set_gauge(
            "mlx_feature_freshness_seconds", freshness_seconds, labels
        )

    def record_cost_metrics(
        self,
        service: str,
        cost_type: str,
        amount: float,
    ):
        """Record cost metrics."""
        self.collector.increment(
            "mlx_cost_total", {"service": service, "cost_type": cost_type}, amount
        )

    def record_llm_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """Record LLM usage metrics."""
        labels_input = {"provider": provider, "model": model, "token_type": "input"}
        labels_output = {"provider": provider, "model": model, "token_type": "output"}

        self.collector.increment("mlx_llm_tokens_total", labels_input, input_tokens)
        self.collector.increment("mlx_llm_tokens_total", labels_output, output_tokens)


# Global metrics collector instance
_metrics_collector = None
_ml_metrics = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_ml_metrics() -> MLMetrics:
    """Get the global ML metrics instance."""
    global _ml_metrics
    if _ml_metrics is None:
        _ml_metrics = MLMetrics(get_metrics_collector())
    return _ml_metrics


def setup_metrics(
    enable_server: bool = True,
    server_port: int = 9090,
    registry: CollectorRegistry | None = None,
) -> MetricsCollector:
    """Set up metrics collection and optionally start server."""
    global _metrics_collector, _ml_metrics

    _metrics_collector = MetricsCollector(registry)
    _ml_metrics = MLMetrics(_metrics_collector)

    if enable_server:
        _metrics_collector.start_metrics_server(server_port)

    # Set system info
    import platform
    import socket

    _metrics_collector.set_info(
        "mlx_system_info",
        {
            "version": "1.0.0",  # Should come from config
            "environment": "development",  # Should come from config
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "python_version": platform.python_version(),
        },
    )

    logger.info("Metrics collection initialized")
    return _metrics_collector


# Decorators for automatic metrics collection
def track_duration(metric_name: str, labels: dict[str, str] | None = None):
    """Decorator to track function execution duration."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            with collector.time_operation(metric_name, labels):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def track_calls(metric_name: str, labels: dict[str, str] | None = None):
    """Decorator to track function calls."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            try:
                result = func(*args, **kwargs)
                success_labels = {**(labels or {}), "status": "success"}
                collector.increment(metric_name, success_labels)
                return result
            except Exception:
                error_labels = {**(labels or {}), "status": "error"}
                collector.increment(metric_name, error_labels)
                raise

        return wrapper

    return decorator
