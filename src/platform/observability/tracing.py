"""Distributed tracing for the MLX platform.

This module provides:
- OpenTelemetry integration for distributed tracing
- Custom trace instrumentation for ML operations
- Trace correlation across services and components
- Performance analysis and debugging support
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

    # Stub implementations
    class _TracerStub:
        def start_span(self, *args, **kwargs):
            return _SpanStub()

        def start_as_current_span(self, *args, **kwargs):
            return _SpanStub()

    class _SpanStub:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_attribute(self, *args):
            pass

        def set_status(self, *args):
            pass

        def add_event(self, *args):
            pass

        def record_exception(self, *args):
            pass

    trace = type(
        "trace",
        (),
        {
            "get_tracer": lambda *args, **kwargs: _TracerStub(),
            "get_current_span": lambda: _SpanStub(),
            "Status": type("Status", (), {"OK": "ok", "ERROR": "error"}),
            "StatusCode": type("StatusCode", (), {"OK": "ok", "ERROR": "error"}),
        },
    )()


class TracingConfig:
    """Configuration for distributed tracing."""

    def __init__(
        self,
        service_name: str = "mlx-platform",
        service_version: str = "1.0.0",
        environment: str = "development",
        jaeger_endpoint: str | None = None,
        otlp_endpoint: str | None = None,
        console_export: bool = False,
        sample_rate: float = 1.0,
        enable_auto_instrumentation: bool = True,
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.jaeger_endpoint = jaeger_endpoint
        self.otlp_endpoint = otlp_endpoint
        self.console_export = console_export
        self.sample_rate = sample_rate
        self.enable_auto_instrumentation = enable_auto_instrumentation

    def setup_tracing(self):
        """Set up OpenTelemetry tracing with configured exporters."""
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available, tracing disabled")
            return

        # Create resource with service information
        resource = Resource.create(
            {
                SERVICE_NAME: self.service_name,
                SERVICE_VERSION: self.service_version,
                "environment": self.environment,
            }
        )

        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Set up exporters
        exporters = []

        if self.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.jaeger_endpoint.split(":")[0],
                agent_port=int(self.jaeger_endpoint.split(":")[1])
                if ":" in self.jaeger_endpoint
                else 14268,
            )
            exporters.append(jaeger_exporter)

        if self.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
            exporters.append(otlp_exporter)

        if self.console_export:
            console_exporter = ConsoleSpanExporter()
            exporters.append(console_exporter)

        # Add span processors
        for exporter in exporters:
            span_processor = BatchSpanProcessor(exporter)
            tracer_provider.add_span_processor(span_processor)

        # Set up propagation
        set_global_textmap(B3MultiFormat())

        # Auto-instrumentation
        if self.enable_auto_instrumentation:
            self._setup_auto_instrumentation()

        logger.info(f"Tracing initialized for service {self.service_name}")

    def _setup_auto_instrumentation(self):
        """Set up automatic instrumentation for common libraries."""
        try:
            RequestsInstrumentor().instrument()
            HTTPXClientInstrumentor().instrument()
            SQLAlchemyInstrumentor().instrument()
            logger.info("Auto-instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to set up auto-instrumentation: {e}")


class MLXTracer:
    """Custom tracer for MLX platform operations."""

    def __init__(self, name: str = "mlx"):
        if OPENTELEMETRY_AVAILABLE:
            self.tracer = trace.get_tracer(name)
        else:
            self.tracer = _TracerStub()
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        attributes: dict[str, Any] | None = None,
        record_exception: bool = True,
    ):
        """Trace an operation with automatic error handling."""
        with self.tracer.start_as_current_span(operation_name) as span:
            if OPENTELEMETRY_AVAILABLE and attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))

            start_time = time.time()

            try:
                yield span

                # Record success
                if OPENTELEMETRY_AVAILABLE:
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    span.set_attribute("operation.success", True)
                    span.set_attribute(
                        "operation.duration_ms", (time.time() - start_time) * 1000
                    )

            except Exception as e:
                # Record error
                if OPENTELEMETRY_AVAILABLE:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.set_attribute("operation.success", False)
                    span.set_attribute("operation.error", str(e))

                    if record_exception:
                        span.record_exception(e)

                raise

    def trace_ml_operation(
        self,
        operation_type: str,
        model_info: dict[str, Any] | None = None,
        data_info: dict[str, Any] | None = None,
        performance_info: dict[str, Any] | None = None,
    ):
        """Trace ML-specific operations with relevant metadata."""
        attributes = {
            "ml.operation_type": operation_type,
        }

        if model_info:
            for key, value in model_info.items():
                attributes[f"ml.model.{key}"] = value

        if data_info:
            for key, value in data_info.items():
                attributes[f"ml.data.{key}"] = value

        if performance_info:
            for key, value in performance_info.items():
                attributes[f"ml.performance.{key}"] = value

        return self.trace_operation(f"ml.{operation_type}", attributes)

    def trace_plugin_execution(
        self,
        plugin_name: str,
        plugin_version: str = "",
        config: dict[str, Any] | None = None,
    ):
        """Trace plugin execution."""
        attributes = {
            "plugin.name": plugin_name,
            "plugin.version": plugin_version,
        }

        if config:
            # Only include non-sensitive config values
            safe_config = {
                k: v
                for k, v in config.items()
                if not any(
                    sensitive in k.lower()
                    for sensitive in ["password", "key", "secret", "token"]
                )
            }
            for key, value in safe_config.items():
                attributes[f"plugin.config.{key}"] = str(value)

        return self.trace_operation(f"plugin.{plugin_name}", attributes)

    def trace_workflow_execution(
        self,
        workflow_name: str,
        workflow_id: str,
        step_count: int,
    ):
        """Trace workflow execution."""
        attributes = {
            "workflow.name": workflow_name,
            "workflow.id": workflow_id,
            "workflow.step_count": step_count,
        }

        return self.trace_operation(f"workflow.{workflow_name}", attributes)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None):
        """Add an event to the current span."""
        if OPENTELEMETRY_AVAILABLE:
            current_span = trace.get_current_span()
            if current_span.is_recording():
                current_span.add_event(name, attributes or {})

    def set_attribute(self, key: str, value: Any):
        """Set an attribute on the current span."""
        if OPENTELEMETRY_AVAILABLE:
            current_span = trace.get_current_span()
            if current_span.is_recording():
                current_span.set_attribute(key, str(value))


# Global tracer instance
_global_tracer = None


def get_tracer(name: str = "mlx") -> MLXTracer:
    """Get a tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = MLXTracer(name)
    return _global_tracer


def setup_tracing(config: TracingConfig):
    """Set up distributed tracing with the given configuration."""
    config.setup_tracing()

    # Initialize global tracer
    global _global_tracer
    _global_tracer = MLXTracer(config.service_name)


# Decorators for automatic tracing
def trace_execution(
    operation_name: str | None = None,
    attributes: dict[str, Any] | None = None,
    record_exception: bool = True,
):
    """Decorator to automatically trace function execution."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            func_attributes = {
                "function.name": func.__name__,
                "function.module": func.__module__,
                **(attributes or {}),
            }

            with tracer.trace_operation(op_name, func_attributes, record_exception):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_ml_function(
    operation_type: str,
    model_info: dict[str, Any] | None = None,
    data_info: dict[str, Any] | None = None,
):
    """Decorator to trace ML functions with ML-specific metadata."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()

            with tracer.trace_ml_operation(operation_type, model_info, data_info):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_plugin_function(plugin_name: str, plugin_version: str = ""):
    """Decorator to trace plugin functions."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()

            with tracer.trace_plugin_execution(plugin_name, plugin_version):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Context managers for manual tracing
@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "mlx",
):
    """Context manager for manual span creation."""
    tracer = get_tracer(tracer_name)
    with tracer.trace_operation(name, attributes):
        yield


@contextmanager
def trace_ml_span(
    operation_type: str,
    model_info: dict[str, Any] | None = None,
    data_info: dict[str, Any] | None = None,
    performance_info: dict[str, Any] | None = None,
):
    """Context manager for ML operation tracing."""
    tracer = get_tracer()
    with tracer.trace_ml_operation(
        operation_type, model_info, data_info, performance_info
    ):
        yield


# Utility functions
def get_trace_id() -> str:
    """Get the current trace ID."""
    if OPENTELEMETRY_AVAILABLE:
        span = trace.get_current_span()
        if span.is_recording():
            return format(span.get_span_context().trace_id, "032x")
    return ""


def get_span_id() -> str:
    """Get the current span ID."""
    if OPENTELEMETRY_AVAILABLE:
        span = trace.get_current_span()
        if span.is_recording():
            return format(span.get_span_context().span_id, "016x")
    return ""


def inject_trace_context(headers: dict[str, str]) -> dict[str, str]:
    """Inject trace context into HTTP headers."""
    if OPENTELEMETRY_AVAILABLE:
        from opentelemetry.propagate import inject

        inject(headers)
    return headers


def extract_trace_context(headers: dict[str, str]):
    """Extract trace context from HTTP headers."""
    if OPENTELEMETRY_AVAILABLE:
        from opentelemetry.propagate import extract

        return extract(headers)
    return {}


# Export main components
__all__ = [
    "TracingConfig",
    "MLXTracer",
    "get_tracer",
    "setup_tracing",
    "trace_execution",
    "trace_ml_function",
    "trace_plugin_function",
    "trace_span",
    "trace_ml_span",
    "get_trace_id",
    "get_span_id",
    "inject_trace_context",
    "extract_trace_context",
]
