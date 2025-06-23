"""Production-grade structured logging for the MLX platform.

This module provides:
- JSON structured logging with OpenTelemetry integration
- Contextual logging with correlation IDs and trace information
- Performance metrics and observability
- Security audit logging
- Multi-destination log routing (console, file, external systems)
"""

import json
import logging
import logging.config
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Context variables for request correlation
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")
component_var: ContextVar[str] = ContextVar("component", default="")
operation_var: ContextVar[str] = ContextVar("operation", default="")


@dataclass
class LogContext:
    """Structured context for logs."""

    request_id: str = ""
    user_id: str = ""
    component: str = ""
    operation: str = ""
    trace_id: str = ""
    span_id: str = ""
    environment: str = "development"
    service_version: str = "1.0.0"


class StructuredFormatter(logging.Formatter):
    """JSON formatter with structured context."""

    def __init__(self, include_context: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add execution context
        if self.include_context:
            log_data.update(
                {
                    "request_id": request_id_var.get(),
                    "user_id": user_id_var.get(),
                    "component": component_var.get(),
                    "operation": operation_var.get(),
                    "environment": getattr(record, "environment", "development"),
                }
            )

        # Add exception information
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields from log call
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
            }:
                extra_fields[key] = value

        if extra_fields:
            log_data["extra"] = extra_fields

        # Add performance metrics if available
        if hasattr(record, "duration"):
            log_data["performance"] = {
                "duration_ms": record.duration,
                "memory_mb": getattr(record, "memory_usage", None),
                "cpu_percent": getattr(record, "cpu_usage", None),
            }

        # Add OpenTelemetry trace context if available
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span.is_recording():
                span_context = span.get_span_context()
                log_data.update(
                    {
                        "trace_id": format(span_context.trace_id, "032x"),
                        "span_id": format(span_context.span_id, "016x"),
                    }
                )
        except ImportError:
            pass

        return json.dumps(log_data, default=str, ensure_ascii=False)


class PerformanceLogger:
    """Context manager for performance logging."""

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        level: int = logging.INFO,
        include_memory: bool = True,
    ):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.include_memory = include_memory
        self.start_time = None
        self.start_memory = None

    def __enter__(self):
        self.start_time = time.perf_counter()

        if self.include_memory:
            try:
                import psutil

                process = psutil.Process()
                self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            except ImportError:
                self.start_memory = None

        # Set operation context
        operation_var.set(self.operation)

        self.logger.log(
            self.level,
            f"Starting operation: {self.operation}",
            extra={"operation_status": "started"},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.perf_counter() - self.start_time) * 1000  # ms

        extra = {
            "operation_status": "completed" if exc_type is None else "failed",
            "duration": duration,
        }

        if self.include_memory and self.start_memory:
            try:
                import psutil

                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                extra.update(
                    {
                        "memory_usage": end_memory,
                        "memory_delta": end_memory - self.start_memory,
                    }
                )
            except ImportError:
                pass

        if exc_type is None:
            self.logger.log(
                self.level,
                f"Completed operation: {self.operation} in {duration:.2f}ms",
                extra=extra,
            )
        else:
            self.logger.error(
                f"Failed operation: {self.operation} after {duration:.2f}ms",
                extra=extra,
                exc_info=(exc_type, exc_val, exc_tb),
            )


class AuditLogger:
    """Security audit logging with compliance features."""

    def __init__(self, logger_name: str = "mlx.audit"):
        self.logger = logging.getLogger(logger_name)

    def log_access(
        self,
        resource: str,
        action: str,
        user_id: str = "",
        result: str = "success",
        **kwargs,
    ):
        """Log resource access for security auditing."""
        self.logger.info(
            f"Resource access: {action} {resource}",
            extra={
                "audit_type": "access",
                "resource": resource,
                "action": action,
                "user_id": user_id or user_id_var.get(),
                "result": result,
                "ip_address": kwargs.get("ip_address"),
                "user_agent": kwargs.get("user_agent"),
                **kwargs,
            },
        )

    def log_data_access(
        self,
        dataset: str,
        operation: str,
        user_id: str = "",
        row_count: int | None = None,
        **kwargs,
    ):
        """Log data access for compliance."""
        extra = {
            "audit_type": "data_access",
            "dataset": dataset,
            "operation": operation,
            "user_id": user_id or user_id_var.get(),
        }

        if row_count is not None:
            extra["row_count"] = row_count

        extra.update(kwargs)

        self.logger.info(f"Data access: {operation} on {dataset}", extra=extra)

    def log_model_operation(
        self,
        model_id: str,
        operation: str,
        user_id: str = "",
        model_version: str | None = None,
        **kwargs,
    ):
        """Log model operations for ML governance."""
        extra = {
            "audit_type": "model_operation",
            "model_id": model_id,
            "operation": operation,
            "user_id": user_id or user_id_var.get(),
        }

        if model_version:
            extra["model_version"] = model_version

        extra.update(kwargs)

        self.logger.info(f"Model operation: {operation} on {model_id}", extra=extra)


class LoggingConfig:
    """Centralized logging configuration for the MLX platform."""

    @staticmethod
    def setup_logging(
        level: str = "INFO",
        format_type: str = "structured",  # "structured" or "simple"
        log_file: Path | None = None,
        enable_console: bool = True,
        enable_audit: bool = True,
        environment: str = "development",
    ) -> None:
        """Set up comprehensive logging configuration."""

        # Convert string level to logging constant
        numeric_level = getattr(logging, level.upper(), logging.INFO)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        # Clear existing handlers
        root_logger.handlers.clear()

        # Create formatters
        if format_type == "structured":
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        # Audit file handler
        if enable_audit:
            audit_logger = logging.getLogger("mlx.audit")
            audit_file = (
                log_file.parent / "audit.log" if log_file else Path("logs/audit.log")
            )
            audit_file.parent.mkdir(parents=True, exist_ok=True)

            audit_handler = logging.FileHandler(audit_file)
            audit_handler.setLevel(logging.INFO)
            audit_handler.setFormatter(formatter)
            audit_logger.addHandler(audit_handler)
            audit_logger.setLevel(logging.INFO)
            audit_logger.propagate = False  # Don't propagate to root logger

        # Set environment context
        logging.getLogger().info(
            "Logging initialized",
            extra={
                "environment": environment,
                "log_level": level,
                "format_type": format_type,
                "log_file": str(log_file) if log_file else None,
            },
        )

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger with consistent configuration."""
        return logging.getLogger(name)

    @staticmethod
    def create_performance_logger(name: str) -> PerformanceLogger:
        """Create a performance logger for the given operation."""
        logger = logging.getLogger(name)
        return PerformanceLogger(logger, name)


def set_request_context(
    request_id: str = "",
    user_id: str = "",
    component: str = "",
    operation: str = "",
) -> None:
    """Set request context for correlation across logs."""
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if component:
        component_var.set(component)
    if operation:
        operation_var.set(operation)


def clear_request_context() -> None:
    """Clear request context."""
    request_id_var.set("")
    user_id_var.set("")
    component_var.set("")
    operation_var.set("")


class RequestLogger:
    """Context manager for request-scoped logging."""

    def __init__(
        self,
        request_id: str,
        user_id: str = "",
        component: str = "",
        operation: str = "",
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.component = component
        self.operation = operation
        self.previous_context = {}

    def __enter__(self):
        # Save previous context
        self.previous_context = {
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
            "component": component_var.get(),
            "operation": operation_var.get(),
        }

        # Set new context
        set_request_context(
            self.request_id,
            self.user_id,
            self.component,
            self.operation,
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        set_request_context(**self.previous_context)


# Convenience functions for common logging patterns
def log_function_call(func):
    """Decorator to log function calls with performance metrics."""

    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        operation = f"{func.__module__}.{func.__name__}"

        with PerformanceLogger(logger, operation):
            return func(*args, **kwargs)

    return wrapper


def log_api_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: str = "",
    **kwargs,
):
    """Log API request with standard format."""
    logger.info(
        f"{method} {path} - {status_code}",
        extra={
            "request_type": "api",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration": duration_ms,
            "user_id": user_id or user_id_var.get(),
            **kwargs,
        },
    )


def log_ml_operation(
    logger: logging.Logger,
    operation: str,
    model_type: str = "",
    data_shape: tuple | None = None,
    performance_metrics: dict[str, float] | None = None,
    **kwargs,
):
    """Log ML operations with relevant metadata."""
    extra = {
        "ml_operation": operation,
        "model_type": model_type,
    }

    if data_shape:
        extra["data_shape"] = data_shape

    if performance_metrics:
        extra["ml_metrics"] = performance_metrics

    extra.update(kwargs)

    logger.info(f"ML operation: {operation}", extra=extra)


# Global audit logger instance
audit_logger = AuditLogger()

# Export main components
__all__ = [
    "LoggingConfig",
    "StructuredFormatter",
    "PerformanceLogger",
    "AuditLogger",
    "RequestLogger",
    "set_request_context",
    "clear_request_context",
    "log_function_call",
    "log_api_request",
    "log_ml_operation",
    "audit_logger",
]
