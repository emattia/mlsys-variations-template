"""Standardized exception hierarchy for the MLX platform.

This module provides a comprehensive exception hierarchy with:
- Structured error context and metadata
- Integration with observability systems
- Recovery strategies and error handling patterns
- Type-safe error handling for expert AI systems engineers
"""

import logging
import traceback
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for observability and alerting."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification and handling."""

    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    DATA = "data"
    MODEL = "model"
    PLUGIN = "plugin"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    RATE_LIMIT = "rate_limit"
    EXTERNAL_API = "external_api"
    TIMEOUT = "timeout"
    RESOURCE = "resource"


class MLXException(Exception):
    """Base exception for all MLX platform errors.

    Provides structured error context, metadata, and integration
    with observability systems for production debugging.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        category: ErrorCategory = ErrorCategory.INFRASTRUCTURE,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: dict[str, Any] | None = None,
        recovery_suggestion: str | None = None,
        cause: Exception | None = None,
    ):
        """Initialize MLX exception with comprehensive context.

        Args:
            message: Human-readable error message
            error_code: Unique error code for tracking
            category: Error category for classification
            severity: Error severity for alerting
            context: Additional context for debugging
            recovery_suggestion: Suggested recovery action
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.recovery_suggestion = recovery_suggestion
        self.cause = cause
        self.stack_trace = traceback.format_exc()

        # Log error for observability
        self._log_error()

    def _generate_error_code(self) -> str:
        """Generate unique error code based on exception type."""
        return f"MLX_{self.__class__.__name__.upper()}_{hash(self.message) % 10000:04d}"

    def _log_error(self) -> None:
        """Log error with structured context for observability."""
        log_data = {
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "recovery_suggestion": self.recovery_suggestion,
        }

        if self.cause:
            log_data["cause"] = str(self.cause)

        if self.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.error("MLX Error occurred", extra=log_data, exc_info=True)
        else:
            logger.warning("MLX Warning occurred", extra=log_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "recovery_suggestion": self.recovery_suggestion,
            "cause": str(self.cause) if self.cause else None,
            "stack_trace": self.stack_trace,
        }


class ValidationError(MLXException):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        field_value: Any = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if field_name:
            context["field_name"] = field_name
        if field_value is not None:
            context["field_value"] = str(field_value)

        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestion="Check input parameters and try again",
            **kwargs,
        )


class ConfigurationError(MLXException):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: str | None = None, **kwargs):
        context = kwargs.get("context", {})
        if config_key:
            context["config_key"] = config_key

        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestion="Check configuration files and environment variables",
            **kwargs,
        )


class DataError(MLXException):
    """Raised when data-related operations fail."""

    def __init__(
        self,
        message: str,
        data_source: str | None = None,
        data_shape: tuple | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if data_source:
            context["data_source"] = data_source
        if data_shape:
            context["data_shape"] = data_shape

        super().__init__(
            message,
            category=ErrorCategory.DATA,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestion="Validate data format, schema, and content",
            **kwargs,
        )


class ModelError(MLXException):
    """Raised when model operations fail."""

    def __init__(
        self,
        message: str,
        model_type: str | None = None,
        model_path: str | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if model_type:
            context["model_type"] = model_type
        if model_path:
            context["model_path"] = model_path

        super().__init__(
            message,
            category=ErrorCategory.MODEL,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestion="Check model compatibility and training parameters",
            **kwargs,
        )


class PluginError(MLXException):
    """Raised when plugin operations fail."""

    def __init__(
        self,
        message: str,
        plugin_name: str | None = None,
        plugin_version: str | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if plugin_name:
            context["plugin_name"] = plugin_name
        if plugin_version:
            context["plugin_version"] = plugin_version

        super().__init__(
            message,
            category=ErrorCategory.PLUGIN,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestion="Check plugin installation and dependencies",
            **kwargs,
        )


class RateLimitError(MLXException):
    """Raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        retry_after: int | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if service_name:
            context["service_name"] = service_name
        if retry_after:
            context["retry_after"] = retry_after

        recovery_msg = "Reduce request rate"
        if retry_after:
            recovery_msg += f" and retry after {retry_after} seconds"

        super().__init__(
            message,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestion=recovery_msg,
            **kwargs,
        )


class ExternalAPIError(MLXException):
    """Raised when external API calls fail."""

    def __init__(
        self,
        message: str,
        api_name: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if api_name:
            context["api_name"] = api_name
        if status_code:
            context["status_code"] = status_code
        if response_body:
            context["response_body"] = response_body[:1000]  # Truncate for logging

        super().__init__(
            message,
            category=ErrorCategory.EXTERNAL_API,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestion="Check API credentials and network connectivity",
            **kwargs,
        )


class TimeoutError(MLXException):
    """Raised when operations timeout."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        timeout_seconds: float | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if operation:
            context["operation"] = operation
        if timeout_seconds:
            context["timeout_seconds"] = timeout_seconds

        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestion="Increase timeout or optimize operation",
            **kwargs,
        )


class ResourceError(MLXException):
    """Raised when resource constraints are exceeded."""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        limit: str | None = None,
        current_usage: str | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if resource_type:
            context["resource_type"] = resource_type
        if limit:
            context["limit"] = limit
        if current_usage:
            context["current_usage"] = current_usage

        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestion="Free up resources or increase limits",
            **kwargs,
        )


class SecurityError(MLXException):
    """Raised when security violations occur."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            recovery_suggestion="Review security configuration and access controls",
            **kwargs,
        )


class WorkflowError(MLXException):
    """Raised when workflow execution fails."""

    def __init__(
        self,
        message: str,
        workflow_id: str | None = None,
        step_id: str | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if workflow_id:
            context["workflow_id"] = workflow_id
        if step_id:
            context["step_id"] = step_id

        super().__init__(
            message,
            category=ErrorCategory.INFRASTRUCTURE,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestion="Check workflow configuration and step dependencies",
            **kwargs,
        )


def handle_exception(
    exc: Exception,
    operation: str,
    context: dict[str, Any] | None = None,
    reraise: bool = True,
) -> MLXException | None:
    """Standardized exception handler that converts exceptions to MLXException.

    Args:
        exc: Original exception
        operation: Description of the operation that failed
        context: Additional context for debugging
        reraise: Whether to reraise the exception

    Returns:
        MLXException instance if not reraising

    Raises:
        MLXException: If reraise is True
    """
    # If already an MLXException, just reraise or return
    if isinstance(exc, MLXException):
        if reraise:
            raise exc
        return exc

    # Map common exceptions to MLX exceptions
    mlx_exc = None

    if isinstance(exc, ValueError):
        mlx_exc = ValidationError(
            f"Validation failed in {operation}: {str(exc)}",
            context=context,
            cause=exc,
        )
    elif isinstance(exc, FileNotFoundError):
        mlx_exc = DataError(
            f"File not found in {operation}: {str(exc)}",
            context=context,
            cause=exc,
        )
    elif isinstance(exc, PermissionError):
        mlx_exc = SecurityError(
            f"Permission denied in {operation}: {str(exc)}",
            context=context,
            cause=exc,
        )
    elif isinstance(exc, TimeoutError):
        mlx_exc = TimeoutError(
            f"Timeout in {operation}: {str(exc)}",
            operation=operation,
            context=context,
            cause=exc,
        )
    else:
        # Generic MLX exception for unknown errors
        mlx_exc = MLXException(
            f"Unexpected error in {operation}: {str(exc)}",
            category=ErrorCategory.INFRASTRUCTURE,
            severity=ErrorSeverity.HIGH,
            context=context,
            cause=exc,
        )

    if reraise:
        raise mlx_exc

    return mlx_exc


def safe_execute(
    func,
    operation: str,
    context: dict[str, Any] | None = None,
    default_return: Any = None,
    allowed_exceptions: list | None = None,
):
    """Safely execute a function with standardized error handling.

    Args:
        func: Function to execute
        operation: Description of the operation
        context: Additional context for debugging
        default_return: Value to return on error
        allowed_exceptions: List of exception types to not convert

    Returns:
        Function result or default_return on error
    """
    try:
        return func()
    except Exception as exc:
        if allowed_exceptions and isinstance(exc, tuple(allowed_exceptions)):
            raise

        handle_exception(exc, operation, context, reraise=True)


class ErrorHandler:
    """Context manager for standardized error handling."""

    def __init__(
        self,
        operation: str,
        context: dict[str, Any] | None = None,
        reraise: bool = True,
    ):
        self.operation = operation
        self.context = context or {}
        self.reraise = reraise
        self.exception: MLXException | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.exception = handle_exception(
                exc_val, self.operation, self.context, self.reraise
            )
            return not self.reraise  # Suppress exception if not reraising
        return False
