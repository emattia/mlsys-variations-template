"""Plugin architecture for extensible MLOps components.

This module provides abstract base classes and interfaces for creating
pluggable components in the MLOps pipeline.
"""

from .base import (
    ComponentResult,
    ComponentStatus,
    DataProcessor,
    ExecutionContext,
    ExperimentTracker,
    MLOpsComponent,
    ModelEvaluator,
    ModelServer,
    ModelTrainer,
    WorkflowOrchestrator,
)
from .registry import (
    PluginRegistry,
    get_plugin,
    get_registry,
    list_plugins,
    register_plugin,
)

__all__ = [
    "MLOpsComponent",
    "DataProcessor",
    "ModelTrainer",
    "ModelEvaluator",
    "ModelServer",
    "WorkflowOrchestrator",
    "ExperimentTracker",
    "ExecutionContext",
    "ComponentResult",
    "ComponentStatus",
    "PluginRegistry",
    "register_plugin",
    "get_plugin",
    "list_plugins",
    "get_registry",
]
