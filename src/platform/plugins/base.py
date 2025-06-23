"""Base classes and interfaces for the MLOps plugin system."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import polars as pl

from src.platform.config.models import AppConfig

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Status of a component execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionContext:
    """Context information for component execution."""

    config: AppConfig
    run_id: str
    component_name: str
    input_data: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts_dir: Path | None = None

    def __post_init__(self):
        """Set up execution context after initialization."""
        if self.artifacts_dir is None:
            self.artifacts_dir = self.config.paths.models_trained / self.run_id
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ComponentResult:
    """Result of a component execution."""

    status: ComponentStatus
    component_name: str
    execution_time: float
    output_data: dict[str, Any] | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def is_success(self) -> bool:
        """Check if the execution was successful."""
        return self.status == ComponentStatus.SUCCESS

    def is_failed(self) -> bool:
        """Check if the execution failed."""
        return self.status == ComponentStatus.FAILED


class MLOpsComponent(ABC):
    """Abstract base class for all MLOps components."""

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """Initialize the component.

        Args:
            name: Name of the component
            config: Component-specific configuration
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def initialize(self, context: ExecutionContext) -> None:
        """Initialize the component with execution context.

        Args:
            context: Execution context containing configuration and metadata
        """
        pass

    @abstractmethod
    def execute(self, context: ExecutionContext) -> ComponentResult:
        """Execute the component's main functionality.

        Args:
            context: Execution context

        Returns:
            Result of the component execution
        """
        pass

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate component configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid
        """
        pass

    @abstractmethod
    def cleanup(self, context: ExecutionContext) -> None:
        """Clean up resources after execution.

        Args:
            context: Execution context
        """
        pass

    def get_dependencies(self) -> list[str]:
        """Get list of dependencies required by this component.

        Returns:
            List of dependency names
        """
        return []

    def get_version(self) -> str:
        """Get component version.

        Returns:
            Version string
        """
        return "1.0.0"


class DataProcessor(MLOpsComponent):
    """Abstract base class for data processing components."""

    @abstractmethod
    def process_data(
        self, input_data: pl.DataFrame, context: ExecutionContext
    ) -> pl.DataFrame:
        """Process input data.

        Args:
            input_data: Input DataFrame
            context: Execution context

        Returns:
            Processed DataFrame
        """
        pass

    @abstractmethod
    def validate_data(self, data: pl.DataFrame, context: ExecutionContext) -> bool:
        """Validate input data.

        Args:
            data: DataFrame to validate
            context: Execution context

        Returns:
            True if data is valid
        """
        pass


class ModelTrainer(MLOpsComponent):
    """Abstract base class for model training components."""

    @abstractmethod
    def train_model(
        self,
        train_data: pl.DataFrame,
        validation_data: pl.DataFrame | None,
        context: ExecutionContext,
    ) -> Any:
        """Train a machine learning model.

        Args:
            train_data: Training data DataFrame
            validation_data: Optional validation data DataFrame
            context: Execution context

        Returns:
            Trained model object
        """
        pass

    @abstractmethod
    def save_model(
        self, model: Any, model_path: Path, context: ExecutionContext
    ) -> None:
        """Save trained model to disk.

        Args:
            model: Trained model object
            model_path: Path to save the model
            context: Execution context
        """
        pass

    @abstractmethod
    def load_model(self, model_path: Path, context: ExecutionContext) -> Any:
        """Load model from disk.

        Args:
            model_path: Path to the saved model
            context: Execution context

        Returns:
            Loaded model object
        """
        pass


class ModelEvaluator(MLOpsComponent):
    """Abstract base class for model evaluation components."""

    @abstractmethod
    def evaluate_model(
        self, model: Any, test_data: pl.DataFrame, context: ExecutionContext
    ) -> dict[str, float]:
        """Evaluate model performance.

        Args:
            model: Trained model object
            test_data: Test data DataFrame
            context: Execution context

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def generate_report(
        self, evaluation_results: dict[str, float], context: ExecutionContext
    ) -> Path:
        """Generate evaluation report.

        Args:
            evaluation_results: Dictionary of evaluation metrics
            context: Execution context

        Returns:
            Path to generated report
        """
        pass


class ModelServer(MLOpsComponent):
    """Abstract base class for model serving components."""

    @abstractmethod
    def load_model_for_serving(
        self, model_path: Path, context: ExecutionContext
    ) -> None:
        """Load model for serving.

        Args:
            model_path: Path to the model
            context: Execution context
        """
        pass

    @abstractmethod
    def predict(
        self, input_data: dict[str, Any], context: ExecutionContext
    ) -> dict[str, Any]:
        """Make predictions using the loaded model.

        Args:
            input_data: Input data for prediction
            context: Execution context

        Returns:
            Prediction results
        """
        pass

    @abstractmethod
    def batch_predict(
        self, input_data: pl.DataFrame, context: ExecutionContext
    ) -> pl.DataFrame:
        """Make batch predictions.

        Args:
            input_data: Input DataFrame
            context: Execution context

        Returns:
            DataFrame with predictions
        """
        pass

    @abstractmethod
    def get_model_info(self, context: ExecutionContext) -> dict[str, Any]:
        """Get information about the loaded model.

        Args:
            context: Execution context

        Returns:
            Dictionary with model information
        """
        pass


class WorkflowOrchestrator(MLOpsComponent):
    """Abstract base class for workflow orchestration components."""

    @abstractmethod
    def create_workflow(
        self, workflow_definition: dict[str, Any], context: ExecutionContext
    ) -> str:
        """Create a new workflow.

        Args:
            workflow_definition: Workflow definition
            context: Execution context

        Returns:
            Workflow ID
        """
        pass

    @abstractmethod
    def execute_workflow(
        self, workflow_id: str, context: ExecutionContext
    ) -> ComponentResult:
        """Execute a workflow.

        Args:
            workflow_id: ID of the workflow to execute
            context: Execution context

        Returns:
            Workflow execution result
        """
        pass

    @abstractmethod
    def get_workflow_status(
        self, workflow_id: str, context: ExecutionContext
    ) -> ComponentStatus:
        """Get status of a workflow.

        Args:
            workflow_id: ID of the workflow
            context: Execution context

        Returns:
            Workflow status
        """
        pass


class ExperimentTracker(MLOpsComponent):
    """Abstract base class for experiment tracking components."""

    @abstractmethod
    def start_experiment(self, experiment_name: str, context: ExecutionContext) -> str:
        """Start a new experiment.

        Args:
            experiment_name: Name of the experiment
            context: Execution context

        Returns:
            Experiment ID
        """
        pass

    @abstractmethod
    def log_parameters(
        self, parameters: dict[str, Any], context: ExecutionContext
    ) -> None:
        """Log experiment parameters.

        Args:
            parameters: Dictionary of parameters
            context: Execution context
        """
        pass

    @abstractmethod
    def log_metrics(
        self, metrics: dict[str, float], step: int | None, context: ExecutionContext
    ) -> None:
        """Log experiment metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
            context: Execution context
        """
        pass

    @abstractmethod
    def log_artifact(
        self, artifact_path: Path, artifact_name: str, context: ExecutionContext
    ) -> None:
        """Log an artifact.

        Args:
            artifact_path: Path to the artifact
            artifact_name: Name of the artifact
            context: Execution context
        """
        pass

    @abstractmethod
    def end_experiment(self, context: ExecutionContext) -> None:
        """End the current experiment.

        Args:
            context: Execution context
        """
        pass
