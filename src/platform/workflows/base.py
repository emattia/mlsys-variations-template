"""Base workflow orchestration system for MLX platform.

This module provides a production-grade workflow engine for ML pipelines with:
- DAG-based execution with dependency management
- Distributed execution with fault tolerance
- Rich observability and monitoring
- Plugin-based step implementations
- State management and recovery
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.platform.plugins.base import ComponentStatus, ExecutionContext
from src.platform.utils.exceptions import WorkflowError

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(Enum):
    """Individual step execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class StepResult:
    """Result of a workflow step execution."""

    step_id: str
    status: StepStatus
    execution_time: float
    output_data: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None
    retry_count: int = 0


@dataclass
class WorkflowResult:
    """Result of a complete workflow execution."""

    workflow_id: str
    status: WorkflowStatus
    execution_time: float
    step_results: dict[str, StepResult] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None


class WorkflowStep(ABC):
    """Abstract base class for workflow steps."""

    def __init__(
        self,
        step_id: str,
        name: str,
        description: str = "",
        dependencies: list[str] | None = None,
        retry_count: int = 0,
        timeout: float | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.step_id = step_id
        self.name = name
        self.description = description
        self.dependencies = dependencies or []
        self.retry_count = retry_count
        self.timeout = timeout
        self.config = config or {}
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    async def execute(
        self, context: ExecutionContext, inputs: dict[str, Any]
    ) -> StepResult:
        """Execute the workflow step.

        Args:
            context: Execution context
            inputs: Input data from previous steps

        Returns:
            StepResult with execution outcome
        """
        pass

    def validate_inputs(self, inputs: dict[str, Any]) -> bool:
        """Validate input data for the step."""
        return True

    def get_required_inputs(self) -> list[str]:
        """Get list of required input keys."""
        return []

    def get_output_keys(self) -> list[str]:
        """Get list of output keys this step produces."""
        return []


class PluginWorkflowStep(WorkflowStep):
    """Workflow step that executes a plugin component."""

    def __init__(
        self,
        step_id: str,
        plugin_name: str,
        plugin_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(step_id, f"Plugin: {plugin_name}", **kwargs)
        self.plugin_name = plugin_name
        self.plugin_config = plugin_config or {}

    async def execute(
        self, context: ExecutionContext, inputs: dict[str, Any]
    ) -> StepResult:
        """Execute plugin as workflow step."""
        start_time = time.time()

        try:
            # Import here to avoid circular imports
            from src.platform.plugins import get_plugin

            # Get plugin instance
            plugin = get_plugin(self.plugin_name, config=self.plugin_config)

            # Update context with step inputs
            step_context = ExecutionContext(
                config=context.config,
                run_id=f"{context.run_id}_{self.step_id}",
                component_name=self.plugin_name,
                input_data=inputs,
                metadata={**context.metadata, "workflow_step": self.step_id},
                artifacts_dir=context.artifacts_dir / self.step_id,
            )

            # Initialize and execute plugin
            plugin.initialize(step_context)
            result = plugin.execute(step_context)

            execution_time = time.time() - start_time

            # Convert ComponentResult to StepResult
            if result.status == ComponentStatus.SUCCESS:
                return StepResult(
                    step_id=self.step_id,
                    status=StepStatus.SUCCESS,
                    execution_time=execution_time,
                    output_data=result.output_data or {},
                    artifacts=result.artifacts or {},
                    metrics=result.metrics or {},
                )
            else:
                return StepResult(
                    step_id=self.step_id,
                    status=StepStatus.FAILED,
                    execution_time=execution_time,
                    error_message=result.error_message,
                )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Plugin step {self.step_id} failed: {e}", exc_info=True)
            return StepResult(
                step_id=self.step_id,
                status=StepStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e),
            )


class WorkflowEngine:
    """Production-grade workflow execution engine."""

    def __init__(self, max_concurrent_steps: int = 5):
        self.max_concurrent_steps = max_concurrent_steps
        self.workflows: dict[str, Workflow] = {}
        self.running_workflows: set[str] = set()
        self.logger = logging.getLogger(__name__)

    async def execute_workflow(
        self,
        workflow: "Workflow",
        context: ExecutionContext,
        inputs: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Execute a workflow with full observability and error handling."""
        workflow_id = workflow.workflow_id
        self.workflows[workflow_id] = workflow
        self.running_workflows.add(workflow_id)

        start_time = time.time()
        step_results = {}
        workflow_outputs = {}
        workflow_artifacts = {}
        workflow_metrics = {}

        try:
            self.logger.info(f"Starting workflow {workflow_id}")

            # Validate workflow
            if not workflow.validate():
                raise WorkflowError(f"Workflow validation failed: {workflow_id}")

            # Build execution graph
            execution_order = workflow.get_execution_order()
            self.logger.info(f"Workflow execution order: {execution_order}")

            # Execute steps in dependency order
            step_outputs = inputs or {}

            for step_batch in execution_order:
                # Execute steps in batch (parallel execution)
                batch_tasks = []

                for step_id in step_batch:
                    step = workflow.steps[step_id]

                    # Prepare step inputs
                    step_inputs = self._prepare_step_inputs(step, step_outputs)

                    # Create execution task
                    task = self._execute_step_with_retry(step, context, step_inputs)
                    batch_tasks.append((step_id, task))

                # Wait for batch completion
                for step_id, task in batch_tasks:
                    try:
                        result = await task
                        step_results[step_id] = result

                        if result.status == StepStatus.SUCCESS:
                            # Merge step outputs for subsequent steps
                            step_outputs.update(result.output_data)
                            workflow_outputs.update(result.output_data)
                            workflow_artifacts.update(result.artifacts)
                            workflow_metrics.update(result.metrics)
                        else:
                            # Step failed - handle based on workflow configuration
                            if workflow.fail_fast:
                                raise WorkflowError(
                                    f"Step {step_id} failed: {result.error_message}"
                                )
                            else:
                                self.logger.warning(
                                    f"Step {step_id} failed but continuing"
                                )

                    except Exception as e:
                        step_results[step_id] = StepResult(
                            step_id=step_id,
                            status=StepStatus.FAILED,
                            execution_time=0,
                            error_message=str(e),
                        )

                        if workflow.fail_fast:
                            raise WorkflowError(f"Step {step_id} failed: {e}") from e

            execution_time = time.time() - start_time

            # Determine final workflow status
            failed_steps = [
                r for r in step_results.values() if r.status == StepStatus.FAILED
            ]
            if failed_steps and workflow.fail_fast:
                status = WorkflowStatus.FAILED
                error_msg = f"Workflow failed due to {len(failed_steps)} failed steps"
            elif failed_steps:
                status = WorkflowStatus.SUCCESS  # Partial success
                error_msg = None
            else:
                status = WorkflowStatus.SUCCESS
                error_msg = None

            self.logger.info(f"Workflow {workflow_id} completed with status {status}")

            return WorkflowResult(
                workflow_id=workflow_id,
                status=status,
                execution_time=execution_time,
                step_results=step_results,
                output_data=workflow_outputs,
                artifacts=workflow_artifacts,
                metrics=workflow_metrics,
                error_message=error_msg,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Workflow {workflow_id} failed: {e}", exc_info=True)

            return WorkflowResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                execution_time=execution_time,
                step_results=step_results,
                error_message=str(e),
            )

        finally:
            self.running_workflows.discard(workflow_id)

    async def _execute_step_with_retry(
        self,
        step: WorkflowStep,
        context: ExecutionContext,
        inputs: dict[str, Any],
    ) -> StepResult:
        """Execute a step with retry logic."""
        last_error = None

        for attempt in range(step.retry_count + 1):
            try:
                self.logger.info(
                    f"Executing step {step.step_id} (attempt {attempt + 1})"
                )

                # Execute with timeout if specified
                if step.timeout:
                    result = await asyncio.wait_for(
                        step.execute(context, inputs), timeout=step.timeout
                    )
                else:
                    result = await step.execute(context, inputs)

                if result.status == StepStatus.SUCCESS:
                    return result

                last_error = result.error_message

                if attempt < step.retry_count:
                    self.logger.warning(
                        f"Step {step.step_id} failed (attempt {attempt + 1}), retrying..."
                    )
                    result.retry_count = attempt + 1
                    await asyncio.sleep(2**attempt)  # Exponential backoff

            except TimeoutError:
                last_error = (
                    f"Step {step.step_id} timed out after {step.timeout} seconds"
                )
                self.logger.error(last_error)
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Step {step.step_id} failed: {e}", exc_info=True)

        # All retries exhausted
        return StepResult(
            step_id=step.step_id,
            status=StepStatus.FAILED,
            execution_time=0,
            error_message=last_error,
            retry_count=step.retry_count,
        )

    def _prepare_step_inputs(
        self,
        step: WorkflowStep,
        available_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare inputs for a step based on available outputs."""
        step_inputs = {}

        # Add step configuration
        step_inputs.update(step.config)

        # Add outputs from dependency steps
        for key, value in available_outputs.items():
            step_inputs[key] = value

        return step_inputs


class Workflow:
    """Represents a complete ML workflow with steps and dependencies."""

    def __init__(
        self,
        workflow_id: str | None = None,
        name: str = "",
        description: str = "",
        fail_fast: bool = True,
    ):
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.fail_fast = fail_fast
        self.steps: dict[str, WorkflowStep] = {}
        self.logger = logging.getLogger(__name__)

    def add_step(self, step: WorkflowStep) -> "Workflow":
        """Add a step to the workflow."""
        self.steps[step.step_id] = step
        return self

    def add_plugin_step(
        self,
        step_id: str,
        plugin_name: str,
        dependencies: list[str] | None = None,
        plugin_config: dict[str, Any] | None = None,
        **kwargs,
    ) -> "Workflow":
        """Add a plugin-based step to the workflow."""
        step = PluginWorkflowStep(
            step_id=step_id,
            plugin_name=plugin_name,
            plugin_config=plugin_config,
            dependencies=dependencies,
            **kwargs,
        )
        return self.add_step(step)

    def validate(self) -> bool:
        """Validate workflow structure and dependencies."""
        if not self.steps:
            self.logger.error("Workflow has no steps")
            return False

        # Check for circular dependencies
        if self._has_circular_dependencies():
            self.logger.error("Workflow has circular dependencies")
            return False

        # Check that all dependencies exist
        for step in self.steps.values():
            for dep in step.dependencies:
                if dep not in self.steps:
                    self.logger.error(
                        f"Step {step.step_id} depends on non-existent step {dep}"
                    )
                    return False

        return True

    def get_execution_order(self) -> list[list[str]]:
        """Get execution order as batches of steps that can run in parallel."""
        if not self.validate():
            raise WorkflowError("Cannot determine execution order for invalid workflow")

        # Topological sort with batching
        remaining_steps = set(self.steps.keys())
        execution_order = []

        while remaining_steps:
            # Find steps with no unresolved dependencies
            ready_steps = []
            for step_id in remaining_steps:
                step = self.steps[step_id]
                if all(dep not in remaining_steps for dep in step.dependencies):
                    ready_steps.append(step_id)

            if not ready_steps:
                raise WorkflowError(
                    "Cannot resolve workflow dependencies - circular dependency detected"
                )

            execution_order.append(ready_steps)
            remaining_steps -= set(ready_steps)

        return execution_order

    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS."""
        white = set(self.steps.keys())  # Unvisited
        gray = set()  # Currently being processed
        black = set()  # Fully processed

        def visit(step_id: str) -> bool:
            if step_id in gray:
                return True  # Back edge found - circular dependency
            if step_id in black:
                return False  # Already processed

            gray.add(step_id)
            white.discard(step_id)

            step = self.steps[step_id]
            for dep in step.dependencies:
                if dep in self.steps and visit(dep):
                    return True

            gray.discard(step_id)
            black.add(step_id)
            return False

        while white:
            if visit(white.pop()):
                return True

        return False

    def visualize(self) -> str:
        """Generate a text-based visualization of the workflow."""
        lines = [f"Workflow: {self.name} ({self.workflow_id})"]
        lines.append("=" * 50)

        execution_order = self.get_execution_order()

        for i, batch in enumerate(execution_order):
            lines.append(f"Batch {i + 1}:")
            for step_id in batch:
                step = self.steps[step_id]
                deps_str = (
                    f" (depends on: {', '.join(step.dependencies)})"
                    if step.dependencies
                    else ""
                )
                lines.append(f"  - {step.name} [{step_id}]{deps_str}")
            lines.append("")

        return "\n".join(lines)


# Predefined workflow templates for common ML patterns
class MLWorkflowTemplates:
    """Collection of predefined ML workflow templates."""

    @staticmethod
    def training_pipeline(
        data_path: str,
        model_type: str = "random_forest",
        problem_type: str = "classification",
    ) -> Workflow:
        """Create a standard ML training pipeline workflow."""
        workflow = Workflow(
            name="ML Training Pipeline",
            description="Complete ML training pipeline from data to model",
        )

        # Data loading and validation
        workflow.add_plugin_step(
            step_id="load_data",
            plugin_name="data_loader",
            plugin_config={"data_path": data_path},
        )

        # Data preprocessing
        workflow.add_plugin_step(
            step_id="preprocess_data",
            plugin_name="data_preprocessor",
            dependencies=["load_data"],
        )

        # Model training
        workflow.add_plugin_step(
            step_id="train_model",
            plugin_name="sklearn_trainer",
            dependencies=["preprocess_data"],
            plugin_config={
                "model_type": model_type,
                "problem_type": problem_type,
            },
        )

        # Model evaluation
        workflow.add_plugin_step(
            step_id="evaluate_model",
            plugin_name="model_evaluator",
            dependencies=["train_model"],
        )

        return workflow

    @staticmethod
    def inference_pipeline(model_path: str) -> Workflow:
        """Create a model inference pipeline workflow."""
        workflow = Workflow(
            name="ML Inference Pipeline",
            description="Model inference pipeline for predictions",
        )

        # Load model
        workflow.add_plugin_step(
            step_id="load_model",
            plugin_name="model_loader",
            plugin_config={"model_path": model_path},
        )

        # Data preprocessing
        workflow.add_plugin_step(
            step_id="preprocess_input",
            plugin_name="data_preprocessor",
            dependencies=["load_model"],
        )

        # Make predictions
        workflow.add_plugin_step(
            step_id="predict",
            plugin_name="model_predictor",
            dependencies=["preprocess_input"],
        )

        return workflow
