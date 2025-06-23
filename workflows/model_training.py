"""Model training workflow for the MLX platform.

This workflow provides production-grade model training with:
- Plugin-based architecture for extensibility
- Comprehensive error handling and observability
- Hyperparameter optimization
- Model validation and persistence
- Integration with the MLX configuration system
- Workflow orchestration with dependency management

This demonstrates both direct plugin usage and workflow orchestration patterns.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from src.data.loading import load_data
from src.ml.training import (
    evaluate_model_cv,
    hyperparameter_tuning,
    save_model,
    train_model,
)
from src.platform.config import ConfigManager
from src.platform.plugins import (
    ComponentResult,
    ComponentStatus,
    ExecutionContext,
    register_plugin,
)
from src.platform.plugins.base import ModelTrainer
from src.platform.utils.common import create_run_id, setup_logging
from src.platform.workflows.base import WorkflowResult

logger = logging.getLogger(__name__)


@register_plugin(
    name="sklearn_trainer",
    category="model_training",
    description="Production-grade scikit-learn model trainer with hyperparameter optimization",
    dependencies=["sklearn", "polars"],
    version="1.2.0",
)
class SklearnModelTrainer(ModelTrainer):
    """Advanced model trainer using scikit-learn with production features."""

    def initialize(self, context: ExecutionContext) -> None:
        """Initialize the trainer with validation."""
        self.logger.info(f"Initializing {self.name} v{self.get_version()}")
        self.model = None
        self.feature_columns = None
        self.training_metadata = {}

    def execute(self, context: ExecutionContext) -> ComponentResult:
        """Execute model training with comprehensive error handling."""
        start_time = time.time()

        try:
            # Load and validate configuration
            config = context.config
            model_config = config.model
            ml_config = config.ml

            # Load and validate data
            df = self._load_and_validate_data(context)

            # Prepare features and target
            X, y, feature_columns = self._prepare_features_target(df, model_config)
            self.feature_columns = feature_columns

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=ml_config.test_size,
                random_state=ml_config.random_seed,
                stratify=y if model_config.problem_type == "classification" else None,
            )

            self.logger.info(
                f"Data split: {len(X_train)} train, {len(X_test)} test samples"
            )

            # Create and train model
            model = self._create_model(model_config, ml_config)

            if ml_config.hyperparameter_search:
                model, tuning_results = self._perform_hyperparameter_tuning(
                    model, X_train, y_train, model_config, ml_config
                )
            else:
                model = train_model(X_train, y_train, model)
                tuning_results = None

            # Evaluate model
            scoring = self._get_scoring_metric(model_config.problem_type)
            cv_metrics = evaluate_model_cv(
                X_train, y_train, model, cv=ml_config.cv_folds, scoring=scoring
            )

            # Save model with metadata
            model_path = self._save_model_with_metadata(
                model, context, config, feature_columns, cv_metrics, tuning_results
            )

            # Store for later use
            self.model = model
            execution_time = time.time() - start_time

            return ComponentResult(
                status=ComponentStatus.SUCCESS,
                component_name=self.name,
                execution_time=execution_time,
                output_data={
                    "model_path": str(model_path),
                    "feature_columns": feature_columns,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "model_type": model_config.model_type,
                    "problem_type": model_config.problem_type,
                },
                artifacts={"model": model_path},
                metrics={
                    "cv_score_mean": cv_metrics["mean_cv_score"],
                    "cv_score_std": cv_metrics["std_cv_score"],
                    "training_time": execution_time,
                    "feature_count": len(feature_columns),
                },
                metadata=self.training_metadata,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Model training failed: {e}", exc_info=True)
            return ComponentResult(
                status=ComponentStatus.FAILED,
                component_name=self.name,
                execution_time=execution_time,
                error_message=str(e),
            )

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate trainer configuration."""
        required_fields = ["model_type", "problem_type", "target_column"]

        for field in required_fields:
            if field not in config:
                self.logger.error(f"Missing required config field: {field}")
                return False

        valid_model_types = ["random_forest", "linear"]
        if config["model_type"] not in valid_model_types:
            self.logger.error(f"Invalid model_type: {config['model_type']}")
            return False

        valid_problem_types = ["classification", "regression"]
        if config["problem_type"] not in valid_problem_types:
            self.logger.error(f"Invalid problem_type: {config['problem_type']}")
            return False

        return True

    def train_model(self, train_data, validation_data, context: ExecutionContext):
        """Train a model (implementation of abstract method)."""
        return self.model

    def save_model(self, model, model_path: Path, context: ExecutionContext) -> None:
        """Save model (implementation of abstract method)."""
        metadata = {"saved_by": self.name, "version": self.get_version()}
        save_model(model, model_path, metadata)

    def load_model(self, model_path: Path, context: ExecutionContext):
        """Load model (implementation of abstract method)."""
        from src.ml.training import load_model

        model, metadata = load_model(model_path)
        return model

    def cleanup(self, context: ExecutionContext) -> None:
        """Clean up resources after execution."""
        self.logger.debug(f"Cleaning up {self.name}")
        # No specific cleanup needed for sklearn models

    def get_version(self) -> str:
        """Get component version."""
        return "1.2.0"

    def _load_and_validate_data(self, context: ExecutionContext):
        """Load and validate input data."""
        if "data_path" in context.input_data:
            data_path = context.input_data["data_path"]
            df = load_data(data_path)
        elif "dataframe" in context.input_data:
            df = context.input_data["dataframe"]
        else:
            raise ValueError(
                "No data provided in context (requires 'data_path' or 'dataframe')"
            )

        if len(df) == 0:
            raise ValueError("Dataset is empty")

        self.logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _prepare_features_target(self, df, model_config):
        """Prepare features and target variables."""
        if model_config.target_column not in df.columns:
            raise ValueError(f"Target column '{model_config.target_column}' not found")

        # Select features
        if model_config.feature_columns is None:
            feature_columns = [
                col for col in df.columns if col != model_config.target_column
            ]
        else:
            feature_columns = model_config.feature_columns
            missing_features = set(feature_columns) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing feature columns: {missing_features}")

        X = df.select(feature_columns)
        y = df[model_config.target_column]

        return X, y, feature_columns

    def _create_model(self, model_config, ml_config):
        """Create model based on configuration."""
        base_params = {
            "random_state": ml_config.random_seed,
            **model_config.model_params,
        }

        if model_config.model_type == "random_forest":
            if model_config.problem_type == "classification":
                return RandomForestClassifier(**base_params)
            else:  # regression
                return RandomForestRegressor(**base_params)
        elif model_config.model_type == "linear":
            if model_config.problem_type == "classification":
                return LogisticRegression(**base_params)
            else:  # regression
                base_params.pop(
                    "random_state", None
                )  # LinearRegression doesn't have random_state
                return LinearRegression(**base_params)
        else:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")

    def _get_scoring_metric(self, problem_type: str) -> str:
        """Get appropriate scoring metric."""
        return (
            "f1_weighted"
            if problem_type == "classification"
            else "neg_mean_squared_error"
        )

    def _perform_hyperparameter_tuning(
        self, model, X_train, y_train, model_config, ml_config
    ):
        """Perform hyperparameter optimization."""
        self.logger.info("Starting hyperparameter optimization")

        param_grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "linear": {
                "classification": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"],
                },
                "regression": {
                    "fit_intercept": [True, False],
                },
            },
        }

        if model_config.model_type == "linear":
            param_grid = param_grids["linear"][model_config.problem_type]
        else:
            param_grid = param_grids.get(model_config.model_type, {})

        if not param_grid:
            self.logger.warning(f"No parameter grid for {model_config.model_type}")
            return train_model(X_train, y_train, model), None

        scoring = self._get_scoring_metric(model_config.problem_type)
        best_model, tuning_results = hyperparameter_tuning(
            X_train, y_train, model, param_grid, cv=ml_config.cv_folds, scoring=scoring
        )

        self.logger.info(f"Best parameters: {tuning_results['best_params']}")
        return best_model, tuning_results

    def _save_model_with_metadata(
        self, model, context, config, feature_columns, cv_metrics, tuning_results
    ):
        """Save model with comprehensive metadata."""
        model_path = (
            context.artifacts_dir
            / f"{config.model.model_type}_{config.model.problem_type}_{context.run_id}.pkl"
        )

        metadata = {
            "model_type": config.model.model_type,
            "problem_type": config.model.problem_type,
            "feature_columns": feature_columns,
            "target_column": config.model.target_column,
            "cv_metrics": cv_metrics,
            "test_size": config.ml.test_size,
            "random_seed": config.ml.random_seed,
            "hyperparameter_search": config.ml.hyperparameter_search,
            "cv_folds": config.ml.cv_folds,
            "trainer_name": self.name,
            "trainer_version": self.get_version(),
            "run_id": context.run_id,
            "training_timestamp": time.time(),
        }

        if tuning_results:
            metadata["hyperparameter_tuning"] = {
                "best_params": tuning_results["best_params"],
                "best_score": tuning_results["best_score"],
            }

        self.training_metadata = metadata
        save_model(model, model_path, metadata)
        self.logger.info(f"Model saved to {model_path}")

        return model_path


def train_and_evaluate_model(
    data_path: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> ComponentResult:
    """Train and evaluate a machine learning model using the plugin system.

    Args:
        data_path: Path to the data file (optional if data in config)
        config_overrides: Configuration overrides
        run_id: Run ID for tracking (if None, generates one)

    Returns:
        ComponentResult with training results
    """
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config(overrides=config_overrides)

    # Setup logging
    setup_logging(config)

    # Generate run ID if not provided
    if run_id is None:
        run_id = create_run_id("train")

    logger.info(f"Starting model training workflow with run ID: {run_id}")

    # Create execution context
    input_data = {}
    if data_path:
        input_data["data_path"] = str(data_path)

    context = ExecutionContext(
        config=config,
        run_id=run_id,
        component_name="sklearn_trainer",
        input_data=input_data,
    )

    # Get trainer plugin and execute
    from src.platform.plugins import get_plugin

    trainer = get_plugin("sklearn_trainer")
    trainer.initialize(context)
    result = trainer.execute(context)

    if result.is_success():
        logger.info("Model training completed successfully")
        logger.info(f"Model saved to: {result.artifacts.get('model', 'N/A')}")
        logger.info(
            f"CV Score: {result.metrics.get('cv_score_mean', 0):.4f} ± "
            f"{result.metrics.get('cv_score_std', 0):.4f}"
        )
    else:
        logger.error(f"Model training failed: {result.error_message}")

    return result


async def train_model_workflow(
    data_path: Path,
    config_overrides: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> WorkflowResult:
    """Execute model training as a structured workflow.

    This demonstrates the workflow orchestration approach for complex ML pipelines
    with proper dependency management and observability.
    """
    from src.platform.workflows.base import MLWorkflowTemplates, WorkflowEngine

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config(overrides=config_overrides)

    # Generate run ID if not provided
    if run_id is None:
        run_id = create_run_id("workflow_train")

    logger.info(f"Starting training workflow with run ID: {run_id}")

    # Create workflow from template
    workflow = MLWorkflowTemplates.training_pipeline(
        data_path=str(data_path),
        model_type=config.model.model_type,
        problem_type=config.model.problem_type,
    )

    # Create execution context
    context = ExecutionContext(
        config=config,
        run_id=run_id,
        component_name="training_workflow",
        input_data={"data_path": str(data_path)},
    )

    # Execute workflow
    engine = WorkflowEngine()
    result = await engine.execute_workflow(workflow, context)

    logger.info(f"Training workflow completed with status: {result.status}")
    return result


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Model training workflow for MLX platform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data_path", help="Path to the data file")
    parser.add_argument(
        "--model-type", choices=["random_forest", "linear"], help="Override model type"
    )
    parser.add_argument(
        "--problem-type",
        choices=["classification", "regression"],
        help="Override problem type",
    )
    parser.add_argument("--target-column", help="Override target column name")
    parser.add_argument("--random-seed", type=int, help="Override random seed")
    parser.add_argument(
        "--hyperparameter-search",
        action="store_true",
        help="Enable hyperparameter search",
    )
    parser.add_argument("--run-id", help="Specify run ID for tracking")

    args = parser.parse_args()

    # Build config overrides from command line arguments
    config_overrides = {}
    if args.model_type:
        config_overrides.setdefault("model", {})["model_type"] = args.model_type
    if args.problem_type:
        config_overrides.setdefault("model", {})["problem_type"] = args.problem_type
    if args.target_column:
        config_overrides.setdefault("model", {})["target_column"] = args.target_column
    if args.random_seed:
        config_overrides.setdefault("ml", {})["random_seed"] = args.random_seed
    if args.hyperparameter_search:
        config_overrides.setdefault("ml", {})["hyperparameter_search"] = True

    # Run the workflow
    result = train_and_evaluate_model(
        data_path=Path(args.data_path),
        config_overrides=config_overrides,
        run_id=args.run_id,
    )

    # Print result summary
    if result.is_success():
        print("✅ Model training successful!")
        print(
            f"📊 CV Score: {result.metrics.get('cv_score_mean', 0):.4f} ± "
            f"{result.metrics.get('cv_score_std', 0):.4f}"
        )
        print(f"💾 Model saved: {result.artifacts.get('model', 'N/A')}")
        print(f"⏱️  Training time: {result.execution_time:.2f}s")
        print(f"📈 Features: {result.metrics.get('feature_count', 0)}")
    else:
        print(f"❌ Model training failed: {result.error_message}")
        exit(1)
