"""Model training workflow updated for the new configuration system.

This workflow handles the training of machine learning models using the
unified configuration system with Pydantic models and Hydra integration.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import AppConfig, load_config
from src.data.loading import load_data
from src.models.training import (
    evaluate_model_cv,
    hyperparameter_tuning,
    save_model,
    train_model,
)
from src.plugins import (
    ComponentResult,
    ComponentStatus,
    ExecutionContext,
    ModelTrainer,
    register_plugin,
)
from src.utils.common import create_run_id, setup_logging

logger = logging.getLogger(__name__)


@register_plugin(
    name="sklearn_trainer",
    category="model_training",
    description="Scikit-learn based model trainer",
    dependencies=["sklearn"],
    version="1.0.0",
)
class SklearnModelTrainer(ModelTrainer):
    """Model trainer using scikit-learn models."""

    def initialize(self, context: ExecutionContext) -> None:
        """Initialize the trainer."""
        self.logger.info(f"Initializing {self.name}")
        self.model = None
        self.feature_columns = None

    def execute(self, context: ExecutionContext) -> ComponentResult:
        """Execute model training."""
        start_time = time.time()

        try:
            # Get configuration
            config = context.config
            model_config = config.model
            ml_config = config.ml

            # Load data from context or file
            if "data_path" in context.input_data:
                data_path = context.input_data["data_path"]
                df = load_data(data_path)
            elif "dataframe" in context.input_data:
                df = context.input_data["dataframe"]
            else:
                raise ValueError("No data provided in context")

            self.logger.info(
                f"Loaded data with {len(df)} rows and {len(df.columns)} columns"
            )

            # Validate target column
            if model_config.target_column not in df.columns:
                raise ValueError(
                    f"Target column '{model_config.target_column}' not found in data"
                )

            # Select features
            if model_config.feature_columns is None:
                feature_columns = [
                    col for col in df.columns if col != model_config.target_column
                ]
            else:
                feature_columns = model_config.feature_columns

            self.feature_columns = feature_columns

            # Split data into features and target
            X = df.select(feature_columns)
            y = df[model_config.target_column]

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ml_config.test_size, random_state=ml_config.random_seed
            )

            self.logger.info(
                f"Split data into {len(X_train)} training and {len(X_test)} testing samples"
            )

            # Train model
            model = self._create_model(model_config, ml_config)

            # Perform hyperparameter search if requested
            if ml_config.hyperparameter_search:
                model, tuning_results = self._perform_hyperparameter_tuning(
                    model, X_train, y_train, model_config, ml_config
                )
            else:
                model = train_model(X_train, y_train, model)
                tuning_results = None

            # Evaluate the model using cross-validation
            scoring = self._get_scoring_metric(model_config.problem_type)
            cv_metrics = evaluate_model_cv(
                X_train, y_train, model, cv=ml_config.cv_folds, scoring=scoring
            )

            self.logger.info(
                f"Cross-validation {scoring}: {cv_metrics['mean_cv_score']:.4f} "
                f"± {cv_metrics['std_cv_score']:.4f}"
            )

            # Save the model
            model_path = (
                context.artifacts_dir
                / f"{model_config.model_type}_{model_config.problem_type}.pkl"
            )
            metadata = self._create_metadata(
                config, feature_columns, cv_metrics, tuning_results
            )

            save_model(model, model_path, metadata)
            self.logger.info(f"Model saved to {model_path}")

            # Store model for later use
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
                },
                artifacts={"model": model_path},
                metrics={
                    "cv_score_mean": cv_metrics["mean_cv_score"],
                    "cv_score_std": cv_metrics["std_cv_score"],
                    "training_time": execution_time,
                },
                metadata=metadata,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Model training failed: {e}")

            return ComponentResult(
                status=ComponentStatus.FAILED,
                component_name=self.name,
                execution_time=execution_time,
                error_message=str(e),
            )

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate trainer configuration."""
        required_fields = ["model_type", "problem_type", "target_column"]
        return all(field in config for field in required_fields)

    def train_model(self, train_data, validation_data, context: ExecutionContext):
        """Train a model (implementation of abstract method)."""
        # This is implemented in execute() method
        return self.model

    def save_model(self, model, model_path: Path, context: ExecutionContext) -> None:
        """Save model (implementation of abstract method)."""
        metadata = {"saved_by": self.name}
        save_model(model, model_path, metadata)

    def load_model(self, model_path: Path, context: ExecutionContext):
        """Load model (implementation of abstract method)."""
        import pickle

        with open(model_path, "rb") as f:
            return pickle.load(f)

    def _create_model(self, model_config, ml_config):
        """Create model based on configuration."""
        if model_config.model_type == "random_forest":
            if model_config.problem_type == "classification":
                return RandomForestClassifier(
                    random_state=ml_config.random_seed, **model_config.model_params
                )
            else:  # regression
                return RandomForestRegressor(
                    random_state=ml_config.random_seed, **model_config.model_params
                )
        elif model_config.model_type == "linear":
            if model_config.problem_type == "classification":
                return LogisticRegression(
                    random_state=ml_config.random_seed, **model_config.model_params
                )
            else:  # regression
                return LinearRegression(**model_config.model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")

    def _get_scoring_metric(self, problem_type: str) -> str:
        """Get scoring metric based on problem type."""
        if problem_type == "classification":
            return "f1_weighted"
        else:  # regression
            return "neg_mean_squared_error"

    def _perform_hyperparameter_tuning(
        self, model, X_train, y_train, model_config, ml_config
    ):
        """Perform hyperparameter tuning."""
        self.logger.info("Performing hyperparameter search")

        # Define parameter grid based on model type
        if model_config.model_type == "random_forest":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        elif model_config.model_type == "linear":
            if model_config.problem_type == "classification":
                param_grid = {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"],
                }
            else:  # regression
                param_grid = {
                    "fit_intercept": [True, False],
                }
        else:
            param_grid = {}

        scoring = self._get_scoring_metric(model_config.problem_type)
        best_model, tuning_results = hyperparameter_tuning(
            X_train, y_train, model, param_grid, cv=ml_config.cv_folds, scoring=scoring
        )

        self.logger.info(f"Best parameters: {tuning_results['best_params']}")
        return best_model, tuning_results

    def _create_metadata(
        self, config: AppConfig, feature_columns, cv_metrics, tuning_results
    ):
        """Create metadata for model saving."""
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
        }

        if tuning_results:
            metadata["hyperparameter_tuning"] = {
                "best_params": tuning_results["best_params"],
                "best_score": tuning_results["best_score"],
            }

        return metadata


def train_and_evaluate_model(
    data_path: Path | None = None,
    config: AppConfig | None = None,
    config_overrides: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> ComponentResult:
    """Train and evaluate a machine learning model using the plugin system.

    Args:
        data_path: Path to the data file (optional if data in config)
        config: Application configuration (if None, loads default)
        config_overrides: Configuration overrides
        run_id: Run ID for tracking (if None, generates one)

    Returns:
        ComponentResult with training results
    """
    # Load configuration
    if config is None:
        config = load_config(overrides=config_overrides)

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
    from src.plugins import get_plugin

    trainer = get_plugin("sklearn_trainer")
    trainer.initialize(context)

    result = trainer.execute(context)

    if result.is_success():
        logger.info("Model training completed successfully")
        logger.info(f"Model saved to: {result.artifacts.get('model', 'N/A')}")
        logger.info(
            f"CV Score: {result.metrics.get('cv_score_mean', 0):.4f} ± {result.metrics.get('cv_score_std', 0):.4f}"
        )
    else:
        logger.error(f"Model training failed: {result.error_message}")

    return result


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Model training workflow")
    parser.add_argument("data_path", help="Path to the data file")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--model-type", help="Override model type")
    parser.add_argument("--problem-type", help="Override problem type")
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

    # Load config with overrides
    if args.config:
        config = load_config(config_path=args.config, overrides=config_overrides)
    else:
        config = load_config(overrides=config_overrides)

    # Run the workflow
    result = train_and_evaluate_model(
        data_path=Path(args.data_path),
        config=config,
        run_id=args.run_id,
    )

    # Print result summary
    if result.is_success():
        print("✅ Model training successful!")
        print(
            f"📊 CV Score: {result.metrics.get('cv_score_mean', 0):.4f} ± {result.metrics.get('cv_score_std', 0):.4f}"
        )
        print(f"💾 Model saved: {result.artifacts.get('model', 'N/A')}")
        print(f"⏱️  Training time: {result.execution_time:.2f}s")
    else:
        print(f"❌ Model training failed: {result.error_message}")
        exit(1)
