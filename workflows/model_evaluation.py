# """Model evaluation workflow.
# This workflow handles the evaluation of trained machine learning models.
# """
# import argparse
# import logging
# from pathlib import Path
# from typing import Any
# import os
# from dotenv import load_dotenv
# from src.data.loading import load_data
# from src.ml.inference import predict
# from src.ml.training import load_model
# from src.utils.common import get_model_path, load_config, setup_logging

# # Load environment variables
# load_dotenv()

# # Configure logging
# setup_logging(
# level=os.getenv("LOG_LEVEL", "INFO"),
# log_file=os.getenv("LOG_FILE", None),
# log_format=os.getenv(
#         "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     ),
# )
# logger = logging.getLogger(__name__)
# def evaluate_model(
# model_path: str | Path,
# data_path: str | Path,
# output_dir: str | Path | None = None,
# target_column: str | None = None,
# feature_columns: list[str] | None = None,
# generate_plots: bool = True,
# config_path: str | Path | None = None,
# ) -> dict[str, Any]:
#     """Evaluate a trained machine learning model.
# Args:
# model_path: Path to the trained model
# data_path: Path to the evaluation data
# output_dir: Directory where evaluation results will be saved (None for default)
# target_column: Name of the target column (None to use model metadata)
# feature_columns: List of feature columns to use (None to use model metadata)
# generate_plots: Whether to generate evaluation plots
# config_path: Path to the configuration file (None for default)
# Returns:
# Dictionary with evaluation results
#     """
# logger.info(f"Starting model evaluation workflow: {model_path}")

#     # Load configuration if provided
# config = None
# if config_path is not None:
# config = load_config(config_path)
# logger.info(f"Loaded configuration from {config_path}")

#         # Override parameters from config if not explicitly provided
# if "target_column" in config and target_column is None:
# target_column = config["target_column"]
# if "feature_columns" in config and feature_columns is None:
# feature_columns = config["feature_columns"]
# if "generate_plots" in config and generate_plots is None:
# generate_plots = config["generate_plots"]

#     # Set default output directory if not provided
# if output_dir is None:
# output_dir = get_model_path("evaluation")
# logger.info(f"Using default output directory: {output_dir}")

#     # Create output directory if it doesn't exist
# output_dir = Path(output_dir)
# output_dir.mkdir(parents=True, exist_ok=True)

#     # Load model
# model, metadata = load_model(model_path)
# logger.info(
# f"Loaded {metadata.get('model_type', 'unknown')} model for "
# f"{metadata.get('problem_type', 'unknown')}"
#     )

#     # Use metadata for target and feature columns if not provided
# if target_column is None and "target_column" in metadata:
# target_column = metadata["target_column"]
# logger.info(f"Using target column from model metadata: {target_column}")
# if feature_columns is None and "feature_columns" in metadata:
# feature_columns = metadata["feature_columns"]
# logger.info(
# f"Using feature columns from model metadata: {len(feature_columns)} columns"
#         )

#     # Load data
# df = load_data(data_path)
# logger.info(
# f"Loaded evaluation data with {len(df)} rows and {len(df.columns)} columns"
#     )

#     # Validate target column
# if target_column is not None and target_column not in df.columns:
# raise ValueError(f"Target column '{target_column}' not found in data")

#     # Select features and target
# if feature_columns is None:
# if target_column is not None:
# feature_columns = [col for col in df.columns if col != target_column]
# else:
# feature_columns = df.columns
# X = df.select(feature_columns)
# if target_column is not None:
# y = df[target_column]
# has_target = True
# else:
# has_target = False

#     # Make predictions
# __y_pred = predict(model, X)

#     # Evaluate model if target is available
# evaluation_results = {}
# if has_target:
# problem_type = metadata.get("problem_type", "classification")
# if problem_type == "classification":
# metrics = evaluate_classification_model(model, X, y)
# logger.info(
# f"Classification metrics: accuracy={metrics['accuracy']:.4f}, "
# f"f1={metrics['f1']:.4f}"
#             )
# else:  # regression
# metrics = evaluate_regression_model(model, X, y)
# logger.info(
# f"Regression metrics: rmse={metrics['rmse']:.4f}, "
# f"r2={metrics['r2']:.4f}"
#             )
# evaluation_results["metrics"] = metrics

#     # Generate plots if requested
# if generate_plots:
# plots_info = {}

#         # Create confusion matrix plot for classification
# if has_target and metadata.get("problem_type") == "classification":
# cm_plot_path = output_dir / "confusion_matrix.png"
# cm = metrics["confusion_matrix"]
# plot_confusion_matrix(cm, output_path=cm_plot_path)
# plots_info["confusion_matrix"] = str(cm_plot_path)
# logger.info(f"Confusion matrix plot saved to {cm_plot_path}")

#         # Create feature importance plot if available
# if hasattr(model, "feature_importances_"):
# importance_plot_path = output_dir / "feature_importance.png"
# plot_feature_importance(
# feature_columns,
# model.feature_importances_,
# output_path=importance_plot_path,
#             )
# plots_info["feature_importance"] = str(importance_plot_path)
# logger.info(f"Feature importance plot saved to {importance_plot_path}")
# evaluation_results["plots"] = plots_info

#     # Save evaluation results
# results_path = output_dir / "evaluation_results.json"
# save_evaluation_results(evaluation_results, results_path)
# logger.info(f"Evaluation results saved to {results_path}")

#     # Return evaluation results
# return {
#         "model_path": str(model_path),
#         "data_path": str(data_path),
#         "output_dir": str(output_dir),
#         "results_path": str(results_path),
#         "model_type": metadata.get("model_type"),
#         "problem_type": metadata.get("problem_type"),
#         "has_target": has_target,
#         "evaluation_results": evaluation_results,
#         "success": True,
#     }
# if __name__ == "__main__":
#     # Parse command line arguments
# parser = argparse.ArgumentParser(description="Model evaluation workflow")
# parser.add_argument("model_path", help="Path to the trained model")
# parser.add_argument("data_path", help="Path to the evaluation data")
# parser.add_argument(
#         "--output-dir", help="Directory where evaluation results will be saved"
#     )
# parser.add_argument("--target-column", help="Name of the target column")
# parser.add_argument(
#         "--feature-columns", nargs="+", help="List of feature columns to use"
#     )
# parser.add_argument(
#         "--no-plots", action="store_true", help="Skip generating evaluation plots"
#     )
# parser.add_argument("--config", help="Path to the configuration file")
# args = parser.parse_args()

#     # Run the workflow
# result = evaluate_model(
# model_path=args.model_path,
# data_path=args.data_path,
# output_dir=args.output_dir,
# target_column=args.target_column,
# feature_columns=args.feature_columns,
# generate_plots=not args.no_plots,
# config_path=args.config,
#     )

#     # Print result summary
# print(
# f"Model evaluation complete for {result['model_type']} "
# f"{result['problem_type']} model"
#     )
# if result["has_target"]:
# if result["problem_type"] == "classification":
# metrics = result["evaluation_results"]["metrics"]
# print(f"Accuracy: {metrics['accuracy']:.4f}")
# print(f"F1 Score: {metrics['f1']:.4f}")
# else:  # regression
# metrics = result["evaluation_results"]["metrics"]
# print(f"RMSE: {metrics['rmse']:.4f}")
# print(f"R²: {metrics['r2']:.4f}")
# print(f"Evaluation results saved to: {result['results_path']}")
