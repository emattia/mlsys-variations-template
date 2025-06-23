# """Batch inference workflow.

# This workflow handles batch predictions using trained machine learning models.
# """

# import argparse
# import logging
# from pathlib import Path
# from typing import Any
# import os

# from dotenv import load_dotenv

# from src.ml.inference import batch_predict
# from src.ml.training import load_model
# from src.utils.common import get_data_path, load_config, setup_logging

# # Load environment variables
# load_dotenv()

# # Configure logging
# setup_logging(
#     level=os.getenv("LOG_LEVEL", "INFO"),
#     log_file=os.getenv("LOG_FILE", None),
#     log_format=os.getenv(
#         "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     ),
# )
# logger = logging.getLogger(__name__)


# def run_batch_inference(
#     model_path: str | Path,
#     input_path: str | Path,
#     output_path: str | Path | None = None,
#     feature_columns: list[str] | None = None,
#     id_column: str | None = None,
#     return_probabilities: bool = False,
#     config_path: str | Path | None = None,
# ) -> dict[str, Any]:
#     """Run batch inference using a trained model.

#     Args:
#         model_path: Path to the trained model
#         input_path: Path to the input data file
#         output_path: Path where the predictions will be saved (None for default)
#         feature_columns: List of feature columns to use (None to use model metadata)
#         id_column: Name of the ID column (None if no ID column)
#         return_probabilities: Whether to return probability estimates (for classifiers)
#         config_path: Path to the configuration file (None for default)

#     Returns:
#         Dictionary with information about the batch inference
#     """
#     logger.info(f"Starting batch inference workflow: {model_path} on {input_path}")

#     # Load configuration if provided
#     config = None
#     if config_path is not None:
#         config = load_config(config_path)
#         logger.info(f"Loaded configuration from {config_path}")

#         # Override parameters from config if not explicitly provided
#         if "feature_columns" in config and feature_columns is None:
#             feature_columns = config["feature_columns"]
#         if "id_column" in config and id_column is None:
#             id_column = config["id_column"]
#         if "return_probabilities" in config and return_probabilities is None:
#             return_probabilities = config["return_probabilities"]

#     # Set default output path if not provided
#     if output_path is None:
#         input_path = Path(input_path)
#         model_path = Path(model_path)
#         output_dir = get_data_path("processed")
#         output_path = output_dir / f"{input_path.stem}_predictions{input_path.suffix}"
#         logger.info(f"Using default output path: {output_path}")

#     # Load model to get metadata
#     model, metadata = load_model(model_path)
#     logger.info(
#         f"Loaded {metadata.get('model_type', 'unknown')} model for "
#         f"{metadata.get('problem_type', 'unknown')} problem"
#     )

#     # Use feature columns from model metadata if not provided
#     if feature_columns is None and "feature_columns" in metadata:
#         feature_columns = metadata["feature_columns"]
#         logger.info(
#             f"Using feature columns from model metadata: {len(feature_columns)} columns"
#         )

#     # Run batch prediction
#     result = batch_predict(
#         model_path=model_path,
#         input_path=input_path,
#         output_path=output_path,
#         feature_columns=feature_columns,
#         id_column=id_column,
#         return_probabilities=return_probabilities,
#     )

#     logger.info(f"Batch inference complete: {result['num_predictions']} predictions")
#     logger.info(f"Predictions saved to {result['output_path']}")

#     return {
#         "model_path": str(model_path),
#         "input_path": str(input_path),
#         "output_path": str(output_path),
#         "num_predictions": result["num_predictions"],
#         "feature_columns": result["feature_columns"],
#         "id_column": id_column,
#         "has_probabilities": result["has_probabilities"],
#         "model_type": result["model_type"],
#         "model_metadata": result["model_metadata"],
#         "success": True,
#     }


# if __name__ == "__main__":
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description="Batch inference workflow")
#     parser.add_argument("model_path", help="Path to the trained model")
#     parser.add_argument("input_path", help="Path to the input data file")
#     parser.add_argument(
#         "--output-path", help="Path where the predictions will be saved"
#     )
#     parser.add_argument(
#         "--feature-columns", nargs="+", help="List of feature columns to use"
#     )
#     parser.add_argument("--id-column", help="Name of the ID column")
#     parser.add_argument(
#         "--return-probabilities",
#         action="store_true",
#         help="Return probability estimates",
#     )
#     parser.add_argument("--config", help="Path to the configuration file")

#     args = parser.parse_args()

#     # Run the workflow
#     result = run_batch_inference(
#         model_path=args.model_path,
#         input_path=args.input_path,
#         output_path=args.output_path,
#         feature_columns=args.feature_columns,
#         id_column=args.id_column,
#         return_probabilities=args.return_probabilities,
#         config_path=args.config,
#     )

#     # Print result summary
#     result = run_batch_inference(model_path, input_path, output_path, id_column)
#     print(f"Batch inference complete: {result['num_predictions']} predictions")
#     print(f"Model type: {result['model_type']}")
#     print(f"Predictions saved to: {result['output_path']}")

#     if result["has_probabilities"]:
#         print("Probability estimates included in predictions")
