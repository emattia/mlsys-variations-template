"""Feature engineering workflow.

This workflow handles the transformation of raw data into features for model training.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.data.loading import load_data, save_data
from src.data.processing import (
    clean_data,
    encode_categorical,
    get_categorical_columns,
    get_numeric_columns,
    normalize_features,
)
from src.utils.common import get_data_path, load_config, setup_logging

# Load environment variables
load_dotenv()

# Configure logging
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", None),
    log_format=os.getenv(
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ),
)
logger = logging.getLogger(__name__)


def engineer_features(
    input_path: str | Path,
    output_path: str | Path | None = None,
    config_path: str | Path | None = None,
    clean: bool = True,
    normalize: bool = True,
    encode: bool = True,
    normalization_method: str = "standard",
    encoding_method: str = "one_hot",
    target_column: str | None = None,
) -> dict[str, Any]:
    """Engineer features from raw data.

    Args:
        input_path: Path to the input data file
        output_path: Path for the engineered data (None for default)
        config_path: Path to the configuration file (None for default)
        clean: Whether to clean the data
        normalize: Whether to normalize numeric features
        encode: Whether to encode categorical features
        normalization_method: Method for normalization ('standard' or 'minmax')
        encoding_method: Method for encoding ('one_hot' or 'label')
        target_column: Name of the target column (None if no target column)

    Returns:
        Dictionary with information about the feature engineering
    """
    logger.info(f"Starting feature engineering workflow: {input_path}")

    # Load configuration if provided
    config = None
    if config_path is not None:
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        # Override parameters from config if not explicitly provided
        if "clean" in config and clean is None:
            clean = config["clean"]
        if "normalize" in config and normalize is None:
            normalize = config["normalize"]
        if "encode" in config and encode is None:
            encode = config["encode"]
        if "normalization_method" in config and normalization_method is None:
            normalization_method = config["normalization_method"]
        if "encoding_method" in config and encoding_method is None:
            encoding_method = config["encoding_method"]
        if "target_column" in config and target_column is None:
            target_column = config["target_column"]

    # Set default output path if not provided
    if output_path is None:
        input_path = Path(input_path)
        processed_data_path = get_data_path("processed")
        output_path = (
            processed_data_path / f"{input_path.stem}_features{input_path.suffix}"
        )
        logger.info(f"Using default output path: {output_path}")

    # Load data
    df = load_data(input_path)
    rows_before = len(df)
    logger.info(f"Loaded data with {rows_before} rows and {len(df.columns)} columns")

    # Separate target column if provided
    target = None
    if target_column is not None and target_column in df.columns:
        target = df[target_column]
        df = df.drop(target_column)
        logger.info(f"Separated target column: {target_column}")

    # Clean data if requested
    if clean:
        logger.info("Cleaning data")
        df = clean_data(df)
        logger.info(f"Data cleaned: {rows_before} rows -> {len(df)} rows")

    # Get numeric and categorical columns
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_categorical_columns(df)
    logger.info(
        f"Found {len(numeric_columns)} numeric and "
        f"{len(categorical_columns)} categorical columns"
    )

    # Normalize numeric features if requested
    if normalize and numeric_columns:
        logger.info(f"Normalizing numeric features using {normalization_method} method")
        df = normalize_features(df, numeric_columns, method=normalization_method)

    # Encode categorical features if requested
    if encode and categorical_columns:
        logger.info(f"Encoding categorical features using {encoding_method} method")
        df = encode_categorical(df, categorical_columns, method=encoding_method)

    # Add target column back if it was separated
    if target is not None:
        df = df.with_columns(target.alias(target_column))
        logger.info(f"Added target column back: {target_column}")

    # Save engineered features
    save_data(df, output_path)
    logger.info(f"Engineered features saved to {output_path}")

    # Return information about the feature engineering
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_before": rows_before,
        "rows_after": len(df),
        "columns_before": rows_before,
        "columns_after": len(df.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "cleaning_applied": clean,
        "normalization_applied": normalize,
        "encoding_applied": encode,
        "normalization_method": normalization_method if normalize else None,
        "encoding_method": encoding_method if encode else None,
        "target_column": target_column,
        "success": True,
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Feature engineering workflow")
    parser.add_argument("input_path", help="Path to the input data file")
    parser.add_argument(
        "--output-path", help="Path where the engineered data will be saved"
    )
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument("--no-clean", action="store_true", help="Skip data cleaning")
    parser.add_argument(
        "--no-normalize", action="store_true", help="Skip normalization"
    )
    parser.add_argument("--no-encode", action="store_true", help="Skip encoding")
    parser.add_argument(
        "--normalization-method",
        choices=["standard", "minmax"],
        default="standard",
        help="Method for normalizing numeric features",
    )
    parser.add_argument(
        "--encoding-method",
        choices=["one_hot", "label"],
        default="one_hot",
        help="Method for encoding categorical features",
    )
    parser.add_argument("--target-column", help="Name of the target column")

    args = parser.parse_args()

    # Run the workflow
    result = engineer_features(
        input_path=args.input_path,
        output_path=args.output_path,
        config_path=args.config,
        clean=not args.no_clean,
        normalize=not args.no_normalize,
        encode=not args.no_encode,
        normalization_method=args.normalization_method,
        encoding_method=args.encoding_method,
        target_column=args.target_column,
    )

    # Print result summary
    print(
        f"Feature engineering complete: {result['rows_after']} rows, "
        f"{result['columns_after']} columns"
    )
    print(f"Output saved to: {result['output_path']}")
