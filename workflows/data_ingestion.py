"""
Data ingestion workflow for ML pipeline.
This module handles loading, validating, and preprocessing data from various sources.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.data.loading import load_data, save_data
from src.data.validation import generate_data_quality_report, validate_schema
from src.platform.utils.common import get_data_path, load_config, setup_logging

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


def ingest_data(
    source_path: str | Path,
    destination_path: str | Path | None = None,
    validate: bool = True,
    generate_report: bool = True,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Ingest data from a source into the raw data directory.

    Args:
        source_path: Path to the source data file
        destination_path: Path where the ingested data will be saved (None for default)
        validate: Whether to validate the data
        generate_report: Whether to generate a data quality report
        config_path: Path to the configuration file (None for default)

    Returns:
        Dictionary with information about the ingestion
    """
    logger.info(f"Starting data ingestion workflow: {source_path}")

    # Load configuration if provided
    config = None
    if config_path is not None:
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")

    # Set default destination path if not provided
    if destination_path is None:
        source_path = Path(source_path)
        raw_data_path = get_data_path("raw")
        destination_path = raw_data_path / source_path.name
        logger.info(f"Using default destination path: {destination_path}")

    # Load data
    df = load_data(source_path)
    rows_count = len(df)
    columns_count = len(df.columns)
    logger.info(f"Loaded data with {rows_count} rows and {columns_count} columns")

    # Validate data if requested
    validation_results = None
    if validate and config is not None and "schema" in config:
        logger.info("Validating data against schema")
        is_valid, errors = validate_schema(df, config["schema"])
        validation_results = {"is_valid": is_valid, "errors": errors}

        if not is_valid:
            logger.warning(f"Data validation failed: {errors}")

    # Generate data quality report if requested
    quality_report = None
    if generate_report:
        logger.info("Generating data quality report")
        quality_report = generate_data_quality_report(df)

    # Save data to destination
    save_data(df, destination_path)
    logger.info(f"Data saved to {destination_path}")

    # Return information about the ingestion
    result = {
        "source_path": str(source_path),
        "destination_path": str(destination_path),
        "rows_count": rows_count,
        "columns_count": columns_count,
        "columns": df.columns.tolist(),
        "success": True,
    }

    if validation_results is not None:
        result["validation"] = validation_results

    if quality_report is not None:
        # Save quality report
        quality_report_path = Path(destination_path).with_suffix(".report.json")
        with open(quality_report_path, "w") as f:
            json.dump(quality_report, f, indent=2)
        logger.info(f"Data quality report saved to {quality_report_path}")

        result["quality_report_path"] = str(quality_report_path)

    return result


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Data ingestion workflow")
    parser.add_argument("source_path", help="Path to the source data file")
    parser.add_argument(
        "--destination-path", help="Path where the ingested data will be saved"
    )
    parser.add_argument(
        "--no-validate", action="store_true", help="Skip data validation"
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Skip data quality report generation"
    )
    parser.add_argument("--config", help="Path to the configuration file")

    args = parser.parse_args()

    # Run the workflow
    result = ingest_data(
        source_path=args.source_path,
        destination_path=args.destination_path,
        validate=not args.no_validate,
        generate_report=not args.no_report,
        config_path=args.config,
    )

    # Print result summary
    print(
        f"Data ingestion complete: {result['rows_count']} rows "
        f"ingested to {result['destination_path']}"
    )

    if "validation" in result:
        if result["validation"]["is_valid"]:
            print("Data validation: PASSED")
        else:
            print(
                "Data validation: FAILED - "
                f"{len(result['validation']['errors'])} errors"
            )

    if "quality_report_path" in result:
        print(f"Data quality report saved to: {result['quality_report_path']}")
