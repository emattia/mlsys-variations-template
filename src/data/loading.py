"""Data loading functions.

This module contains functions for loading data from various sources.
"""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def load_data(file_path: str | Path, file_format: str | None = None) -> pl.DataFrame:
    """Load data from a file into a Polars DataFrame.

    Args:
        file_path: Path to the data file
        file_format: Format of the file (csv, parquet etc.). If None
            inferred from extension.

    Returns:
        Polars DataFrame containing the data
    """
    file_path = Path(file_path)

    if file_format is None:
        file_format = file_path.suffix.lstrip(".")

    logger.info(f"Loading data from {file_path} with format {file_format}")

    if file_format.lower() == "csv":
        return pl.read_csv(file_path)
    elif file_format.lower() == "parquet":
        return pl.read_parquet(file_path)
    elif file_format.lower() == "json":
        return pl.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def load_local_dataset(
    file_path: str | Path,
    columns: list[str] | None = None,
    sample_frac: float | None = None,
    random_seed: int = 42,
) -> pl.DataFrame:
    """Load a dataset from a file with optional column selection and sampling.

    Args:
        file_path: Path to the data file
        columns: List of columns to select (None for all columns)
        sample_frac: Fraction of rows to sample (None for all rows)
        random_seed: Random seed for reproducible sampling

    Returns:
        Polars DataFrame with the loaded data
    """
    file_path = Path(file_path)
    file_format = file_path.suffix.lstrip(".")

    # Load the data based on file format
    if file_format.lower() == "csv":
        df = pl.read_csv(file_path)
    elif file_format.lower() == "parquet":
        df = pl.read_parquet(file_path)
    elif file_format.lower() == "json":
        df = pl.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    # Select columns if specified
    if columns is not None:
        df = df.select(columns)

    # Sample if specified
    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("sample_frac must be between 0 and 1")
        df = df.sample(fraction=sample_frac, seed=random_seed)

    return df


def save_data(
    df: pl.DataFrame, output_path: str | Path, file_format: str | None = None
) -> Path:
    """Save the DataFrame to a file.

    Args:
        df: DataFrame to save
        output_path: Path where the file will be saved
        file_format: Format to save as (csv, parquet etc.). If None
            inferred from extension.

    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format is None:
        file_format = output_path.suffix.lstrip(".")

    logger.info(f"Saving data to {output_path} with format {file_format}")

    if file_format.lower() == "csv":
        df.write_csv(output_path)
    elif file_format.lower() == "parquet":
        df.write_parquet(output_path)
    elif file_format.lower() == "json":
        df.write_json(output_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    return output_path
