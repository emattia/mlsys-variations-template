"""Data processing functions.

This module contains functions for cleaning, transforming, and processing data.
"""

import logging
from pathlib import Path
from typing import Any

import polars as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data.loading import load_data, save_data

logger = logging.getLogger(__name__)


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """Clean the data by handling missing values and outliers.

    Args:
        df: Input DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data")

    # Example cleaning operations
    # 1. Drop rows with any null values
    df_clean = df.drop_nulls()

    # 2. Log the number of rows dropped
    logger.info(f"Dropped {len(df) - len(df_clean)} rows with null values")

    return df_clean


def transform_data(df: pl.DataFrame) -> pl.DataFrame:
    """Transform the data by creating new features, normalizing, etc.

    Args:
        df: Input DataFrame

    Returns:
        Transformed DataFrame
    """
    logger.info("Transforming data")

    # This is a placeholder for actual transformations
    # In a real project, you would implement specific transformations here

    return df


def get_numeric_columns(df: pl.DataFrame) -> list[str]:
    """Get a list of numeric column names from a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        List of column names with numeric data types
    """
    numeric_dtypes = [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]
    return [col for col in df.columns if df.schema[col] in numeric_dtypes]


def get_categorical_columns(df: pl.DataFrame) -> list[str]:
    """Get a list of categorical column names from a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        List of column names with categorical data types
    """
    categorical_dtypes = [pl.Categorical, pl.Utf8, pl.Boolean]
    return [col for col in df.columns if df.schema[col] in categorical_dtypes]


def normalize_features(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    method: str = "standard",
    return_scaler: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, Any]:
    """Normalize numeric features in a DataFrame.

    Args:
        df: Input DataFrame
        columns: List of columns to normalize (None for all numeric columns)
        method: Normalization method ('standard' or 'minmax')
        return_scaler: Whether to return the scaler object

    Returns:
        Normalized DataFrame, and optionally the scaler object
    """
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = get_numeric_columns(df)

    # Return unchanged DataFrame if no columns to normalize
    if not columns:
        if return_scaler:
            return df, None
        else:
            return df

    # Convert to numpy for sklearn
    features = df.select(columns).to_numpy()

    # Apply normalization
    if method.lower() == "standard":
        scaler = StandardScaler()
    elif method.lower() == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    normalized = scaler.fit_transform(features)

    # Create a new DataFrame with normalized values
    normalized_df = df.clone()
    for i, col in enumerate(columns):
        normalized_df = normalized_df.with_columns(
            pl.Series(name=col, values=normalized[:, i])
        )

    if return_scaler:
        return normalized_df, scaler
    else:
        return normalized_df


def encode_categorical(
    df: pl.DataFrame, columns: list[str] | None = None, method: str = "one_hot"
) -> pl.DataFrame:
    """Encode categorical variables in a DataFrame.

    Args:
        df: Input DataFrame
        columns: List of columns to encode (None for all categorical columns)
        method: Encoding method ('one_hot' or 'label')

    Returns:
        DataFrame with encoded categorical variables
    """
    # If no columns specified, use all categorical columns
    if columns is None:
        columns = get_categorical_columns(df)

    result_df = df.clone()

    if method.lower() == "one_hot":
        # One-hot encode each categorical column
        for col in columns:
            # Get unique values for this column
            unique_values = result_df[col].unique().to_list()

            # Create dummy columns for each unique value
            for value in unique_values:
                dummy_col_name = f"{col}_{value}"
                result_df = result_df.with_columns(
                    (pl.col(col) == value).cast(pl.Int32).alias(dummy_col_name)
                )

        # Drop original categorical columns
        result_df = result_df.drop(columns)

    elif method.lower() == "label":
        # Label encode each categorical column
        for col in columns:
            unique_values = result_df[col].unique().to_list()
            # Create mapping dictionary
            mapping = {val: i for i, val in enumerate(unique_values)}
            result_df = result_df.with_columns(
                pl.col(col)
                .map_elements(
                    lambda x, mapping=mapping: mapping[x], return_dtype=pl.Int32
                )
                .alias(f"{col}_encoded")
            )

        # Drop original categorical columns
        result_df = result_df.drop(columns)

    else:
        raise ValueError(f"Unsupported encoding method: {method}")

    return result_df


def process_data(
    input_path: str | Path,
    output_path: str | Path,
    clean: bool = True,
    transform: bool = True,
) -> dict[str, Any]:
    """Process data from input path to output path.

    This function orchestrates the entire data processing workflow.

    Args:
        input_path: Path to the input data file
        output_path: Path where the processed data will be saved
        clean: Whether to clean the data
        transform: Whether to transform the data

    Returns:
        Dictionary with information about the processing
    """
    logger.info(f"Starting data processing workflow: {input_path} -> {output_path}")

    # Load data
    df = load_data(input_path)
    rows_before = len(df)

    # Clean data if requested
    if clean:
        df = clean_data(df)

    # Transform data if requested
    if transform:
        df = transform_data(df)

    # Save processed data
    __output_file = save_data(df, output_path)

    # Return information about the processing
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_before": rows_before,
        "rows_after": len(df),
        "cleaning_applied": clean,
        "transformation_applied": transform,
        "success": True,
    }
