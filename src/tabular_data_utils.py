"""Utility functions for data processing and analysis.

This module contains reusable functions for data manipulation,
feature engineering, and other common data science tasks.
"""

import logging
from pathlib import Path
from typing import Any

import polars as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler

__logger = logging.getLogger(__name__)


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


def split_train_test(
    df: pl.DataFrame, target_column: str, test_size: float = 0.2, random_seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    """Split a DataFrame into training and testing sets.
    A slightly custom, polars-specific version of the legend:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

    Args:
        df: Input DataFrame
        target_column: Name of the target column
        test_size: Fraction of data to use for testing
        random_seed: Random seed for reproducible splitting

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.sample.html
    df_shuffled = df.sample(fraction=1.0, seed=random_seed)

    # Calculate split index
    split_idx = int(len(df) * (1 - test_size))

    # Split into train and test
    train_df = df_shuffled.slice(0, split_idx)
    test_df = df_shuffled.slice(split_idx, len(df) - split_idx)

    # Extract features and target
    X_train = train_df.drop(target_column)
    y_train = train_df[target_column]
    X_test = test_df.drop(target_column)
    y_test = test_df[target_column]

    return X_train, X_test, y_train, y_test


def calculate_feature_importance(
    feature_names: list[str],
    importance_values: list[float],
    top_n: int | None = None,
) -> dict[str, float]:
    """Calculate and sort feature importance.

    Args:
        feature_names: List of feature names
        importance_values: List of importance values
        top_n: Number of top features to return (None for all)

    Returns:
        Dictionary mapping feature names to importance values, sorted by importance
    """
    if len(feature_names) != len(importance_values):
        raise ValueError("Length of feature_names and importance_values must match")

    # Create dictionary of feature importances
    importance_dict = dict(zip(feature_names, importance_values, strict=True))

    # Sort by importance (descending)
    sorted_importance = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    # Return top N if specified
    if top_n is not None:
        return dict(list(sorted_importance.items())[:top_n])
    else:
        return sorted_importance
