"""Data validation functions.

This module contains functions for validating data quality and integrity.
"""

import logging
from typing import Any

import polars as pl

__logger = logging.getLogger(__name__)


def validate_schema(
    df: pl.DataFrame, expected_schema: dict[str, type]
) -> tuple[bool, list[str]]:
    """Validate that a DataFrame has the expected schema.

    Args:
        df: DataFrame to validate
        expected_schema: Dictionary mapping column names to expected types

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Check for missing columns
    missing_columns = set(expected_schema.keys()) - set(df.columns)
    if missing_columns:
        errors.append(f"Missing columns: {', '.join(missing_columns)}")

    # Check column types
    for col_name, expected_type in expected_schema.items():
        if col_name in df.columns:
            actual_type = df.schema[col_name]
            # This is a simplified type check - in practice,
            # you might need more nuanced type checking
            if not isinstance(actual_type, expected_type):
                errors.append(
                    f"'{col_name}' has type {actual_type}, expected {expected_type}"
                )

    return len(errors) == 0, errors


def check_missing_values(df: pl.DataFrame) -> dict[str, float]:
    """Check for missing values in each column of a DataFrame.

    Args:
        df: DataFrame to check

    Returns:
        Dictionary mapping column names to percentage of missing values
    """
    result = {}
    for col in df.columns:
        null_count = df[col].null_count()
        percentage = (null_count / len(df)) * 100 if len(df) > 0 else 0
        result[col] = percentage

    return result


def check_duplicates(
    df: pl.DataFrame, subset: list[str] | None = None
) -> tuple[int, float]:
    """Check for duplicate rows in a DataFrame.

    Args:
        df: DataFrame to check
        subset: Optional list of columns to consider for duplicates

    Returns:
        Tuple of (number of duplicates, percentage of duplicates)
    """
    if len(df) == 0:
        return 0, 0.0

    if subset is None:
        # Consider all columns - count total rows minus unique rows
        unique_count = df.unique().height
        duplicate_count = len(df) - unique_count
    else:
        # Consider only the specified columns
        unique_count = df.select(subset).unique().height
        duplicate_count = len(df) - unique_count

    percentage = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0

    return duplicate_count, percentage


def check_value_ranges(
    df: pl.DataFrame, ranges: dict[str, tuple[float, float]]
) -> dict[str, list[dict[str, Any]]]:
    """Check if values in specified columns are within expected ranges.

    Args:
        df: DataFrame to check
        ranges: Dictionary mapping column names to (min, max) tuples

    Returns:
        Dict mapping cols to lists of out-of-range values with their counts
    """
    results: dict[str, list[dict[str, Any]]] = {}

    for col, (min_val, max_val) in ranges.items():
        if col not in df.columns:
            results[col] = [{"error": f"Column '{col}' not found in DataFrame"}]
            continue

        # Check for values below minimum
        below_min = df.filter(pl.col(col) < min_val)
        if len(below_min) > 0:
            results.setdefault(col, []).append(
                {
                    "type": "below_minimum",
                    "min_value": min_val,
                    "count": len(below_min),
                    "percentage": (len(below_min) / len(df)) * 100,
                }
            )

        # Check for values above maximum
        above_max = df.filter(pl.col(col) > max_val)
        if len(above_max) > 0:
            results.setdefault(col, []).append(
                {
                    "type": "above_maximum",
                    "max_value": max_val,
                    "count": len(above_max),
                    "percentage": (len(above_max) / len(df)) * 100,
                }
            )

    return results


def validate_categorical_values(
    df: pl.DataFrame, valid_values: dict[str, set[str]]
) -> dict[str, dict[str, Any]]:
    """Check if categorical columns contain only valid values.

    Args:
        df: DataFrame to check
        valid_values: Dictionary mapping column names to sets of valid values

    Returns:
        Dictionary with validation results for each column
    """
    results: dict[str, dict[str, Any]] = {}

    for col, allowed_values in valid_values.items():
        if col not in df.columns:
            results[col] = {"error": f"Column '{col}' not found in DataFrame"}
            continue

        # Get unique values in the column
        unique_values = set(df[col].unique().to_list())

        # Find invalid values
        invalid_values = unique_values - allowed_values

        if invalid_values:
            # Count rows with invalid values
            invalid_mask = df[col].is_in(list(invalid_values))
            invalid_count = df.filter(invalid_mask).height

            results[col] = {
                "invalid_values": list(invalid_values),
                "invalid_count": invalid_count,
                "invalid_percentage": (invalid_count / len(df)) * 100
                if len(df) > 0
                else 0,
            }
        else:
            results[col] = {"valid": True}

    return results


def generate_data_quality_report(df: pl.DataFrame) -> dict[str, Any]:
    """Generate a comprehensive data quality report for a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with data quality metrics
    """
    report: dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": check_missing_values(df),
        "duplicates": check_duplicates(df)[0],
        "column_stats": {},
    }

    # Generate statistics for each column
    for col in df.columns:
        col_type = str(df.schema[col])

        # Basic statistics that apply to all column types
        col_stats: dict[str, Any] = {
            "type": col_type,
            "missing_count": df[col].null_count(),
            "missing_percentage": (df[col].null_count() / len(df)) * 100
            if len(df) > 0
            else 0,
        }

        # Add type-specific statistics
        if "float" in col_type.lower() or "int" in col_type.lower():
            # Numeric column
            non_null = df.filter(~pl.col(col).is_null())
            if len(non_null) > 0:
                col_stats.update(
                    {
                        "min": float(non_null[col].min()),
                        "max": float(non_null[col].max()),
                        "mean": float(non_null[col].mean()),
                        "median": float(non_null[col].median()),
                        "std_dev": float(non_null[col].std()),
                    }
                )
        elif (
            "str" in col_type.lower()
            or "utf8" in col_type.lower()
            or "categorical" in col_type.lower()
        ):
            # String or categorical column
            non_null = df.filter(~pl.col(col).is_null())
            if len(non_null) > 0:
                unique_values = non_null[col].unique()
                col_stats.update(
                    {
                        "unique_count": len(unique_values),
                        "unique_percentage": (len(unique_values) / len(non_null)) * 100,
                    }
                )

        report["column_stats"][col] = col_stats

    return report
