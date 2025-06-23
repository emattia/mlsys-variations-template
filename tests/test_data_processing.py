"""Test cases for src/data/processing.py.
This module contains comprehensive tests for the data processing functions.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.processing import (
    clean_data,
    encode_categorical,
    get_categorical_columns,
    get_numeric_columns,
    normalize_features,
    process_data,
    transform_data,
)


def test_clean_data_basic():
    """Test basic cleaning of data with null values."""
    df = pl.DataFrame(
        {"A": [1, 2, None, 4], "B": [1.0, None, 3.0, 4.0], "C": ["a", "b", "c", None]}
    )
    result = clean_data(df)

    # Policy decision: only rows with no null values
    assert len(result) == 1
    assert result["A"][0] == 1
    assert result["B"][0] == 1.0
    assert result["C"][0] == "a"


def test_get_numeric_columns():
    """Test getting numeric columns from DataFrame."""
    df = pl.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
    )
    result = get_numeric_columns(df)
    assert set(result) == {"int_col", "float_col"}


def test_get_categorical_columns():
    """Test getting categorical columns from DataFrame."""
    df = pl.DataFrame(
        {
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
            "cat_col": pl.Series(["x", "y", "z"], dtype=pl.Categorical),
        }
    )
    result = get_categorical_columns(df)
    assert set(result) == {"str_col", "bool_col", "cat_col"}


def test_normalize_features_standard():
    """Test standard normalization."""
    df = pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [10.0, 20.0, 30.0, 40.0, 50.0],
            "C": ["a", "b", "c", "d", "e"],
        }
    )
    result = normalize_features(df, method="standard")

    # Check that numeric columns are normalized
    a_values = result["A"].to_numpy()
    b_values = result["B"].to_numpy()

    # Standard normalization should give mean ≈ 0, std ≈ 1
    assert abs(np.mean(a_values)) < 1e-10
    assert abs(np.std(a_values, ddof=0) - 1.0) < 1e-10
    assert abs(np.mean(b_values)) < 1e-10
    assert abs(np.std(b_values, ddof=0) - 1.0) < 1e-10


def test_encode_categorical_one_hot():
    """Test one-hot encoding."""
    df = pl.DataFrame(
        {"category": ["A", "B", "A", "C", "B"], "numeric": [1, 2, 3, 4, 5]}
    )
    result = encode_categorical(df, columns=["category"], method="one_hot")

    # Should have dummy columns for each category
    expected_cols = {"numeric", "category_A", "category_B", "category_C"}
    assert set(result.columns) == expected_cols

    # Check dummy values
    assert result["category_A"].to_list() == [1, 0, 1, 0, 0]
    assert result["category_B"].to_list() == [0, 1, 0, 0, 1]
    assert result["category_C"].to_list() == [0, 0, 0, 1, 0]


def test_process_data_workflow():
    """Test complete data processing workflow."""
    test_df = pl.DataFrame(
        {"A": [1, 2, None, 4], "B": [1.0, 2.0, 3.0, None], "C": ["x", "y", "z", "w"]}
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input.csv"
        output_path = Path(temp_dir) / "output.csv"

        # Save test data
        test_df.write_csv(input_path)

        # Process data
        result = process_data(
            input_path=input_path, output_path=output_path, clean=True, transform=True
        )

        # Check result metadata
        assert result["input_path"] == str(input_path)
        assert result["output_path"] == str(output_path)
        assert result["rows_before"] == 4
        assert result["rows_after"] == 2  # 2 rows dropped due to nulls
        assert result["cleaning_applied"] is True
        assert result["transformation_applied"] is True
        assert result["success"] is True

        # Check that output file exists
        assert output_path.exists()


class TestCleanData:
    """Test cases for clean_data function."""

    def test_clean_data_no_nulls(self):
        """Test cleaning data with no null values."""
        df = pl.DataFrame(
            {"A": [1, 2, 3, 4], "B": [1.0, 2.0, 3.0, 4.0], "C": ["a", "b", "c", "d"]}
        )
        result = clean_data(df)

        # Should keep all rows
        assert len(result) == 4
        assert result.equals(df)

    def test_clean_data_all_nulls(self):
        """Test cleaning data where all rows have null values."""
        df = pl.DataFrame({"A": [None, None, None], "B": [None, None, None]})
        result = clean_data(df)

        # Should return empty DataFrame
        assert len(result) == 0

    def test_clean_data_empty_dataframe(self):
        """Test cleaning empty DataFrame."""
        df = pl.DataFrame()
        result = clean_data(df)
        assert len(result) == 0
        assert result.columns == []

    @patch("src.data.processing.logger")
    def test_clean_data_logging(self, mock_logger):
        """Test that cleaning logs appropriate messages."""
        df = pl.DataFrame({"A": [1, 2, None, 4], "B": [1.0, None, 3.0, 4.0]})
        clean_data(df)

        # Check logging calls
        mock_logger.info.assert_any_call("Cleaning data")
        mock_logger.info.assert_any_call("Dropped 2 rows with null values")


class TestTransformData:
    """Test cases for transform_data function."""

    def test_transform_data_basic(self):
        """Test basic transformation (placeholder)."""
        df = pl.DataFrame({"A": [1, 2, 3, 4], "B": [1.0, 2.0, 3.0, 4.0]})
        result = transform_data(df)

        # Currently just returns the same DataFrame
        assert result.equals(df)

    @patch("src.data.processing.logger")
    def test_transform_data_logging(self, mock_logger):
        """Test that transformation logs appropriate messages."""
        df = pl.DataFrame({"A": [1, 2, 3]})
        transform_data(df)
        mock_logger.info.assert_called_with("Transforming data")


class TestGetNumericColumns:
    """Test cases for get_numeric_columns function."""

    def test_get_numeric_columns_mixed_types(self):
        """Test getting numeric columns from mixed-type DataFrame."""
        df = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.0, 2.0, 3.0],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )
        result = get_numeric_columns(df)
        assert set(result) == {"int_col", "float_col"}

    def test_get_numeric_columns_all_numeric(self):
        """Test getting numeric columns from all-numeric DataFrame."""
        df = pl.DataFrame(
            {
                "int32": pl.Series([1, 2, 3], dtype=pl.Int32),
                "int64": pl.Series([1, 2, 3], dtype=pl.Int64),
                "float32": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
                "float64": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
                "uint32": pl.Series([1, 2, 3], dtype=pl.UInt32),
                "uint64": pl.Series([1, 2, 3], dtype=pl.UInt64),
            }
        )
        result = get_numeric_columns(df)
        assert len(result) == 6
        assert set(result) == set(df.columns)

    def test_get_numeric_columns_no_numeric(self):
        """Test getting numeric columns from non-numeric DataFrame."""
        df = pl.DataFrame(
            {
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "cat_col": pl.Series(["x", "y", "z"], dtype=pl.Categorical),
            }
        )
        result = get_numeric_columns(df)
        assert result == []

    def test_get_numeric_columns_empty_dataframe(self):
        """Test getting numeric columns from empty DataFrame."""
        df = pl.DataFrame()
        result = get_numeric_columns(df)
        assert result == []


class TestGetCategoricalColumns:
    """Test cases for get_categorical_columns function."""

    def test_get_categorical_columns_mixed_types(self):
        """Test getting categorical columns from mixed-type DataFrame."""
        df = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "cat_col": pl.Series(["x", "y", "z"], dtype=pl.Categorical),
            }
        )
        result = get_categorical_columns(df)
        assert set(result) == {"str_col", "bool_col", "cat_col"}

    def test_get_categorical_columns_no_categorical(self):
        """Test getting categorical columns from non-categorical DataFrame."""
        df = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.0, 2.0, 3.0],
            }
        )
        result = get_categorical_columns(df)
        assert result == []

    def test_get_categorical_columns_all_categorical(self):
        """Test getting categorical columns from all-categorical DataFrame."""
        df = pl.DataFrame(
            {
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "cat_col": pl.Series(["x", "y", "z"], dtype=pl.Categorical),
            }
        )
        result = get_categorical_columns(df)
        assert set(result) == set(df.columns)

    def test_get_categorical_columns_empty_dataframe(self):
        """Test getting categorical columns from empty DataFrame."""
        df = pl.DataFrame()
        result = get_categorical_columns(df)
        assert result == []


class TestNormalizeFeatures:
    """Test cases for normalize_features function."""

    def test_normalize_features_minmax(self):
        """Test min-max normalization."""
        df = pl.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0, 5.0],
                "B": [10.0, 20.0, 30.0, 40.0, 50.0],
                "C": ["a", "b", "c", "d", "e"],
            }
        )
        result = normalize_features(df, method="minmax")

        # Check that numeric columns are normalized to [0, 1]
        a_values = result["A"].to_numpy()
        b_values = result["B"].to_numpy()

        assert np.min(a_values) == 0.0
        assert np.max(a_values) == 1.0
        assert np.min(b_values) == 0.0
        assert np.max(b_values) == 1.0

    def test_normalize_features_specific_columns(self):
        """Test normalization with specific columns."""
        df = pl.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0, 5.0],
                "B": [10.0, 20.0, 30.0, 40.0, 50.0],
                "C": ["a", "b", "c", "d", "e"],
            }
        )
        result = normalize_features(df, columns=["A"], method="standard")

        # Only column A should be normalized
        assert not np.array_equal(result["A"].to_numpy(), df["A"].to_numpy())
        assert np.array_equal(result["B"].to_numpy(), df["B"].to_numpy())

    def test_normalize_features_invalid_method(self):
        """Test invalid normalization method."""
        df = pl.DataFrame({"A": [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            normalize_features(df, method="invalid_method")

    def test_normalize_features_return_scaler(self):
        """Test returning scaler object."""
        df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result, scaler = normalize_features(df, method="standard", return_scaler=True)
        assert isinstance(result, pl.DataFrame)
        assert isinstance(scaler, StandardScaler)

    def test_normalize_features_no_numeric_columns(self):
        """Test normalization with no numeric columns."""
        df = pl.DataFrame({"A": ["a", "b", "c"], "B": [True, False, True]})
        result = normalize_features(df)
        # Should return original DataFrame unchanged
        assert result.equals(df)


class TestEncodeCategorical:
    """Test cases for encode_categorical function."""

    def test_encode_categorical_label(self):
        """Test label encoding."""
        df = pl.DataFrame(
            {"category": ["A", "B", "A", "C", "B"], "numeric": [1, 2, 3, 4, 5]}
        )
        result = encode_categorical(df, columns=["category"], method="label")

        # Should replace categorical column with numeric labels - with _encoded suffix
        assert "category_encoded" in result.columns
        assert "category" not in result.columns  # Original should be dropped
        assert result["category_encoded"].dtype in [
            pl.Int32,
            pl.Int64,
            pl.UInt32,
            pl.UInt64,
        ]

        # Check that unique values are encoded
        unique_values = result["category_encoded"].unique().sort()
        assert len(unique_values) == 3  # A, B, C

    def test_encode_categorical_auto_detect(self):
        """Test automatic categorical column detection."""
        df = pl.DataFrame(
            {
                "cat1": ["A", "B", "A"],
                "cat2": ["X", "Y", "X"],
                "numeric": [1, 2, 3],
            }
        )
        result = encode_categorical(df, method="label")

        # Should auto-detect and encode categorical columns
        assert "cat1_encoded" in result.columns
        assert "cat2_encoded" in result.columns
        assert "cat1" not in result.columns  # Original should be dropped
        assert "cat2" not in result.columns  # Original should be dropped
        assert "numeric" in result.columns  # Numeric should remain

        # Categorical columns should now be numeric
        assert result["cat1_encoded"].dtype in [
            pl.Int32,
            pl.Int64,
            pl.UInt32,
            pl.UInt64,
        ]
        assert result["cat2_encoded"].dtype in [
            pl.Int32,
            pl.Int64,
            pl.UInt32,
            pl.UInt64,
        ]

    def test_encode_categorical_invalid_method(self):
        """Test invalid encoding method."""
        df = pl.DataFrame({"category": ["A", "B", "C"]})
        with pytest.raises(ValueError, match="Unsupported encoding method"):
            encode_categorical(df, method="invalid_method")

    def test_encode_categorical_no_categorical_columns(self):
        """Test encoding with no categorical columns."""
        df = pl.DataFrame({"A": [1, 2, 3], "B": [1.0, 2.0, 3.0]})
        result = encode_categorical(df)
        # Should return original DataFrame unchanged
        assert result.equals(df)
