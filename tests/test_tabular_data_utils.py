"""Comprehensive tests for tabular_data_utils module.
This module tests all functions in src.tabular_data_utils with focus on:
- Data loading from different formats
- Feature engineering utilities
- Data transformation functions
- Edge cases and error conditions
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl
import pytest
from sklearn.preprocessing import StandardScaler
import tempfile
from src.tabular_data_utils import (
    encode_categorical,
    get_categorical_columns,
    get_numeric_columns,
    load_local_dataset,
    normalize_features,
)


class TestLoadLocalDataset:
    """Test suite for load_local_dataset function."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample data for testing."""
        return pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "numeric_col": [1.0, 2.0, 3.0, 4.0, 5.0],
                "category_col": ["A", "B", "A", "C", "B"],
                "target": [0, 1, 0, 1, 0],
            }
        )

    @pytest.fixture
    def csv_file(self, sample_data: pl.DataFrame) -> str:
        """Create temporary CSV file."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            sample_data.write_csv(f.name)
            return f.name

    @pytest.fixture
    def parquet_file(self, sample_data: pl.DataFrame) -> str:
        """Create temporary Parquet file."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            sample_data.write_parquet(f.name)
            return f.name

    @pytest.fixture
    def json_file(self, sample_data: pl.DataFrame) -> str:
        """Create temporary JSON file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            sample_data.write_json(f.name)
            return f.name

    @pytest.mark.parametrize("file_fixture", ["csv_file", "parquet_file", "json_file"])
    def test_load_local_dataset_formats(self, request, file_fixture: str):
        """Test loading different file formats."""
        file_path = request.getfixturevalue(file_fixture)
        df = load_local_dataset(file_path)
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 5
        assert df.shape[1] == 4
        assert "id" in df.columns
        assert "numeric_col" in df.columns

    def test_load_local_dataset_with_columns(self, csv_file: str):
        """Test loading with specific columns."""
        columns = ["id", "numeric_col"]
        df = load_local_dataset(csv_file, columns=columns)
        assert df.shape[1] == 2
        assert list(df.columns) == columns

    @pytest.mark.parametrize(
        "sample_frac,expected_min,expected_max",
        [
            (0.2, 1, 1),  # 20% of 5 rows = 1 row
            (0.5, 2, 3),  # 50% of 5 rows = 2-3 rows
            (1.0, 5, 5),  # 100% of 5 rows = 5 rows
        ],
    )
    def test_load_local_dataset_sampling(
        self, csv_file: str, sample_frac: float, expected_min: int, expected_max: int
    ):
        """Test data sampling functionality."""
        df = load_local_dataset(csv_file, sample_frac=sample_frac, random_seed=42)
        assert expected_min <= df.shape[0] <= expected_max

    def test_load_local_dataset_invalid_sample_frac(self, csv_file: str):
        """Test invalid sample_frac values."""
        with pytest.raises(ValueError, match="sample_frac must be between 0 and 1"):
            load_local_dataset(csv_file, sample_frac=0.0)
        with pytest.raises(ValueError, match="sample_frac must be between 0 and 1"):
            load_local_dataset(csv_file, sample_frac=1.5)

    def test_load_local_dataset_unsupported_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx") as f:
            with pytest.raises(ValueError, match="Unsupported file format: xlsx"):
                load_local_dataset(f.name)

    def test_load_local_dataset_path_object(self, csv_file: str):
        """Test loading with Path object."""
        path_obj = Path(csv_file)
        df = load_local_dataset(path_obj)
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 5


class TestColumnDetection:
    """Test suite for column type detection functions."""

    @pytest.fixture
    def mixed_data(self) -> pl.DataFrame:
        """Create DataFrame with mixed column types."""
        return pl.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "string_col": ["A", "B", "C", "D", "E"],
                "bool_col": [True, False, True, False, True],
                "categorical_col": pl.Series(
                    ["X", "Y", "X", "Z", "Y"], dtype=pl.Categorical
                ),
            }
        )

    def test_get_numeric_columns(self, mixed_data: pl.DataFrame):
        """Test numeric column detection."""
        numeric_cols = get_numeric_columns(mixed_data)
        assert "int_col" in numeric_cols
        assert "float_col" in numeric_cols
        assert "string_col" not in numeric_cols
        assert "bool_col" not in numeric_cols
        assert "categorical_col" not in numeric_cols

    def test_get_categorical_columns(self, mixed_data: pl.DataFrame):
        """Test categorical column detection."""
        categorical_cols = get_categorical_columns(mixed_data)
        assert "string_col" in categorical_cols
        assert "bool_col" in categorical_cols
        assert "categorical_col" in categorical_cols
        assert "int_col" not in categorical_cols
        assert "float_col" not in categorical_cols

    def test_empty_dataframe(self):
        """Test column detection with empty DataFrame."""
        empty_df = pl.DataFrame()
        assert get_numeric_columns(empty_df) == []
        assert get_categorical_columns(empty_df) == []


class TestNormalizeFeatures:
    """Test suite for normalize_features function."""

    @pytest.fixture
    def numeric_data(self) -> pl.DataFrame:
        """Create DataFrame with numeric data."""
        return pl.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "category": ["A", "B", "C", "D", "E"],
            }
        )

    @pytest.mark.parametrize("method", ["standard", "minmax"])
    def test_normalize_features_methods(self, numeric_data: pl.DataFrame, method: str):
        """Test different normalization methods."""
        result = normalize_features(numeric_data, method=method)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == numeric_data.shape

        # Check that numeric columns are normalized
        normalized_col = result["feature1"].to_numpy()
        if method == "standard":
            assert abs(np.mean(normalized_col)) < 1e-10  # Mean should be ~0
            assert abs(np.std(normalized_col) - 1.0) < 1e-10  # Std should be ~1
        elif method == "minmax":
            assert np.min(normalized_col) >= 0.0  # Min should be 0
            assert np.max(normalized_col) <= 1.0  # Max should be 1

    def test_normalize_features_specific_columns(self, numeric_data: pl.DataFrame):
        """Test normalization with specific columns."""
        columns = ["feature1"]
        result = normalize_features(numeric_data, columns=columns)

        # feature1 should be normalized, feature2 should remain unchanged
        assert not np.array_equal(
            result["feature1"].to_numpy(), numeric_data["feature1"].to_numpy()
        )
        assert np.array_equal(
            result["feature2"].to_numpy(), numeric_data["feature2"].to_numpy()
        )

    def test_normalize_features_return_scaler(self, numeric_data: pl.DataFrame):
        """Test returning scaler object."""
        result, scaler = normalize_features(
            numeric_data, method="standard", return_scaler=True
        )
        assert isinstance(result, pl.DataFrame)
        assert isinstance(scaler, StandardScaler)

    def test_normalize_features_invalid_method(self, numeric_data: pl.DataFrame):
        """Test invalid normalization method."""
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            normalize_features(numeric_data, method="invalid")


class TestEncodeCategorical:
    """Test suite for encode_categorical function."""

    @pytest.fixture
    def categorical_data(self) -> pl.DataFrame:
        """Create DataFrame with categorical data."""
        return pl.DataFrame(
            {
                "category1": ["A", "B", "A", "C", "B"],
                "category2": ["X", "Y", "X", "X", "Z"],
                "numeric": [1, 2, 3, 4, 5],
            }
        )

    def test_encode_categorical_one_hot(self, categorical_data: pl.DataFrame):
        """Test one-hot encoding."""
        result = encode_categorical(
            categorical_data, columns=["category1"], method="one_hot"
        )

        # Should have dummy columns for each category value
        expected_cols = {
            "numeric",
            "category2",
            "category1_A",
            "category1_B",
            "category1_C",
        }
        assert set(result.columns) == expected_cols

        # Check dummy values for first category
        assert result["category1_A"].to_list() == [1, 0, 1, 0, 0]
        assert result["category1_B"].to_list() == [0, 1, 0, 0, 1]
        assert result["category1_C"].to_list() == [0, 0, 0, 1, 0]

    def test_encode_categorical_label(self, categorical_data: pl.DataFrame):
        """Test label encoding."""
        result = encode_categorical(
            categorical_data, columns=["category1"], method="label"
        )

        # Should replace categorical column with numeric labels - with _encoded suffix
        assert "category1_encoded" in result.columns
        assert "category1" not in result.columns  # Original should be dropped
        assert result["category1_encoded"].dtype in [
            pl.Int32,
            pl.Int64,
            pl.UInt32,
            pl.UInt64,
        ]

        # Check that unique values are encoded
        unique_values = result["category1_encoded"].unique().sort()
        assert len(unique_values) == 3  # A, B, C

    def test_encode_categorical_multiple_columns(self, categorical_data: pl.DataFrame):
        """Test encoding multiple columns."""
        result = encode_categorical(
            categorical_data, columns=["category1", "category2"], method="one_hot"
        )

        # Should have dummy columns for both categorical columns
        category1_dummies = {"category1_A", "category1_B", "category1_C"}
        category2_dummies = {"category2_X", "category2_Y", "category2_Z"}
        expected_cols = {"numeric"} | category1_dummies | category2_dummies
        assert set(result.columns) == expected_cols

    def test_encode_categorical_invalid_method(self, categorical_data: pl.DataFrame):
        """Test invalid encoding method."""
        with pytest.raises(ValueError, match="Unsupported encoding method"):
            encode_categorical(categorical_data, method="invalid")

    def test_encode_categorical_no_columns_specified(
        self, categorical_data: pl.DataFrame
    ):
        """Test encoding when no columns specified."""
        # Should auto-detect categorical columns
        result = encode_categorical(categorical_data, method="label")

        # category1 and category2 should be encoded (numeric column unchanged)
        assert "numeric" in result.columns
        assert "category1_encoded" in result.columns
        assert "category2_encoded" in result.columns
        assert "category1" not in result.columns  # Original should be dropped
        assert "category2" not in result.columns  # Original should be dropped

        # Categorical columns should now be numeric
        assert result["category1_encoded"].dtype in [
            pl.Int32,
            pl.Int64,
            pl.UInt32,
            pl.UInt64,
        ]
        assert result["category2_encoded"].dtype in [
            pl.Int32,
            pl.Int64,
            pl.UInt32,
            pl.UInt64,
        ]
