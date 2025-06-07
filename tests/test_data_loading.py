"""Tests for data loading module.

This module tests the functions in src.data.loading.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

from src.data.loading import (
    load_data,
    load_local_dataset,
    save_data,
)


class TestLoadData:
    """Test suite for load_data function."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample data for testing."""
        return pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                "score": [85.5, 92.0, 78.5, 96.0, 89.5],
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

    def test_load_csv(self, csv_file: str, sample_data: pl.DataFrame):
        """Test loading CSV file."""
        df = load_data(csv_file)
        assert isinstance(df, pl.DataFrame)
        assert df.shape == sample_data.shape

    def test_load_parquet(self, parquet_file: str, sample_data: pl.DataFrame):
        """Test loading Parquet file."""
        df = load_data(parquet_file)
        assert isinstance(df, pl.DataFrame)
        assert df.shape == sample_data.shape

    def test_load_json(self, json_file: str, sample_data: pl.DataFrame):
        """Test loading JSON file."""
        df = load_data(json_file)
        assert isinstance(df, pl.DataFrame)
        assert df.shape == sample_data.shape

    def test_load_data_explicit_format(self, csv_file: str):
        """Test loading with explicit format."""
        df = load_data(csv_file, file_format="csv")
        assert isinstance(df, pl.DataFrame)

    def test_load_data_unsupported_format(self):
        """Test loading unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx") as f:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_data(f.name)

    def test_load_data_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(Exception):  # Could be FileNotFoundError or other
            load_data("nonexistent.csv")


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

    def test_load_local_dataset_basic(self, csv_file: str):
        """Test basic loading."""
        df = load_local_dataset(csv_file)
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 5

    def test_load_local_dataset_with_columns(self, csv_file: str):
        """Test loading with specific columns."""
        columns = ["id", "numeric_col"]
        df = load_local_dataset(csv_file, columns=columns)
        assert df.shape[1] == 2
        assert list(df.columns) == columns

    def test_load_local_dataset_with_sampling(self, csv_file: str):
        """Test loading with sampling."""
        df = load_local_dataset(csv_file, sample_frac=0.6, random_seed=42)
        assert df.shape[0] <= 5  # Should be <= original

    def test_load_local_dataset_invalid_sample_frac(self, csv_file: str):
        """Test invalid sample_frac."""
        with pytest.raises(ValueError, match="sample_frac must be between 0 and 1"):
            load_local_dataset(csv_file, sample_frac=0.0)


class TestSaveData:
    """Test suite for save_data function."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample data for testing."""
        return pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                "score": [85.5, 92.0, 78.5, 96.0, 89.5],
            }
        )

    def test_save_csv(self, sample_data: pl.DataFrame):
        """Test saving to CSV."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            result_path = save_data(sample_data, f.name)
            assert result_path == Path(f.name)
            assert Path(f.name).exists()

    def test_save_parquet(self, sample_data: pl.DataFrame):
        """Test saving to Parquet."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            result_path = save_data(sample_data, f.name)
            assert result_path == Path(f.name)
            assert Path(f.name).exists()

    def test_save_json(self, sample_data: pl.DataFrame):
        """Test saving to JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result_path = save_data(sample_data, f.name)
            assert result_path == Path(f.name)
            assert Path(f.name).exists()

    def test_save_explicit_format(self, sample_data: pl.DataFrame):
        """Test saving with explicit format."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            result_path = save_data(sample_data, f.name, file_format="csv")
            assert result_path == Path(f.name)

    def test_save_unsupported_format(self, sample_data: pl.DataFrame):
        """Test saving unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx") as f:
            with pytest.raises(ValueError, match="Unsupported file format"):
                save_data(sample_data, f.name)

    def test_round_trip(self, sample_data: pl.DataFrame):
        """Test saving and loading maintains data integrity."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            # Save data
            save_data(sample_data, f.name)

            # Load it back
            loaded_data = load_data(f.name)

            # Verify integrity
            assert loaded_data.shape == sample_data.shape
            assert list(loaded_data.columns) == list(sample_data.columns)
