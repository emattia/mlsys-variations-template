"""Tests for data validation module.

This module tests the functions in src.data.validation.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.data.validation import (
    check_duplicates,
    check_missing_values,
    check_value_ranges,
    generate_data_quality_report,
    validate_categorical_values,
    validate_schema,
)


class TestCheckMissingValues:
    """Test suite for check_missing_values function."""

    def test_check_missing_values_no_missing(self):
        """Test with no missing values."""
        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            }
        )
        result = check_missing_values(df)
        assert result["a"] == 0.0
        assert result["b"] == 0.0

    def test_check_missing_values_with_missing(self):
        """Test with missing values."""
        df = pl.DataFrame(
            {
                "a": [1, None, 3],
                "b": ["x", "y", None],
            }
        )
        result = check_missing_values(df)
        assert result["a"] == pytest.approx(33.33, abs=0.1)
        assert result["b"] == pytest.approx(33.33, abs=0.1)


class TestValidateSchema:
    """Test suite for validate_schema function."""

    def test_validate_schema_valid(self):
        """Test with valid schema."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["a", "b", "c"],
            }
        )
        expected_schema = {"id": pl.Int64, "name": pl.Utf8}
        is_valid, errors = validate_schema(df, expected_schema)
        assert is_valid is True
        assert errors == []

    def test_validate_schema_missing_column(self):
        """Test with missing column."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
            }
        )
        expected_schema = {"id": pl.Int64, "name": pl.Utf8}
        is_valid, errors = validate_schema(df, expected_schema)
        assert is_valid is False
        assert len(errors) > 0


class TestCheckDuplicates:
    """Test suite for check_duplicates function."""

    def test_check_duplicates_no_duplicates(self):
        """Test with no duplicates."""
        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            }
        )
        count, percentage = check_duplicates(df)
        assert count >= 0
        assert percentage >= 0.0


class TestCheckValueRanges:
    """Test suite for check_value_ranges function."""

    def test_check_value_ranges_basic(self):
        """Test basic value range checking."""
        df = pl.DataFrame(
            {
                "age": [25, 30, 35],
            }
        )
        ranges = {
            "age": (20, 40),
        }
        result = check_value_ranges(df, ranges)
        # Should work without errors
        assert isinstance(result, dict)


class TestValidateCategoricalValues:
    """Test suite for validate_categorical_values function."""

    def test_validate_categorical_values_basic(self):
        """Test basic categorical validation."""
        df = pl.DataFrame(
            {
                "category": ["A", "B", "A"],
            }
        )
        valid_values = {
            "category": {"A", "B", "C"},
        }
        result = validate_categorical_values(df, valid_values)
        assert isinstance(result, dict)


class TestGenerateDataQualityReport:
    """Test suite for generate_data_quality_report function."""

    def test_generate_data_quality_report_basic(self):
        """Test basic data quality report generation."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
            }
        )
        report = generate_data_quality_report(df)

        assert report["row_count"] == 3
        assert report["column_count"] == 2
        assert "missing_values" in report
