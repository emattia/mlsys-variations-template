"""Example test file to demonstrate pytest usage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def test_project_structure() -> None:
    """Test that the project structure is set up correctly."""
    # Check that key directories exist
    assert Path("data").exists(), "Data directory not found"
    assert Path("src").exists(), "Source directory not found"
    assert Path("tests").exists(), "Tests directory not found"
    assert Path("notebooks").exists(), "Notebooks directory not found"
    assert Path("reports").exists(), "Reports directory not found"


@pytest.fixture
def sample_data() -> Any:
    """Fixture to provide sample data for tests."""
    return {"features": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]}


def test_data_processing(sample_data: dict[str, list[int]]) -> None:
    """Test a simple data processing function."""
    # This is a placeholder for an actual test of your data processing code
    features = sample_data["features"]
    target = sample_data["target"]

    # Example assertion
    assert len(features) == len(target), (
        "Features and target should have the same length"
    )
    assert sum(features) > 0, "Sum of features should be positive"


@pytest.mark.parametrize(
    "input_value,expected",
    [
        (2, 4),
        (0, 0),
        (-1, 1),
        (10, 100),
    ],
)
def test_parametrized_example(input_value: int, expected: int) -> None:
    """Demonstrate parametrized testing."""
    # This would typically test a function from your src module
    # For example: from src.utils import square
    # result = square(input_value)

    # For demonstration, we'll just square the input directly
    result = input_value**2
    assert result == expected, f"Square of {input_value} should be {expected}"
