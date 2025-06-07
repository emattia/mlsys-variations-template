# Tests Directory

This directory contains the test suite for the project, ensuring code quality and correctness.

## Purpose

The tests directory serves to:

1. **Verify correctness**: Ensure code behaves as expected
2. **Prevent regressions**: Catch bugs before they affect users
3. **Document behavior**: Tests serve as executable documentation
4. **Enable refactoring**: Allow code improvements with confidence
5. **Support CI/CD**: Automate testing in continuous integration

## Structure

The test suite is organized to mirror the structure of the source code:

- **unit/**: Tests for individual functions and classes
- **integration/**: Tests for interactions between components
- **functional/**: Tests for end-to-end functionality
- **fixtures/**: Test data and fixtures
- **conftest.py**: Shared pytest fixtures and configuration

## Testing Framework

This project uses pytest as the primary testing framework:

- **pytest**: Main testing framework
- **pytest-cov**: Coverage reporting
- **pytest-xdist**: Parallel test execution
- **pytest-mock**: Mocking functionality

## Writing Tests

### Test File Naming

Test files should be named with the `test_` prefix followed by the name of the module being tested:

- `test_data_utils.py` for testing `src/data_utils.py`
- `test_model.py` for testing `src/models/model.py`

### Test Function Naming

Test functions should be named descriptively to indicate what they're testing:

- `test_function_name_expected_behavior`
- `test_class_method_condition_expected_behavior`

### Example Test

```python
"""Tests for the data_utils module."""

import pytest
import polars as pl
import numpy as np
from src.data_utils import normalize_features, get_numeric_columns

def test_get_numeric_columns_returns_correct_columns():
    """Test that get_numeric_columns correctly identifies numeric columns."""
    # Arrange
    data = {
        "numeric1": pl.Series([1, 2, 3], dtype=pl.Int64),
        "numeric2": pl.Series([1.1, 2.2, 3.3], dtype=pl.Float64),
        "string": pl.Series(["a", "b", "c"], dtype=pl.Utf8),
        "bool": pl.Series([True, False, True], dtype=pl.Boolean),
    }
    df = pl.DataFrame(data)

    # Act
    result = get_numeric_columns(df)

    # Assert
    assert set(result) == {"numeric1", "numeric2"}
    assert "string" not in result
    assert "bool" not in result

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pl.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "category": ["A", "B", "A", "B", "C"]
    })

def test_normalize_features_standard_method(sample_dataframe):
    """Test that normalize_features correctly standardizes numeric features."""
    # Arrange
    columns = ["feature1", "feature2"]

    # Act
    result = normalize_features(sample_dataframe, columns, method="standard")

    # Assert
    # Check that the result is a DataFrame
    assert isinstance(result, pl.DataFrame)

    # Check that the normalized columns have mean ≈ 0 and std ≈ 1
    for col in columns:
        values = result[col].to_numpy()
        assert np.isclose(np.mean(values), 0, atol=1e-10)
        assert np.isclose(np.std(values), 1, atol=1e-10)

    # Check that non-normalized columns are unchanged
    assert result["category"].to_list() == sample_dataframe["category"].to_list()
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=src
```

### Run Specific Tests

```bash
# Run tests in a specific file
pytest tests/test_data_utils.py

# Run a specific test
pytest tests/test_data_utils.py::test_normalize_features_standard_method

# Run tests matching a pattern
pytest -k "normalize"
```

### Run Tests in Parallel

```bash
pytest -xvs -n auto
```

## Test Coverage

Aim for high test coverage, but focus on critical paths:

1. **Core functionality**: Ensure all core functionality is well-tested
2. **Edge cases**: Test boundary conditions and error handling
3. **Complex logic**: Focus on complex or error-prone code

To generate a coverage report:

```bash
pytest --cov=src --cov-report=html
```

This will create an HTML coverage report in the `htmlcov/` directory.

## Continuous Integration

Tests are automatically run in GitHub Actions on every push and pull request. See `.github/workflows/tests.yml` for the configuration.

## Best Practices

1. **Keep tests fast**: Tests should run quickly to encourage frequent testing
2. **Make tests independent**: Tests should not depend on each other
3. **Use fixtures**: Share setup code with pytest fixtures
4. **Test one thing per test**: Each test should verify a single behavior
5. **Use descriptive names**: Test names should describe what they're testing
6. **Follow AAA pattern**: Arrange, Act, Assert
7. **Mock external dependencies**: Use mocks for external services
8. **Test both success and failure cases**: Ensure code handles errors correctly
