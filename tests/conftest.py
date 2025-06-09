"""Pytest configuration and fixtures for the MLOps testing framework."""

import asyncio
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient

from src.api.app import app
from src.config import Config, ConfigManager
from src.plugins import ExecutionContext


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def config_manager(temp_dir: Path) -> ConfigManager:
    """Create a test configuration manager."""
    config_dir = temp_dir / "conf"
    manager = ConfigManager(config_dir=config_dir)
    manager.create_default_config_files()
    return manager


@pytest.fixture(scope="session")
def test_config(config_manager: ConfigManager) -> Config:
    """Create a test configuration."""
    return config_manager.load_config()


@pytest.fixture
def execution_context(test_config: Config) -> ExecutionContext:
    """Create a test execution context."""
    return ExecutionContext(
        config=test_config,
        run_id="test-run-001",
        component_name="test-component",
        metadata={"test": True},
    )


@pytest.fixture(scope="session")
async def async_client(test_config: Config) -> AsyncClient:
    """Create an async test client for the API."""
    app.state.config = test_config
    async with LifespanManager(app):
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Create a sample DataFrame for testing."""
    import numpy as np

    np.random.seed(42)
    n_samples = 100

    data = {
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.uniform(0, 1, n_samples),
        "categorical_feature": np.random.choice(["A", "B", "C"], n_samples),
        "target": np.random.choice([0, 1], n_samples),
    }

    return pl.DataFrame(data)


@pytest.fixture
def sample_regression_dataframe() -> pl.DataFrame:
    """Create a sample DataFrame for regression testing."""
    import numpy as np

    np.random.seed(42)
    n_samples = 100

    # Create correlated features for regression
    feature_1 = np.random.randn(n_samples)
    feature_2 = np.random.randn(n_samples)
    target = 2 * feature_1 + 1.5 * feature_2 + np.random.randn(n_samples) * 0.1

    data = {
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": np.random.uniform(0, 1, n_samples),
        "categorical_feature": np.random.choice(["X", "Y", "Z"], n_samples),
        "target": target,
    }

    return pl.DataFrame(data)


@pytest.fixture
def sample_csv_file(sample_dataframe: pl.DataFrame, temp_dir: Path) -> Path:
    """Create a sample CSV file for testing."""
    csv_path = temp_dir / "sample_data.csv"
    sample_dataframe.write_csv(csv_path)
    return csv_path


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean the plugin registry before each test."""
    from src.plugins.registry import get_registry

    registry = get_registry()
    registry.clear()
    yield
    registry.clear()


@pytest.fixture
def mock_model():
    """Create a mock trained model for testing."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    # Create synthetic data
    X, y = make_classification(
        n_samples=100, n_features=4, n_classes=2, random_state=42
    )

    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    return model


@pytest.fixture
def config_override() -> dict[str, Any]:
    """Provide configuration overrides for testing."""
    return {
        "ml": {
            "random_seed": 123,
            "test_size": 0.3,
        },
        "model": {
            "model_type": "linear",
            "problem_type": "classification",
        },
    }


# Test markers
pytest.mark.integration = pytest.mark.integration
pytest.mark.unit = pytest.mark.unit
pytest.mark.slow = pytest.mark.slow


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers."""
    for item in items:
        # Add unit marker by default if no other marker is present
        if not any(
            mark.name in ["integration", "slow", "unit"] for mark in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)
