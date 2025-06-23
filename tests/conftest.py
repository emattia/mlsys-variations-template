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

# Import from installed package (no sys.path manipulation needed)
from src.api.app import app
from src.config import Config, ConfigManager
from src.plugins import ExecutionContext


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def config() -> Config:
    """Provide a test configuration instance."""
    # Create a temporary config for testing
    test_config = Config(
        app_name="test_app",
        debug=True,
        testing=True,
    )
    return test_config


@pytest.fixture
def config_manager() -> ConfigManager:
    """Provide a test configuration manager."""
    return ConfigManager()


@pytest.fixture
def execution_context(config: Config) -> ExecutionContext:
    """Provide a test execution context."""
    with tempfile.TemporaryDirectory() as temp_dir:
        context = ExecutionContext(
            config=config,
            run_id="test_run_123",
            component_name="test_component",
            artifacts_dir=Path(temp_dir),
        )
        yield context


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Provide a sample polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 28, 32],
            "score": [85.5, 90.2, 78.8, 92.1, 87.3],
        }
    )


@pytest.fixture
async def async_client() -> Generator[AsyncClient, None, None]:
    """Provide an async HTTP client for API testing."""
    async with LifespanManager(app):
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def mock_api_key() -> str:
    """Provide a mock API key for testing."""
    return "test_api_key_12345"


# Test configuration for different environments
@pytest.fixture(params=["development", "testing", "production"])
def env_config(request) -> str:
    """Provide different environment configurations for parametrized tests."""
    return request.param


@pytest.fixture
def temp_file_path() -> Generator[Path, None, None]:
    """Provide a temporary file path for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    yield tmp_path
    if tmp_path.exists():
        tmp_path.unlink()


@pytest.fixture
def temp_dir_path() -> Generator[Path, None, None]:
    """Provide a temporary directory path for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


# Markers for different test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "api: mark test as an API test")
    config.addinivalue_line(
        "markers", "benchmark: mark test as a performance benchmark"
    )


# Custom test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers."""
    for item in items:
        # Add unit marker to tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add integration marker to tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        # Add api marker to API tests
        if "api" in str(item.fspath) or "test_api" in item.name:
            item.add_marker(pytest.mark.api)


# Async fixtures for database/external service testing
@pytest.fixture
async def mock_database():
    """Provide a mock database connection for testing."""

    # This would typically set up a test database
    # For now, return a simple mock
    class MockDB:
        def __init__(self):
            self.data = {}

        async def get(self, key: str):
            return self.data.get(key)

        async def set(self, key: str, value: Any):
            self.data[key] = value

        async def delete(self, key: str):
            self.data.pop(key, None)

    db = MockDB()
    yield db
    # Cleanup would happen here


@pytest.fixture
def mock_external_service():
    """Provide a mock external service for testing."""

    class MockService:
        def __init__(self):
            self.calls = []

        async def call_api(self, endpoint: str, data: dict = None):
            self.calls.append({"endpoint": endpoint, "data": data})
            return {"status": "success", "data": data}

        def get_call_count(self):
            return len(self.calls)

    return MockService()


# Performance testing fixtures
@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        "iterations": 100,
        "timeout": 30,
        "memory_limit": "1GB",
    }
