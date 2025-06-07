"""Unit tests for the plugin architecture."""

from typing import Any

import polars as pl

from src.config import AppConfig
from src.plugins import (
    ComponentResult,
    ComponentStatus,
    DataProcessor,
    ExecutionContext,
)
from src.plugins.registry import PluginRegistry


class TestDataProcessor(DataProcessor):
    """Test implementation of DataProcessor."""

    def initialize(self, context: ExecutionContext) -> None:
        """Initialize the component."""
        pass

    def execute(self, context: ExecutionContext) -> ComponentResult:
        """Execute the component."""
        return ComponentResult(
            status=ComponentStatus.SUCCESS,
            component_name=self.name,
            execution_time=0.1,
        )

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate configuration."""
        return True

    def process_data(
        self, input_data: pl.DataFrame, context: ExecutionContext
    ) -> pl.DataFrame:
        """Process data."""
        return input_data

    def validate_data(self, data: pl.DataFrame, context: ExecutionContext) -> bool:
        """Validate data."""
        return len(data) > 0


class TestPluginRegistry:
    """Test PluginRegistry functionality."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = PluginRegistry()

        assert len(registry._plugins) == 0
        assert len(registry._instances) == 0

    def test_plugin_registration(self):
        """Test plugin registration."""
        registry = PluginRegistry()

        registry.register(
            name="test-processor",
            plugin_class=TestDataProcessor,
            category="data_processing",
            description="Test data processor",
            version="1.0.0",
        )

        assert "test-processor" in registry._plugins
        plugin_info = registry._plugins["test-processor"]
        assert plugin_info["class"] == TestDataProcessor
        assert plugin_info["category"] == "data_processing"


class TestExecutionContext:
    """Test ExecutionContext dataclass."""

    def test_context_creation(self, test_config: AppConfig):
        """Test creating execution context."""
        context = ExecutionContext(
            config=test_config,
            run_id="test-run",
            component_name="test-component",
        )

        assert context.config == test_config
        assert context.run_id == "test-run"
        assert context.component_name == "test-component"


class TestComponentResult:
    """Test ComponentResult dataclass."""

    def test_result_creation(self):
        """Test creating component result."""
        result = ComponentResult(
            status=ComponentStatus.SUCCESS,
            component_name="test-component",
            execution_time=1.5,
        )

        assert result.status == ComponentStatus.SUCCESS
        assert result.component_name == "test-component"
        assert result.execution_time == 1.5
        assert result.is_success() is True
        assert result.is_failed() is False
