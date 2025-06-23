"""Unit tests for plugin architecture."""

from src.plugins.base import MLOpsComponent
from src.plugins.registry import PluginRegistry


class MockPlugin(MLOpsComponent):
    """Mock plugin for testing."""

    def __init__(self, name="mock_plugin", config=None):
        super().__init__(name, config)

    @property
    def version(self):
        """Plugin version."""
        return "1.0.0"

    def initialize(self, context=None):
        """Initialize the plugin."""
        return True

    def validate_config(self, config=None):
        """Validate plugin configuration."""
        return True

    def execute(self, context):
        """Mock execute method."""
        return {"status": "success", "context": context}


class TestPluginRegistry:
    """Test cases for PluginRegistry."""

    def test_register_plugin(self):
        """Test plugin registration."""
        registry = PluginRegistry()
        plugin = MockPlugin("test_plugin")
        registry.register("test_plugin", plugin.__class__)
        assert "test_plugin" in registry._plugins

    def test_get_plugin(self):
        """Test getting registered plugin."""
        registry = PluginRegistry()
        plugin = MockPlugin("test_plugin")
        registry.register("test_plugin", plugin.__class__)
        retrieved = registry.get_plugin("test_plugin")
        assert retrieved is not None

    def test_get_nonexistent_plugin(self):
        """Test getting non-existent plugin."""
        registry = PluginRegistry()
        result = registry.get_plugin("nonexistent")
        assert result is None

    def test_list_plugins(self):
        """Test listing all plugins."""
        registry = PluginRegistry()
        plugin1 = MockPlugin("plugin1")
        plugin2 = MockPlugin("plugin2")
        registry.register("plugin1", plugin1.__class__)
        registry.register("plugin2", plugin2.__class__)
        plugins = registry.list_plugins()
        assert len(plugins) >= 2
        assert "plugin1" in plugins
        assert "plugin2" in plugins


class TestMLOpsComponent:
    """Test cases for MLOpsComponent."""

    def test_plugin_initialization(self):
        """Test plugin initialization."""
        plugin = MockPlugin("test")
        assert plugin.name == "test"
        assert plugin.version == "1.0.0"

    def test_plugin_execute(self):
        """Test plugin execution."""
        plugin = MockPlugin("test")
        context = {"input": "test_data"}
        result = plugin.execute(context)
        assert result["status"] == "success"
        assert result["context"] == context
