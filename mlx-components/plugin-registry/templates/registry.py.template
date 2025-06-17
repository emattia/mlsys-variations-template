"""Plugin registry for managing MLOps components."""

import logging
from typing import Any

from .base import MLOpsComponent

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing MLOps component plugins."""

    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: dict[str, dict[str, Any]] = {}
        self._instances: dict[str, MLOpsComponent] = {}

    def register(
        self,
        name: str,
        plugin_class: type[MLOpsComponent],
        category: str = "general",
        description: str = "",
        config_schema: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
        version: str = "1.0.0",
    ) -> None:
        """Register a plugin component.

        Args:
            name: Unique name for the plugin
            plugin_class: Plugin class (must inherit from MLOpsComponent)
            category: Category of the plugin (e.g., 'data_processor', 'model_trainer')
            description: Description of the plugin
            config_schema: Configuration schema for the plugin
            dependencies: List of required dependencies
            version: Plugin version

        Raises:
            ValueError: If plugin name already exists or class is invalid
        """
        if name in self._plugins:
            raise ValueError(f"Plugin '{name}' is already registered")

        if not issubclass(plugin_class, MLOpsComponent):
            raise ValueError("Plugin class must inherit from MLOpsComponent")

        # Validate that the class can be instantiated
        try:
            # Check if the class has required abstract methods implemented
            abstract_methods = getattr(plugin_class, "__abstractmethods__", set())
            if abstract_methods:
                logger.warning(
                    f"Plugin '{name}' has unimplemented abstract methods: {abstract_methods}"
                )
        except Exception as e:
            logger.warning(f"Could not validate plugin '{name}': {e}")

        self._plugins[name] = {
            "class": plugin_class,
            "category": category,
            "description": description,
            "config_schema": config_schema or {},
            "dependencies": dependencies or [],
            "version": version,
            "registered_at": None,  # Could add timestamp
        }

        logger.info(f"Registered plugin '{name}' in category '{category}'")

    def unregister(self, name: str) -> None:
        """Unregister a plugin.

        Args:
            name: Name of the plugin to unregister

        Raises:
            KeyError: If plugin is not registered
        """
        if name not in self._plugins:
            raise KeyError(f"Plugin '{name}' is not registered")

        # Clean up any cached instances
        if name in self._instances:
            del self._instances[name]

        del self._plugins[name]
        logger.info(f"Unregistered plugin '{name}'")

    def get_plugin_class(self, name: str) -> type[MLOpsComponent]:
        """Get plugin class by name.

        Args:
            name: Name of the plugin

        Returns:
            Plugin class

        Raises:
            KeyError: If plugin is not registered
        """
        if name not in self._plugins:
            raise KeyError(f"Plugin '{name}' is not registered")

        return self._plugins[name]["class"]

    def create_instance(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        cache: bool = True,
    ) -> MLOpsComponent:
        """Create an instance of a plugin.

        Args:
            name: Name of the plugin
            config: Configuration for the plugin instance
            cache: Whether to cache the instance for reuse

        Returns:
            Plugin instance

        Raises:
            KeyError: If plugin is not registered
            Exception: If plugin cannot be instantiated
        """
        if name not in self._plugins:
            raise KeyError(f"Plugin '{name}' is not registered")

        # Check if we have a cached instance
        if cache and name in self._instances:
            return self._instances[name]

        plugin_info = self._plugins[name]
        plugin_class = plugin_info["class"]

        try:
            # Create instance with provided config
            instance = plugin_class(name=name, config=config)

            # Cache instance if requested
            if cache:
                self._instances[name] = instance

            logger.debug(f"Created instance of plugin '{name}'")
            return instance

        except Exception as e:
            logger.error(f"Failed to create instance of plugin '{name}': {e}")
            raise

    def get_plugin_info(self, name: str) -> dict[str, Any]:
        """Get information about a registered plugin.

        Args:
            name: Name of the plugin

        Returns:
            Dictionary with plugin information

        Raises:
            KeyError: If plugin is not registered
        """
        if name not in self._plugins:
            raise KeyError(f"Plugin '{name}' is not registered")

        info = self._plugins[name].copy()
        # Remove the class reference for JSON serialization
        info["class_name"] = info["class"].__name__
        info["module"] = info["class"].__module__
        del info["class"]

        return info

    def list_plugins(
        self,
        category: str | None = None,
        with_info: bool = False,
    ) -> list[str] | dict[str, dict[str, Any]]:
        """List registered plugins.

        Args:
            category: Filter by category (None for all categories)
            with_info: Whether to include plugin information

        Returns:
            List of plugin names or dict with plugin info
        """
        plugins = self._plugins

        # Filter by category if specified
        if category:
            plugins = {
                name: info
                for name, info in plugins.items()
                if info["category"] == category
            }

        if with_info:
            return {name: self.get_plugin_info(name) for name in plugins.keys()}
        else:
            return list(plugins.keys())

    def list_categories(self) -> list[str]:
        """List all plugin categories.

        Returns:
            List of unique categories
        """
        categories = {info["category"] for info in self._plugins.values()}
        return sorted(categories)

    def validate_dependencies(self, name: str) -> dict[str, bool]:
        """Validate plugin dependencies.

        Args:
            name: Name of the plugin

        Returns:
            Dictionary mapping dependency names to availability status

        Raises:
            KeyError: If plugin is not registered
        """
        if name not in self._plugins:
            raise KeyError(f"Plugin '{name}' is not registered")

        dependencies = self._plugins[name]["dependencies"]
        validation_results = {}

        for dep in dependencies:
            try:
                # Try to import the dependency
                __import__(dep)
                validation_results[dep] = True
            except ImportError:
                validation_results[dep] = False

        return validation_results

    def clear(self) -> None:
        """Clear all registered plugins."""
        self._plugins.clear()
        self._instances.clear()
        logger.info("Cleared all registered plugins")


# Global plugin registry instance
_registry = PluginRegistry()


def register_plugin(
    name: str,
    plugin_class: type[MLOpsComponent] | None = None,
    category: str = "general",
    description: str = "",
    config_schema: dict[str, Any] | None = None,
    dependencies: list[str] | None = None,
    version: str = "1.0.0",
):
    """Register a plugin component.

    Can be used as a decorator or function call.

    Args:
        name: Unique name for the plugin
        plugin_class: Plugin class (if None, used as decorator)
        category: Category of the plugin
        description: Description of the plugin
        config_schema: Configuration schema for the plugin
        dependencies: List of required dependencies
        version: Plugin version

    Returns:
        Plugin class (when used as decorator) or None
    """

    def _register(cls: type[MLOpsComponent]) -> type[MLOpsComponent]:
        _registry.register(
            name=name,
            plugin_class=cls,
            category=category,
            description=description,
            config_schema=config_schema,
            dependencies=dependencies,
            version=version,
        )
        return cls

    if plugin_class is None:
        # Used as decorator
        return _register
    else:
        # Used as function call
        _register(plugin_class)
        return None


def get_plugin(
    name: str,
    config: dict[str, Any] | None = None,
    cache: bool = True,
) -> MLOpsComponent:
    """Get a plugin instance.

    Args:
        name: Name of the plugin
        config: Configuration for the plugin
        cache: Whether to cache the instance

    Returns:
        Plugin instance
    """
    return _registry.create_instance(name, config, cache)


def list_plugins(
    category: str | None = None,
    with_info: bool = False,
) -> list[str] | dict[str, dict[str, Any]]:
    """List registered plugins."""
    return _registry.list_plugins(category, with_info)


def get_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _registry
