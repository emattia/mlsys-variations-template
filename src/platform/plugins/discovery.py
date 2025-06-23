"""Plugin auto-discovery system for the MLX platform.

This module provides:
- Automatic discovery of plugins from multiple sources
- Plugin validation and compatibility checking
- Dynamic loading and hot-reloading capabilities
- Plugin dependency resolution
- Integration with package managers and external registries
"""

import importlib
import importlib.util
import inspect
import logging
import pkgutil
import sys
from pathlib import Path
from typing import Any

from src.platform.plugins.base import MLOpsComponent
from src.platform.plugins.registry import PluginRegistry, register_plugin

logger = logging.getLogger(__name__)


class PluginDiscovery:
    """Automatic plugin discovery and loading system."""

    def __init__(self, registry: PluginRegistry | None = None):
        self.registry = registry or PluginRegistry()
        self.discovered_plugins: dict[str, dict[str, Any]] = {}
        self.discovery_sources: list[str] = []
        self.logger = logging.getLogger(__name__)

    def add_discovery_source(self, source: str | Path) -> None:
        """Add a source for plugin discovery.

        Args:
            source: Python module/package path or filesystem path
        """
        source_str = str(source)
        if source_str not in self.discovery_sources:
            self.discovery_sources.append(source_str)
            self.logger.info(f"Added plugin discovery source: {source_str}")

    def discover_plugins(
        self,
        auto_register: bool = True,
        validate: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Discover plugins from all configured sources.

        Args:
            auto_register: Whether to automatically register discovered plugins
            validate: Whether to validate plugins before registration

        Returns:
            Dictionary of discovered plugins with metadata
        """
        self.discovered_plugins.clear()

        for source in self.discovery_sources:
            try:
                if self._is_filesystem_path(source):
                    self._discover_from_filesystem(source)
                else:
                    self._discover_from_module(source)
            except Exception as e:
                self.logger.error(f"Failed to discover plugins from {source}: {e}")

        if validate:
            self._validate_discovered_plugins()

        if auto_register:
            self._register_discovered_plugins()

        self.logger.info(f"Discovered {len(self.discovered_plugins)} plugins")
        return self.discovered_plugins

    def discover_from_entry_points(
        self,
        group: str = "mlx.plugins",
        auto_register: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Discover plugins from setuptools entry points.

        Args:
            group: Entry point group name
            auto_register: Whether to automatically register plugins

        Returns:
            Dictionary of discovered plugins
        """
        try:
            import pkg_resources

            discovered = {}

            for entry_point in pkg_resources.iter_entry_points(group):
                try:
                    plugin_class = entry_point.load()

                    if self._is_valid_plugin_class(plugin_class):
                        plugin_info = self._extract_plugin_metadata(plugin_class)
                        plugin_info.update(
                            {
                                "name": entry_point.name,
                                "entry_point": str(entry_point),
                                "distribution": entry_point.dist.project_name,
                                "version": entry_point.dist.version,
                            }
                        )

                        discovered[entry_point.name] = plugin_info

                        if auto_register:
                            self._register_plugin_class(
                                entry_point.name, plugin_class, plugin_info
                            )

                except Exception as e:
                    self.logger.error(
                        f"Failed to load plugin from entry point {entry_point}: {e}"
                    )

            self.logger.info(f"Discovered {len(discovered)} plugins from entry points")
            return discovered

        except ImportError:
            self.logger.warning(
                "pkg_resources not available, skipping entry point discovery"
            )
            return {}

    def discover_from_namespace(
        self,
        namespace: str,
        auto_register: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Discover plugins from a namespace package.

        Args:
            namespace: Namespace package (e.g., 'mlx_plugins')
            auto_register: Whether to automatically register plugins

        Returns:
            Dictionary of discovered plugins
        """
        discovered = {}

        try:
            # Import the namespace package
            namespace_module = importlib.import_module(namespace)

            # Iterate through all modules in the namespace
            for _finder, module_name, _ispkg in pkgutil.iter_modules(
                namespace_module.__path__, namespace + "."
            ):
                try:
                    module = importlib.import_module(module_name)
                    plugin_classes = self._find_plugin_classes_in_module(module)

                    for name, plugin_class in plugin_classes.items():
                        plugin_info = self._extract_plugin_metadata(plugin_class)
                        plugin_info.update(
                            {
                                "name": name,
                                "module": module_name,
                                "namespace": namespace,
                            }
                        )

                        discovered[name] = plugin_info

                        if auto_register:
                            self._register_plugin_class(name, plugin_class, plugin_info)

                except Exception as e:
                    self.logger.error(
                        f"Failed to load plugin module {module_name}: {e}"
                    )

        except ImportError:
            self.logger.warning(f"Namespace package {namespace} not found")

        self.logger.info(
            f"Discovered {len(discovered)} plugins from namespace {namespace}"
        )
        return discovered

    def hot_reload_plugin(self, plugin_name: str) -> bool:
        """Hot reload a plugin by name.

        Args:
            plugin_name: Name of the plugin to reload

        Returns:
            True if reload successful, False otherwise
        """
        try:
            # Get plugin info
            if plugin_name not in self.discovered_plugins:
                self.logger.error(
                    f"Plugin {plugin_name} not found in discovered plugins"
                )
                return False

            plugin_info = self.discovered_plugins[plugin_name]
            module_name = plugin_info.get("module")

            if not module_name:
                self.logger.error(f"No module information for plugin {plugin_name}")
                return False

            # Unregister existing plugin
            try:
                self.registry.unregister(plugin_name)
            except KeyError:
                pass  # Plugin not registered yet

            # Reload module
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)

            # Re-discover and register
            module = sys.modules[module_name]
            plugin_classes = self._find_plugin_classes_in_module(module)

            if plugin_name in plugin_classes:
                plugin_class = plugin_classes[plugin_name]
                updated_info = self._extract_plugin_metadata(plugin_class)
                updated_info.update(plugin_info)  # Preserve discovery metadata

                self._register_plugin_class(plugin_name, plugin_class, updated_info)
                self.discovered_plugins[plugin_name] = updated_info

                self.logger.info(f"Successfully reloaded plugin {plugin_name}")
                return True
            else:
                self.logger.error(f"Plugin class {plugin_name} not found after reload")
                return False

        except Exception as e:
            self.logger.error(f"Failed to reload plugin {plugin_name}: {e}")
            return False

    def validate_plugin_compatibility(
        self,
        plugin_name: str,
        target_version: str | None = None,
    ) -> dict[str, Any]:
        """Validate plugin compatibility and dependencies.

        Args:
            plugin_name: Name of plugin to validate
            target_version: Target platform version to check against

        Returns:
            Validation result with compatibility information
        """
        if plugin_name not in self.discovered_plugins:
            return {
                "valid": False,
                "error": f"Plugin {plugin_name} not found",
            }

        plugin_info = self.discovered_plugins[plugin_name]
        validation_result = {
            "valid": True,
            "plugin_name": plugin_name,
            "warnings": [],
            "errors": [],
        }

        try:
            # Check if plugin class can be instantiated
            plugin_class = plugin_info.get("class")
            if plugin_class:
                try:
                    # Try to create instance with minimal config
                    plugin_class(name=plugin_name, config={})

                    # Check abstract methods implementation
                    abstract_methods = getattr(
                        plugin_class, "__abstractmethods__", set()
                    )
                    if abstract_methods:
                        validation_result["errors"].append(
                            f"Plugin has unimplemented abstract methods: {abstract_methods}"
                        )
                        validation_result["valid"] = False

                except Exception as e:
                    validation_result["errors"].append(
                        f"Cannot instantiate plugin: {e}"
                    )
                    validation_result["valid"] = False

            # Validate dependencies
            dependency_validation = self.registry.validate_dependencies(plugin_name)

            for dep, available in dependency_validation.items():
                if not available:
                    validation_result["errors"].append(f"Missing dependency: {dep}")
                    validation_result["valid"] = False

            # Check version compatibility if specified
            if target_version:
                plugin_version = plugin_info.get("version", "unknown")
                # Simple version comparison - could be enhanced with semantic versioning
                if plugin_version != "unknown" and plugin_version != target_version:
                    validation_result["warnings"].append(
                        f"Version mismatch: plugin {plugin_version}, platform {target_version}"
                    )

        except Exception as e:
            validation_result["errors"].append(f"Validation error: {e}")
            validation_result["valid"] = False

        return validation_result

    def _is_filesystem_path(self, source: str) -> bool:
        """Check if source is a filesystem path."""
        return "/" in source or "\\" in source or source.startswith(".")

    def _discover_from_filesystem(self, path: str) -> None:
        """Discover plugins from filesystem path."""
        path_obj = Path(path)

        if not path_obj.exists():
            self.logger.warning(f"Plugin path does not exist: {path}")
            return

        if path_obj.is_file() and path_obj.suffix == ".py":
            # Single Python file
            self._load_plugin_from_file(path_obj)
        elif path_obj.is_dir():
            # Directory with Python files
            for py_file in path_obj.rglob("*.py"):
                if py_file.name != "__init__.py":
                    self._load_plugin_from_file(py_file)

    def _discover_from_module(self, module_path: str) -> None:
        """Discover plugins from Python module/package."""
        try:
            module = importlib.import_module(module_path)

            if hasattr(module, "__path__"):
                # Package - iterate through submodules
                for _finder, module_name, _ispkg in pkgutil.iter_modules(
                    module.__path__, module_path + "."
                ):
                    try:
                        submodule = importlib.import_module(module_name)
                        plugin_classes = self._find_plugin_classes_in_module(submodule)

                        for name, plugin_class in plugin_classes.items():
                            plugin_info = self._extract_plugin_metadata(plugin_class)
                            plugin_info["module"] = module_name
                            self.discovered_plugins[name] = plugin_info

                    except Exception as e:
                        self.logger.error(
                            f"Failed to load submodule {module_name}: {e}"
                        )
            else:
                # Single module
                plugin_classes = self._find_plugin_classes_in_module(module)

                for name, plugin_class in plugin_classes.items():
                    plugin_info = self._extract_plugin_metadata(plugin_class)
                    plugin_info["module"] = module_path
                    self.discovered_plugins[name] = plugin_info

        except ImportError as e:
            self.logger.error(f"Failed to import module {module_path}: {e}")

    def _load_plugin_from_file(self, file_path: Path) -> None:
        """Load plugin from a Python file."""
        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                plugin_classes = self._find_plugin_classes_in_module(module)

                for name, plugin_class in plugin_classes.items():
                    plugin_info = self._extract_plugin_metadata(plugin_class)
                    plugin_info.update(
                        {
                            "file_path": str(file_path),
                            "module": file_path.stem,
                        }
                    )
                    self.discovered_plugins[name] = plugin_info

        except Exception as e:
            self.logger.error(f"Failed to load plugin from {file_path}: {e}")

    def _find_plugin_classes_in_module(self, module) -> dict[str, type[MLOpsComponent]]:
        """Find all plugin classes in a module."""
        plugin_classes = {}

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if self._is_valid_plugin_class(obj):
                # Use class name as plugin name if not specified
                plugin_name = getattr(obj, "_plugin_name", name)
                plugin_classes[plugin_name] = obj

        return plugin_classes

    def _is_valid_plugin_class(self, cls) -> bool:
        """Check if a class is a valid plugin."""
        return (
            inspect.isclass(cls)
            and issubclass(cls, MLOpsComponent)
            and cls != MLOpsComponent
            and not inspect.isabstract(cls)
        )

    def _extract_plugin_metadata(
        self, plugin_class: type[MLOpsComponent]
    ) -> dict[str, Any]:
        """Extract metadata from plugin class."""
        metadata = {
            "class": plugin_class,
            "class_name": plugin_class.__name__,
            "module": plugin_class.__module__,
            "description": plugin_class.__doc__ or "",
            "version": getattr(plugin_class, "_plugin_version", "1.0.0"),
            "category": getattr(plugin_class, "_plugin_category", "general"),
            "dependencies": getattr(plugin_class, "_plugin_dependencies", []),
            "config_schema": getattr(plugin_class, "_plugin_config_schema", {}),
        }

        return metadata

    def _validate_discovered_plugins(self) -> None:
        """Validate all discovered plugins."""
        invalid_plugins = []

        for plugin_name, _plugin_info in self.discovered_plugins.items():
            validation = self.validate_plugin_compatibility(plugin_name)

            if not validation["valid"]:
                self.logger.warning(
                    f"Invalid plugin {plugin_name}: {validation['errors']}"
                )
                invalid_plugins.append(plugin_name)

        # Remove invalid plugins
        for plugin_name in invalid_plugins:
            del self.discovered_plugins[plugin_name]

    def _register_discovered_plugins(self) -> None:
        """Register all valid discovered plugins."""
        for plugin_name, plugin_info in self.discovered_plugins.items():
            try:
                self._register_plugin_class(
                    plugin_name, plugin_info["class"], plugin_info
                )
            except Exception as e:
                self.logger.error(f"Failed to register plugin {plugin_name}: {e}")

    def _register_plugin_class(
        self,
        name: str,
        plugin_class: type[MLOpsComponent],
        plugin_info: dict[str, Any],
    ) -> None:
        """Register a plugin class with the registry."""
        self.registry.register(
            name=name,
            plugin_class=plugin_class,
            category=plugin_info.get("category", "general"),
            description=plugin_info.get("description", ""),
            config_schema=plugin_info.get("config_schema", {}),
            dependencies=plugin_info.get("dependencies", []),
            version=plugin_info.get("version", "1.0.0"),
        )


# Global discovery instance
_discovery = PluginDiscovery()


def get_discovery() -> PluginDiscovery:
    """Get the global plugin discovery instance."""
    return _discovery


def auto_discover_plugins(
    sources: list[str | Path] | None = None,
    include_entry_points: bool = True,
    include_namespace: bool = True,
    namespace: str = "mlx_plugins",
) -> dict[str, dict[str, Any]]:
    """Convenience function for automatic plugin discovery.

    Args:
        sources: Additional discovery sources
        include_entry_points: Whether to discover from entry points
        include_namespace: Whether to discover from namespace packages
        namespace: Namespace package name for discovery

    Returns:
        Dictionary of all discovered plugins
    """
    discovery = get_discovery()

    # Add default sources
    default_sources = [
        "src.platform.plugins",
        "workflows",
    ]

    all_sources = default_sources + (sources or [])

    for source in all_sources:
        discovery.add_discovery_source(source)

    # Discover from all sources
    discovered = discovery.discover_plugins()

    # Discover from entry points
    if include_entry_points:
        entry_point_plugins = discovery.discover_from_entry_points()
        discovered.update(entry_point_plugins)

    # Discover from namespace packages
    if include_namespace:
        namespace_plugins = discovery.discover_from_namespace(namespace)
        discovered.update(namespace_plugins)

    logger.info(f"Auto-discovery completed: {len(discovered)} total plugins discovered")
    return discovered


# Plugin decorator for auto-discovery
def plugin(
    name: str | None = None,
    category: str = "general",
    description: str = "",
    version: str = "1.0.0",
    dependencies: list[str] | None = None,
    config_schema: dict[str, Any] | None = None,
):
    """Decorator to mark a class as a discoverable plugin.

    This sets metadata attributes that the discovery system uses.
    """

    def decorator(cls):
        cls._plugin_name = name or cls.__name__
        cls._plugin_category = category
        cls._plugin_description = description
        cls._plugin_version = version
        cls._plugin_dependencies = dependencies or []
        cls._plugin_config_schema = config_schema or {}

        # Also register with the registry if using the decorator
        try:
            register_plugin(
                name=cls._plugin_name,
                plugin_class=cls,
                category=category,
                description=description,
                version=version,
                dependencies=dependencies,
                config_schema=config_schema,
            )
        except Exception as e:
            logger.warning(f"Failed to auto-register plugin {cls._plugin_name}: {e}")

        return cls

    return decorator
