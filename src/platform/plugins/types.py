"""Plugin type system for the MLX platform.

This module defines the plugin type hierarchy and conflict resolution system
to ensure proper composability and prevent incompatible plugins from running
simultaneously.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.platform.plugins.base import MLOpsComponent
from src.platform.utils.exceptions import PluginError


class PluginType(Enum):
    """Core plugin types with conflict resolution rules."""

    # Data layer
    DATA_SOURCE = "data_source"  # Single active per source type
    DATA_PROCESSOR = "data_processor"  # Multiple allowed, chainable
    FEATURE_STORE = "feature_store"  # Single active

    # ML platform layer
    ML_PLATFORM = "ml_platform"  # Single active per platform type
    EXPERIMENT_TRACKER = "experiment_tracker"  # Single active
    MODEL_REGISTRY = "model_registry"  # Single active

    # Compute layer
    WORKFLOW_ENGINE = "workflow_engine"  # Single active
    COMPUTE_BACKEND = "compute_backend"  # Multiple allowed, different purposes

    # Serving layer
    SERVING_PLATFORM = "serving_platform"  # Multiple allowed, different endpoints
    VECTOR_DATABASE = "vector_database"  # Single active per use case

    # AI/LLM layer
    LLM_PROVIDER = "llm_provider"  # Multiple allowed, different models

    # Infrastructure layer
    CLOUD_PROVIDER = "cloud_provider"  # Multiple allowed, different regions
    MONITORING = "monitoring"  # Multiple allowed, different aspects
    SECURITY = "security"  # Multiple allowed, different layers

    # Generic
    GENERAL = "general"  # Multiple allowed


@dataclass
class PluginMetadata:
    """Rich metadata for plugin registration and discovery."""

    name: str
    plugin_type: PluginType
    version: str
    description: str

    # Capabilities and integration
    capabilities: list[str] = field(default_factory=list)
    integration_points: list[str] = field(default_factory=list)
    data_formats: list[str] = field(default_factory=list)

    # Conflict resolution
    conflicts_with: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    optional_deps: list[str] = field(default_factory=list)

    # Platform compatibility
    supported_platforms: list[str] = field(
        default_factory=lambda: ["linux", "darwin", "win32"]
    )
    min_python_version: str = "3.9"

    # Resource requirements
    min_memory_mb: int = 512
    min_cpu_cores: int = 1
    gpu_required: bool = False

    # Configuration
    config_schema: dict[str, Any] = field(default_factory=dict)
    default_config: dict[str, Any] = field(default_factory=dict)

    # Documentation and support
    documentation_url: str | None = None
    repository_url: str | None = None
    license: str = "MIT"
    maintainer: str = ""
    maintainer_email: str = ""

    # Quality indicators
    test_coverage: float = 0.0
    performance_benchmarks: dict[str, float] = field(default_factory=dict)
    security_audit_date: str | None = None


class PluginConflictResolver:
    """Resolves conflicts between plugins based on type and metadata."""

    # Plugin types that allow only one active instance
    SINGLETON_TYPES = {
        PluginType.ML_PLATFORM,
        PluginType.EXPERIMENT_TRACKER,
        PluginType.MODEL_REGISTRY,
        PluginType.WORKFLOW_ENGINE,
        PluginType.FEATURE_STORE,
    }

    # Plugin types that allow multiple instances but may have sub-conflicts
    MULTI_INSTANCE_TYPES = {
        PluginType.DATA_PROCESSOR,
        PluginType.COMPUTE_BACKEND,
        PluginType.SERVING_PLATFORM,
        PluginType.LLM_PROVIDER,
        PluginType.CLOUD_PROVIDER,
        PluginType.MONITORING,
        PluginType.SECURITY,
        PluginType.GENERAL,
    }

    def __init__(self):
        self.active_plugins: dict[str, PluginMetadata] = {}
        self.type_instances: dict[PluginType, set[str]] = {}

    def can_activate(self, metadata: PluginMetadata) -> tuple[bool, str | None]:
        """Check if a plugin can be activated given current state.

        Returns:
            (can_activate, reason_if_not)
        """
        # Check singleton type conflicts
        if metadata.plugin_type in self.SINGLETON_TYPES:
            existing = self.type_instances.get(metadata.plugin_type, set())
            if existing:
                existing_plugin = list(existing)[0]
                return (
                    False,
                    f"Plugin type {metadata.plugin_type.value} already has active plugin: {existing_plugin}",
                )

        # Check explicit conflicts
        for active_plugin_name in self.active_plugins:
            if active_plugin_name in metadata.conflicts_with:
                return False, f"Conflicts with active plugin: {active_plugin_name}"

            active_metadata = self.active_plugins[active_plugin_name]
            if metadata.name in active_metadata.conflicts_with:
                return (
                    False,
                    f"Active plugin {active_plugin_name} conflicts with {metadata.name}",
                )

        # Check requirements
        for required_plugin in metadata.requires:
            if required_plugin not in self.active_plugins:
                return False, f"Required plugin not active: {required_plugin}"

        # Check platform compatibility
        import sys

        if sys.platform not in metadata.supported_platforms:
            return False, f"Platform {sys.platform} not supported"

        # Check Python version
        import sys

        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if current_version < metadata.min_python_version:
            return (
                False,
                f"Python {metadata.min_python_version}+ required, have {current_version}",
            )

        return True, None

    def activate_plugin(self, metadata: PluginMetadata) -> bool:
        """Activate a plugin if no conflicts exist.

        Returns:
            True if activated successfully
        """
        can_activate, reason = self.can_activate(metadata)
        if not can_activate:
            raise PluginError(f"Cannot activate plugin {metadata.name}: {reason}")

        self.active_plugins[metadata.name] = metadata

        # Track by type
        if metadata.plugin_type not in self.type_instances:
            self.type_instances[metadata.plugin_type] = set()
        self.type_instances[metadata.plugin_type].add(metadata.name)

        return True

    def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate a plugin and check for dependency violations.

        Returns:
            True if deactivated successfully
        """
        if plugin_name not in self.active_plugins:
            return False

        # Check if any active plugins depend on this one
        dependent_plugins = []
        for active_name, active_metadata in self.active_plugins.items():
            if plugin_name in active_metadata.requires:
                dependent_plugins.append(active_name)

        if dependent_plugins:
            raise PluginError(
                f"Cannot deactivate plugin {plugin_name}: required by {dependent_plugins}"
            )

        # Remove from tracking
        metadata = self.active_plugins[plugin_name]
        del self.active_plugins[plugin_name]

        if metadata.plugin_type in self.type_instances:
            self.type_instances[metadata.plugin_type].discard(plugin_name)
            if not self.type_instances[metadata.plugin_type]:
                del self.type_instances[metadata.plugin_type]

        return True

    def get_active_plugins_by_type(self, plugin_type: PluginType) -> list[str]:
        """Get list of active plugins of a specific type."""
        return list(self.type_instances.get(plugin_type, set()))

    def get_composition_summary(self) -> dict[str, Any]:
        """Get summary of current plugin composition."""
        summary = {
            "total_active": len(self.active_plugins),
            "by_type": {},
            "potential_conflicts": [],
            "missing_requirements": [],
        }

        for plugin_type, instances in self.type_instances.items():
            summary["by_type"][plugin_type.value] = list(instances)

        # Check for potential issues
        for metadata in self.active_plugins.values():
            # Check optional dependencies
            for optional_dep in metadata.optional_deps:
                if optional_dep not in self.active_plugins:
                    summary["missing_requirements"].append(
                        {
                            "plugin": metadata.name,
                            "missing_optional": optional_dep,
                        }
                    )

        return summary


class TypedPlugin(MLOpsComponent, ABC):
    """Base class for typed plugins with rich metadata."""

    def __init__(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        metadata: PluginMetadata | None = None,
    ):
        super().__init__(name, config)
        self.metadata = metadata or self.get_default_metadata()
        self.resolver: PluginConflictResolver | None = None

    @abstractmethod
    def get_default_metadata(self) -> PluginMetadata:
        """Get default metadata for this plugin type."""
        pass

    def set_resolver(self, resolver: PluginConflictResolver):
        """Set the conflict resolver for this plugin."""
        self.resolver = resolver

    def can_activate(self) -> tuple[bool, str | None]:
        """Check if this plugin can be activated."""
        if self.resolver:
            return self.resolver.can_activate(self.metadata)
        return True, None

    def get_integration_points(self) -> list[str]:
        """Get integration points for this plugin."""
        return self.metadata.integration_points

    def get_capabilities(self) -> list[str]:
        """Get capabilities provided by this plugin."""
        return self.metadata.capabilities

    def check_compatibility(self, other_plugin: "TypedPlugin") -> dict[str, Any]:
        """Check compatibility with another plugin."""
        compatibility = {
            "compatible": True,
            "conflicts": [],
            "synergies": [],
            "warnings": [],
        }

        # Check explicit conflicts
        if other_plugin.name in self.metadata.conflicts_with:
            compatibility["compatible"] = False
            compatibility["conflicts"].append(
                f"Explicit conflict with {other_plugin.name}"
            )

        # Check type conflicts
        if (
            self.metadata.plugin_type in PluginConflictResolver.SINGLETON_TYPES
            and self.metadata.plugin_type == other_plugin.metadata.plugin_type
        ):
            compatibility["compatible"] = False
            compatibility["conflicts"].append(
                f"Both are singleton type {self.metadata.plugin_type.value}"
            )

        # Check integration synergies
        common_integration_points = set(self.metadata.integration_points) & set(
            other_plugin.metadata.integration_points
        )
        if common_integration_points:
            compatibility["synergies"].extend(list(common_integration_points))

        return compatibility


# Factory for creating typed plugins
class PluginTypeFactory:
    """Factory for creating typed plugins with proper metadata."""

    _type_registry: dict[PluginType, type[TypedPlugin]] = {}

    @classmethod
    def register_type(cls, plugin_type: PluginType, plugin_class: type[TypedPlugin]):
        """Register a plugin class for a specific type."""
        cls._type_registry[plugin_type] = plugin_class

    @classmethod
    def create_plugin(
        cls,
        plugin_type: PluginType,
        name: str,
        config: dict[str, Any] | None = None,
        metadata_overrides: dict[str, Any] | None = None,
    ) -> TypedPlugin:
        """Create a typed plugin instance."""
        if plugin_type not in cls._type_registry:
            raise PluginError(f"No plugin class registered for type {plugin_type}")

        plugin_class = cls._type_registry[plugin_type]
        plugin = plugin_class(name, config)

        # Apply metadata overrides
        if metadata_overrides:
            for key, value in metadata_overrides.items():
                if hasattr(plugin.metadata, key):
                    setattr(plugin.metadata, key, value)

        return plugin


# Global conflict resolver
_global_resolver = PluginConflictResolver()


def get_conflict_resolver() -> PluginConflictResolver:
    """Get the global plugin conflict resolver."""
    return _global_resolver


# Decorator for plugin type registration
def register_plugin_type(plugin_type: PluginType):
    """Decorator to register a plugin class for a specific type."""

    def decorator(plugin_class: type[TypedPlugin]):
        PluginTypeFactory.register_type(plugin_type, plugin_class)
        return plugin_class

    return decorator
