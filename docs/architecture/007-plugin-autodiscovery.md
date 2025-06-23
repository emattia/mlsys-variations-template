# ADR-007: Plugin Auto-Discovery Mechanism

## Status
Accepted (2025-06-23)

## Context

The MLX platform needed an efficient way to discover and register plugins without requiring manual configuration updates. Key requirements:

1. **Developer Experience**: New plugins should be automatically available without manual registration
2. **Namespace Support**: Support for namespace packages and third-party plugins
3. **Hot Reloading**: Development-time ability to reload plugins without restart
4. **Performance**: Minimize startup time impact from plugin discovery
5. **Validation**: Ensure discovered plugins meet interface requirements
6. **Conflict Prevention**: Detect and handle plugin conflicts during discovery

Manual registration had significant problems:
- **Maintenance Burden**: Every new plugin required registry updates
- **Error-Prone**: Easy to forget registration, causing runtime failures
- **Development Friction**: Slowed down plugin development iteration
- **Third-Party Integration**: Difficult to integrate external plugins

Example of problematic manual registration:
```python
# Manual plugin registry - error prone and high maintenance
PLUGIN_REGISTRY = {
    "snowflake": "src.plugins.snowflake.SnowflakePlugin",
    "databricks": "src.plugins.databricks.DatabricksPlugin",
    # Easy to forget new plugins, causing runtime errors
}
```

## Decision

We implement an **automatic plugin discovery system** using Python entry points and namespace scanning:

### Discovery Mechanisms

1. **Entry Points**: Standard Python packaging discovery mechanism
2. **Namespace Scanning**: Automatic scanning of plugin namespaces
3. **File System Discovery**: Local development plugin discovery
4. **Import Hooks**: Dynamic plugin loading with validation

### Implementation Architecture

```python
class PluginDiscoveryEngine:
    def __init__(self):
        self.discovered_plugins: Dict[str, PluginMetadata] = {}
        self.discovery_sources = [
            EntryPointDiscovery(),
            NamespaceDiscovery(),
            FileSystemDiscovery(),
        ]

    async def discover_plugins(self,
                             cache_enabled: bool = True,
                             include_sources: List[str] = None) -> List[PluginMetadata]:
        """Discover all available plugins from configured sources."""

        if cache_enabled and self._cache_valid():
            return list(self.discovered_plugins.values())

        discovered = []
        for source in self.discovery_sources:
            if include_sources and source.name not in include_sources:
                continue

            try:
                plugins = await source.discover()
                discovered.extend(plugins)
            except Exception as e:
                logger.warning(f"Discovery source {source.name} failed: {e}")

        # Validate and deduplicate
        self.discovered_plugins = self._validate_and_deduplicate(discovered)
        return list(self.discovered_plugins.values())
```

### Entry Point Configuration
```python
# setup.py or pyproject.toml
entry_points = {
    "mlx.plugins.data_source": [
        "snowflake = mlx_snowflake:SnowflakePlugin",
        "postgres = mlx_postgres:PostgresPlugin",
    ],
    "mlx.plugins.ml_platform": [
        "databricks = mlx_databricks:DatabricksPlugin",
        "sagemaker = mlx_sagemaker:SageMakerPlugin",
    ],
}
```

### Hot Reloading Support
```python
class HotReloadManager:
    def __init__(self, watch_directories: List[Path]):
        self.watchers = []
        self.reload_callbacks = []

    async def start_watching(self):
        """Start watching for plugin file changes."""
        for directory in self.watch_directories:
            watcher = FileWatcher(directory, self._on_file_change)
            await watcher.start()
            self.watchers.append(watcher)

    async def _on_file_change(self, changed_file: Path):
        """Handle plugin file changes."""
        if self._is_plugin_file(changed_file):
            await self._reload_plugin(changed_file)
```

## Alternatives Considered

### 1. Manual Registry Pattern
- **Pros**: Simple, explicit, full control over registration
- **Cons**: High maintenance, error-prone, development friction
- **Rejected**: Too much overhead for plugin ecosystem

### 2. Import-Time Registration (Decorators)
- **Pros**: Automatic registration, no separate discovery step
- **Cons**: Forces import of all plugins, poor error isolation
- **Rejected**: Performance and reliability concerns

### 3. Configuration-Based Discovery
- **Pros**: Explicit control, easy to understand
- **Cons**: Manual maintenance, doesn't support third-party plugins well
- **Rejected**: Doesn't meet auto-discovery requirements

### 4. Reflection-Based Discovery
- **Pros**: Automatic detection of plugin classes
- **Cons**: Performance overhead, fragile to code changes
- **Rejected**: Too brittle and slow

## Consequences

### Positive
1. **Zero-Config Registration**: New plugins automatically discovered
2. **Third-Party Support**: Easy integration of external plugins
3. **Developer Experience**: Faster development iteration cycles
4. **Standard Compliance**: Uses Python packaging standards (entry points)
5. **Namespace Support**: Clean separation of plugin categories
6. **Hot Reloading**: Development-time plugin reloading
7. **Validation**: Automatic validation of plugin interfaces
8. **Performance**: Caching and lazy loading minimize startup impact

### Negative
1. **Discovery Overhead**: Initial discovery adds startup time
2. **Complexity**: More complex than simple manual registration
3. **Debugging**: Discovery failures can be hard to debug
4. **Security Concerns**: Automatic loading of code from discovered plugins
5. **Dependency Management**: Plugin dependencies must be carefully managed

### Implementation Details

#### Entry Point Discovery
```python
class EntryPointDiscovery:
    def discover(self) -> List[PluginMetadata]:
        """Discover plugins via entry points."""
        discovered = []

        for group_name in ["mlx.plugins.data_source", "mlx.plugins.ml_platform"]:
            entry_points = pkg_resources.iter_entry_points(group_name)

            for entry_point in entry_points:
                try:
                    plugin_class = entry_point.load()
                    metadata = self._extract_metadata(plugin_class, entry_point)
                    discovered.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to load plugin {entry_point.name}: {e}")

        return discovered
```

#### Namespace Discovery
```python
class NamespaceDiscovery:
    def discover(self) -> List[PluginMetadata]:
        """Discover plugins in namespace packages."""
        discovered = []

        # Scan mlx_plugins namespace
        for finder, name, ispkg in pkgutil.iter_modules(mlx_plugins.__path__,
                                                       mlx_plugins.__name__ + "."):
            if self._is_plugin_module(name):
                try:
                    module = importlib.import_module(name)
                    plugin_class = self._find_plugin_class(module)
                    if plugin_class:
                        metadata = self._extract_metadata(plugin_class)
                        discovered.append(metadata)
                except ImportError as e:
                    logger.debug(f"Could not import {name}: {e}")

        return discovered
```

#### Development Hot Reloading
```python
class DevelopmentPluginLoader:
    def __init__(self, development_paths: List[Path]):
        self.development_paths = development_paths
        self.file_watcher = FileWatcher()

    async def enable_hot_reload(self):
        """Enable hot reloading for development."""
        for path in self.development_paths:
            await self.file_watcher.watch(path, self._on_plugin_change)

    async def _on_plugin_change(self, changed_file: Path):
        """Reload plugin when file changes."""
        if changed_file.suffix == '.py':
            plugin_name = self._extract_plugin_name(changed_file)
            await self._reload_plugin(plugin_name)
```

#### Plugin Validation
```python
def validate_plugin(plugin_class: Type) -> ValidationResult:
    """Validate plugin meets interface requirements."""
    errors = []

    # Check inheritance
    if not issubclass(plugin_class, TypedPlugin):
        errors.append("Plugin must inherit from TypedPlugin")

    # Check required methods
    required_methods = ['execute', 'initialize', 'cleanup']
    for method in required_methods:
        if not hasattr(plugin_class, method):
            errors.append(f"Plugin missing required method: {method}")

    # Check metadata
    try:
        instance = plugin_class("test", {})
        metadata = instance.get_default_metadata()
        validate_metadata(metadata)
    except Exception as e:
        errors.append(f"Invalid metadata: {e}")

    return ValidationResult(valid=len(errors) == 0, errors=errors)
```

### Performance Optimizations
1. **Lazy Loading**: Only load plugins when actually needed
2. **Caching**: Cache discovery results with invalidation
3. **Parallel Discovery**: Discover from multiple sources concurrently
4. **Metadata Caching**: Cache plugin metadata separately from code

### Security Considerations
1. **Trusted Sources**: Only discover from trusted package sources
2. **Code Signing**: Verify plugin signatures when available
3. **Sandboxing**: Load untrusted plugins in restricted environments
4. **Audit Logging**: Log all plugin discovery and loading events

This auto-discovery system dramatically improves the developer experience while maintaining security and performance requirements for production use.
