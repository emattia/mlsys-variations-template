# ADR-004: Plugin Type System and Conflict Resolution

## Status
Accepted (2025-06-23)

## Context

The MLX platform supports composing multiple plugins to create complete ML workflows. However, unrestricted plugin composition can lead to:

1. **Resource Conflicts**: Multiple plugins competing for the same resources
2. **Functional Conflicts**: Plugins providing overlapping capabilities (e.g., MLflow vs Wandb)
3. **Dependency Issues**: Plugins requiring specific versions of shared dependencies
4. **Integration Problems**: Plugins not designed to work together
5. **Performance Degradation**: Too many plugins running simultaneously

The platform needed a system to:
- **Prevent Conflicts**: Automatically detect and prevent incompatible combinations
- **Enable Synergies**: Identify plugins that work well together
- **Guide Composition**: Help users choose complementary plugins
- **Enforce Constraints**: Apply business rules about plugin usage

## Decision

We implement a **Plugin Type System** with automatic conflict resolution based on:

### 1. Plugin Type Hierarchy
```python
class PluginType(Enum):
    # Singleton types (only one active per type)
    ML_PLATFORM = "ml_platform"              # Databricks OR Sagemaker
    EXPERIMENT_TRACKER = "experiment_tracker" # MLflow OR Wandb
    MODEL_REGISTRY = "model_registry"         # Single model store

    # Multi-instance types (multiple allowed)
    DATA_SOURCE = "data_source"              # Snowflake + S3 + PostgreSQL
    LLM_PROVIDER = "llm_provider"           # OpenAI + Anthropic + Local
    MONITORING = "monitoring"               # Multiple monitoring tools
```

### 2. Rich Plugin Metadata
```python
@dataclass
class PluginMetadata:
    name: str
    plugin_type: PluginType
    capabilities: List[str]
    conflicts_with: List[str]
    requires: List[str]
    optional_deps: List[str]
    integration_points: List[str]
    resource_requirements: Dict[str, Any]
```

### 3. Automatic Conflict Resolution
```python
class PluginConflictResolver:
    def can_activate(self, metadata: PluginMetadata) -> tuple[bool, Optional[str]]:
        # Check singleton conflicts
        # Check explicit conflicts
        # Check missing requirements
        # Check resource availability
```

### 4. Composition Validation
- Pre-activation validation prevents invalid states
- Runtime monitoring detects emergent conflicts
- Recommendation engine suggests compatible plugins

## Alternatives Considered

### 1. No Type System (Free Composition)
- **Pros**: Maximum flexibility, no restrictions
- **Cons**: Runtime conflicts, poor user experience, debugging complexity
- **Rejected**: Too many failure modes in production

### 2. Strict Categories (Mutually Exclusive)
- **Pros**: Simple model, no conflicts by design
- **Cons**: Too restrictive, prevents valid combinations
- **Rejected**: Real ML workflows need multiple tools per category

### 3. Manual Configuration
- **Pros**: User has full control, no system complexity
- **Cons**: Error-prone, requires deep knowledge, poor UX
- **Rejected**: Too much cognitive load on users

### 4. Runtime Detection Only
- **Pros**: Flexible, handles edge cases
- **Cons**: Failures discovered late, poor debugging experience
- **Rejected**: Prevention better than cure

## Consequences

### Positive
1. **Conflict Prevention**: Catches incompatible combinations before activation
2. **Better UX**: Clear error messages with resolution suggestions
3. **Composition Guidance**: Helps users discover compatible plugins
4. **Resource Management**: Prevents resource over-allocation
5. **Faster Debugging**: Type system narrows problem space
6. **Documentation**: Metadata serves as plugin documentation
7. **Ecosystem Health**: Encourages well-designed plugin interfaces

### Negative
1. **System Complexity**: Additional layer of abstraction to maintain
2. **Plugin Development Overhead**: Authors must define comprehensive metadata
3. **False Positives**: Type system may prevent valid combinations
4. **Performance Impact**: Validation overhead on plugin operations
5. **Learning Curve**: Users must understand type system concepts

### Implementation Guidelines

#### Plugin Type Assignment
```python
# Singleton types - only one active instance
SINGLETON_TYPES = {
    PluginType.ML_PLATFORM,          # Databricks vs Sagemaker
    PluginType.EXPERIMENT_TRACKER,   # MLflow vs Wandb
    PluginType.MODEL_REGISTRY,       # Centralized model store
    PluginType.WORKFLOW_ENGINE,      # Temporal vs Airflow
}

# Multi-instance types - multiple instances allowed
MULTI_INSTANCE_TYPES = {
    PluginType.DATA_SOURCE,          # Multiple data sources
    PluginType.LLM_PROVIDER,         # Multiple LLM providers
    PluginType.MONITORING,           # Multiple monitoring tools
}
```

#### Conflict Definition
```python
# Explicit conflicts
snowflake_metadata = PluginMetadata(
    conflicts_with=["bigquery-plugin"],  # Same capability, different vendors
    requires=["aws-credentials"],        # Dependency requirements
    optional_deps=["dbt-plugin"]         # Enhanced functionality
)

# Type-based conflicts (automatic)
databricks_metadata = PluginMetadata(
    plugin_type=PluginType.ML_PLATFORM  # Conflicts with other ML platforms
)
```

#### Integration Points
```python
# Define how plugins can integrate
databricks_metadata = PluginMetadata(
    integration_points=["data_sources", "model_registries", "feature_stores"],
    capabilities=["spark_processing", "ml_training", "model_serving"]
)
```

### Evolution Strategy
1. **Start Conservative**: Begin with known conflicts, expand over time
2. **User Feedback**: Collect feedback on false positives/negatives
3. **Machine Learning**: Eventually learn conflicts from usage patterns
4. **Versioning**: Support plugin metadata versioning for evolution

### Monitoring and Metrics
- Track conflict resolution effectiveness
- Monitor false positive rates
- Measure plugin composition success rates
- Analyze user override patterns

This type system provides the foundation for a reliable, composable plugin ecosystem while maintaining flexibility for legitimate use cases.
