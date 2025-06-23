# ADR-001: Dual-Layer Plugin Architecture

## Status
Accepted (2025-06-23)

## Context

The MLX platform needed an architecture that could balance:

1. **Rapid Development**: Teams need to bootstrap ML projects quickly with proven patterns
2. **Flexibility**: Runtime behavior must be configurable and swappable without code changes
3. **Consistency**: Code structure and interfaces should be standardized across teams
4. **Composability**: Different ML tools and services need to integrate seamlessly

Traditional approaches faced limitations:
- **Pure Templates**: Static code generation lacks runtime flexibility
- **Pure Plugins**: No structural consistency, teams reinvent boilerplate
- **Monolithic Frameworks**: Too rigid for diverse ML platform requirements

## Decision

We adopt a **dual-layer architecture** combining:

### Layer 1: MLX Components (Static Templates)
- **Purpose**: Code generation and project structure
- **Location**: `@mlx-components/` registry
- **Provides**: Templates, schemas, best practices, documentation
- **Used at**: Development time

### Layer 2: Runtime Plugins (Dynamic Components)
- **Purpose**: Business logic and service integration
- **Location**: `src/platform/plugins/`
- **Provides**: Live functionality, external integrations, configurable behavior
- **Used at**: Runtime

### Integration Pattern
```mermaid
graph LR
    A[MLX Component] --> B[Generated Plugin Scaffold]
    B --> C[Developer Implementation]
    C --> D[Runtime Plugin Registration]
    D --> E[Dynamic Execution]
```

## Alternatives Considered

### 1. Pure Plugin System
- **Pros**: Maximum runtime flexibility
- **Cons**: No structural consistency, teams reinvent boilerplate
- **Rejected**: Too much duplication and inconsistency

### 2. Pure Template System
- **Pros**: Consistent structure, rapid development
- **Cons**: No runtime flexibility, requires recompilation for changes
- **Rejected**: Insufficient flexibility for ML platform needs

### 3. Monolithic Framework
- **Pros**: Complete integration, battle-tested patterns
- **Cons**: Vendor lock-in, limited extensibility
- **Rejected**: Too rigid for diverse team requirements

## Consequences

### Positive
1. **Best of Both Worlds**: Structure + flexibility
2. **Rapid Bootstrapping**: Components provide proven starting points
3. **Runtime Flexibility**: Plugins enable hot-swapping and configuration
4. **Consistency**: Components ensure standardized interfaces
5. **Composability**: Type system prevents incompatible combinations
6. **Evolution**: Can update templates without breaking runtime behavior

### Negative
1. **Complexity**: Two layers to understand and maintain
2. **Learning Curve**: Developers must understand both components and plugins
3. **Potential Confusion**: Clear boundaries between layers must be maintained
4. **Tooling Overhead**: Need CLI tools to manage component-to-plugin workflow

### Mitigation Strategies
1. **Documentation**: Comprehensive guides explaining the dual-layer pattern
2. **CLI Tooling**: Automate component-to-plugin generation
3. **Examples**: Provide clear examples of the full workflow
4. **Type System**: Strong typing to catch misuse early
