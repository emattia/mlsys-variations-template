# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records documenting significant architectural decisions made during the development of the MLX platform.

## ADR Format

Each ADR follows the standard format:

1. **Status**: Proposed, Accepted, Deprecated, Superseded
2. **Context**: The technical and business context that led to the decision
3. **Decision**: The architectural decision made
4. **Consequences**: The positive and negative consequences of the decision

## Index of ADRs

| Number | Title | Status | Date |
|--------|-------|--------|------|
| [ADR-001](./001-dual-layer-architecture.md) | Dual-Layer Plugin Architecture | Accepted | 2025-06-23 |
| [ADR-002](./002-temporal-workflow-orchestration.md) | Temporal for Workflow Orchestration | Accepted | 2025-06-23 |
| [ADR-003](./003-opentelemetry-observability.md) | OpenTelemetry for Observability | Accepted | 2025-06-23 |
| [ADR-004](./004-plugin-type-system.md) | Plugin Type System and Conflict Resolution | Accepted | 2025-06-23 |
| [ADR-005](./005-hydra-pydantic-configuration.md) | Hydra + Pydantic Configuration Management | Accepted | 2025-06-23 |
| [ADR-006](./006-exception-hierarchy.md) | Standardized Exception Hierarchy | Accepted | 2025-06-23 |
| [ADR-007](./007-plugin-autodiscovery.md) | Plugin Auto-Discovery Mechanism | Accepted | 2025-06-23 |

## Decision Process

1. **Identification**: Technical challenges requiring architectural decisions
2. **Research**: Analysis of alternatives and trade-offs
3. **Proposal**: ADR draft with recommendation
4. **Review**: Team review and feedback
5. **Decision**: Final decision and documentation
6. **Implementation**: Execute the architectural decision

## Guidelines

- ADRs should be written when making significant architectural decisions
- Focus on decisions that are hard to reverse or have broad impact
- Include context about why the decision was needed
- Document alternatives considered and why they were rejected
- Update status when decisions are superseded or deprecated
