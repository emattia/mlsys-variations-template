"""Base monitoring and cost tracking classes."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class EventType(Enum):
    """Types of events to monitor."""

    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    LLM_CALL = "llm_call"
    RAG_QUERY = "rag_query"
    TOOL_USE = "tool_use"


@dataclass
class MonitoringEvent:
    """Represents a monitoring event."""

    event_type: EventType
    timestamp: datetime
    agent_name: str
    data: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    user_id: str | None = None
    cost_usd: float | None = None
    tokens_used: int | None = None
    duration_ms: int | None = None


@dataclass
class CostBreakdown:
    """Breakdown of costs by component."""

    llm_calls: float = 0.0
    vector_store: float = 0.0
    tools: float = 0.0
    total: float = 0.0
    currency: str = "USD"


class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self):
        self.events: list[MonitoringEvent] = []
        self.logger = logging.getLogger(__name__)

    def record_event(self, event: MonitoringEvent) -> None:
        """Record a monitoring event."""
        self.events.append(event)
        self.logger.debug(f"Recorded event: {event.event_type.value}")

    def get_events(
        self,
        event_type: EventType | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
    ) -> list[MonitoringEvent]:
        """Get events with optional filtering."""
        filtered_events = self.events

        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]

        if agent_name:
            filtered_events = [e for e in filtered_events if e.agent_name == agent_name]

        if since:
            filtered_events = [e for e in filtered_events if e.timestamp >= since]

        return filtered_events

    def get_metrics_summary(self, time_window_hours: int = 24) -> dict[str, Any]:
        """Get summary metrics for a time window."""
        since = datetime.now() - timedelta(hours=time_window_hours)
        recent_events = self.get_events(since=since)

        # Count events by type
        event_counts = {}
        for event_type in EventType:
            event_counts[event_type.value] = len(
                [e for e in recent_events if e.event_type == event_type]
            )

        # Calculate costs and tokens
        total_cost = sum(e.cost_usd or 0 for e in recent_events)
        total_tokens = sum(e.tokens_used or 0 for e in recent_events)

        # Calculate average duration
        durations = [e.duration_ms for e in recent_events if e.duration_ms]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Count unique agents
        unique_agents = len({e.agent_name for e in recent_events})

        return {
            "time_window_hours": time_window_hours,
            "total_events": len(recent_events),
            "event_counts": event_counts,
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "avg_duration_ms": avg_duration,
            "unique_agents": unique_agents,
            "events_per_hour": len(recent_events) / time_window_hours,
        }


class CostTracker:
    """Tracks and analyzes costs for AI operations."""

    def __init__(self):
        self.cost_events: list[MonitoringEvent] = []
        self.logger = logging.getLogger(__name__)

    def track_llm_cost(
        self,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        session_id: str | None = None,
    ) -> None:
        """Track cost of an LLM call."""
        event = MonitoringEvent(
            event_type=EventType.LLM_CALL,
            timestamp=datetime.now(),
            agent_name=agent_name,
            data={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            session_id=session_id,
            cost_usd=cost_usd,
            tokens_used=input_tokens + output_tokens,
        )

        self.cost_events.append(event)
        self.logger.info(f"Tracked LLM cost: ${cost_usd:.4f} for {agent_name}")

    def track_rag_cost(
        self,
        agent_name: str,
        query: str,
        documents_retrieved: int,
        vector_store_cost: float = 0.0,
        session_id: str | None = None,
    ) -> None:
        """Track cost of a RAG query."""
        event = MonitoringEvent(
            event_type=EventType.RAG_QUERY,
            timestamp=datetime.now(),
            agent_name=agent_name,
            data={
                "query": query[:100],  # Truncate for privacy
                "documents_retrieved": documents_retrieved,
            },
            session_id=session_id,
            cost_usd=vector_store_cost,
        )

        self.cost_events.append(event)
        self.logger.info(f"Tracked RAG cost: ${vector_store_cost:.4f} for {agent_name}")

    def get_cost_breakdown(
        self, agent_name: str | None = None, time_window_hours: int = 24
    ) -> CostBreakdown:
        """Get cost breakdown for an agent or all agents."""
        since = datetime.now() - timedelta(hours=time_window_hours)

        filtered_events = [
            e
            for e in self.cost_events
            if e.timestamp >= since and (not agent_name or e.agent_name == agent_name)
        ]

        llm_cost = sum(
            e.cost_usd or 0
            for e in filtered_events
            if e.event_type == EventType.LLM_CALL
        )

        vector_cost = sum(
            e.cost_usd or 0
            for e in filtered_events
            if e.event_type == EventType.RAG_QUERY
        )

        tool_cost = sum(
            e.cost_usd or 0
            for e in filtered_events
            if e.event_type == EventType.TOOL_USE
        )

        total_cost = llm_cost + vector_cost + tool_cost

        return CostBreakdown(
            llm_calls=llm_cost,
            vector_store=vector_cost,
            tools=tool_cost,
            total=total_cost,
        )

    def get_cost_trends(self, days: int = 7) -> dict[str, list[float]]:
        """Get daily cost trends."""
        trends = {"dates": [], "costs": []}

        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)

            day_events = [
                e for e in self.cost_events if day_start <= e.timestamp < day_end
            ]

            day_cost = sum(e.cost_usd or 0 for e in day_events)

            trends["dates"].insert(0, date.strftime("%Y-%m-%d"))
            trends["costs"].insert(0, day_cost)

        return trends

    def estimate_monthly_cost(self) -> float:
        """Estimate monthly cost based on recent usage."""
        # Use last 7 days to estimate monthly cost
        breakdown = self.get_cost_breakdown(time_window_hours=24 * 7)
        weekly_cost = breakdown.total

        # Extrapolate to monthly (4.33 weeks per month)
        monthly_estimate = weekly_cost * 4.33

        return monthly_estimate

    def get_top_cost_agents(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get agents with highest costs."""
        agent_costs = {}

        for event in self.cost_events:
            agent_name = event.agent_name
            cost = event.cost_usd or 0

            if agent_name not in agent_costs:
                agent_costs[agent_name] = {
                    "agent_name": agent_name,
                    "total_cost": 0,
                    "call_count": 0,
                }

            agent_costs[agent_name]["total_cost"] += cost
            agent_costs[agent_name]["call_count"] += 1

        # Sort by cost and return top N
        sorted_agents = sorted(
            agent_costs.values(), key=lambda x: x["total_cost"], reverse=True
        )

        return sorted_agents[:limit]


class AgentMonitor:
    """Main monitoring class that combines metrics and cost tracking."""

    def __init__(self):
        self.metrics = MetricsCollector()
        self.costs = CostTracker()
        self.logger = logging.getLogger(__name__)
        self.active_sessions: dict[str, datetime] = {}

    def start_session(self, agent_name: str, session_id: str) -> None:
        """Start monitoring a session."""
        self.active_sessions[session_id] = datetime.now()

        event = MonitoringEvent(
            event_type=EventType.AGENT_START,
            timestamp=datetime.now(),
            agent_name=agent_name,
            session_id=session_id,
        )

        self.metrics.record_event(event)
        self.logger.info(f"Started monitoring session {session_id} for {agent_name}")

    def end_session(
        self,
        agent_name: str,
        session_id: str,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """End monitoring a session."""
        start_time = self.active_sessions.get(session_id)
        duration_ms = None

        if start_time:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            del self.active_sessions[session_id]

        event_type = EventType.AGENT_COMPLETE if success else EventType.AGENT_ERROR

        event = MonitoringEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            agent_name=agent_name,
            session_id=session_id,
            duration_ms=duration_ms,
            data={"error_message": error_message} if error_message else {},
        )

        self.metrics.record_event(event)
        self.logger.info(
            f"Ended session {session_id} for {agent_name} (success: {success})"
        )

    def log_execution(
        self, agent_name: str, task: str, result: str, metadata: dict[str, Any] = None
    ) -> None:
        """Log agent execution details."""
        event = MonitoringEvent(
            event_type=EventType.AGENT_COMPLETE,
            timestamp=datetime.now(),
            agent_name=agent_name,
            data={
                "task": task[:200],  # Truncate for storage
                "result": result[:500],  # Truncate for storage
                "metadata": metadata or {},
            },
        )

        self.metrics.record_event(event)

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        metrics_summary = self.metrics.get_metrics_summary()
        cost_breakdown = self.costs.get_cost_breakdown()
        cost_trends = self.costs.get_cost_trends()
        top_agents = self.costs.get_top_cost_agents()

        return {
            "metrics": metrics_summary,
            "costs": {
                "breakdown": cost_breakdown.__dict__,
                "trends": cost_trends,
                "monthly_estimate": self.costs.estimate_monthly_cost(),
                "top_agents": top_agents,
            },
            "active_sessions": len(self.active_sessions),
        }

    def export_data(self, format: str = "json") -> str:
        """Export monitoring data."""
        if format == "json":
            data = {
                "events": [
                    {
                        "event_type": e.event_type.value,
                        "timestamp": e.timestamp.isoformat(),
                        "agent_name": e.agent_name,
                        "data": e.data,
                        "session_id": e.session_id,
                        "cost_usd": e.cost_usd,
                        "tokens_used": e.tokens_used,
                        "duration_ms": e.duration_ms,
                    }
                    for e in self.metrics.events
                ],
                "stats": self.get_stats(),
            }
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
