"""Tests for AI agent monitoring and analytics."""

from datetime import datetime, timedelta

import pytest

from src.ai.monitoring import AgentMonitor, CostTracker, MetricsCollector
from src.ai.monitoring.base import CostBreakdown, EventType, MonitoringEvent


class TestEventType:
    """Test EventType enum."""

    def test_event_type_values(self):
        """Test that event types have correct values."""
        assert EventType.AGENT_START.value == "agent_start"
        assert EventType.AGENT_COMPLETE.value == "agent_complete"
        assert EventType.AGENT_ERROR.value == "agent_error"
        assert EventType.LLM_CALL.value == "llm_call"
        assert EventType.RAG_QUERY.value == "rag_query"
        assert EventType.TOOL_USE.value == "tool_use"


class TestMonitoringEvent:
    """Test MonitoringEvent dataclass."""

    def test_event_creation(self):
        """Test basic monitoring event creation."""
        timestamp = datetime.now()
        event = MonitoringEvent(
            event_type=EventType.AGENT_START,
            timestamp=timestamp,
            agent_name="test_agent",
            data={"task": "test_task"},
        )

        assert event.event_type == EventType.AGENT_START
        assert event.timestamp == timestamp
        assert event.agent_name == "test_agent"
        assert event.data["task"] == "test_task"
        assert event.session_id is None
        assert event.cost_usd is None

    def test_event_with_all_fields(self):
        """Test event with all optional fields."""
        timestamp = datetime.now()
        event = MonitoringEvent(
            event_type=EventType.LLM_CALL,
            timestamp=timestamp,
            agent_name="test_agent",
            data={"model": "gpt-4"},
            session_id="session_123",
            user_id="user_456",
            cost_usd=0.001,
            tokens_used=100,
            duration_ms=500,
        )

        assert event.session_id == "session_123"
        assert event.user_id == "user_456"
        assert event.cost_usd == 0.001
        assert event.tokens_used == 100
        assert event.duration_ms == 500


class TestCostBreakdown:
    """Test CostBreakdown dataclass."""

    def test_cost_breakdown_defaults(self):
        """Test cost breakdown with default values."""
        breakdown = CostBreakdown()

        assert breakdown.llm_calls == 0.0
        assert breakdown.vector_store == 0.0
        assert breakdown.tools == 0.0
        assert breakdown.total == 0.0
        assert breakdown.currency == "USD"

    def test_cost_breakdown_custom_values(self):
        """Test cost breakdown with custom values."""
        breakdown = CostBreakdown(
            llm_calls=0.05, vector_store=0.01, tools=0.002, total=0.062, currency="EUR"
        )

        assert breakdown.llm_calls == 0.05
        assert breakdown.vector_store == 0.01
        assert breakdown.tools == 0.002
        assert breakdown.total == 0.062
        assert breakdown.currency == "EUR"


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    @pytest.fixture
    def collector(self):
        """Create metrics collector instance."""
        return MetricsCollector()

    def test_collector_initialization(self, collector):
        """Test collector initialization."""
        assert collector.events == []
        assert collector.logger is not None

    def test_record_event(self, collector):
        """Test recording events."""
        event = MonitoringEvent(
            event_type=EventType.AGENT_START,
            timestamp=datetime.now(),
            agent_name="test_agent",
        )

        collector.record_event(event)

        assert len(collector.events) == 1
        assert collector.events[0] == event

    def test_get_events_no_filter(self, collector):
        """Test getting all events without filter."""
        event1 = MonitoringEvent(EventType.AGENT_START, datetime.now(), "agent1")
        event2 = MonitoringEvent(EventType.LLM_CALL, datetime.now(), "agent2")

        collector.record_event(event1)
        collector.record_event(event2)

        events = collector.get_events()

        assert len(events) == 2
        assert event1 in events
        assert event2 in events

    def test_get_events_with_type_filter(self, collector):
        """Test getting events filtered by type."""
        event1 = MonitoringEvent(EventType.AGENT_START, datetime.now(), "agent1")
        event2 = MonitoringEvent(EventType.LLM_CALL, datetime.now(), "agent1")
        event3 = MonitoringEvent(EventType.AGENT_START, datetime.now(), "agent2")

        collector.record_event(event1)
        collector.record_event(event2)
        collector.record_event(event3)

        start_events = collector.get_events(event_type=EventType.AGENT_START)

        assert len(start_events) == 2
        assert event1 in start_events
        assert event3 in start_events
        assert event2 not in start_events

    def test_get_events_with_agent_filter(self, collector):
        """Test getting events filtered by agent name."""
        event1 = MonitoringEvent(EventType.AGENT_START, datetime.now(), "agent1")
        event2 = MonitoringEvent(EventType.LLM_CALL, datetime.now(), "agent1")
        event3 = MonitoringEvent(EventType.AGENT_START, datetime.now(), "agent2")

        collector.record_event(event1)
        collector.record_event(event2)
        collector.record_event(event3)

        agent1_events = collector.get_events(agent_name="agent1")

        assert len(agent1_events) == 2
        assert event1 in agent1_events
        assert event2 in agent1_events
        assert event3 not in agent1_events

    def test_get_events_with_time_filter(self, collector):
        """Test getting events filtered by time."""
        now = datetime.now()
        old_time = now - timedelta(hours=2)
        recent_time = now - timedelta(minutes=30)

        old_event = MonitoringEvent(EventType.AGENT_START, old_time, "agent1")
        recent_event = MonitoringEvent(EventType.LLM_CALL, recent_time, "agent1")

        collector.record_event(old_event)
        collector.record_event(recent_event)

        since = now - timedelta(hours=1)
        recent_events = collector.get_events(since=since)

        assert len(recent_events) == 1
        assert recent_event in recent_events
        assert old_event not in recent_events

    def test_get_metrics_summary(self, collector):
        """Test getting metrics summary."""
        # Create events with costs and tokens
        event1 = MonitoringEvent(
            EventType.LLM_CALL,
            datetime.now(),
            "agent1",
            cost_usd=0.001,
            tokens_used=100,
            duration_ms=500,
        )
        event2 = MonitoringEvent(
            EventType.AGENT_START, datetime.now(), "agent2", duration_ms=200
        )

        collector.record_event(event1)
        collector.record_event(event2)

        summary = collector.get_metrics_summary()

        assert summary["total_events"] == 2
        assert summary["total_cost_usd"] == 0.001
        assert summary["total_tokens"] == 100
        assert summary["avg_duration_ms"] == 350  # (500 + 200) / 2
        assert summary["unique_agents"] == 2
        assert summary["event_counts"]["llm_call"] == 1
        assert summary["event_counts"]["agent_start"] == 1


class TestCostTracker:
    """Test CostTracker functionality."""

    @pytest.fixture
    def tracker(self):
        """Create cost tracker instance."""
        return CostTracker()

    def test_tracker_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.cost_events == []
        assert tracker.logger is not None

    def test_track_llm_cost(self, tracker):
        """Test tracking LLM costs."""
        tracker.track_llm_cost(
            agent_name="test_agent",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.006,
            session_id="session_123",
        )

        assert len(tracker.cost_events) == 1
        event = tracker.cost_events[0]

        assert event.event_type == EventType.LLM_CALL
        assert event.agent_name == "test_agent"
        assert event.data["model"] == "gpt-4"
        assert event.data["input_tokens"] == 100
        assert event.data["output_tokens"] == 50
        assert event.data["total_tokens"] == 150
        assert event.cost_usd == 0.006
        assert event.tokens_used == 150
        assert event.session_id == "session_123"

    def test_track_rag_cost(self, tracker):
        """Test tracking RAG costs."""
        tracker.track_rag_cost(
            agent_name="rag_agent",
            query="What is machine learning?",
            documents_retrieved=5,
            vector_store_cost=0.001,
            session_id="session_456",
        )

        assert len(tracker.cost_events) == 1
        event = tracker.cost_events[0]

        assert event.event_type == EventType.RAG_QUERY
        assert event.agent_name == "rag_agent"
        assert event.data["query"] == "What is machine learning?"
        assert event.data["documents_retrieved"] == 5
        assert event.cost_usd == 0.001
        assert event.session_id == "session_456"

    def test_get_cost_breakdown(self, tracker):
        """Test getting cost breakdown."""
        # Add different types of cost events
        tracker.track_llm_cost("agent1", "gpt-4", 100, 50, 0.005)
        tracker.track_rag_cost("agent1", "test query", 3, 0.001)
        tracker.track_llm_cost("agent2", "gpt-3.5", 200, 100, 0.002)

        # Get overall breakdown
        breakdown = tracker.get_cost_breakdown()

        assert breakdown.llm_calls == 0.007  # 0.005 + 0.002
        assert breakdown.vector_store == 0.001
        assert breakdown.tools == 0.0
        assert breakdown.total == 0.008

        # Get breakdown for specific agent
        agent1_breakdown = tracker.get_cost_breakdown(agent_name="agent1")

        assert agent1_breakdown.llm_calls == 0.005
        assert agent1_breakdown.vector_store == 0.001
        assert agent1_breakdown.total == 0.006

    def test_get_cost_trends(self, tracker):
        """Test getting cost trends."""
        # Add some historical events
        tracker.track_llm_cost("agent1", "gpt-4", 100, 50, 0.005)

        trends = tracker.get_cost_trends(days=3)

        assert "dates" in trends
        assert "costs" in trends
        assert len(trends["dates"]) == 3
        assert len(trends["costs"]) == 3
        # Today should have some cost
        assert trends["costs"][-1] == 0.005

    def test_estimate_monthly_cost(self, tracker):
        """Test monthly cost estimation."""
        # Add events to simulate weekly usage
        for _ in range(7):
            tracker.track_llm_cost("agent1", "gpt-4", 100, 50, 0.01)

        monthly_estimate = tracker.estimate_monthly_cost()

        # Should be weekly cost (0.07) * 4.33 weeks
        expected = 0.07 * 4.33
        assert abs(monthly_estimate - expected) < 0.01

    def test_get_top_cost_agents(self, tracker):
        """Test getting top cost agents."""
        # Add costs for different agents
        tracker.track_llm_cost("expensive_agent", "gpt-4", 1000, 500, 0.1)
        tracker.track_llm_cost("cheap_agent", "gpt-3.5", 100, 50, 0.001)
        tracker.track_llm_cost("expensive_agent", "gpt-4", 500, 250, 0.05)

        top_agents = tracker.get_top_cost_agents(limit=2)

        assert len(top_agents) == 2
        assert top_agents[0]["agent_name"] == "expensive_agent"
        assert (
            abs(top_agents[0]["total_cost"] - 0.15) < 0.001
        )  # Floating point precision
        assert top_agents[0]["call_count"] == 2
        assert top_agents[1]["agent_name"] == "cheap_agent"
        assert top_agents[1]["total_cost"] == 0.001


class TestAgentMonitor:
    """Test AgentMonitor main class."""

    @pytest.fixture
    def monitor(self):
        """Create agent monitor instance."""
        return AgentMonitor()

    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.metrics is not None
        assert monitor.costs is not None
        assert monitor.logger is not None
        assert monitor.active_sessions == {}

    def test_start_session(self, monitor):
        """Test starting a monitoring session."""
        monitor.start_session("test_agent", "session_123")

        assert "session_123" in monitor.active_sessions
        assert len(monitor.metrics.events) == 1

        event = monitor.metrics.events[0]
        assert event.event_type == EventType.AGENT_START
        assert event.agent_name == "test_agent"
        assert event.session_id == "session_123"

    def test_end_session_success(self, monitor):
        """Test ending a session successfully."""
        # Start session first
        monitor.start_session("test_agent", "session_123")

        # End session
        monitor.end_session("test_agent", "session_123", success=True)

        assert "session_123" not in monitor.active_sessions
        assert len(monitor.metrics.events) == 2

        end_event = monitor.metrics.events[1]
        assert end_event.event_type == EventType.AGENT_COMPLETE
        assert end_event.agent_name == "test_agent"
        assert end_event.session_id == "session_123"
        assert end_event.duration_ms is not None

    def test_end_session_error(self, monitor):
        """Test ending a session with error."""
        monitor.start_session("test_agent", "session_123")

        monitor.end_session(
            "test_agent",
            "session_123",
            success=False,
            error_message="Something went wrong",
        )

        end_event = monitor.metrics.events[1]
        assert end_event.event_type == EventType.AGENT_ERROR
        assert end_event.data["error_message"] == "Something went wrong"

    def test_log_execution(self, monitor):
        """Test logging agent execution."""
        monitor.log_execution(
            agent_name="test_agent",
            task="Test task description",
            result="Test result",
            metadata={"key": "value"},
        )

        assert len(monitor.metrics.events) == 1
        event = monitor.metrics.events[0]

        assert event.event_type == EventType.AGENT_COMPLETE
        assert event.agent_name == "test_agent"
        assert "Test task description" in event.data["task"]
        assert "Test result" in event.data["result"]
        assert event.data["metadata"]["key"] == "value"

    def test_get_stats(self, monitor):
        """Test getting comprehensive stats."""
        # Add some activity
        monitor.start_session("agent1", "session1")
        monitor.costs.track_llm_cost("agent1", "gpt-4", 100, 50, 0.005)
        monitor.end_session("agent1", "session1", success=True)

        stats = monitor.get_stats()

        assert "metrics" in stats
        assert "costs" in stats
        assert "active_sessions" in stats

        assert stats["metrics"]["total_events"] == 2
        assert stats["costs"]["breakdown"]["llm_calls"] == 0.005
        assert stats["active_sessions"] == 0

    def test_export_data_json(self, monitor):
        """Test exporting data as JSON."""
        monitor.log_execution("test_agent", "task", "result")

        exported = monitor.export_data(format="json")

        assert isinstance(exported, str)
        # Should be valid JSON
        import json

        data = json.loads(exported)

        assert "events" in data
        assert "stats" in data
        assert len(data["events"]) == 1

    def test_export_data_unsupported_format(self, monitor):
        """Test exporting with unsupported format."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            monitor.export_data(format="xml")


class TestMonitoringIntegration:
    """Integration tests for monitoring components."""

    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        monitor = AgentMonitor()

        # Start session
        monitor.start_session("integration_agent", "session_test")

        # Track LLM cost
        monitor.costs.track_llm_cost(
            "integration_agent", "gpt-4", 100, 50, 0.005, "session_test"
        )

        # Track RAG cost
        monitor.costs.track_rag_cost(
            "integration_agent", "test query", 3, 0.001, "session_test"
        )

        # Log execution
        monitor.log_execution("integration_agent", "Integration test", "Success")

        # End session
        monitor.end_session("integration_agent", "session_test", success=True)

        # Verify comprehensive tracking
        stats = monitor.get_stats()

        assert stats["metrics"]["total_events"] == 3  # start, log, end
        assert stats["costs"]["breakdown"]["llm_calls"] == 0.005
        assert stats["costs"]["breakdown"]["vector_store"] == 0.001
        assert stats["costs"]["breakdown"]["total"] == 0.006
        assert stats["active_sessions"] == 0

    def test_multiple_agents_monitoring(self):
        """Test monitoring multiple agents simultaneously."""
        monitor = AgentMonitor()

        # Start multiple sessions
        monitor.start_session("agent1", "session1")
        monitor.start_session("agent2", "session2")

        # Track different activities
        monitor.costs.track_llm_cost("agent1", "gpt-4", 100, 50, 0.005)
        monitor.costs.track_llm_cost("agent2", "gpt-3.5", 200, 100, 0.002)

        # End sessions
        monitor.end_session("agent1", "session1", success=True)
        monitor.end_session(
            "agent2", "session2", success=False, error_message="Test error"
        )

        # Verify stats
        stats = monitor.get_stats()

        assert stats["metrics"]["unique_agents"] == 2
        assert stats["costs"]["breakdown"]["llm_calls"] == 0.007

        # Check top agents
        top_agents = monitor.costs.get_top_cost_agents()
        assert len(top_agents) == 2
        assert top_agents[0]["agent_name"] == "agent1"  # Higher cost


if __name__ == "__main__":
    pytest.main([__file__])
