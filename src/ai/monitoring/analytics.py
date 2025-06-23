"""Performance analytics and insights for AI agents."""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Any

from .base import AgentMonitor, EventType, MonitoringEvent


class PerformanceAnalytics:
    """Advanced analytics for agent performance and optimization."""

    def __init__(self, monitor: AgentMonitor):
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)

    def analyze_performance(
        self, agent_name: str | None = None, days: int = 7
    ) -> dict[str, Any]:
        """Comprehensive performance analysis."""
        since = datetime.now() - timedelta(days=days)
        events = self.monitor.metrics.get_events(agent_name=agent_name, since=since)

        if not events:
            return {"error": "No events found for analysis"}

        return {
            "summary": self._generate_summary(events),
            "performance_metrics": self._calculate_performance_metrics(events),
            "cost_analysis": self._analyze_costs(events),
            "usage_patterns": self._analyze_usage_patterns(events),
            "reliability_metrics": self._calculate_reliability(events),
            "recommendations": self._generate_recommendations(events),
        }

    def _generate_summary(self, events: list[MonitoringEvent]) -> dict[str, Any]:
        """Generate high-level summary."""
        total_events = len(events)
        unique_agents = len({e.agent_name for e in events})

        # Time range
        timestamps = [e.timestamp for e in events]
        time_range = {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat(),
            "duration_hours": (max(timestamps) - min(timestamps)).total_seconds()
            / 3600,
        }

        # Event type distribution
        event_distribution = {}
        for event_type in EventType:
            count = len([e for e in events if e.event_type == event_type])
            event_distribution[event_type.value] = count

        return {
            "total_events": total_events,
            "unique_agents": unique_agents,
            "time_range": time_range,
            "event_distribution": event_distribution,
        }

    def _calculate_performance_metrics(
        self, events: list[MonitoringEvent]
    ) -> dict[str, Any]:
        """Calculate performance metrics."""
        # Duration analysis
        durations = [e.duration_ms for e in events if e.duration_ms is not None]

        if durations:
            duration_stats = {
                "avg_duration_ms": statistics.mean(durations),
                "median_duration_ms": statistics.median(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "p95_duration_ms": self._percentile(durations, 95),
                "p99_duration_ms": self._percentile(durations, 99),
            }
        else:
            duration_stats = {"note": "No duration data available"}

        # Token usage analysis
        tokens = [e.tokens_used for e in events if e.tokens_used is not None]

        if tokens:
            token_stats = {
                "total_tokens": sum(tokens),
                "avg_tokens": statistics.mean(tokens),
                "median_tokens": statistics.median(tokens),
                "max_tokens": max(tokens),
                "min_tokens": min(tokens),
            }
        else:
            token_stats = {"note": "No token data available"}

        # Throughput analysis
        throughput = self._calculate_throughput(events)

        return {
            "duration_stats": duration_stats,
            "token_stats": token_stats,
            "throughput": throughput,
        }

    def _analyze_costs(self, events: list[MonitoringEvent]) -> dict[str, Any]:
        """Analyze cost patterns and trends."""
        cost_events = [e for e in events if e.cost_usd is not None]

        if not cost_events:
            return {"note": "No cost data available"}

        costs = [e.cost_usd for e in cost_events]
        total_cost = sum(costs)

        # Cost by agent
        agent_costs = {}
        for event in cost_events:
            agent = event.agent_name
            if agent not in agent_costs:
                agent_costs[agent] = 0
            agent_costs[agent] += event.cost_usd

        # Cost by event type
        type_costs = {}
        for event in cost_events:
            event_type = event.event_type.value
            if event_type not in type_costs:
                type_costs[event_type] = 0
            type_costs[event_type] += event.cost_usd

        # Cost efficiency (cost per successful operation)
        successful_events = [
            e for e in events if e.event_type == EventType.AGENT_COMPLETE
        ]
        cost_per_success = (
            total_cost / len(successful_events) if successful_events else 0
        )

        return {
            "total_cost": total_cost,
            "avg_cost_per_event": statistics.mean(costs),
            "cost_per_success": cost_per_success,
            "cost_by_agent": dict(
                sorted(agent_costs.items(), key=lambda x: x[1], reverse=True)
            ),
            "cost_by_type": type_costs,
            "cost_trend": self._calculate_cost_trend(cost_events),
        }

    def _analyze_usage_patterns(self, events: list[MonitoringEvent]) -> dict[str, Any]:
        """Analyze usage patterns and trends."""
        # Hourly distribution
        hourly_usage = {}
        for event in events:
            hour = event.timestamp.hour
            hourly_usage[hour] = hourly_usage.get(hour, 0) + 1

        # Daily distribution
        daily_usage = {}
        for event in events:
            day = event.timestamp.weekday()  # 0 = Monday
            daily_usage[day] = daily_usage.get(day, 0) + 1

        # Peak usage times
        peak_hour = (
            max(hourly_usage.items(), key=lambda x: x[1]) if hourly_usage else (0, 0)
        )
        peak_day = (
            max(daily_usage.items(), key=lambda x: x[1]) if daily_usage else (0, 0)
        )

        # Agent usage distribution
        agent_usage = {}
        for event in events:
            agent = event.agent_name
            agent_usage[agent] = agent_usage.get(agent, 0) + 1

        return {
            "hourly_distribution": hourly_usage,
            "daily_distribution": daily_usage,
            "peak_hour": {"hour": peak_hour[0], "count": peak_hour[1]},
            "peak_day": {"day": peak_day[0], "count": peak_day[1]},
            "agent_usage": dict(
                sorted(agent_usage.items(), key=lambda x: x[1], reverse=True)
            ),
        }

    def _calculate_reliability(self, events: list[MonitoringEvent]) -> dict[str, Any]:
        """Calculate reliability and error metrics."""
        total_completions = len(
            [
                e
                for e in events
                if e.event_type in [EventType.AGENT_COMPLETE, EventType.AGENT_ERROR]
            ]
        )
        successful_completions = len(
            [e for e in events if e.event_type == EventType.AGENT_COMPLETE]
        )
        error_events = len([e for e in events if e.event_type == EventType.AGENT_ERROR])

        success_rate = (
            successful_completions / total_completions if total_completions > 0 else 0
        )
        error_rate = error_events / total_completions if total_completions > 0 else 0

        # Error analysis by agent
        agent_errors = {}
        for event in events:
            if event.event_type == EventType.AGENT_ERROR:
                agent = event.agent_name
                agent_errors[agent] = agent_errors.get(agent, 0) + 1

        # Mean time between failures (MTBF)
        mtbf = self._calculate_mtbf(events)

        return {
            "success_rate": success_rate,
            "error_rate": error_rate,
            "total_completions": total_completions,
            "successful_completions": successful_completions,
            "error_events": error_events,
            "mtbf_hours": mtbf,
            "errors_by_agent": agent_errors,
        }

    def _generate_recommendations(self, events: list[MonitoringEvent]) -> list[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Performance recommendations
        durations = [e.duration_ms for e in events if e.duration_ms is not None]
        if durations:
            avg_duration = statistics.mean(durations)
            if avg_duration > 5000:  # 5 seconds
                recommendations.append(
                    "Consider optimizing agent response time - average duration is over 5 seconds"
                )

            p95_duration = self._percentile(durations, 95)
            if p95_duration > 10000:  # 10 seconds
                recommendations.append(
                    "95th percentile response time is high - investigate long-running operations"
                )

        # Cost recommendations
        cost_events = [e for e in events if e.cost_usd is not None]
        if cost_events:
            total_cost = sum(e.cost_usd for e in cost_events)
            if total_cost > 100:  # $100
                recommendations.append(
                    "High costs detected - consider optimizing token usage or using cheaper models"
                )

        # Error rate recommendations
        error_events = [e for e in events if e.event_type == EventType.AGENT_ERROR]
        total_events = len(events)
        if total_events > 0:
            error_rate = len(error_events) / total_events
            if error_rate > 0.1:  # 10%
                recommendations.append(
                    "High error rate detected - investigate and fix common failure modes"
                )

        # Usage pattern recommendations
        agent_usage = {}
        for event in events:
            agent = event.agent_name
            agent_usage[agent] = agent_usage.get(agent, 0) + 1

        if len(agent_usage) == 1:
            recommendations.append(
                "Consider diversifying agent usage for better load distribution"
            )

        # Token efficiency recommendations
        tokens = [e.tokens_used for e in events if e.tokens_used is not None]
        if tokens:
            avg_tokens = statistics.mean(tokens)
            if avg_tokens > 1500:
                recommendations.append(
                    "High token usage detected - consider prompt optimization"
                )

        return recommendations[:5]  # Return top 5 recommendations

    def _calculate_throughput(self, events: list[MonitoringEvent]) -> dict[str, float]:
        """Calculate throughput metrics."""
        if not events:
            return {}

        timestamps = [e.timestamp for e in events]
        time_span = (max(timestamps) - min(timestamps)).total_seconds()

        if time_span == 0:
            return {"events_per_second": 0, "events_per_hour": 0}

        events_per_second = len(events) / time_span
        events_per_hour = events_per_second * 3600

        return {
            "events_per_second": events_per_second,
            "events_per_hour": events_per_hour,
        }

    def _calculate_cost_trend(
        self, cost_events: list[MonitoringEvent]
    ) -> list[dict[str, Any]]:
        """Calculate cost trend over time."""
        if not cost_events:
            return []

        # Group events by day
        daily_costs = {}
        for event in cost_events:
            date = event.timestamp.date()
            if date not in daily_costs:
                daily_costs[date] = 0
            daily_costs[date] += event.cost_usd

        # Convert to trend format
        trend = []
        for date, cost in sorted(daily_costs.items()):
            trend.append({"date": date.isoformat(), "cost": cost})

        return trend

    def _calculate_mtbf(self, events: list[MonitoringEvent]) -> float:
        """Calculate Mean Time Between Failures."""
        error_events = [e for e in events if e.event_type == EventType.AGENT_ERROR]

        if len(error_events) < 2:
            return float("inf")  # No failures or only one failure

        # Calculate time between consecutive errors
        error_times = [e.timestamp for e in error_events]
        error_times.sort()

        intervals = []
        for i in range(1, len(error_times)):
            interval = (
                error_times[i] - error_times[i - 1]
            ).total_seconds() / 3600  # Convert to hours
            intervals.append(interval)

        return statistics.mean(intervals) if intervals else float("inf")

    def _percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile of a dataset."""
        if not data:
            return 0

        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (percentile / 100)
        f = int(k)
        c = k - f

        if f + 1 < len(sorted_data):
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
        else:
            return sorted_data[f]

    def generate_report(self, agent_name: str | None = None, days: int = 7) -> str:
        """Generate a comprehensive text report."""
        analysis = self.analyze_performance(agent_name, days)

        if "error" in analysis:
            return f"Performance Report Error: {analysis['error']}"

        report = []
        report.append("🤖 AI Agent Performance Report")
        report.append("=" * 40)
        report.append("")

        # Summary
        summary = analysis["summary"]
        report.append(f"📊 Summary ({days} days)")
        report.append(f"• Total Events: {summary['total_events']}")
        report.append(f"• Unique Agents: {summary['unique_agents']}")
        report.append(
            f"• Time Range: {summary['time_range']['duration_hours']:.1f} hours"
        )
        report.append("")

        # Performance
        perf = analysis["performance_metrics"]
        if "avg_duration_ms" in perf["duration_stats"]:
            report.append("⚡ Performance Metrics")
            report.append(
                f"• Average Duration: {perf['duration_stats']['avg_duration_ms']:.0f}ms"
            )
            report.append(
                f"• 95th Percentile: {perf['duration_stats']['p95_duration_ms']:.0f}ms"
            )

        if "total_tokens" in perf["token_stats"]:
            report.append(f"• Total Tokens: {perf['token_stats']['total_tokens']:,}")
            report.append(f"• Average Tokens: {perf['token_stats']['avg_tokens']:.0f}")
        report.append("")

        # Costs
        costs = analysis["cost_analysis"]
        if "total_cost" in costs:
            report.append("💰 Cost Analysis")
            report.append(f"• Total Cost: ${costs['total_cost']:.2f}")
            report.append(f"• Cost per Success: ${costs['cost_per_success']:.3f}")
            report.append("")

        # Reliability
        reliability = analysis["reliability_metrics"]
        report.append("🛡️ Reliability")
        report.append(f"• Success Rate: {reliability['success_rate']:.1%}")
        report.append(f"• Error Rate: {reliability['error_rate']:.1%}")
        report.append("")

        # Recommendations
        recommendations = analysis["recommendations"]
        if recommendations:
            report.append("🎯 Recommendations")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")

        return "\n".join(report)
