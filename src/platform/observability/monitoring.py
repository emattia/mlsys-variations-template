"""Advanced monitoring and alerting for the MLX platform.

This module provides:
- Model performance monitoring and drift detection
- System resource monitoring
- Alert management and notification
- SLA tracking and reporting
- Anomaly detection for ML systems
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from src.platform.observability.metrics import get_metrics_collector, get_ml_metrics

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Represents an alert."""

    id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    value: float | None = None
    threshold: float | None = None


@dataclass
class AlertRule:
    """Defines an alert rule."""

    name: str
    description: str
    condition: Callable[[dict[str, Any]], bool]
    severity: AlertSeverity
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    cooldown_seconds: int = 300  # 5 minutes
    enabled: bool = True


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self):
        self.active_alerts: dict[str, Alert] = {}
        self.alert_rules: dict[str, AlertRule] = {}
        self.alert_history: list[Alert] = []
        self.notification_handlers: list[Callable[[Alert], None]] = []
        self.logger = logging.getLogger(__name__)
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Set up default alert rules."""
        # System resource alerts
        self.add_rule(
            AlertRule(
                name="high_memory_usage",
                description="Memory usage is above 90%",
                condition=lambda metrics: metrics.get("memory_usage_percent", 0) > 90,
                severity=AlertSeverity.WARNING,
                labels={"type": "system", "resource": "memory"},
            )
        )

        self.add_rule(
            AlertRule(
                name="high_cpu_usage",
                description="CPU usage is above 95%",
                condition=lambda metrics: metrics.get("cpu_usage_percent", 0) > 95,
                severity=AlertSeverity.CRITICAL,
                labels={"type": "system", "resource": "cpu"},
            )
        )

        # ML-specific alerts
        self.add_rule(
            AlertRule(
                name="model_accuracy_drop",
                description="Model accuracy dropped below acceptable threshold",
                condition=lambda metrics: metrics.get("model_accuracy", 1.0) < 0.8,
                severity=AlertSeverity.WARNING,
                labels={"type": "ml", "metric": "accuracy"},
            )
        )

        self.add_rule(
            AlertRule(
                name="high_prediction_latency",
                description="Model prediction latency is too high",
                condition=lambda metrics: metrics.get("prediction_latency_ms", 0)
                > 1000,
                severity=AlertSeverity.WARNING,
                labels={"type": "ml", "metric": "latency"},
            )
        )

        self.add_rule(
            AlertRule(
                name="data_drift_detected",
                description="Significant data drift detected",
                condition=lambda metrics: metrics.get("data_drift_score", 0) > 0.8,
                severity=AlertSeverity.WARNING,
                labels={"type": "ml", "metric": "drift"},
            )
        )

    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler."""
        self.notification_handlers.append(handler)

    def check_alerts(self, metrics: dict[str, Any]):
        """Check all alert rules against current metrics."""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue

            try:
                if rule.condition(metrics):
                    self._trigger_alert(rule, metrics)
                else:
                    self._resolve_alert(rule_name)
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule_name}: {e}")

    def _trigger_alert(self, rule: AlertRule, metrics: dict[str, Any]):
        """Trigger an alert."""
        alert_id = f"{rule.name}_{int(time.time())}"

        # Check if alert is already active and within cooldown
        if rule.name in self.active_alerts:
            existing_alert = self.active_alerts[rule.name]
            cooldown_end = existing_alert.triggered_at + timedelta(
                seconds=rule.cooldown_seconds
            )
            if datetime.utcnow() < cooldown_end:
                return  # Still in cooldown

        alert = Alert(
            id=alert_id,
            name=rule.name,
            description=rule.description,
            severity=rule.severity,
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy(),
        )

        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)

        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error sending notification: {e}")

        self.logger.warning(f"Alert triggered: {alert.name} - {alert.description}")

    def _resolve_alert(self, rule_name: str):
        """Resolve an active alert."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()

            del self.active_alerts[rule_name]

            # Send resolution notification
            for handler in self.notification_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Error sending resolution notification: {e}")

            self.logger.info(f"Alert resolved: {alert.name}")

    def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> list[Alert]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history if alert.triggered_at >= cutoff_time
        ]


class Monitor(ABC):
    """Abstract base class for monitors."""

    def __init__(self, name: str, check_interval: int = 60):
        self.name = name
        self.check_interval = check_interval
        self.last_check = None
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.metrics_collector = get_metrics_collector()

    @abstractmethod
    def collect_metrics(self) -> dict[str, Any]:
        """Collect metrics for monitoring."""
        pass

    @abstractmethod
    def check_health(self) -> bool:
        """Check if the monitored component is healthy."""
        pass

    def should_check(self) -> bool:
        """Check if it's time to run the monitor."""
        if self.last_check is None:
            return True
        return time.time() - self.last_check >= self.check_interval

    def run_check(self) -> dict[str, Any]:
        """Run a monitoring check."""
        if not self.should_check():
            return {}

        self.last_check = time.time()

        try:
            metrics = self.collect_metrics()
            health = self.check_health()

            metrics["health"] = health
            metrics["monitor_name"] = self.name
            metrics["check_timestamp"] = self.last_check

            return metrics
        except Exception as e:
            self.logger.error(f"Monitor check failed: {e}")
            return {"health": False, "error": str(e)}


class SystemMonitor(Monitor):
    """Monitor system resources and performance."""

    def __init__(self):
        super().__init__("system", check_interval=30)
        self._setup_psutil()

    def _setup_psutil(self):
        """Set up psutil for system monitoring."""
        try:
            import psutil

            self.psutil = psutil
            self.psutil_available = True
        except ImportError:
            self.logger.warning("psutil not available, system monitoring limited")
            self.psutil_available = False

    def collect_metrics(self) -> dict[str, Any]:
        """Collect system metrics."""
        metrics = {}

        if not self.psutil_available:
            return metrics

        try:
            # CPU metrics
            cpu_percent = self.psutil.cpu_percent(interval=1)
            metrics["cpu_usage_percent"] = cpu_percent
            self.metrics_collector.set_gauge(
                "mlx_cpu_usage_percent", cpu_percent, {"core": "total"}
            )

            # Memory metrics
            memory = self.psutil.virtual_memory()
            metrics["memory_usage_percent"] = memory.percent
            metrics["memory_available_bytes"] = memory.available
            metrics["memory_used_bytes"] = memory.used

            self.metrics_collector.set_gauge(
                "mlx_memory_usage_bytes", memory.used, {"type": "used"}
            )
            self.metrics_collector.set_gauge(
                "mlx_memory_usage_bytes", memory.available, {"type": "available"}
            )

            # Disk metrics
            disk = self.psutil.disk_usage("/")
            metrics["disk_usage_percent"] = (disk.used / disk.total) * 100
            metrics["disk_free_bytes"] = disk.free

            # Network metrics
            network = self.psutil.net_io_counters()
            metrics["network_bytes_sent"] = network.bytes_sent
            metrics["network_bytes_recv"] = network.bytes_recv

        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")

        return metrics

    def check_health(self) -> bool:
        """Check system health."""
        if not self.psutil_available:
            return True  # Assume healthy if we can't check

        try:
            cpu_percent = self.psutil.cpu_percent()
            memory_percent = self.psutil.virtual_memory().percent

            # Consider unhealthy if CPU > 95% or memory > 95%
            return cpu_percent < 95 and memory_percent < 95
        except Exception:
            return False


class ModelMonitor(Monitor):
    """Monitor ML model performance and behavior."""

    def __init__(self, model_id: str, model_version: str = "latest"):
        super().__init__(f"model_{model_id}", check_interval=300)  # 5 minutes
        self.model_id = model_id
        self.model_version = model_version
        self.ml_metrics = get_ml_metrics()
        self.baseline_metrics = {}
        self.drift_threshold = 0.8

    def collect_metrics(self) -> dict[str, Any]:
        """Collect model performance metrics."""
        metrics = {
            "model_id": self.model_id,
            "model_version": self.model_version,
        }

        # This would typically query your model registry or monitoring database
        # For now, we'll simulate some metrics

        try:
            # Simulate collecting metrics from model monitoring system
            metrics.update(
                {
                    "prediction_count": 1000,  # Would come from actual metrics
                    "avg_prediction_latency_ms": 50,
                    "accuracy_score": 0.85,
                    "precision_score": 0.82,
                    "recall_score": 0.88,
                    "f1_score": 0.85,
                    "data_drift_score": 0.1,
                    "model_drift_score": 0.05,
                }
            )

            # Update Prometheus metrics
            self.ml_metrics.record_prediction_metrics(
                self.model_id,
                self.model_version,
                metrics["avg_prediction_latency_ms"] / 1000,
                metrics["prediction_count"],
            )

            self.ml_metrics.record_drift_metrics(
                model_id=self.model_id,
                model_version=self.model_version,
                drift_score=metrics["model_drift_score"],
                drift_type="model",
            )

        except Exception as e:
            self.logger.error(f"Failed to collect model metrics: {e}")

        return metrics

    def check_health(self) -> bool:
        """Check model health based on performance metrics."""
        try:
            metrics = self.collect_metrics()

            # Check various health indicators
            accuracy_ok = metrics.get("accuracy_score", 0) > 0.7
            latency_ok = metrics.get("avg_prediction_latency_ms", 0) < 1000
            drift_ok = metrics.get("data_drift_score", 0) < self.drift_threshold

            return accuracy_ok and latency_ok and drift_ok
        except Exception:
            return False

    def detect_drift(self, current_metrics: dict[str, Any]) -> dict[str, float]:
        """Detect drift in model behavior."""
        drift_scores = {}

        if not self.baseline_metrics:
            self.baseline_metrics = current_metrics.copy()
            return drift_scores

        # Compare current metrics with baseline
        for metric_name in ["accuracy_score", "precision_score", "recall_score"]:
            if metric_name in current_metrics and metric_name in self.baseline_metrics:
                current_value = current_metrics[metric_name]
                baseline_value = self.baseline_metrics[metric_name]

                # Calculate relative change
                if baseline_value > 0:
                    relative_change = (
                        abs(current_value - baseline_value) / baseline_value
                    )
                    drift_scores[metric_name] = relative_change

        return drift_scores


class MonitoringService:
    """Central monitoring service that orchestrates all monitors."""

    def __init__(self):
        self.monitors: dict[str, Monitor] = {}
        self.alert_manager = AlertManager()
        self.logger = logging.getLogger(__name__)
        self.running = False

        # Set up default monitors
        self.add_monitor(SystemMonitor())

        # Set up default notification handlers
        self.alert_manager.add_notification_handler(self._log_alert)

    def add_monitor(self, monitor: Monitor):
        """Add a monitor to the service."""
        self.monitors[monitor.name] = monitor
        self.logger.info(f"Added monitor: {monitor.name}")

    def remove_monitor(self, monitor_name: str):
        """Remove a monitor from the service."""
        if monitor_name in self.monitors:
            del self.monitors[monitor_name]
            self.logger.info(f"Removed monitor: {monitor_name}")

    def add_model_monitor(self, model_id: str, model_version: str = "latest"):
        """Add a model monitor."""
        monitor = ModelMonitor(model_id, model_version)
        self.add_monitor(monitor)

    def run_all_checks(self) -> dict[str, dict[str, Any]]:
        """Run all monitoring checks."""
        all_metrics = {}

        for monitor_name, monitor in self.monitors.items():
            try:
                metrics = monitor.run_check()
                if metrics:
                    all_metrics[monitor_name] = metrics

                    # Check alerts for this monitor's metrics
                    self.alert_manager.check_alerts(metrics)

            except Exception as e:
                self.logger.error(f"Monitor {monitor_name} failed: {e}")
                all_metrics[monitor_name] = {"error": str(e), "health": False}

        return all_metrics

    def get_system_status(self) -> dict[str, Any]:
        """Get overall system status."""
        all_metrics = self.run_all_checks()

        # Determine overall health
        overall_health = all(
            metrics.get("health", False) for metrics in all_metrics.values()
        )

        # Get active alerts
        active_alerts = self.alert_manager.get_active_alerts()

        return {
            "overall_health": overall_health,
            "monitors": all_metrics,
            "active_alerts": [
                {
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "description": alert.description,
                    "triggered_at": alert.triggered_at.isoformat(),
                }
                for alert in active_alerts
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _log_alert(self, alert: Alert):
        """Default alert notification handler that logs alerts."""
        if alert.status == AlertStatus.ACTIVE:
            self.logger.warning(
                f"ALERT: {alert.name} ({alert.severity.value}) - {alert.description}"
            )
        else:
            self.logger.info(f"RESOLVED: {alert.name} - {alert.description}")


# Global monitoring service instance
_monitoring_service = None


def get_monitoring_service() -> MonitoringService:
    """Get the global monitoring service instance."""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service


def setup_monitoring(
    enable_system_monitor: bool = True,
    model_monitors: list[dict[str, str]] | None = None,
) -> MonitoringService:
    """Set up monitoring with specified monitors."""
    service = get_monitoring_service()

    if model_monitors:
        for model_info in model_monitors:
            service.add_model_monitor(
                model_info["model_id"], model_info.get("model_version", "latest")
            )

    logger.info("Monitoring service initialized")
    return service
