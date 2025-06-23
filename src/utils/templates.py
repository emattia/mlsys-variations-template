"""Template management system for versioned prompts and configurations.

This module provides:
- Versioned prompt templates with rollback capability
- A/B testing framework for prompts
- Performance tracking and analytics
- Template validation and testing
"""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class PromptVersion:
    """Prompt version metadata."""

    template: str
    version: str
    created: str
    description: str
    parameters: List[str] = None
    performance_metrics: Dict[str, float] = None
    usage_count: int = 0

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.performance_metrics is None:
            self.performance_metrics = {}


@dataclass
class PromptTestResult:
    """Results from prompt testing."""

    prompt_name: str
    version: str
    test_input: Dict[str, Any]
    output: str
    metrics: Dict[str, float]
    timestamp: str
    success: bool


class TemplateManager:
    """Manager for versioned prompt templates."""

    def __init__(self, config_path: Union[str, Path] = "config/prompt_templates.yaml"):
        """Initialize template manager.

        Args:
            config_path: Path to prompt templates configuration
        """
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Load templates
        self.templates = self._load_templates()

        # Performance tracking
        self.performance_log = []

        # Active experiments (A/B testing)
        self.active_experiments = {}

    def _load_templates(self) -> Dict[str, Any]:
        """Load templates from configuration file."""
        if not self.config_path.exists():
            logger.warning(f"Template config not found: {self.config_path}")
            return {"prompts": {}, "metadata": {}}

        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)

            # Handle exported template format (with "templates" wrapper)
            if isinstance(data, dict) and "templates" in data:
                return data["templates"]

            # Handle regular template format
            return data if data else {"prompts": {}, "metadata": {}}

        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            return {"prompts": {}, "metadata": {}}

    def _save_templates(self) -> None:
        """Save templates to configuration file."""
        try:
            with open(self.config_path, "w") as f:
                yaml.dump(self.templates, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.error(f"Error saving templates: {e}")

    def get_template(self, name: str, version: Optional[str] = None) -> Optional[str]:
        """Get a template by name and version.

        Args:
            name: Template name
            version: Version (if None, uses default)

        Returns:
            Template string or None if not found
        """
        if "prompts" not in self.templates:
            return None

        prompts = self.templates["prompts"]

        if version is None:
            # Use default version
            version = self.templates.get("metadata", {}).get("default_version", "v1")

        if version in prompts and name in prompts[version]:
            template_data = prompts[version][name]
            if isinstance(template_data, dict):
                # Update usage count
                template_data["usage_count"] = template_data.get("usage_count", 0) + 1
                self._save_templates()
                return template_data.get("template")
            else:
                return template_data

        return None

    def render_template(
        self, name: str, variables: Dict[str, Any], version: Optional[str] = None
    ) -> Optional[str]:
        """Render a template with variables.

        Args:
            name: Template name
            variables: Variables to substitute
            version: Template version

        Returns:
            Rendered template or None if not found
        """
        template_str = self.get_template(name, version)
        if template_str is None:
            logger.error(f"Template not found: {name} (version: {version})")
            return None

        try:
            return template_str.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing variable in template {name}: {e}")
            # Return template with missing variables as-is for safe_substitute behavior
            return template_str
        except Exception as e:
            logger.error(f"Error rendering template {name}: {e}")
            return None

    def add_template(
        self,
        name: str,
        template: str,
        version: str = "v1",
        description: str = "",
        parameters: List[str] = None,
    ) -> bool:
        """Add a new template version.

        Args:
            name: Template name
            template: Template string
            version: Version identifier
            description: Template description
            parameters: Required parameters

        Returns:
            True if successful
        """
        try:
            if "prompts" not in self.templates:
                self.templates["prompts"] = {}

            if version not in self.templates["prompts"]:
                self.templates["prompts"][version] = {}

            self.templates["prompts"][version][name] = {
                "template": template,
                "version": version,
                "created": datetime.now().isoformat(),
                "description": description,
                "parameters": parameters or [],
                "usage_count": 0,
                "performance_metrics": {},
            }

            self._save_templates()
            logger.info(f"Added template: {name} (version: {version})")
            return True

        except Exception as e:
            logger.error(f"Error adding template: {e}")
            return False

    def list_templates(
        self, version: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """List all templates.

        Args:
            version: Filter by version (None for all versions)

        Returns:
            Dictionary of templates
        """
        if "prompts" not in self.templates:
            return {}

        if version:
            return self.templates["prompts"].get(version, {})
        else:
            return self.templates["prompts"]

    def get_template_versions(self, name: str) -> List[str]:
        """Get all versions of a template.

        Args:
            name: Template name

        Returns:
            List of version identifiers
        """
        versions = []
        for version, templates in self.templates.get("prompts", {}).items():
            if name in templates:
                versions.append(version)
        return sorted(versions)

    def test_template(
        self,
        name: str,
        test_inputs: List[Dict[str, Any]],
        version: Optional[str] = None,
    ) -> List[PromptTestResult]:
        """Test a template with multiple inputs.

        Args:
            name: Template name
            test_inputs: List of test input dictionaries
            version: Template version

        Returns:
            List of test results
        """
        results = []

        for test_input in test_inputs:
            try:
                rendered = self.render_template(name, test_input, version)

                result = PromptTestResult(
                    prompt_name=name,
                    version=version or "default",
                    test_input=test_input,
                    output=rendered if rendered is not None else "Template not found",
                    metrics={"length": len(rendered) if rendered else 0},
                    timestamp=datetime.now().isoformat(),
                    success=rendered is not None,
                )

                results.append(result)

            except Exception as e:
                result = PromptTestResult(
                    prompt_name=name,
                    version=version or "default",
                    test_input=test_input,
                    output=f"Error: {str(e)}",
                    metrics={},
                    timestamp=datetime.now().isoformat(),
                    success=False,
                )
                results.append(result)

        return results

    def rollback_template(self, name: str, to_version: str) -> bool:
        """Rollback a template to a previous version.

        Args:
            name: Template name
            to_version: Version to rollback to

        Returns:
            True if successful
        """
        try:
            # Check if target version exists
            if to_version not in self.templates.get("prompts", {}):
                logger.error(f"Version {to_version} not found")
                return False

            if name not in self.templates["prompts"][to_version]:
                logger.error(f"Template {name} not found in version {to_version}")
                return False

            # Set as default version
            if "metadata" not in self.templates:
                self.templates["metadata"] = {}

            old_default = self.templates["metadata"].get("default_version", "v1")
            self.templates["metadata"]["default_version"] = to_version

            self._save_templates()

            logger.info(f"Rolled back {name} from {old_default} to {to_version}")
            return True

        except Exception as e:
            logger.error(f"Error rolling back template: {e}")
            return False

    def start_ab_test(
        self, name: str, version_a: str, version_b: str, traffic_split: float = 0.5
    ) -> str:
        """Start A/B test between two template versions.

        Args:
            name: Template name
            version_a: First version
            version_b: Second version
            traffic_split: Fraction of traffic for version A (0.0-1.0)

        Returns:
            Experiment ID
        """
        import uuid

        experiment_id = str(uuid.uuid4())[:8]

        self.active_experiments[experiment_id] = {
            "template_name": name,
            "version_a": version_a,
            "version_b": version_b,
            "traffic_split": traffic_split,
            "started_at": datetime.now().isoformat(),
            "results_a": [],
            "results_b": [],
        }

        logger.info(
            f"Started A/B test {experiment_id}: {name} ({version_a} vs {version_b})"
        )
        return experiment_id

    def get_ab_template(self, experiment_id: str, user_id: str = None) -> Optional[str]:
        """Get template version based on A/B test assignment.

        Args:
            experiment_id: Experiment ID
            user_id: User identifier for consistent assignment

        Returns:
            Template version to use
        """
        if experiment_id not in self.active_experiments:
            return None

        experiment = self.active_experiments[experiment_id]

        # Deterministic assignment based on user_id
        if user_id:
            import hashlib

            hash_val = int(hashlib.md5(user_id.encode(), usedforsecurity=False).hexdigest(), 16)
            use_version_a = (hash_val % 100) < (experiment["traffic_split"] * 100)
        else:
            import random

            use_version_a = random.random() < experiment["traffic_split"]

        return experiment["version_a"] if use_version_a else experiment["version_b"]

    def record_ab_result(
        self, experiment_id: str, version: str, metrics: Dict[str, float]
    ) -> None:
        """Record A/B test results.

        Args:
            experiment_id: Experiment ID
            version: Version used
            metrics: Performance metrics
        """
        if experiment_id not in self.active_experiments:
            return

        experiment = self.active_experiments[experiment_id]

        result = {"timestamp": datetime.now().isoformat(), "metrics": metrics}

        if version == experiment["version_a"]:
            experiment["results_a"].append(result)
        elif version == experiment["version_b"]:
            experiment["results_b"].append(result)

    def get_ab_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get A/B test results summary.

        Args:
            experiment_id: Experiment ID

        Returns:
            Results summary
        """
        if experiment_id not in self.active_experiments:
            return None

        experiment = self.active_experiments[experiment_id]

        def calculate_avg_metrics(results):
            if not results:
                return {}

            all_metrics = {}
            for result in results:
                for metric, value in result["metrics"].items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)

            return {
                metric: sum(values) / len(values)
                for metric, values in all_metrics.items()
            }

        return {
            "experiment_id": experiment_id,
            "template_name": experiment["template_name"],
            "version_a": experiment["version_a"],
            "version_b": experiment["version_b"],
            "started_at": experiment["started_at"],
            "sample_size_a": len(experiment["results_a"]),
            "sample_size_b": len(experiment["results_b"]),
            "avg_metrics_a": calculate_avg_metrics(experiment["results_a"]),
            "avg_metrics_b": calculate_avg_metrics(experiment["results_b"]),
        }

    def get_template_analytics(self, name: str) -> Dict[str, Any]:
        """Get analytics for a template across all versions.

        Args:
            name: Template name

        Returns:
            Analytics data
        """
        analytics = {
            "template_name": name,
            "versions": {},
            "total_usage": 0,
            "most_used_version": None,
        }

        max_usage = 0

        for version, templates in self.templates.get("prompts", {}).items():
            if name in templates:
                template_data = templates[name]
                if isinstance(template_data, dict):
                    usage_count = template_data.get("usage_count", 0)
                    analytics["versions"][version] = {
                        "usage_count": usage_count,
                        "created": template_data.get("created"),
                        "description": template_data.get("description", ""),
                        "performance_metrics": template_data.get(
                            "performance_metrics", {}
                        ),
                    }

                    analytics["total_usage"] += usage_count

                    if usage_count > max_usage:
                        max_usage = usage_count
                        analytics["most_used_version"] = version

        return analytics

    def stop_ab_test(self, experiment_id: str) -> Dict[str, Any]:
        """Stop an A/B test and return final results.

        Args:
            experiment_id: Experiment ID

        Returns:
            Final results summary
        """
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Get final results
        results = self.get_ab_results(experiment_id)

        # Remove from active experiments
        del self.active_experiments[experiment_id]

        logger.info(f"Stopped A/B test {experiment_id}")
        return results

    def compare_templates(
        self, name: str, version_a: str, version_b: str
    ) -> Dict[str, Any]:
        """Compare two template versions.

        Args:
            name: Template name
            version_a: First version
            version_b: Second version

        Returns:
            Comparison results
        """
        template_a = self.get_template(name, version_a)
        template_b = self.get_template(name, version_b)

        if template_a is None or template_b is None:
            raise ValueError("One or both template versions not found")

        # Basic comparison
        comparison = {
            "template_name": name,
            "version_a": version_a,
            "version_b": version_b,
            "template_a": template_a,
            "template_b": template_b,
            "length_diff": len(template_b) - len(template_a),
            "character_diff": abs(len(template_b) - len(template_a)),
            "are_identical": template_a == template_b,
        }

        # Get analytics for each version
        analytics_a = (
            self.get_template_analytics(name).get("versions", {}).get(version_a, {})
        )
        analytics_b = (
            self.get_template_analytics(name).get("versions", {}).get(version_b, {})
        )

        comparison["analytics_a"] = analytics_a
        comparison["analytics_b"] = analytics_b

        return comparison

    def import_templates(self, data: Dict[str, Any]) -> bool:
        """Import templates from data.

        Args:
            data: Template data to import

        Returns:
            True if successful
        """
        try:
            if "prompts" not in data:
                raise ValueError("Invalid template data format")

            # Merge imported templates
            for version, templates in data["prompts"].items():
                if version not in self.templates.get("prompts", {}):
                    if "prompts" not in self.templates:
                        self.templates["prompts"] = {}
                    self.templates["prompts"][version] = {}

                for template_name, template_data in templates.items():
                    self.templates["prompts"][version][template_name] = template_data

            # Merge metadata
            if "metadata" in data:
                if "metadata" not in self.templates:
                    self.templates["metadata"] = {}
                self.templates["metadata"].update(data["metadata"])

            self._save_templates()
            logger.info("Templates imported successfully")
            return True

        except Exception as e:
            logger.error(f"Error importing templates: {e}")
            return False

    def backup_templates(self, backup_path: Union[str, Path]) -> bool:
        """Create a backup of all templates.

        Args:
            backup_path: Path to save backup

        Returns:
            True if successful
        """
        try:
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            with open(backup_path, "w") as f:
                yaml.dump(self.templates, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Templates backed up to {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error backing up templates: {e}")
            return False

    def restore_templates(self, backup_path: Union[str, Path]) -> bool:
        """Restore templates from a backup file.

        Args:
            backup_path: Path to backup file

        Returns:
            True if successful
        """
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")

            with open(backup_path, "r") as f:
                backup_data = yaml.safe_load(f)

            # Restore templates
            self.templates = backup_data
            self._save_templates()

            logger.info(f"Templates restored from {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error restoring templates: {e}")
            return False

    def get_performance_analytics(self, name: str, version: str) -> Dict[str, Any]:
        """Get performance analytics for a specific template version.

        Args:
            name: Template name
            version: Template version

        Returns:
            Performance analytics
        """
        template_data = self.templates.get("prompts", {}).get(version, {}).get(name)

        if template_data is None:
            raise ValueError(f"Template {name} version {version} not found")

        if isinstance(template_data, str):
            # Simple string template
            return {
                "template_name": name,
                "version": version,
                "usage_count": 0,
                "performance_metrics": {},
                "created": None,
            }

        return {
            "template_name": name,
            "version": version,
            "usage_count": template_data.get("usage_count", 0),
            "performance_metrics": template_data.get("performance_metrics", {}),
            "created": template_data.get("created"),
            "description": template_data.get("description", ""),
        }

    def export_templates(self, output_path: Union[str, Path]) -> bool:
        """Export templates to a file.

        Args:
            output_path: Output file path

        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)

            export_data = {
                "exported_at": datetime.now().isoformat(),
                "templates": self.templates,
            }

            if output_path.suffix.lower() == ".json":
                with open(output_path, "w") as f:
                    json.dump(export_data, f, indent=2)
            else:
                with open(output_path, "w") as f:
                    yaml.dump(export_data, f, default_flow_style=False)

            logger.info(f"Exported templates to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting templates: {e}")
            return False


# Global template manager instance
_template_manager = None


def get_template_manager() -> TemplateManager:
    """Get global template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager


def render_prompt(
    name: str, variables: Dict[str, Any], version: Optional[str] = None
) -> Optional[str]:
    """Convenience function to render a prompt template.

    Args:
        name: Template name
        variables: Variables to substitute
        version: Template version

    Returns:
        Rendered prompt
    """
    manager = get_template_manager()
    return manager.render_template(name, variables, version)
