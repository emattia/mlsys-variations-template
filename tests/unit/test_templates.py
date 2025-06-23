"""Unit tests for the template management system."""

from unittest.mock import patch
import tempfile
import yaml
from pathlib import Path
import pytest
import os

from src.utils.templates import PromptVersion, PromptTestResult, TemplateManager


class TestPromptVersion:
    """Test PromptVersion dataclass."""

    def test_prompt_version_creation(self):
        """Test PromptVersion creation with required fields."""
        version = PromptVersion(
            template="Test template: {variable}",
            version="v1.0",
            created="2024-01-01",
            description="Test prompt",
        )
        assert version.template == "Test template: {variable}"
        assert version.version == "v1.0"
        assert version.description == "Test prompt"
        assert version.parameters == []
        assert version.performance_metrics == {}
        assert version.usage_count == 0

    def test_prompt_version_with_optional_fields(self):
        """Test PromptVersion with optional fields."""
        version = PromptVersion(
            template="Test template",
            version="v1.0",
            created="2024-01-01",
            description="Test prompt",
            parameters=["variable1", "variable2"],
            performance_metrics={"accuracy": 0.95},
            usage_count=10,
        )
        assert version.parameters == ["variable1", "variable2"]
        assert version.performance_metrics == {"accuracy": 0.95}
        assert version.usage_count == 10


class TestPromptTestResult:
    """Test PromptTestResult dataclass."""

    def test_prompt_test_result_creation(self):
        """Test PromptTestResult creation."""
        result = PromptTestResult(
            prompt_name="test_prompt",
            version="v1.0",
            test_input={"variable": "test"},
            output="Test output",
            metrics={"length": 11},
            timestamp="2024-01-01T00:00:00",
            success=True,
        )
        assert result.prompt_name == "test_prompt"
        assert result.version == "v1.0"
        assert result.test_input == {"variable": "test"}
        assert result.output == "Test output"
        assert result.metrics == {"length": 11}
        assert result.success is True


class TestTemplateManager:
    """Test TemplateManager class."""

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "prompts": {
                    "v1": {
                        "test_prompt": {
                            "template": "Hello {name}, how are you?",
                            "version": "1.0.0",
                            "created": "2024-01-01",
                            "description": "Test greeting prompt",
                            "usage_count": 0,
                            "performance_metrics": {},
                        }
                    }
                },
                "metadata": {
                    "default_version": "v1",
                    "versioning_strategy": "semantic",
                },
            }
            yaml.dump(config, f)
            yield Path(f.name)
            Path(f.name).unlink()

    @pytest.fixture
    def template_manager(self, temp_config_file):
        """Template manager instance with temp config."""
        return TemplateManager(config_path=temp_config_file)

    @pytest.fixture
    def empty_template_manager(self):
        """Template manager with non-existent config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nonexistent.yaml"
            yield TemplateManager(config_path=config_path)

    def test_template_manager_initialization(self, temp_config_file):
        """Test template manager initialization."""
        manager = TemplateManager(config_path=temp_config_file)
        assert manager.config_path == temp_config_file
        assert "prompts" in manager.templates
        assert "metadata" in manager.templates
        assert manager.active_experiments == {}
        assert manager.performance_log == []

    def test_initialization_with_missing_config(self, empty_template_manager):
        """Test initialization with missing config file."""
        manager = empty_template_manager
        assert manager.templates == {"prompts": {}, "metadata": {}}

    def test_get_template_existing(self, template_manager):
        """Test getting existing template."""
        template = template_manager.get_template("test_prompt", "v1")
        assert template == "Hello {name}, how are you?"

    def test_get_template_nonexistent(self, template_manager):
        """Test getting non-existent template."""
        template = template_manager.get_template("nonexistent", "v1")
        assert template is None

    def test_get_template_default_version(self, template_manager):
        """Test getting template with default version."""
        template = template_manager.get_template("test_prompt")
        assert template == "Hello {name}, how are you?"

    def test_render_template_success(self, template_manager):
        """Test successful template rendering."""
        rendered = template_manager.render_template(
            "test_prompt", {"name": "Alice"}, "v1"
        )
        assert rendered == "Hello Alice, how are you?"

    def test_render_template_missing_variable(self, template_manager):
        """Test template rendering with missing variable."""
        rendered = template_manager.render_template("test_prompt", {}, "v1")
        assert (
            rendered == "Hello {name}, how are you?"
        )  # safe_substitute leaves missing vars

    def test_render_template_nonexistent(self, template_manager):
        """Test rendering non-existent template."""
        rendered = template_manager.render_template("nonexistent", {"name": "Alice"})
        assert rendered is None

    def test_add_template_new(self, template_manager):
        """Test adding new template."""
        success = template_manager.add_template(
            name="new_prompt",
            template="Welcome {user}!",
            version="v1",
            description="Welcome prompt",
            parameters=["user"],
        )
        assert success is True

        # Verify template was added
        template = template_manager.get_template("new_prompt", "v1")
        assert template == "Welcome {user}!"

    def test_add_template_new_version(self, template_manager):
        """Test adding new version of existing template."""
        success = template_manager.add_template(
            name="test_prompt",
            template="Hi {name}!",
            version="v2",
            description="Shorter greeting",
        )
        assert success is True

        # Verify new version
        template_v2 = template_manager.get_template("test_prompt", "v2")
        assert template_v2 == "Hi {name}!"

        # Verify old version still exists
        template_v1 = template_manager.get_template("test_prompt", "v1")
        assert template_v1 == "Hello {name}, how are you?"

    def test_list_templates_all(self, template_manager):
        """Test listing all templates."""
        templates = template_manager.list_templates()
        assert "v1" in templates
        assert "test_prompt" in templates["v1"]

    def test_list_templates_by_version(self, template_manager):
        """Test listing templates by version."""
        templates = template_manager.list_templates(version="v1")
        assert "test_prompt" in templates
        assert len(templates) == 1

    def test_get_template_versions(self, template_manager):
        """Test getting all versions of a template."""
        # Add another version
        template_manager.add_template("test_prompt", "Hey {name}!", "v2")
        versions = template_manager.get_template_versions("test_prompt")
        assert set(versions) == {"v1", "v2"}

    def test_test_template_success(self, template_manager):
        """Test template testing with successful cases."""
        test_inputs = [{"name": "Alice"}, {"name": "Bob"}]
        results = template_manager.test_template("test_prompt", test_inputs, "v1")
        assert len(results) == 2
        assert all(result.success for result in results)
        assert results[0].output == "Hello Alice, how are you?"
        assert results[1].output == "Hello Bob, how are you?"

    def test_test_template_with_error(self, template_manager):
        """Test template testing with errors."""
        results = template_manager.test_template(
            "nonexistent", [{"name": "World"}], version="v1"
        )

        assert len(results) == 1
        assert not results[0].success
        # Check that output is not None and contains error information
        assert results[0].output is not None
        assert (
            "error" in results[0].output.lower()
            or "not found" in results[0].output.lower()
        )

    def test_compare_templates(self, template_manager):
        """Test template comparison."""
        # Add a second version first
        template_manager.add_template(
            "test_prompt", "Goodbye {name}!", "v2", "Farewell prompt"
        )

        comparison = template_manager.compare_templates("test_prompt", "v1", "v2")

        assert comparison["template_name"] == "test_prompt"
        assert comparison["version_a"] == "v1"
        assert comparison["version_b"] == "v2"
        assert "analytics_a" in comparison
        assert "analytics_b" in comparison
        assert comparison["are_identical"] is False

    def test_start_ab_test(self, template_manager):
        """Test starting an A/B test."""
        # Add second version
        template_manager.add_template("test_prompt", "Hi {name}!", "v2")

        experiment_id = template_manager.start_ab_test(
            name="test_prompt", version_a="v1", version_b="v2", traffic_split=0.5
        )

        assert experiment_id is not None
        assert experiment_id in template_manager.active_experiments
        experiment = template_manager.active_experiments[experiment_id]
        assert experiment["template_name"] == "test_prompt"
        assert experiment["version_a"] == "v1"
        assert experiment["version_b"] == "v2"
        assert experiment["traffic_split"] == 0.5

    def test_get_ab_template(self, template_manager):
        """Test getting template version for A/B test."""
        # Add second version and start test
        template_manager.add_template("test_prompt", "Hi {name}!", "v2")
        experiment_id = template_manager.start_ab_test("test_prompt", "v1", "v2", 0.5)

        # Test with deterministic user ID
        version = template_manager.get_ab_template(experiment_id, "user_123")
        assert version in ["v1", "v2"]

        # Same user should get same version
        version2 = template_manager.get_ab_template(experiment_id, "user_123")
        assert version == version2

    def test_stop_ab_test(self, template_manager):
        """Test stopping A/B test."""
        # Add second version and start A/B test
        template_manager.add_template("test_prompt", "Hi {name}!", "v2")
        experiment_id = template_manager.start_ab_test("test_prompt", "v1", "v2")

        # Stop the test
        results = template_manager.stop_ab_test(experiment_id)

        # Check the structure matches what get_ab_results returns
        assert "experiment_id" in results
        assert results["template_name"] == "test_prompt"
        assert results["version_a"] == "v1"
        assert results["version_b"] == "v2"
        assert "sample_size_a" in results
        assert "sample_size_b" in results

    def test_export_templates(self, template_manager):
        """Test template export functionality."""
        # Create a temporary file for export
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            temp_path = tmp.name

        try:
            # Export templates
            result = template_manager.export_templates(temp_path)
            assert result is True

            # Check file was created
            assert os.path.exists(temp_path)

            # Clean up temp file
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_import_templates(self, template_manager):
        """Test template import functionality."""
        import_data = {
            "prompts": {
                "v1": {
                    "imported_prompt": {
                        "template": "Imported: {text}",
                        "version": "1.0.0",
                        "created": "2024-01-02",
                        "description": "Imported prompt",
                        "usage_count": 0,
                        "performance_metrics": {},
                    }
                }
            }
        }

        success = template_manager.import_templates(import_data)
        assert success is True

        # Verify import
        template = template_manager.get_template("imported_prompt", "v1")
        assert template == "Imported: {text}"

    def test_backup_and_restore(self, template_manager):
        """Test backup and restore functionality."""
        # Create backup
        backup_path = Path(tempfile.gettempdir()) / "template_backup.yaml"
        template_manager.backup_templates(backup_path)
        assert backup_path.exists()

        # Modify templates
        template_manager.add_template("backup_test", "Test {var}", "v1")

        # Restore from backup
        template_manager.restore_templates(backup_path)

        # Verify restoration
        template = template_manager.get_template("test_prompt", "v1")
        assert template == "Hello {name}, how are you?"

        # Backup test template should be gone
        backup_template = template_manager.get_template("backup_test", "v1")
        assert backup_template is None

        # Clean up
        backup_path.unlink()

    def test_performance_analytics(self, template_manager):
        """Test performance analytics functionality."""
        # Use the template multiple times to generate usage
        template_manager.get_template("test_prompt", "v1")
        template_manager.get_template("test_prompt", "v1")

        # Get analytics
        analytics = template_manager.get_performance_analytics("test_prompt", "v1")

        # Check structure matches what get_performance_analytics returns
        assert analytics["template_name"] == "test_prompt"
        assert analytics["version"] == "v1"
        assert "usage_count" in analytics
        assert "performance_metrics" in analytics
        assert "created" in analytics

    @patch("builtins.open", side_effect=IOError("File not accessible"))
    def test_file_access_error(self, mock_open):
        """Test handling of file access errors."""
        manager = TemplateManager(config_path=Path("test.yaml"))
        # Should create default templates when file can't be accessed
        assert manager.templates == {"prompts": {}, "metadata": {}}

    def test_invalid_yaml_handling(self):
        """Test handling of invalid YAML content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            invalid_config_path = Path(f.name)

        try:
            manager = TemplateManager(config_path=invalid_config_path)
            # Should fallback to default templates
            assert manager.templates == {"prompts": {}, "metadata": {}}
        finally:
            invalid_config_path.unlink()
