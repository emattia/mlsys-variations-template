"""Unit tests for the template management system."""

import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

import pytest

from src.utils.templates import (
    PromptVersion,
    PromptTestResult,
    TemplateManager,
    get_template_manager,
    render_prompt
)


class TestPromptVersion:
    """Test PromptVersion dataclass."""
    
    def test_prompt_version_creation(self):
        """Test PromptVersion creation with required fields."""
        version = PromptVersion(
            template="Test template: {variable}",
            version="v1.0",
            created="2024-01-01",
            description="Test prompt"
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
            usage_count=10
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
            success=True
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                "prompts": {
                    "v1": {
                        "test_prompt": {
                            "template": "Hello {name}, how are you?",
                            "version": "1.0.0",
                            "created": "2024-01-01",
                            "description": "Test greeting prompt",
                            "usage_count": 0,
                            "performance_metrics": {}
                        }
                    }
                },
                "metadata": {
                    "default_version": "v1",
                    "versioning_strategy": "semantic"
                }
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
        rendered = template_manager.render_template("test_prompt", {"name": "Alice"}, "v1")
        assert rendered == "Hello Alice, how are you?"
    
    def test_render_template_missing_variable(self, template_manager):
        """Test template rendering with missing variable."""
        rendered = template_manager.render_template("test_prompt", {}, "v1")
        assert rendered == "Hello {name}, how are you?"  # safe_substitute leaves missing vars
    
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
            parameters=["user"]
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
            description="Shorter greeting"
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
        test_inputs = [
            {"name": "Alice"},
            {"name": "Bob"}
        ]
        
        results = template_manager.test_template("test_prompt", test_inputs, "v1")
        
        assert len(results) == 2
        assert all(result.success for result in results)
        assert results[0].output == "Hello Alice, how are you?"
        assert results[1].output == "Hello Bob, how are you?"
    
    def test_test_template_with_error(self, template_manager):
        """Test template testing with error case."""
        # Test with non-existent template
        test_inputs = [{"name": "Alice"}]
        
        results = template_manager.test_template("nonexistent", test_inputs, "v1")
        
        assert len(results) == 1
        assert not results[0].success
    
    def test_rollback_template_success(self, template_manager):
        """Test successful template rollback."""
        # Add v2 version
        template_manager.add_template("test_prompt", "Hi {name}!", "v2")
        
        # Rollback to v1
        success = template_manager.rollback_template("test_prompt", "v1")
        
        assert success is True
        assert template_manager.templates["metadata"]["default_version"] == "v1"
    
    def test_rollback_template_nonexistent_version(self, template_manager):
        """Test rollback to non-existent version."""
        success = template_manager.rollback_template("test_prompt", "v999")
        assert success is False
    
    def test_rollback_template_nonexistent_template(self, template_manager):
        """Test rollback of non-existent template."""
        success = template_manager.rollback_template("nonexistent", "v1")
        assert success is False
    
    def test_start_ab_test(self, template_manager):
        """Test starting A/B test."""
        # Add another version for testing
        template_manager.add_template("test_prompt", "Hi {name}!", "v2")
        
        experiment_id = template_manager.start_ab_test(
            name="test_prompt",
            version_a="v1",
            version_b="v2",
            traffic_split=0.6
        )
        
        assert experiment_id in template_manager.active_experiments
        experiment = template_manager.active_experiments[experiment_id]
        assert experiment["template_name"] == "test_prompt"
        assert experiment["version_a"] == "v1"
        assert experiment["version_b"] == "v2"
        assert experiment["traffic_split"] == 0.6
    
    def test_get_ab_template_deterministic(self, template_manager):
        """Test A/B test template assignment with user ID."""
        template_manager.add_template("test_prompt", "Hi {name}!", "v2")
        
        experiment_id = template_manager.start_ab_test(
            name="test_prompt",
            version_a="v1",
            version_b="v2",
            traffic_split=0.5
        )
        
        # Same user should get same version
        version1 = template_manager.get_ab_template(experiment_id, "user123")
        version2 = template_manager.get_ab_template(experiment_id, "user123")
        
        assert version1 == version2
        assert version1 in ["v1", "v2"]
    
    def test_record_ab_result(self, template_manager):
        """Test recording A/B test results."""
        template_manager.add_template("test_prompt", "Hi {name}!", "v2")
        
        experiment_id = template_manager.start_ab_test(
            name="test_prompt",
            version_a="v1",
            version_b="v2"
        )
        
        # Record results
        template_manager.record_ab_result(experiment_id, "v1", {"accuracy": 0.85})
        template_manager.record_ab_result(experiment_id, "v2", {"accuracy": 0.90})
        
        experiment = template_manager.active_experiments[experiment_id]
        assert len(experiment["results_a"]) == 1
        assert len(experiment["results_b"]) == 1
        assert experiment["results_a"][0]["metrics"]["accuracy"] == 0.85
        assert experiment["results_b"][0]["metrics"]["accuracy"] == 0.90
    
    def test_get_ab_results(self, template_manager):
        """Test getting A/B test results summary."""
        template_manager.add_template("test_prompt", "Hi {name}!", "v2")
        
        experiment_id = template_manager.start_ab_test(
            name="test_prompt",
            version_a="v1",
            version_b="v2"
        )
        
        # Record multiple results
        for i in range(3):
            template_manager.record_ab_result(experiment_id, "v1", {"accuracy": 0.8 + i * 0.05})
            template_manager.record_ab_result(experiment_id, "v2", {"accuracy": 0.85 + i * 0.05})
        
        results = template_manager.get_ab_results(experiment_id)
        
        assert results["experiment_id"] == experiment_id
        assert results["sample_size_a"] == 3
        assert results["sample_size_b"] == 3
        assert results["avg_metrics_a"]["accuracy"] == pytest.approx(0.85)  # (0.8 + 0.85 + 0.9) / 3
        assert results["avg_metrics_b"]["accuracy"] == pytest.approx(0.9)   # (0.85 + 0.9 + 0.95) / 3
    
    def test_get_template_analytics(self, template_manager):
        """Test getting template analytics."""
        # Add usage by getting templates
        template_manager.get_template("test_prompt", "v1")
        template_manager.get_template("test_prompt", "v1")
        
        # Add another version
        template_manager.add_template("test_prompt", "Hi {name}!", "v2")
        template_manager.get_template("test_prompt", "v2")
        
        analytics = template_manager.get_template_analytics("test_prompt")
        
        assert analytics["template_name"] == "test_prompt"
        assert analytics["total_usage"] == 3  # 2 from v1, 1 from v2
        assert analytics["most_used_version"] == "v1"
        assert "v1" in analytics["versions"]
        assert "v2" in analytics["versions"]
    
    def test_export_templates_yaml(self, template_manager):
        """Test exporting templates to YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            success = template_manager.export_templates(output_path)
            assert success is True
            
            # Verify exported content
            with open(output_path, 'r') as f:
                exported = yaml.safe_load(f)
            
            assert "exported_at" in exported
            assert "templates" in exported
            assert exported["templates"]["prompts"]["v1"]["test_prompt"]["template"] == "Hello {name}, how are you?"
        finally:
            output_path.unlink()
    
    def test_export_templates_json(self, template_manager):
        """Test exporting templates to JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            success = template_manager.export_templates(output_path)
            assert success is True
            
            # Verify exported content
            with open(output_path, 'r') as f:
                exported = json.load(f)
            
            assert "exported_at" in exported
            assert "templates" in exported
        finally:
            output_path.unlink()


class TestGlobalTemplateManager:
    """Test global template manager functionality."""
    
    def test_get_template_manager_singleton(self):
        """Test that get_template_manager returns singleton."""
        manager1 = get_template_manager()
        manager2 = get_template_manager()
        assert manager1 is manager2
    
    def test_render_prompt_convenience_function(self):
        """Test convenience function for rendering prompts."""
        with patch('src.utils.templates.get_template_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.render_template.return_value = "Rendered prompt"
            mock_get_manager.return_value = mock_manager
            
            result = render_prompt("test_prompt", {"var": "value"}, "v1")
            
            assert result == "Rendered prompt"
            mock_manager.render_template.assert_called_once_with("test_prompt", {"var": "value"}, "v1")


@pytest.mark.integration
class TestTemplateManagerIntegration:
    """Integration tests for template manager."""
    
    def test_full_template_lifecycle(self):
        """Test complete template management lifecycle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "templates.yaml"
            manager = TemplateManager(config_path=config_path)
            
            # 1. Add initial template
            success = manager.add_template(
                name="greeting",
                template="Hello {name}, welcome to {platform}!",
                version="v1",
                description="Initial greeting",
                parameters=["name", "platform"]
            )
            assert success is True
            
            # 2. Test template
            test_inputs = [
                {"name": "Alice", "platform": "our service"},
                {"name": "Bob", "platform": "the platform"}
            ]
            
            results = manager.test_template("greeting", test_inputs, "v1")
            assert all(result.success for result in results)
            assert "Hello Alice, welcome to our service!" in results[0].output
            
            # 3. Add improved version
            manager.add_template(
                name="greeting",
                template="Hi {name}! Welcome to {platform} 🎉",
                version="v2",
                description="Improved greeting with emoji"
            )
            
            # 4. Start A/B test
            experiment_id = manager.start_ab_test("greeting", "v1", "v2", 0.5)
            
            # 5. Simulate usage and record metrics
            for i in range(10):
                user_id = f"user_{i}"
                version = manager.get_ab_template(experiment_id, user_id)
                
                # Simulate different performance for different versions
                if version == "v1":
                    manager.record_ab_result(experiment_id, version, {"satisfaction": 0.7 + (i % 3) * 0.1})
                else:
                    manager.record_ab_result(experiment_id, version, {"satisfaction": 0.8 + (i % 3) * 0.1})
            
            # 6. Analyze results
            results = manager.get_ab_results(experiment_id)
            assert results["sample_size_a"] + results["sample_size_b"] == 10
            
            # v2 should perform better on average
            avg_a = results["avg_metrics_a"]["satisfaction"]
            avg_b = results["avg_metrics_b"]["satisfaction"]
            
            # 7. Get analytics
            analytics = manager.get_template_analytics("greeting")
            assert analytics["template_name"] == "greeting"
            assert len(analytics["versions"]) == 2
            
            # 8. Export configuration
            export_path = Path(temp_dir) / "exported_templates.yaml"
            success = manager.export_templates(export_path)
            assert success is True
            assert export_path.exists()
            
            # 9. Create new manager from exported config
            new_manager = TemplateManager(config_path=export_path)
            
            # Should be able to access templates
            template = new_manager.get_template("greeting", "v1")
            assert template == "Hello {name}, welcome to {platform}!"
    
    def test_concurrent_template_access(self):
        """Test concurrent access to templates."""
        import threading
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "templates.yaml"
            manager = TemplateManager(config_path=config_path)
            
            # Add initial template
            manager.add_template("test", "Hello {name}!", "v1")
            
            results = []
            errors = []
            
            def template_worker(worker_id):
                try:
                    for i in range(10):
                        # Render template
                        rendered = manager.render_template("test", {"name": f"User{worker_id}_{i}"})
                        if rendered:
                            results.append(rendered)
                        
                        # Add new template version occasionally
                        if i == 5:
                            manager.add_template(
                                "test",
                                f"Hi {{name}} from worker {worker_id}!",
                                f"worker_v{worker_id}"
                            )
                except Exception as e:
                    errors.append(e)
            
            # Create multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=template_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            # Check results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 30  # 3 workers × 10 renders
            
            # Check that new versions were added
            versions = manager.get_template_versions("test")
            assert len(versions) >= 4  # v1 + 3 worker versions
    
    def test_template_performance_tracking(self):
        """Test template performance tracking over time."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "templates.yaml"
            manager = TemplateManager(config_path=config_path)
            
            # Add template with performance tracking
            manager.add_template(
                name="performance_test",
                template="Process {task} with {method}",
                version="v1"
            )
            
            # Simulate usage with performance metrics
            performance_data = []
            
            for i in range(100):
                # Render template (increases usage count)
                rendered = manager.render_template(
                    "performance_test", 
                    {"task": f"task_{i}", "method": "method_a"}
                )
                
                # Simulate performance measurement
                performance_score = 0.8 + (i % 10) * 0.02  # Gradually improving
                performance_data.append(performance_score)
            
            # Update template performance metrics
            avg_performance = sum(performance_data) / len(performance_data)
            templates = manager.templates["prompts"]["v1"]["performance_test"]
            templates["performance_metrics"]["avg_score"] = avg_performance
            templates["performance_metrics"]["sample_size"] = len(performance_data)
            manager._save_templates()
            
            # Get analytics
            analytics = manager.get_template_analytics("performance_test")
            
            assert analytics["total_usage"] == 100
            assert analytics["versions"]["v1"]["performance_metrics"]["avg_score"] > 0.8
            assert analytics["versions"]["v1"]["performance_metrics"]["sample_size"] == 100 