"""Comprehensive unit tests for MLOps configuration management."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.platform.config.manager import ConfigManager
from src.platform.config.models import APIConfig, DetailedModelConfig


class TestConfigModels:
    """Test configuration model classes."""

    def test_api_config_creation(self):
        """Test APIConfig model creation."""
        config = APIConfig(
            host="localhost",
            port=8000,
            cors_origins=["*"],
            security={"api_key_enabled": True},
        )
        assert config.host == "localhost"
        assert config.port == 8000
        assert config.cors_origins == ["*"]

    def test_api_config_validation(self):
        """Test APIConfig validation."""
        # Valid config
        config = APIConfig(host="localhost", port=8000)
        assert config.port == 8000

        # Invalid port
        with pytest.raises(ValueError):
            APIConfig(
                host="localhost",
                port=-1,  # Invalid port
            )

    def test_model_config_creation(self):
        """Test DetailedModelConfig model creation."""
        config = DetailedModelConfig(
            name="test_model",
            type="RandomForest",
            parameters={"n_estimators": 100, "max_depth": 10},
            features=["age", "income", "category"],
        )
        assert config.name == "test_model"
        assert config.type == "RandomForest"
        assert config.parameters["n_estimators"] == 100

    def test_model_config_defaults(self):
        """Test DetailedModelConfig default values."""
        config = DetailedModelConfig(name="test_model", type="LogisticRegression")
        assert config.name == "test_model"
        assert config.type == "LogisticRegression"


class TestConfigManager:
    """Test ConfigManager functionality."""

    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        manager = ConfigManager()
        assert hasattr(manager, "_config")

    def test_load_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config_data = {
            "api": {"host": "localhost", "port": 8000},
            "model": {
                "name": "test_model",
                "type": "RandomForest",
                "parameters": {"n_estimators": 50},
            },
        }

        manager = ConfigManager()
        manager.load_from_dict(config_data)

        assert manager.get("api.host") == "localhost"
        assert manager.get("model.name") == "test_model"

    def test_load_config_from_file(self):
        """Test loading configuration from file."""
        config_data = {
            "app_name": "test_app",
            "version": "1.0.0",
            "settings": {"debug": True, "log_level": "INFO"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            manager = ConfigManager()
            manager.load_from_file(str(config_file))

            assert manager.get("app_name") == "test_app"
            assert manager.get("settings.debug") is True

    def test_get_config_value(self):
        """Test getting configuration values."""
        config_data = {
            "level1": {"level2": {"value": "test_value"}},
            "simple_value": 42,
        }

        manager = ConfigManager()
        manager.load_from_dict(config_data)

        # Test nested access
        assert manager.get("level1.level2.value") == "test_value"

        # Test simple access
        assert manager.get("simple_value") == 42

        # Test default value
        assert manager.get("nonexistent", "default") == "default"

    def test_set_config_value(self):
        """Test setting configuration values."""
        manager = ConfigManager()

        # Set simple value
        manager.set("simple_key", "simple_value")
        assert manager.get("simple_key") == "simple_value"

        # Set nested value
        manager.set("nested.key", "nested_value")
        assert manager.get("nested.key") == "nested_value"

    def test_save_config_to_file(self):
        """Test saving configuration to file."""
        config_data = {
            "app": {"name": "test_app", "version": "1.0"},
            "api": {"host": "localhost", "port": 8000},
        }

        manager = ConfigManager()
        manager.load_from_dict(config_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "saved_config.yaml"
            manager.save_to_file(str(config_file))

            assert config_file.exists()

            # Verify saved content
            with open(config_file) as f:
                saved_data = yaml.safe_load(f)
                assert saved_data["app"]["name"] == "test_app"

    def test_config_validation_integration(self):
        """Test integration with config validation."""
        valid_config = {"host": "localhost", "port": 8000}

        # Should not raise any errors
        api_config = APIConfig(**valid_config)
        assert api_config.host == "localhost"


def validate_config(config_data, schema):
    """Mock validation function for testing."""
    missing_fields = []
    type_errors = []

    for field, expected_type in schema.items():
        if field not in config_data:
            missing_fields.append(f"Missing field: {field}")
        elif not isinstance(config_data[field], expected_type):
            type_errors.append(f"Field {field} has wrong type")

    errors = missing_fields + type_errors
    return len(errors) == 0, errors


class TestConfigValidation:
    """Test configuration validation functions."""

    def test_validate_config_success(self):
        """Test successful configuration validation."""
        config_data = {
            "required_field": "value",
            "numeric_field": 42,
            "nested": {"field": "nested_value"},
        }

        schema = {"required_field": str, "numeric_field": int, "nested": dict}

        is_valid, errors = validate_config(config_data, schema)
        assert is_valid is True
        assert errors == []

    def test_validate_config_missing_field(self):
        """Test validation with missing required field."""
        config_data = {
            "field1": "value1"
            # Missing required_field
        }

        schema = {
            "field1": str,
            "required_field": str,  # This is missing
        }

        is_valid, errors = validate_config(config_data, schema)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_config_wrong_type(self):
        """Test validation with wrong field type."""
        config_data = {
            "string_field": 123,  # Should be string
            "int_field": "not_int",  # Should be int
        }

        schema = {
            "string_field": str,
            "int_field": int,
        }

        is_valid, errors = validate_config(config_data, schema)
        assert is_valid is False
        assert len(errors) >= 2

    def test_validate_nested_config(self):
        """Test validation of nested configuration."""
        config_data = {
            "api": {"host": "localhost", "port": 8000},
            "database": {"url": "postgresql://localhost"},
        }

        schema = {
            "api": dict,
            "database": dict,
        }

        is_valid, errors = validate_config(config_data, schema)
        assert is_valid is True


class TestConfigIntegration:
    """Integration tests for configuration management."""

    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        # Create configuration
        config_data = {
            "app": {
                "name": "integration_test",
                "version": "1.0.0",
                "debug": True,
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "cors_origins": ["http://localhost:3000"],
            },
            "models": {
                "default": {
                    "name": "default_model",
                    "type": "RandomForest",
                    "parameters": {"n_estimators": 100},
                }
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "integration_config.yaml"

            # Save configuration
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Load through ConfigManager
            manager = ConfigManager()
            manager.load_from_file(str(config_file))

            # Verify loaded values
            assert manager.get("app.name") == "integration_test"
            assert manager.get("api.port") == 8000
            assert manager.get("models.default.type") == "RandomForest"

            # Modify configuration
            manager.set("app.debug", False)
            manager.set("api.port", 9000)

            # Save modified configuration
            modified_file = Path(temp_dir) / "modified_config.yaml"
            manager.save_to_file(str(modified_file))

            # Load modified configuration
            manager2 = ConfigManager()
            manager2.load_from_file(str(modified_file))

            assert manager2.get("app.debug") is False
            assert manager2.get("api.port") == 9000

    def test_config_model_integration(self):
        """Test integration between config manager and models."""
        # Create config with model parameters that exist in ModelParametersConfig
        config_data = {
            "model": {
                "name": "test_model",
                "type": "classification",
                "parameters": {
                    "n_estimators": 50,  # This exists in ModelParametersConfig
                    "max_depth": 10,  # This exists in ModelParametersConfig
                    "random_state": 123,  # This exists in ModelParametersConfig
                },
            }
        }

        manager = ConfigManager()
        manager.load_from_dict(config_data)
        config = manager.get_config()

        # Test that the model configuration works
        model_config = config.model
        assert model_config.name == "test_model"
        assert model_config.type == "classification"

        # Test parameters access - use parameters that actually exist
        assert model_config.parameters["n_estimators"] == 50
        assert model_config.parameters["max_depth"] == 10
        assert model_config.parameters["random_state"] == 123
