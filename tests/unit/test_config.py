"""Unit tests for the configuration system."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import AppConfig, ConfigManager
from src.config.models import (
    DataConfig,
    LoggingConfig,
    MLConfig,
    ModelConfig,
    PathsConfig,
)


class TestPathsConfig:
    """Test PathsConfig model."""

    def test_default_paths(self):
        """Test default path configuration."""
        config = PathsConfig()

        assert config.data_raw == Path("data/raw")
        assert config.data_processed == Path("data/processed")
        assert config.models_trained == Path("models/trained")
        assert config.reports_figures == Path("reports/figures")

    def test_custom_paths(self):
        """Test custom path configuration."""
        custom_paths = {
            "data_raw": "/custom/data/raw",
            "models_trained": "/custom/models",
        }

        config = PathsConfig(**custom_paths)

        assert config.data_raw == Path("/custom/data/raw")
        assert config.models_trained == Path("/custom/models")
        # Default values should still be present
        assert config.data_processed == Path("data/processed")


class TestLoggingConfig:
    """Test LoggingConfig model."""

    def test_default_logging(self):
        """Test default logging configuration."""
        config = LoggingConfig()

        assert config.level == "INFO"
        assert config.console is True
        assert config.file is None
        assert "%(asctime)s" in config.format

    def test_custom_logging(self):
        """Test custom logging configuration."""
        custom_config = {
            "level": "debug",  # Should be converted to uppercase
            "console": False,
            "file": "/tmp/test.log",
        }

        config = LoggingConfig(**custom_config)

        assert config.level == "DEBUG"
        assert config.console is False
        assert config.file == Path("/tmp/test.log")

    def test_invalid_log_level(self):
        """Test validation of log level."""
        with pytest.raises(ValidationError):
            LoggingConfig(level="INVALID")


class TestMLConfig:
    """Test MLConfig model."""

    def test_default_ml_config(self):
        """Test default ML configuration."""
        config = MLConfig()

        assert config.random_seed == 42
        assert config.test_size == 0.2
        assert config.cv_folds == 5
        assert config.hyperparameter_search is False
        assert config.primary_metric == "accuracy"

    def test_validation_constraints(self):
        """Test validation constraints."""
        # Valid configuration
        config = MLConfig(test_size=0.3, cv_folds=10)
        assert config.test_size == 0.3
        assert config.cv_folds == 10

        # Invalid test_size (> 1.0)
        with pytest.raises(ValidationError):
            MLConfig(test_size=1.5)

        # Invalid cv_folds (< 2)
        with pytest.raises(ValidationError):
            MLConfig(cv_folds=1)


class TestModelConfig:
    """Test ModelConfig model."""

    def test_default_model_config(self):
        """Test default model configuration."""
        config = ModelConfig()

        assert config.model_type == "random_forest"
        assert config.problem_type == "classification"
        assert config.target_column == "target"
        assert config.feature_columns is None
        assert config.model_params == {}

    def test_problem_type_validation(self):
        """Test problem type validation."""
        # Valid problem types
        for problem_type in ["classification", "regression", "clustering"]:
            config = ModelConfig(problem_type=problem_type)
            assert config.problem_type == problem_type

        # Invalid problem type
        with pytest.raises(ValidationError):
            ModelConfig(problem_type="invalid")


class TestDataConfig:
    """Test DataConfig model."""

    def test_default_data_config(self):
        """Test default data configuration."""
        config = DataConfig()

        assert config.file_format == "csv"
        assert config.encoding == "utf-8"
        assert config.separator == ","
        assert config.missing_value_strategy == "drop"
        assert config.scaling_method == "standard"
        assert config.validate_schema is True

    def test_strategy_validation(self):
        """Test validation of strategy fields."""
        # Valid missing value strategy
        config = DataConfig(missing_value_strategy="mean")
        assert config.missing_value_strategy == "mean"

        # Invalid missing value strategy
        with pytest.raises(ValidationError):
            DataConfig(missing_value_strategy="invalid")

        # Valid scaling method
        config = DataConfig(scaling_method="minmax")
        assert config.scaling_method == "minmax"

        # Invalid scaling method
        with pytest.raises(ValidationError):
            DataConfig(scaling_method="invalid")


class TestAppConfig:
    """Test AppConfig model."""

    def test_default_app_config(self):
        """Test default application configuration."""
        config = AppConfig()

        assert config.app_name == "mlsys-variations-template"
        assert config.environment == "development"
        assert config.num_workers == 4

        # Check sub-configurations
        assert isinstance(config.paths, PathsConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.ml, MLConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)

    def test_environment_validation(self):
        """Test environment validation."""
        # Valid environments
        for env in ["development", "staging", "production"]:
            config = AppConfig(environment=env)
            assert config.environment == env

        # Invalid environment
        with pytest.raises(ValidationError):
            AppConfig(environment="invalid")

    def test_nested_config_override(self):
        """Test overriding nested configuration."""
        config_data = {
            "app_name": "test-app",
            "ml": {
                "random_seed": 123,
                "test_size": 0.3,
            },
            "model": {
                "model_type": "linear",
                "problem_type": "regression",
            },
        }

        config = AppConfig(**config_data)

        assert config.app_name == "test-app"
        assert config.ml.random_seed == 123
        assert config.ml.test_size == 0.3
        assert config.model.model_type == "linear"
        assert config.model.problem_type == "regression"

    def test_create_directories(self, temp_dir: Path):
        """Test directory creation."""
        config = AppConfig(
            paths={
                "data_raw": temp_dir / "data" / "raw",
                "models_trained": temp_dir / "models" / "trained",
                "logs_dir": temp_dir / "logs",
            }
        )

        config.create_directories()

        assert (temp_dir / "data" / "raw").exists()
        assert (temp_dir / "models" / "trained").exists()
        assert (temp_dir / "logs").exists()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = AppConfig(app_name="test-app")
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["app_name"] == "test-app"
        assert "paths" in config_dict
        assert "ml" in config_dict

    def test_to_yaml_dict(self):
        """Test conversion to YAML-friendly dictionary."""
        config = AppConfig()
        yaml_dict = config.to_yaml_dict()

        assert isinstance(yaml_dict, dict)
        # Path objects should be converted to strings
        paths = yaml_dict["paths"]
        assert isinstance(paths["data_raw"], str)
        assert isinstance(paths["models_trained"], str)


class TestConfigManager:
    """Test ConfigManager functionality."""

    def test_initialization(self, temp_dir: Path):
        """Test ConfigManager initialization."""
        config_dir = temp_dir / "conf"
        manager = ConfigManager(config_dir=config_dir)

        assert manager.config_dir == config_dir
        assert config_dir.exists()

    def test_create_default_config_files(self, config_manager: ConfigManager):
        """Test creation of default config files."""
        config_manager.create_default_config_files()

        config_dir = config_manager.config_dir

        # Check main config file
        assert (config_dir / "config.yaml").exists()

        # Check sub-config files
        assert (config_dir / "paths" / "default.yaml").exists()
        assert (config_dir / "logging" / "default.yaml").exists()
        assert (config_dir / "ml" / "default.yaml").exists()
        assert (config_dir / "model" / "default.yaml").exists()
        assert (config_dir / "data" / "default.yaml").exists()

    def test_load_config_defaults(self, config_manager: ConfigManager):
        """Test loading configuration with defaults."""
        config = config_manager.load_config()

        assert isinstance(config, AppConfig)
        assert config.app_name == "mlsys-variations-template"
        assert config.environment == "development"

    def test_load_config_with_overrides(self, config_manager: ConfigManager):
        """Test loading configuration with overrides."""
        overrides = {
            "app_name": "test-overrides",
            "ml": {"random_seed": 999},
        }

        config = config_manager.load_config(overrides=overrides)

        assert config.app_name == "test-overrides"
        assert config.ml.random_seed == 999
        # Other values should remain default
        assert config.environment == "development"

    def test_save_and_load_config(self, config_manager: ConfigManager, temp_dir: Path):
        """Test saving and loading configuration."""
        # Create a config
        original_config = AppConfig(app_name="save-test", ml={"random_seed": 777})

        # Save config
        config_path = temp_dir / "saved_config.yaml"
        config_manager.save_config(original_config, config_path)

        assert config_path.exists()

        # Load config from file
        loaded_config = config_manager.load_config(config_path=config_path)

        assert loaded_config.app_name == "save-test"
        assert loaded_config.ml.random_seed == 777

    def test_update_config(self, config_manager: ConfigManager):
        """Test updating configuration."""
        # Load initial config
        config = config_manager.load_config()
        initial_seed = config.ml.random_seed

        # Update config
        updates = {"ml": {"random_seed": 555}}
        updated_config = config_manager.update_config(updates)

        assert updated_config.ml.random_seed == 555
        assert updated_config.ml.random_seed != initial_seed

    def test_config_validation_error_fallback(self, temp_dir: Path):
        """Test fallback to default config on validation error."""
        # Create invalid YAML file
        invalid_config_path = temp_dir / "invalid.yaml"
        with open(invalid_config_path, "w") as f:
            f.write("invalid: { unclosed")

        manager = ConfigManager(config_dir=temp_dir)

        # Should fall back to default config without raising
        config = manager.load_config(config_path=invalid_config_path)
        assert isinstance(config, AppConfig)
        assert config.app_name == "mlsys-variations-template"  # Default value
