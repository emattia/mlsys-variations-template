"""Unit tests for the configuration system."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import AppConfig, ConfigManager
from src.config.manager import get_config_manager
from src.config.models import (
    Config,
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
        """Test that derived paths respect custom roots."""
        custom_config = PathsConfig(
            data_root="/custom/data",
            model_root="/custom/models",
            reports_root="/custom/reports",
        )

        assert custom_config.data_raw == Path("/custom/data/raw")
        assert custom_config.data_processed == Path("/custom/data/processed")
        assert custom_config.models_trained == Path("/custom/models/trained")
        assert custom_config.reports_figures == Path("/custom/reports/figures")
        # Default root should still be used for paths not overridden
        assert custom_config.logs_dir == Path("logs")


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
        paths_config = PathsConfig(
            data_root=str(temp_dir / "data"),
            model_root=str(temp_dir / "models"),
            logs_root=str(temp_dir / "logs"),
        )
        # Create a dummy file in one of the target directories to ensure it gets created
        (temp_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
        (temp_dir / "data" / "raw" / "test.txt").touch()

        # Create directories
        for path_attr in dir(paths_config):
            if path_attr.startswith("_"):
                continue
            path = getattr(paths_config, path_attr)
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)

        assert (temp_dir / "data" / "raw").exists()
        assert (temp_dir / "models").exists()
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
        yaml_dict = config.model_dump(mode="json")

        assert isinstance(yaml_dict, dict)
        # Path objects should be converted to strings
        paths = yaml_dict["paths"]
        assert isinstance(paths["data_root"], str)


@pytest.mark.usefixtures("config_manager")
class TestConfigManager:
    """Unit tests for the ConfigManager."""

    def test_initialization(self, config_manager: ConfigManager):
        """Test ConfigManager initialization."""
        assert config_manager.config_dir.exists()
        assert config_manager.config_name == "config"

    def test_create_default_config_files(self, config_manager: ConfigManager):
        """Test the creation of default configuration files."""
        config_manager.create_default_config_files()
        config_dir = config_manager.config_dir
        assert (config_dir / "config.yaml").exists()
        assert (config_dir / "paths" / "default.yaml").exists()
        assert (config_dir / "logging" / "default.yaml").exists()
        assert (config_dir / "ml" / "default.yaml").exists()
        assert (config_dir / "features" / "default.yaml").exists()
        assert (config_dir / "model" / "default.yaml").exists()
        assert (config_dir / "api" / "default.yaml").exists()

    def test_load_config_defaults(self, config_manager: ConfigManager):
        """Test loading configuration with defaults."""
        config = config_manager.load_config()

        assert isinstance(config, Config)

    def test_load_config_with_overrides(self, temp_dir: Path):
        """Test loading configuration with overrides."""
        # Force re-creation of the config manager for this test
        from src.config import manager

        manager._config_manager = None

        config_manager = get_config_manager(config_dir=temp_dir)
        config_manager.create_default_config_files()
        overrides = [
            "project_name=test-overrides",
            "ml.random_seed=999",
        ]

        config = config_manager.load_config(overrides=overrides)

        assert config.project_name == "test-overrides"
        assert config.ml.random_seed == 999

    def test_fallback_mechanism_with_complex_overrides(self, temp_dir: Path):
        """Test fallback mechanism with various override types."""
        from src.config import manager

        manager._config_manager = None

        # Use a non-existent config directory to force fallback
        non_existent_dir = temp_dir / "non_existent"
        config_manager = get_config_manager(config_dir=non_existent_dir)

        overrides = [
            "project_name=fallback-test",
            "version=2.0.0",
            "ml.random_seed=42",
            "ml.test_size=0.3",
            "logging.level=DEBUG",
        ]

        config = config_manager.load_config(overrides=overrides)

        # Verify fallback applied overrides correctly
        assert config.project_name == "fallback-test"
        assert config.version == "2.0.0"
        assert config.ml.random_seed == 42
        assert config.ml.test_size == 0.3
        assert config.logging.level == "DEBUG"

    def test_fallback_with_invalid_overrides(self, temp_dir: Path):
        """Test fallback mechanism handles invalid overrides gracefully."""
        from src.config import manager

        manager._config_manager = None

        non_existent_dir = temp_dir / "non_existent"
        config_manager = get_config_manager(config_dir=non_existent_dir)

        overrides = [
            "project_name=valid-override",
            "invalid_format_no_equals",
            "=empty_key",
            "nested.very.deep.key=value",
        ]

        # Should not raise an exception
        config = config_manager.load_config(overrides=overrides)

        # Valid override should still be applied
        assert config.project_name == "valid-override"
