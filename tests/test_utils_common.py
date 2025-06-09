"""Test cases for src/utils/common.py."""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from src.utils.common import (
    get_data_path,
    get_model_path,
    get_project_root,
    get_reports_path,
    load_config,
    load_environment_variables,
    load_yaml_config,
    save_config,
    setup_logging,
)


def test_setup_logging_default():
    """Test default logging setup."""
    # Reset logging configuration
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.NOTSET)

    setup_logging()

    # Check that root logger level is set
    root_logger = logging.getLogger()
    assert root_logger.level == logging.INFO


def test_setup_logging_with_level():
    """Test logging setup with custom level."""
    # Reset logging configuration
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.NOTSET)

    setup_logging(level="DEBUG")

    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG


def test_setup_logging_with_file():
    """Test logging setup with log file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"

        # Reset logging configuration
        logging.getLogger().handlers.clear()
        logging.getLogger().setLevel(logging.NOTSET)

        setup_logging(log_file=str(log_file))

        # Log a test message
        logger = logging.getLogger("test")
        logger.info("Test message")

        # Force flush any buffered logs
        for handler in logging.getLogger().handlers:
            handler.flush()

        # Check that log file was created
        assert log_file.exists()

        # Check that message is in log file
        with open(log_file) as f:
            content = f.read()
            assert "Test message" in content


def test_setup_logging_with_custom_format():
    """Test logging setup with custom format."""
    custom_format = "%(levelname)s: %(message)s"

    setup_logging(log_format=custom_format)

    # This test mainly checks that no errors occur
    # Format testing would require capturing log output


def test_load_config_yaml():
    """Test loading YAML configuration."""
    config_data = {
        "database": {"host": "localhost", "port": 5432},
        "api_key": "secret123",
        "debug": True,
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        loaded_config = load_config(config_path)

        assert loaded_config == config_data
        assert loaded_config["database"]["host"] == "localhost"
        assert loaded_config["api_key"] == "secret123"
        assert loaded_config["debug"] is True


def test_load_config_json():
    """Test loading JSON configuration."""
    config_data = {
        "model": {
            "type": "RandomForest",
            "parameters": {"n_estimators": 100, "max_depth": 10},
        },
        "training": {"test_size": 0.2, "random_state": 42},
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.json"

        with open(config_path, "w") as f:
            json.dump(config_data, f)

        loaded_config = load_config(config_path)

        assert loaded_config == config_data
        assert loaded_config["model"]["type"] == "RandomForest"
        assert loaded_config["training"]["test_size"] == 0.2


def test_load_config_file_not_found():
    """Test error when config file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")


def test_load_config_unsupported_format():
    """Test error with unsupported file format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.txt"
        config_path.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            load_config(config_path)


def test_save_config_yaml():
    """Test saving configuration as YAML."""
    config_data = {
        "model_type": "LogisticRegression",
        "hyperparameters": {"C": 1.0, "max_iter": 1000},
        "features": ["age", "income", "category"],
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "saved_config.yaml"

        saved_path = save_config(config_data, config_path, format="yaml")

        assert saved_path.exists()
        assert saved_path == config_path

        # Load and verify
        with open(config_path) as f:
            loaded_data = yaml.safe_load(f)

        assert loaded_data == config_data


def test_save_config_json():
    """Test saving configuration as JSON."""
    config_data = {
        "preprocessing": {"normalize": True, "encoding": "one_hot"},
        "evaluation": {"metrics": ["accuracy", "f1_score"], "cv_folds": 5},
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "saved_config.json"

        saved_path = save_config(config_data, config_path, format="json")

        assert saved_path.exists()

        # Load and verify
        with open(config_path) as f:
            loaded_data = json.load(f)

        assert loaded_data == config_data


def test_save_config_with_numpy():
    """Test saving config with numpy data types."""
    config_data = {
        "float64_value": np.float64(3.14),
        "int32_value": np.int32(42),
        "array": np.array([1, 2, 3, 4]),
        "nested": {"numpy_float": np.float32(2.71)},
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "numpy_config.json"

        save_config(config_data, config_path, format="json")

        # Load and verify that numpy types were converted
        with open(config_path) as f:
            loaded_data = json.load(f)

        assert isinstance(loaded_data["float64_value"], float)
        assert isinstance(loaded_data["int32_value"], int)
        assert isinstance(loaded_data["array"], list)
        assert isinstance(loaded_data["nested"]["numpy_float"], float)


def test_save_config_unsupported_format():
    """Test error with unsupported save format."""
    config_data = {"key": "value"}

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.xml"

        with pytest.raises(ValueError, match="Unsupported format"):
            save_config(config_data, config_path, format="xml")


def test_save_config_creates_directory():
    """Test that save_config creates directories if they don't exist."""
    config_data = {"test": "data"}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a nested path that doesn't exist
        config_path = Path(temp_dir) / "configs" / "models" / "config.yaml"

        saved_path = save_config(config_data, config_path)

        assert saved_path.exists()
        assert saved_path.parent.exists()


def test_get_project_root():
    """Test getting project root directory."""
    root = get_project_root()

    assert isinstance(root, Path)
    # Should be three levels up from src/utils/common.py
    # Handle both local development and CI environments
    assert root.name in ["analysis-template", "mlsys-variations-template"]


def test_get_data_path():
    """Test getting data directory paths."""
    # Test specific types
    raw_path = get_data_path("raw")
    assert raw_path.name == "raw"
    assert raw_path.parent.name == "data"

    processed_path = get_data_path("processed")
    assert processed_path.name == "processed"

    interim_path = get_data_path("interim")
    assert interim_path.name == "interim"

    external_path = get_data_path("external")
    assert external_path.name == "external"


def test_get_data_path_invalid_type():
    """Test error with invalid data type."""
    with pytest.raises(ValueError, match="Invalid data_type"):
        get_data_path("invalid_type")


def test_get_model_path():
    """Test getting model directory paths."""
    # Test specific types
    trained_path = get_model_path("trained")
    assert trained_path.name == "trained"
    assert trained_path.parent.name == "models"

    # Test evaluation
    eval_path = get_model_path("evaluation")
    assert eval_path.name == "evaluation"


def test_get_model_path_invalid_type():
    """Test error with invalid model type."""
    with pytest.raises(ValueError, match="Invalid model_type"):
        get_model_path("invalid_type")


def test_get_reports_path():
    """Test getting report directory paths."""
    # Test specific types
    figures_path = get_reports_path("figures")
    assert figures_path.name == "figures"
    assert figures_path.parent.name == "reports"

    # Test other types
    tables_path = get_reports_path("tables")
    assert tables_path.name == "tables"

    documents_path = get_reports_path("documents")
    assert documents_path.name == "documents"


def test_get_reports_path_invalid_type():
    """Test error with invalid report type."""
    with pytest.raises(ValueError, match="Invalid report_type"):
        get_reports_path("invalid_type")


@patch("dotenv.load_dotenv")
def test_load_environment_variables_default(mock_load_dotenv):
    """Test loading environment variables with default .env file."""
    mock_load_dotenv.return_value = True

    load_environment_variables()

    # Should call load_dotenv with project root .env
    mock_load_dotenv.assert_called_once()
    call_args = mock_load_dotenv.call_args[0]
    assert call_args[0].name == ".env"


@patch("dotenv.load_dotenv")
def test_load_environment_variables_custom_file(mock_load_dotenv):
    """Test loading environment variables from custom file."""
    mock_load_dotenv.return_value = True

    with tempfile.TemporaryDirectory() as temp_dir:
        env_file = Path(temp_dir) / "custom.env"
        env_file.write_text("TEST_VAR=test_value")

        load_environment_variables(env_file)

        mock_load_dotenv.assert_called_once_with(env_file)


@patch("src.utils.common.logger")
def test_logging_calls(mock_logger):
    """Test that utility functions log appropriate messages."""
    config_data = {"test": "data"}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test load_config logging
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        load_config(config_path)
        mock_logger.info.assert_any_call(f"Loaded configuration from {config_path}")

        # Test save_config logging
        save_path = Path(temp_dir) / "save_config.yaml"
        save_config(config_data, save_path)
        mock_logger.info.assert_any_call(f"Saved configuration to {save_path}")


def test_integration_config_roundtrip():
    """Test complete configuration save/load roundtrip."""
    original_config = {
        "model": {
            "type": "RandomForestClassifier",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
            },
        },
        "data": {
            "train_path": "/path/to/train.csv",
            "test_path": "/path/to/test.csv",
            "features": ["feature1", "feature2", "feature3"],
        },
        "training": {"validation_split": 0.2, "batch_size": 32, "epochs": 100},
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1_score"],
            "cross_validation": {"enabled": True, "folds": 5},
        },
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test YAML roundtrip
        yaml_path = Path(temp_dir) / "config.yaml"
        save_config(original_config, yaml_path, format="yaml")
        loaded_yaml = load_config(yaml_path)

        assert loaded_yaml == original_config

        # Test JSON roundtrip
        json_path = Path(temp_dir) / "config.json"
        save_config(original_config, json_path, format="json")
        loaded_json = load_config(json_path)

        assert loaded_json == original_config

        # Both loaded configs should be identical
        assert loaded_yaml == loaded_json


def test_path_utilities_integration():
    """Test integration of path utility functions."""
    # Get project root
    project_root = get_project_root()

    # Test that all path functions return paths under project root
    data_raw = get_data_path("raw")
    data_processed = get_data_path("processed")
    models_trained = get_model_path("trained")
    reports_figures = get_reports_path("figures")

    # All paths should be under project root
    assert project_root in data_raw.parents
    assert project_root in data_processed.parents
    assert project_root in models_trained.parents
    assert project_root in reports_figures.parents

    # Test path structure
    assert data_raw == project_root / "data" / "raw"
    assert data_processed == project_root / "data" / "processed"
    assert models_trained == project_root / "models" / "trained"
    assert reports_figures == project_root / "reports" / "figures"


def test_numpy_conversion_edge_cases():
    """Test numpy conversion with edge cases."""
    config_with_complex_numpy = {
        "nested_arrays": {
            "matrix": np.array([[1, 2], [3, 4]]),
            "vector": np.array([1.0, 2.0, 3.0]),
        },
        "mixed_list": [
            np.int64(42),
            "string",
            np.float32(3.14),
            {"nested_numpy": np.array([1, 2, 3])},
        ],
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "complex_config.json"

        save_config(config_with_complex_numpy, config_path, format="json")

        # Load and verify all numpy types were converted
        with open(config_path) as f:
            loaded_data = json.load(f)

        # Check that nested arrays became lists
        assert isinstance(loaded_data["nested_arrays"]["matrix"], list)
        assert isinstance(loaded_data["nested_arrays"]["vector"], list)

        # Check mixed list conversion
        mixed_list = loaded_data["mixed_list"]
        assert isinstance(mixed_list[0], int)  # np.int64 -> int
        assert isinstance(mixed_list[1], str)  # string unchanged
        assert isinstance(mixed_list[2], float)  # np.float32 -> float
        assert isinstance(mixed_list[3]["nested_numpy"], list)  # nested array -> list
