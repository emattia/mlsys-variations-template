"""Common utility functions updated for the new configuration system."""

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.config import AppConfig, ConfigManager

logger = logging.getLogger(__name__)


def setup_logging(
    config: AppConfig | None = None,
    level: str | None = None,
    log_file: str | None = None,
    log_format: str | None = None,
) -> None:
    """Set up logging configuration.

    Args:
        config: Application configuration (if None, loads default)
        level: Override log level
        log_file: Override log file path
        log_format: Override log format
    """
    if config is None:
        try:
            config = ConfigManager().get_config()
        except RuntimeError:
            # Fallback to basic logging setup if no config available
            logging.basicConfig(
                level=getattr(logging, (level or "INFO").upper()),
                format=log_format
                or "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler()]
                if not log_file
                else [logging.FileHandler(log_file)],
                force=True,
            )
            return

    # Use config values with overrides
    actual_level = level or config.logging.level
    actual_format = log_format or config.logging.format
    actual_file = log_file or (
        str(config.logging.file) if config.logging.file else None
    )

    # Configure logging
    handlers = []

    if config.logging.console:
        handlers.append(logging.StreamHandler())

    if actual_file:
        # Ensure log directory exists
        log_path = Path(actual_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(actual_file))

    logging.basicConfig(
        level=getattr(logging, actual_level.upper()),
        format=actual_format,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary with configuration values
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load based on file extension
    config: dict[str, Any] = {}
    if config_path.suffix.lower() in [".yaml", ".yml"]:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    elif config_path.suffix.lower() == ".json":
        with open(config_path) as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(
    config: dict[str, Any], config_path: str | Path, format: str = "yaml"
) -> Path:
    """Save configuration to a file.

    Args:
        config: Configuration dictionary
        config_path: Path where the configuration will be saved
        format: File format (yaml or json)

    Returns:
        Path to the saved configuration file
    """
    config_path = Path(config_path)

    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types
    def convert_numpy(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, np.int64 | np.int32 | np.float64 | np.float32):
            return float(obj) if np.issubdtype(type(obj), np.floating) else int(obj)
        else:
            return obj

    config = convert_numpy(config)

    # Save based on format
    if format.lower() == "yaml":
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    elif format.lower() == "json":
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved configuration to {config_path}")
    return config_path


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to the project root directory
    """
    # This assumes that this file is in src/utils/common.py
    return Path(__file__).parent.parent.parent


def get_data_path(
    data_type: str,
    config: AppConfig | None = None,
) -> Path:
    """Get path for data of specified type.

    Args:
        data_type: Type of data ('raw', 'processed', 'interim', 'external')
        config: Application configuration (if None, loads default)

    Returns:
        Path to data directory

    Raises:
        ValueError: If data_type is not valid
    """
    if config is None:
        try:
            config = ConfigManager().get_config()
        except RuntimeError:
            # Fallback to project root-based paths
            project_root = get_project_root()
            data_type_mapping = {
                "raw": project_root / "data" / "raw",
                "processed": project_root / "data" / "processed",
                "interim": project_root / "data" / "interim",
                "external": project_root / "data" / "external",
            }

            if data_type not in data_type_mapping:
                raise ValueError(
                    f"Invalid data_type: {data_type}. Must be one of {list(data_type_mapping.keys())}"
                )

            path = data_type_mapping[data_type]
            path.mkdir(parents=True, exist_ok=True)
            return path

    data_type_mapping = {
        "raw": config.paths.data_raw,
        "processed": config.paths.data_processed,
        "interim": config.paths.data_interim,
        "external": config.paths.data_external,
    }

    if data_type not in data_type_mapping:
        raise ValueError(
            f"Invalid data_type: {data_type}. Must be one of {list(data_type_mapping.keys())}"
        )

    path = data_type_mapping[data_type]
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_path(
    model_type: str,
    config: AppConfig | None = None,
) -> Path:
    """Get path for models of specified type.

    Args:
        model_type: Type of model ('trained', 'evaluation')
        config: Application configuration (if None, loads default)

    Returns:
        Path to model directory

    Raises:
        ValueError: If model_type is not valid
    """
    if config is None:
        try:
            config = ConfigManager().get_config()
        except RuntimeError:
            # Fallback to project root-based paths
            project_root = get_project_root()
            model_type_mapping = {
                "trained": project_root / "models" / "trained",
                "evaluation": project_root / "models" / "evaluation",
            }

            if model_type not in model_type_mapping:
                raise ValueError(
                    f"Invalid model_type: {model_type}. Must be one of {list(model_type_mapping.keys())}"
                )

            path = model_type_mapping[model_type]
            path.mkdir(parents=True, exist_ok=True)
            return path

    model_type_mapping = {
        "trained": config.paths.models_trained,
        "evaluation": config.paths.models_evaluation,
    }

    if model_type not in model_type_mapping:
        raise ValueError(
            f"Invalid model_type: {model_type}. Must be one of {list(model_type_mapping.keys())}"
        )

    path = model_type_mapping[model_type]
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_reports_path(
    report_type: str,
    config: AppConfig | None = None,
) -> Path:
    """Get path for reports of specified type.

    Args:
        report_type: Type of report ('figures', 'tables', 'documents')
        config: Application configuration (if None, loads default)

    Returns:
        Path to reports directory

    Raises:
        ValueError: If report_type is not valid
    """
    if config is None:
        try:
            config = ConfigManager().get_config()
        except RuntimeError:
            # Fallback to project root-based paths
            project_root = get_project_root()
            report_type_mapping = {
                "figures": project_root / "reports" / "figures",
                "tables": project_root / "reports" / "tables",
                "documents": project_root / "reports" / "documents",
            }

            if report_type not in report_type_mapping:
                raise ValueError(
                    f"Invalid report_type: {report_type}. Must be one of {list(report_type_mapping.keys())}"
                )

            path = report_type_mapping[report_type]
            path.mkdir(parents=True, exist_ok=True)
            return path

    report_type_mapping = {
        "figures": config.paths.reports_figures,
        "tables": config.paths.reports_tables,
        "documents": config.paths.reports_documents,
    }

    if report_type not in report_type_mapping:
        raise ValueError(
            f"Invalid report_type: {report_type}. Must be one of {list(report_type_mapping.keys())}"
        )

    path = report_type_mapping[report_type]
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_environment_variables(env_file: str | Path | None = None) -> dict[str, str]:
    """Load environment variables from a .env file.

    Args:
        env_file: Path to the .env file (None for default .env)

    Returns:
        Dictionary with environment variables
    """
    from dotenv import load_dotenv

    if env_file is None:
        env_file = get_project_root() / ".env"

    # Load environment variables
    load_dotenv(env_file)

    # Get all environment variables
    env_vars = dict(os.environ.items())

    logger.info(f"Loaded environment variables from {env_file}")
    return env_vars


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance.

    Returns:
        ConfigManager instance
    """
    from src.config.manager import get_config_manager

    return get_config_manager()


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file (deprecated - use ConfigManager instead).

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Note:
        This function is deprecated. Use ConfigManager.load_config() instead.
    """
    import warnings

    warnings.warn(
        "load_yaml_config is deprecated. Use ConfigManager.load_config() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    import yaml

    config_path = Path(config_path)

    if not config_path.exists():
        return {}

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def create_run_id(prefix: str = "run") -> str:
    """Create a unique run ID for tracking experiments.

    Args:
        prefix: Prefix for the run ID

    Returns:
        Unique run ID string
    """
    import uuid
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]

    return f"{prefix}_{timestamp}_{short_uuid}"


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        The same path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
