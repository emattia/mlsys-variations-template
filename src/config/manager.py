"""Configuration manager for unified config handling with Hydra integration."""

import logging
from pathlib import Path
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .models import AppConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Unified configuration manager integrating Pydantic models with Hydra."""

    def __init__(
        self,
        config_dir: str | Path | None = None,
        config_name: str = "config",
        version_base: str | None = None,
    ):
        """Initialize the configuration manager.

        Args:
            config_dir: Directory containing Hydra config files (defaults to ./conf)
            config_name: Name of the main config file
            version_base: Hydra version base for compatibility
        """
        self.config_dir = Path(config_dir) if config_dir else Path("conf").resolve()
        self.config_name = config_name
        self.version_base = version_base
        self._hydra_config: DictConfig | None = None
        self._app_config: AppConfig | None = None

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def initialize_hydra(self, overrides: list | None = None) -> DictConfig:
        """Initialize Hydra and load configuration.

        Args:
            overrides: List of config overrides (e.g., ['key=value'])

        Returns:
            Hydra DictConfig object
        """
        overrides = overrides or []

        # Clear any existing Hydra instance
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()

        try:
            # Initialize Hydra with config directory
            with initialize_config_dir(
                config_dir=str(self.config_dir), version_base=self.version_base
            ):
                # Compose configuration with overrides
                self._hydra_config = compose(
                    config_name=self.config_name, overrides=overrides
                )

            logger.info(f"Hydra configuration loaded from {self.config_dir}")
            return self._hydra_config

        except Exception as e:
            logger.warning(f"Failed to load Hydra config: {e}")
            logger.info("Falling back to default configuration")
            self._hydra_config = OmegaConf.create({})
            return self._hydra_config

    def load_config(
        self,
        config_path: str | Path | None = None,
        overrides: dict | None = None,
        hydra_overrides: list | None = None,
    ) -> AppConfig:
        """Load and validate configuration using both Hydra and Pydantic.

        Args:
            config_path: Path to YAML config file (optional)
            overrides: Dictionary of config overrides
            hydra_overrides: List of Hydra-style overrides

        Returns:
            Validated AppConfig instance
        """
        config_dict = {}

        # 1. Load from Hydra (if available)
        if not self._hydra_config:
            self.initialize_hydra(hydra_overrides)

        if self._hydra_config:
            config_dict.update(OmegaConf.to_object(self._hydra_config))

        # 2. Load from YAML file (if provided)
        if config_path:
            yaml_config = self._load_yaml_config(config_path)
            config_dict.update(yaml_config)

        # 3. Apply manual overrides
        if overrides:
            config_dict.update(overrides)

        # 4. Create and validate Pydantic model
        try:
            self._app_config = AppConfig(**config_dict)
            logger.info("Configuration loaded and validated successfully")

            # Create directories
            self._app_config.create_directories()

            return self._app_config

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            # Fall back to default configuration
            self._app_config = AppConfig()
            self._app_config.create_directories()
            return self._app_config

    def _load_yaml_config(self, config_path: str | Path) -> dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}

        try:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load YAML config from {config_path}: {e}")
            return {}

    def save_config(
        self, config: AppConfig, output_path: str | Path, format: str = "yaml"
    ) -> None:
        """Save configuration to file.

        Args:
            config: AppConfig instance to save
            output_path: Output file path
            format: Output format ('yaml' or 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "yaml":
            config_dict = config.to_yaml_dict()
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(config.json(indent=2))
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Configuration saved to {output_path}")

    def get_config(self) -> AppConfig:
        """Get the current application configuration.

        Returns:
            Current AppConfig instance

        Raises:
            RuntimeError: If no configuration has been loaded
        """
        if self._app_config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")
        return self._app_config

    def update_config(self, updates: dict[str, Any]) -> AppConfig:
        """Update the current configuration with new values.

        Args:
            updates: Dictionary of updates to apply

        Returns:
            Updated AppConfig instance
        """
        if self._app_config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")

        # Get current config as dict
        current_dict = self._app_config.to_dict()

        # Apply updates (nested dictionary merge)
        updated_dict = self._deep_merge(current_dict, updates)

        # Create new config instance
        self._app_config = AppConfig(**updated_dict)
        self._app_config.create_directories()

        logger.info("Configuration updated successfully")
        return self._app_config

    def _deep_merge(
        self, base: dict[str, Any], update: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            update: Dictionary with updates

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def create_default_config_files(self) -> None:
        """Create default Hydra configuration files."""
        # Main config file
        main_config = {
            "defaults": [
                "_self_",
                "paths: default",
                "logging: default",
                "ml: default",
                "model: default",
                "data: default",
            ],
            "hydra": {"run": {"dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"}},
        }

        # Sub-config files
        configs = {
            "config.yaml": main_config,
            "paths/default.yaml": {
                "data_raw": "data/raw",
                "data_processed": "data/processed",
                "data_interim": "data/interim",
                "data_external": "data/external",
                "models_trained": "models/trained",
                "models_evaluation": "models/evaluation",
                "reports_figures": "reports/figures",
                "reports_tables": "reports/tables",
                "reports_documents": "reports/documents",
                "logs_dir": "logs",
            },
            "logging/default.yaml": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "console": True,
            },
            "ml/default.yaml": {
                "random_seed": 42,
                "test_size": 0.2,
                "validation_size": 0.2,
                "cv_folds": 5,
                "hyperparameter_search": False,
                "early_stopping": True,
                "patience": 10,
                "primary_metric": "accuracy",
                "additional_metrics": [],
            },
            "model/default.yaml": {
                "model_type": "random_forest",
                "problem_type": "classification",
                "target_column": "target",
                "categorical_features": [],
                "numerical_features": [],
                "model_params": {},
            },
            "data/default.yaml": {
                "file_format": "csv",
                "encoding": "utf-8",
                "separator": ",",
                "missing_value_strategy": "drop",
                "outlier_detection": False,
                "scaling_method": "standard",
                "validate_schema": True,
                "min_rows": 1,
                "max_missing_percentage": 0.5,
            },
        }

        # Create config files
        for filename, config_data in configs.items():
            config_path = self.config_dir / filename
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)

            logger.info(f"Created default config file: {config_path}")


# Global configuration manager instance
_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(
    config_path: str | Path | None = None,
    overrides: dict | None = None,
    hydra_overrides: list | None = None,
) -> AppConfig:
    """Load configuration using the global config manager."""
    manager = get_config_manager()
    return manager.load_config(config_path, overrides, hydra_overrides)
