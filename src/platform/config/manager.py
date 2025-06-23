"""Configuration manager for unified config handling with Hydra integration."""

import logging
import os
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .models import (
    APIConfig,
    Config,
    DetailedModelConfig,
    FeaturesConfig,
    LoggingConfig,
    MLConfig,
    PathsConfig,
)

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
        self._app_config: Config | None = None
        self._config: Config | None = (
            None  # Add _config attribute for test compatibility
        )

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def initialize_hydra(self, hydra_overrides: list[str] | None = None) -> None:
        """Initialize Hydra."""
        if GlobalHydra().is_initialized():
            logger.debug("Hydra is already initialized. Clearing.")
            GlobalHydra.instance().clear()

        try:
            logger.debug(
                f"Initializing Hydra with config_dir: {self.config_dir}, config_name: {self.config_name}"
            )
            with hydra.initialize(
                config_path=str(self.config_dir.absolute()),
                job_name="app",
                version_base=None,
            ):
                self._hydra_config = hydra.compose(
                    config_name=self.config_name, overrides=hydra_overrides or []
                )
                logger.info("Hydra initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to load Hydra config: {e}")
            logger.info("Falling back to default configuration")
            self._hydra_config = OmegaConf.create({})

    def load_config(self, overrides: list[str] | None = None) -> Config:
        """Load configuration from Hydra.

        Args:
            overrides: List of configuration overrides in the format "key=value".
                      Supports nested keys using dot notation (e.g., "ml.random_seed=42").

        Returns:
            Config: The loaded configuration object.

        Note:
            If Hydra fails to load the configuration (e.g., due to missing files or
            path issues), this method will fall back to a default configuration and
            attempt to apply the provided overrides manually. This ensures the
            application can still start even in degraded environments.

        Examples:
            >>> config_manager.load_config(["project_name=my-project", "ml.random_seed=42"])
        """
        GlobalHydra.instance().clear()

        # Try to load config with Hydra first
        config = self._try_hydra_load(overrides)
        if config is not None:
            return config

        # Fall back to default config with manual override application
        return self._create_fallback_config(overrides)

    def _try_hydra_load(self, overrides: list[str] | None = None) -> Config | None:
        """Try to load configuration using Hydra.

        Returns:
            Config object if successful, None if failed
        """
        try:
            # Calculate relative path from cwd to config_dir
            rel_path = self._get_hydra_config_path()

            hydra.initialize(config_path=rel_path, job_name="app", version_base=None)
            self._hydra_config = hydra.compose(
                config_name=self.config_name, overrides=overrides or []
            )
            logger.info("Hydra initialized and config composed successfully")

            # Load and resolve config
            config_dict = OmegaConf.to_container(self._hydra_config, resolve=True)

            # Create and validate Pydantic model
            self._app_config = Config(**config_dict if config_dict else {})
            logger.info("Configuration loaded and validated successfully")
            return self._app_config

        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            return None

    def _get_hydra_config_path(self) -> str:
        """Get the config path for Hydra initialization."""
        try:
            # Always use relative path for Hydra (required by Hydra)
            return os.path.relpath(self.config_dir, Path.cwd())
        except ValueError:
            # This happens on Windows if the paths are on different drives
            return str(self.config_dir.absolute())

    def _create_fallback_config(self, overrides: list[str] | None = None) -> Config:
        """Create fallback config with manual override application."""
        logger.info("Falling back to default configuration")
        self._app_config = Config()

        if overrides:
            self._apply_overrides_to_fallback(overrides)

        return self._app_config

    def _apply_overrides_to_fallback(self, overrides: list[str]) -> None:
        """Apply overrides to the fallback configuration."""
        try:
            updates = self._parse_overrides(overrides)

            if updates:
                # Apply updates to the fallback config
                current_dict = self._app_config.model_dump()
                updated_dict = self._deep_merge(current_dict, updates)
                self._app_config = Config(**updated_dict)
                logger.info(f"Applied overrides to fallback config: {updates}")

        except Exception as override_error:
            logger.warning(
                f"Failed to apply overrides to fallback config: {override_error}"
            )

    def _parse_overrides(self, overrides: list[str]) -> dict[str, Any]:
        """Parse override strings into a nested dictionary."""
        updates = {}

        for override in overrides:
            if not self._is_valid_override_format(override):
                continue

            key, value = override.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key:
                logger.warning(f"Skipping override with empty key: {override}")
                continue

            self._apply_single_override(updates, key, value, override)

        return updates

    def _is_valid_override_format(self, override: str) -> bool:
        """Check if override has valid format."""
        if "=" not in override:
            logger.warning(f"Skipping invalid override format: {override}")
            return False
        return True

    def _apply_single_override(
        self, updates: dict[str, Any], key: str, value: str, original_override: str
    ) -> None:
        """Apply a single override to the updates dictionary."""
        if "." in key:
            self._apply_nested_override(updates, key, value, original_override)
        else:
            updates[key] = self._convert_override_value(value)

    def _apply_nested_override(
        self, updates: dict[str, Any], key: str, value: str, original_override: str
    ) -> None:
        """Apply a nested override (with dots in key)."""
        keys = [k.strip() for k in key.split(".")]

        # Skip if any key part is empty
        if any(not k for k in keys):
            logger.warning(
                f"Skipping override with empty key part: {original_override}"
            )
            return

        nested_dict = updates
        for k in keys[:-1]:
            if k not in nested_dict:
                nested_dict[k] = {}
            nested_dict = nested_dict[k]

        nested_dict[keys[-1]] = self._convert_override_value(value)

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
        self, config: Config, output_path: str | Path, format: str = "yaml"
    ) -> None:
        """Save configuration to file.

        Args:
            config: Config instance to save
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

    def get_config(self) -> Config:
        """Get the current application configuration.

        Returns:
            Current Config instance

        Raises:
            RuntimeError: If no configuration has been loaded
        """
        if self._app_config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")
        return self._app_config

    def update_config(self, updates: dict[str, Any]) -> Config:
        """Update the current configuration with new values.

        Args:
            updates: Dictionary of updates to apply

        Returns:
            Updated Config instance
        """
        if self._app_config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")

        # Get current config as dict
        current_dict = self._app_config.dict()

        # Apply updates (nested dictionary merge)
        updated_dict = self._deep_merge(current_dict, updates)

        # Create new config instance
        self._app_config = Config(**updated_dict)

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
        main_config_path = self.config_dir / self.config_name
        if not main_config_path.with_suffix(".yaml").exists():
            main_config = {
                "defaults": [
                    "_self_",
                    {"paths": "default"},
                    {"logging": "default"},
                    {"ml": "default"},
                    {"features": "default"},
                    {"model": "default"},
                    {"api": "default"},
                ],
                "project_name": "new-analysis-project",
                "version": "0.1.0",
                "environment": "development",
            }
            self._save_yaml(main_config, main_config_path.with_suffix(".yaml"))

        # Sub-config files
        default_configs = {
            "paths/default.yaml": PathsConfig().model_dump(),
            "logging/default.yaml": LoggingConfig().model_dump(),
            "ml/default.yaml": MLConfig().model_dump(),
            "features/default.yaml": FeaturesConfig().model_dump(),
            "model/default.yaml": DetailedModelConfig().model_dump(),
            "api/default.yaml": APIConfig().model_dump(),
        }

        for path_str, data in default_configs.items():
            path = self.config_dir / path_str
            path.parent.mkdir(parents=True, exist_ok=True)
            self._save_yaml(data, path)
            logger.info(f"Created default config file: {path}")

    def _save_yaml(self, data: dict, path: Path) -> None:
        """Save dictionary to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
        logger.info(f"Created default config file: {path}")

    @property
    def config(self) -> Config:
        """Get the current application configuration.

        Returns:
            Current Config instance

        Raises:
            RuntimeError: If no configuration has been loaded
        """
        if self._app_config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")
        return self._app_config

    def _convert_override_value(self, value: str) -> Any:
        """Convert a string value to the appropriate type."""
        try:
            # Try int first
            return int(value)
        except ValueError:
            try:
                # Try float
                return float(value)
            except ValueError:
                # Keep as string
                return value

    def load_from_dict(self, config_data: dict[str, Any]) -> None:
        """Load configuration from a dictionary.

        Args:
            config_data: Dictionary containing configuration data
        """
        try:
            # Try to create Config model first
            config_instance = Config(**config_data)

            # Check if the config data was actually used or if defaults were applied
            # by comparing a few key fields that should be different if data was used
            model_dump = config_instance.model_dump()
            data_was_used = False

            # Check if any of the provided keys exist in the model
            for key in config_data.keys():
                if key in model_dump:
                    data_was_used = True
                    break

            if data_was_used:
                self._app_config = config_instance
                self._config = self._app_config
                # Also store raw config for flexibility
                self._raw_config = config_data
                logger.info("Configuration loaded from dictionary successfully")
            else:
                # The data wasn't recognized by the Config model, store as raw
                raise ValueError("Config data doesn't match expected schema")

        except Exception as e:
            # If it fails, store as raw dict for flexibility
            logger.warning(f"Could not create Config model: {e}, storing as raw config")
            self._raw_config = config_data
            self._config = None
            self._app_config = None
            logger.info("Configuration loaded from dictionary as raw data")

    def load_from_file(self, file_path: str | Path) -> None:
        """Load configuration from a file.

        Args:
            file_path: Path to the configuration file (YAML or JSON)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            config_data = self._load_yaml_config(file_path)
            self.load_from_dict(config_data)
            logger.info(f"Configuration loaded from file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from file {file_path}: {e}")
            raise

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Value to set
        """
        # Initialize raw config if needed
        if not hasattr(self, "_raw_config"):
            if self._app_config:
                self._raw_config = self._app_config.model_dump()
            else:
                self._raw_config = {}

        # Handle nested keys
        keys = key.split(".")
        current = self._raw_config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

        # Try to recreate Config model
        try:
            self._app_config = Config(**self._raw_config)
            self._config = self._app_config
        except Exception:
            # Keep as raw config if model creation fails
            pass

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        # Try raw config first (more flexible)
        if hasattr(self, "_raw_config") and self._raw_config:
            current = self._raw_config
            keys = key.split(".")
            try:
                for k in keys:
                    current = current[k]
                return current
            except (KeyError, TypeError):
                pass

        # Try structured config
        if self._app_config is not None:
            keys = key.split(".")
            current = self._app_config.model_dump()
            try:
                for k in keys:
                    current = current[k]
                return current
            except (KeyError, TypeError):
                pass

        return default

    def save_to_file(self, file_path: str | Path) -> None:
        """Save configuration to a file.

        Args:
            file_path: Path where to save the configuration
        """
        file_path = Path(file_path)

        if hasattr(self, "_raw_config") and self._raw_config:
            config_data = self._raw_config
        elif self._app_config:
            config_data = self._app_config.model_dump()
        else:
            config_data = {}

        try:
            self._save_yaml(config_data, file_path)
            logger.info(f"Configuration saved to file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to file {file_path}: {e}")
            raise


# Global configuration manager instance
_config_manager: ConfigManager | None = None


def get_config_manager(
    config_dir: str | Path | None = None,
    config_name: str = "config",
    version_base: str | None = "1.3",
) -> "ConfigManager":
    """Get a singleton instance of the ConfigManager."""
    global _config_manager
    if _config_manager is None or (
        config_dir is not None
        and _config_manager.config_dir != Path(config_dir).resolve()
    ):
        _config_manager = ConfigManager(
            config_dir=config_dir,
            config_name=config_name,
            version_base=version_base,
        )
    return _config_manager


def load_config(overrides: list[str] | None = None) -> "Config":
    """Load the application configuration."""
    return get_config_manager().load_config(overrides=overrides)
