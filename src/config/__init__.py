"""Configuration management system.

This module provides a unified configuration system using Pydantic and Hydra
for type-safe, hierarchical configuration management.
"""

from .manager import ConfigManager, get_config_manager, load_config
from .models import (
    AppConfig,
    Config,
    DataConfig,
    LoggingConfig,
    MLConfig,
    ModelConfig,
    PathsConfig,
)

__all__ = [
    "AppConfig",
    "DataConfig",
    "LoggingConfig",
    "MLConfig",
    "ModelConfig",
    "PathsConfig",
    "ConfigManager",
    "load_config",
]
