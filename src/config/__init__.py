"""Configuration management system.

This module provides a unified configuration system using Pydantic and Hydra
for type-safe, hierarchical configuration management.
"""

from .manager import ConfigManager, load_config
from .models import (
    AppConfig,
    Config,
    DataConfig,
    DetailedModelConfig,
    FeaturesConfig,
    LoggingConfig,
    MLConfig,
    PathsConfig,
)

__all__ = [
    "AppConfig",
    "Config",
    "DataConfig",
    "DetailedModelConfig",
    "FeaturesConfig",
    "LoggingConfig",
    "MLConfig",
    "PathsConfig",
    "ConfigManager",
    "load_config",
]
