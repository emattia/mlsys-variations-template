"""Pydantic configuration models for type-safe configuration management."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class PathsConfig(BaseModel):
    """Configuration for file and directory paths."""

    # Data paths
    data_raw: Path = Field(default=Path("data/raw"), description="Raw data directory")
    data_processed: Path = Field(
        default=Path("data/processed"), description="Processed data directory"
    )
    data_interim: Path = Field(
        default=Path("data/interim"), description="Interim data directory"
    )
    data_external: Path = Field(
        default=Path("data/external"), description="External data directory"
    )

    # Model paths
    models_trained: Path = Field(
        default=Path("models/trained"), description="Trained models directory"
    )
    models_evaluation: Path = Field(
        default=Path("models/evaluation"), description="Model evaluation directory"
    )

    # Output paths
    reports_figures: Path = Field(
        default=Path("reports/figures"), description="Figures output directory"
    )
    reports_tables: Path = Field(
        default=Path("reports/tables"), description="Tables output directory"
    )
    reports_documents: Path = Field(
        default=Path("reports/documents"), description="Documents output directory"
    )

    # Logs
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")

    @validator("*", pre=True)
    def convert_to_path(cls, v):
        """Convert strings to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging settings."""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    file: Path | None = Field(default=None, description="Log file path")
    console: bool = Field(default=True, description="Enable console logging")

    @validator("level")
    def validate_level(cls, v):
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class MLConfig(BaseModel):
    """Configuration for machine learning parameters."""

    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    test_size: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Test set size fraction"
    )
    validation_size: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Validation set size fraction"
    )
    cv_folds: int = Field(
        default=5, ge=2, description="Number of cross-validation folds"
    )

    # Model training
    hyperparameter_search: bool = Field(
        default=False, description="Enable hyperparameter search"
    )
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(default=10, ge=1, description="Early stopping patience")

    # Metrics
    primary_metric: str = Field(
        default="accuracy", description="Primary evaluation metric"
    )
    additional_metrics: list[str] = Field(
        default_factory=list, description="Additional metrics to track"
    )


class ModelConfig(BaseModel):
    """Configuration for model-specific parameters."""

    model_type: str = Field(default="random_forest", description="Type of model to use")
    problem_type: str = Field(
        default="classification", description="Type of ML problem"
    )

    # Feature engineering
    feature_columns: list[str] | None = Field(
        default=None, description="List of feature columns"
    )
    target_column: str = Field(default="target", description="Target column name")
    categorical_features: list[str] = Field(
        default_factory=list, description="Categorical feature columns"
    )
    numerical_features: list[str] = Field(
        default_factory=list, description="Numerical feature columns"
    )

    # Model parameters (flexible for different model types)
    model_params: dict[str, Any] = Field(
        default_factory=dict, description="Model-specific parameters"
    )

    @validator("problem_type")
    def validate_problem_type(cls, v):
        """Validate problem type."""
        valid_types = {"classification", "regression", "clustering"}
        if v not in valid_types:
            raise ValueError(f"Invalid problem type: {v}. Must be one of {valid_types}")
        return v


class DataConfig(BaseModel):
    """Configuration for data processing parameters."""

    # Data loading
    file_format: str = Field(default="csv", description="Data file format")
    encoding: str = Field(default="utf-8", description="File encoding")
    separator: str = Field(default=",", description="CSV separator")

    # Data processing
    missing_value_strategy: str = Field(
        default="drop", description="Strategy for handling missing values"
    )
    outlier_detection: bool = Field(
        default=False, description="Enable outlier detection"
    )
    scaling_method: str = Field(
        default="standard", description="Feature scaling method"
    )

    # Data validation
    validate_schema: bool = Field(
        default=True, description="Enable data schema validation"
    )
    min_rows: int = Field(
        default=1, ge=1, description="Minimum number of rows required"
    )
    max_missing_percentage: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Maximum allowed missing data percentage",
    )

    @validator("missing_value_strategy")
    def validate_missing_strategy(cls, v):
        """Validate missing value strategy."""
        valid_strategies = {
            "drop",
            "mean",
            "median",
            "mode",
            "forward_fill",
            "backward_fill",
        }
        if v not in valid_strategies:
            raise ValueError(
                f"Invalid missing value strategy: {v}. Must be one of {valid_strategies}"
            )
        return v

    @validator("scaling_method")
    def validate_scaling_method(cls, v):
        """Validate scaling method."""
        valid_methods = {"standard", "minmax", "robust", "quantile", "none"}
        if v not in valid_methods:
            raise ValueError(
                f"Invalid scaling method: {v}. Must be one of {valid_methods}"
            )
        return v


class AppConfig(BaseSettings):
    """Main application configuration combining all sub-configurations."""

    # Application metadata
    app_name: str = Field(
        default="mlsys-variations-template", description="Application name"
    )
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)",
    )

    # Sub-configurations
    paths: PathsConfig = Field(
        default_factory=PathsConfig, description="Path configurations"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configurations"
    )
    ml: MLConfig = Field(
        default_factory=MLConfig, description="Machine learning configurations"
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig, description="Model configurations"
    )
    data: DataConfig = Field(
        default_factory=DataConfig, description="Data processing configurations"
    )

    # Database (optional)
    db_host: str | None = Field(default=None, description="Database host")
    db_port: int | None = Field(default=None, description="Database port")
    db_name: str | None = Field(default=None, description="Database name")
    db_user: str | None = Field(default=None, description="Database user")
    db_password: str | None = Field(default=None, description="Database password")

    # Cloud/API settings
    aws_access_key_id: str | None = Field(default=None, description="AWS access key ID")
    aws_secret_access_key: str | None = Field(
        default=None, description="AWS secret access key"
    )
    aws_region: str = Field(default="us-west-2", description="AWS region")
    s3_bucket: str | None = Field(default=None, description="S3 bucket name")

    api_key: str | None = Field(default=None, description="API key")
    api_secret: str | None = Field(default=None, description="API secret")

    # Computation
    num_workers: int = Field(default=4, ge=1, description="Number of parallel workers")

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = {"development", "staging", "production", "testing"}
        if v not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v

    def create_directories(self) -> None:
        """Create all configured directories if they don't exist."""
        paths_to_create = [
            self.paths.data_raw,
            self.paths.data_processed,
            self.paths.data_interim,
            self.paths.data_external,
            self.paths.models_trained,
            self.paths.models_evaluation,
            self.paths.reports_figures,
            self.paths.reports_tables,
            self.paths.reports_documents,
            self.paths.logs_dir,
        ]

        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert configuration to YAML-friendly dictionary."""
        config_dict = self.dict()

        # Convert Path objects to strings for YAML serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        return convert_paths(config_dict)
