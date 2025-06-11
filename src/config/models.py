"""Pydantic configuration models for type-safe configuration management."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# These Enums can be used to provide strong typing for config fields
# and ensure only valid values are used.


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ScalingMethod(str, Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"


class CategoricalEncoding(str, Enum):
    ONE_HOT = "one_hot"
    TARGET = "target"
    ORDINAL = "ordinal"


class FeatureSelection(str, Enum):
    MUTUAL_INFO = "mutual_info"
    CHI2 = "chi2"
    F_CLASSIF = "f_classif"


class MissingHandling(str, Enum):
    MEDIAN = "median"
    MEAN = "mean"
    MODE = "mode"
    DROP = "drop"


class PathsConfig(BaseModel):
    """Configuration for project paths."""

    data_root: str = "data"
    model_root: str = "models"
    reports_root: str = "reports"
    logs_root: str = "logs"

    class Config:
        protected_namespaces = ()

    # Derived path properties for convenience and backward compatibility
    @property
    def data_raw(self) -> Path:
        """Path to raw data."""
        return Path(self.data_root) / "raw"

    @property
    def data_processed(self) -> Path:
        """Path to processed data."""
        return Path(self.data_root) / "processed"

    @property
    def data_interim(self) -> Path:
        """Path to interim data."""
        return Path(self.data_root) / "interim"

    @property
    def data_external(self) -> Path:
        """Path to external data."""
        return Path(self.data_root) / "external"

    @property
    def models_trained(self) -> Path:
        """Path to trained models."""
        return Path(self.model_root) / "trained"

    @property
    def models_evaluation(self) -> Path:
        """Path to model evaluation artifacts."""
        return Path(self.model_root) / "evaluation"

    @property
    def reports_figures(self) -> Path:
        """Path to report figures."""
        return Path(self.reports_root) / "figures"

    @property
    def reports_tables(self) -> Path:
        """Path to report tables."""
        return Path(self.reports_root) / "tables"

    @property
    def reports_documents(self) -> Path:
        """Path to report documents."""
        return Path(self.reports_root) / "documents"

    @property
    def logs_dir(self) -> Path:
        """Path to log directory."""
        return Path(self.logs_root)


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console: bool = True
    file: Path | None = None

    @field_validator("level", mode="before")
    def validate_level(cls, v):
        """Ensure log level is valid and uppercase."""
        v = v.upper()
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in allowed_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {allowed_levels}")
        return v

    class Config:
        protected_namespaces = ()


class FeaturesConfig(BaseModel):
    """Configuration for feature engineering."""

    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    categorical_encoding: CategoricalEncoding = CategoricalEncoding.ONE_HOT
    feature_selection: FeatureSelection = FeatureSelection.MUTUAL_INFO
    handle_missing: MissingHandling = MissingHandling.MEDIAN


class ModelParametersConfig(BaseModel):
    """Parameters for the ML model."""

    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "auto"
    bootstrap: bool = True
    random_state: int = 42
    n_jobs: int = -1


class HyperparameterRangesConfig(BaseModel):
    """Ranges for hyperparameter tuning."""

    n_estimators: dict[str, Any] = Field(
        default_factory=lambda: {"type": "int", "low": 10, "high": 200}
    )
    max_depth: dict[str, Any] = Field(
        default_factory=lambda: {"type": "int", "low": 3, "high": 50}
    )
    min_samples_split: dict[str, Any] = Field(
        default_factory=lambda: {"type": "int", "low": 2, "high": 20}
    )
    min_samples_leaf: dict[str, Any] = Field(
        default_factory=lambda: {"type": "int", "low": 1, "high": 20}
    )


class ModelTrainingConfig(BaseModel):
    """Configuration specific to model training."""

    early_stopping: bool = False
    validation_metric: str = "accuracy"
    save_best_only: bool = True


class DetailedModelConfig(BaseModel):
    """Configuration for a specific model."""

    model_type: str = "RandomForest"
    algorithm: str = "ensemble.RandomForestClassifier"
    parameters: ModelParametersConfig = Field(default_factory=ModelParametersConfig)
    hyperparameter_ranges: HyperparameterRangesConfig = Field(
        default_factory=HyperparameterRangesConfig
    )
    training: ModelTrainingConfig = Field(default_factory=ModelTrainingConfig)

    class Config:
        protected_namespaces = ()


class APISecurityConfig(BaseModel):
    enable_cors: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    api_key_header: str = "X-API-Key"
    rate_limit_enabled: bool = False
    rate_limits: dict[str, str] | None = None


class APIModelsConfig(BaseModel):
    auto_load: bool = True
    model_directory: str = "models"
    max_models: int = 10
    model_timeout: int = 300
    lazy_loading: bool = True

    class Config:
        protected_namespaces = ()


class APIMonitoringConfig(BaseModel):
    enable_metrics: bool = True
    metrics_endpoint: str = "/metrics"
    health_endpoint: str = "/health"
    log_requests: bool = True
    prometheus_port: int = 9090


class APICachingConfig(BaseModel):
    enabled: bool = False
    backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    default_ttl: int = 300


class APIConfig(BaseModel):
    """Configuration for the API server."""

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    timeout: int = 120
    max_request_size: int = 1048576  # 1MB
    security: APISecurityConfig = Field(default_factory=APISecurityConfig)
    models: APIModelsConfig = Field(default_factory=APIModelsConfig)
    monitoring: APIMonitoringConfig = Field(default_factory=APIMonitoringConfig)
    caching: APICachingConfig = Field(default_factory=APICachingConfig)


# Main Configuration Model that brings everything together


class MLConfig(BaseModel):
    """Configuration for machine learning processes."""

    problem_type: str = "classification"
    target_column: str = "target"
    test_size: float = Field(default=0.2, ge=0.0, le=1.0)
    validation_size: float = Field(default=0.2, ge=0.0, le=1.0)
    cv_folds: int = Field(default=5, ge=2)
    random_seed: int = 42
    hyperparameter_search: bool = False
    early_stopping: bool = False
    primary_metric: str = "accuracy"

    class Config:
        protected_namespaces = ()

    @field_validator("problem_type")
    def validate_problem_type(cls, v):
        """Validate problem type."""
        valid_types = {"classification", "regression", "clustering"}
        if v not in valid_types:
            raise ValueError(f"Invalid problem type: {v}. Must be one of {valid_types}")
        return v


class Config(BaseModel):
    """The main configuration model for the entire application."""

    project_name: str = "mlops-project"
    version: str = "0.1.0"
    environment: Environment = Environment.DEVELOPMENT
    ml: MLConfig = Field(default_factory=MLConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: DetailedModelConfig = Field(default_factory=DetailedModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    # The 'defaults' list from Hydra is not part of the validated config content,
    # so we don't define it here. Hydra processes it before Pydantic sees it.


class SimpleModelConfig(BaseModel):
    """Configuration for model-specific parameters."""

    model_type: str = Field(default="random_forest", description="Type of model to use")
    problem_type: str = Field(
        default="classification", description="Type of ML problem"
    )

    # Feature engineering
    feature_columns: list[str] | None = Field(
        default=None, description="List of feature columns"
    )
    categorical_features: list[str] = Field(
        default_factory=list, description="Categorical feature columns"
    )
    numerical_features: list[str] = Field(
        default_factory=list, description="Numerical feature columns"
    )

    # Target column (supervised problems)
    target_column: str = Field(
        default="target", description="Name of the target column"
    )

    # Model parameters (flexible for different model types)
    model_params: dict[str, Any] = Field(
        default_factory=dict, description="Model-specific parameters"
    )

    @field_validator("problem_type")
    def validate_problem_type(cls, v):
        """Validate problem type."""
        valid_types = {"classification", "regression", "clustering"}
        if v not in valid_types:
            raise ValueError(f"Invalid problem type: {v}. Must be one of {valid_types}")
        return v

    class Config:
        protected_namespaces = ()


# ---------------------------------------------------------------------------
# Backwards-compatibility alias
# ---------------------------------------------------------------------------

# Re-export ``SimpleModelConfig`` under the historical name ``ModelConfig`` so that
# external modules importing ``ModelConfig`` continue to work without code
# changes. This preserves API stability while resolving the previous duplicate
# class definition issue.

ModelConfig = SimpleModelConfig


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

    @field_validator("missing_value_strategy")
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

    @field_validator("scaling_method")
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
    model: SimpleModelConfig = Field(
        default_factory=SimpleModelConfig, description="Model configurations"
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

    @field_validator("environment")
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
