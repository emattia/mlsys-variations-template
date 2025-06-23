"""Concrete implementations of DataSource plugins."""

from pathlib import Path

import polars as pl

from .base import DataSource, ExecutionContext
from .registry import register_plugin


@register_plugin(
    name="local_file",
    category="data_source",
    description="Load data from a local file.",
)
class LocalFileDataSource(DataSource):
    """
    A data source that loads data from a local file (e.g., CSV, Parquet).
    """

    def initialize(self, context: ExecutionContext) -> None:
        """Initializes the data source, checking for file path in config."""
        self.logger.info(f"Initializing LocalFileDataSource with config: {self.config}")
        if "path" not in self.config:
            raise ValueError(
                "'path' not specified in the configuration for LocalFileDataSource."
            )

        file_path = Path(self.config["path"])
        if not file_path.exists():
            raise FileNotFoundError(f"The file specified does not exist: {file_path}")

    def execute(self, context: ExecutionContext) -> pl.DataFrame:
        """Delegates to load_data for execution."""
        return self.load_data(context)

    def load_data(self, context: ExecutionContext) -> pl.DataFrame:
        """Loads data from the file specified in the configuration."""
        file_path = self.config.get("path")
        self.logger.info(f"Loading data from {file_path}")

        try:
            if file_path.endswith(".csv"):
                df = pl.read_csv(file_path)
            elif file_path.endswith(".parquet"):
                df = pl.read_parquet(file_path)
            else:
                raise ValueError(
                    f"Unsupported file type: {file_path}. Only .csv and .parquet are supported."
                )

            self.logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            raise

    def validate_config(self, config: dict) -> bool:
        """Validates the configuration for the component."""
        return "path" in config and isinstance(config["path"], str)


@register_plugin(
    name="snowflake", category="data_source", description="Load data from Snowflake."
)
class SnowflakeDataSource(DataSource):
    """
    A data source that loads data from Snowflake.
    This is a placeholder implementation.
    """

    def initialize(self, context: ExecutionContext) -> None:
        self.logger.info("Initializing SnowflakeDataSource.")
        # In a real implementation, you would check for credentials and connection details.

    def execute(self, context: ExecutionContext) -> pl.DataFrame:
        return self.load_data(context)

    def load_data(self, context: ExecutionContext) -> pl.DataFrame:
        self.logger.warning(
            "SnowflakeDataSource is a placeholder and does not load real data."
        )
        # In a real implementation, you would connect to Snowflake and run a query.
        # Example:
        # conn = snowflake.connector.connect(...)
        # query = self.config.get("query")
        # df = pd.read_sql(query, conn)
        # return pl.from_pandas(df)
        return pl.DataFrame(
            {"feature1": [1, 2, 3], "feature2": ["A", "B", "C"], "target": [0, 1, 0]}
        )

    def validate_config(self, config: dict) -> bool:
        # In a real implementation, you would validate connection details and query.
        return True
