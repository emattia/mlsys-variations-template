"""Batch processor endpoint.

This module provides a service for processing batch prediction requests.
"""

import time
import os
import json
from datetime import datetime
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.models.inference import batch_predict
from src.utils.common import get_model_path, load_config, setup_logging

# Load environment variables
load_dotenv()

# Configure logging
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", None),
    log_format=os.getenv(
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ),
)
logger = logging.getLogger(__name__)


class BatchRequestHandler(FileSystemEventHandler):
    """Handler for batch prediction request files."""

    def __init__(self, input_dir: Path, output_dir: Path, model_dir: Path):
        """Initialize the handler.

        Args:
            input_dir: Directory to watch for input files
            output_dir: Directory to save output files
            model_dir: Directory containing trained models
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_dir = model_dir

        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Process any existing files
        self._process_existing_files()

    def _process_existing_files(self) -> None:
        """Process any existing files in the input directory."""
        for file_path in self.input_dir.glob("*.json"):
            logger.info(f"Processing existing file: {file_path}")
            self._process_request_file(file_path)

    def on_created(self, event: FileSystemEventHandler) -> None:
        """Handle file creation events."""
        if not event.is_directory:
            file_path = Path(event.src_path)

            # Only process JSON files
            if file_path.suffix.lower() == ".json":
                logger.info(f"New request file detected: {file_path}")
                self._process_request_file(file_path)

    def _process_request_file(self, file_path: Path) -> None:
        """Process a batch prediction request file.

        Args:
            file_path: Path to the request file
        """
        try:
            # Load request
            with open(file_path) as f:
                request = json.load(f)

            # Validate request
            if "data_path" not in request:
                raise ValueError("Request must contain 'data_path'")

            # Get request parameters
            data_path = request["data_path"]
            model_name = request.get("model_name")
            feature_columns = request.get("feature_columns")
            id_column = request.get("id_column")
            return_probabilities = request.get("return_probabilities", False)

            # Find model path
            if model_name is None:
                # Use the first available model
                model_files = list(self.model_dir.glob("*.pkl"))

                if not model_files:
                    raise ValueError("No models found")

                model_path = model_files[0]
            else:
                # Look for the specified model
                model_path = self.model_dir / f"{model_name}.pkl"

                if not model_path.exists():
                    raise ValueError(f"Model '{model_name}' not found")

            # Generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"predictions_{timestamp}.csv"
            output_path = self.output_dir / output_filename

            # Run batch prediction
            _result = batch_predict(
                model_path=model_path,
                input_path=data_path,
                output_path=output_path,
                feature_columns=feature_columns,
                id_column=id_column,
                return_probabilities=return_probabilities,
            )

            # Create response
            response = {
                "request_id": file_path.stem,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "model_path": str(model_path),
                "data_path": data_path,
                "output_path": str(output_path),
                "num_predictions": _result["num_predictions"],
                "has_probabilities": _result["has_probabilities"],
            }

            # Save response
            response_path = self.output_dir / f"{file_path.stem}_response.json"
            with open(response_path, "w") as f:
                json.dump(response, f, indent=2)

            logger.info(
                f"Batch prediction complete: {_result['num_predictions']} "
                f"predictions saved to {output_path}"
            )

            # Move request file to processed directory
            processed_dir = self.input_dir / "processed"
            processed_dir.mkdir(exist_ok=True)
            processed_path = processed_dir / file_path.name
            file_path.rename(processed_path)

        except Exception as e:
            logger.error(f"Error processing request file {file_path}: {e}")

            # Create error response
            error_response = {
                "request_id": file_path.stem,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

            # Save error response
            error_path = self.output_dir / f"{file_path.stem}_error.json"
            with open(error_path, "w") as f:
                json.dump(error_response, f, indent=2)

            # Move request file to failed directory
            failed_dir = self.input_dir / "failed"
            failed_dir.mkdir(exist_ok=True)
            failed_path = failed_dir / file_path.name
            file_path.rename(failed_path)


def run_batch_processor(
    input_dir: str | Path,
    output_dir: str | Path,
    model_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> None:
    """Run the batch processor service.

    Args:
        input_dir: Directory to watch for input files
        output_dir: Directory to save output files
        model_dir: Directory containing trained models (None for default)
        config_path: Path to the configuration file (None for default)
    """
    logger.info("Starting batch processor service")

    # Load configuration if provided
    config = None
    if config_path is not None:
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        # Override parameters from config if not explicitly provided
        if "input_dir" in config and input_dir is None:
            input_dir = config["input_dir"]
        if "output_dir" in config and output_dir is None:
            output_dir = config["output_dir"]
        if "model_dir" in config and model_dir is None:
            model_dir = config["model_dir"]

    # Set default model directory if not provided
    if model_dir is None:
        model_dir = get_model_path("trained")

    # Convert paths to Path objects
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)

    logger.info(f"Watching for request files in: {input_dir}")
    logger.info(f"Saving output files to: {output_dir}")
    logger.info(f"Using models from: {model_dir}")

    # Create event handler and observer
    event_handler = BatchRequestHandler(input_dir, output_dir, model_dir)
    observer = Observer()
    observer.schedule(event_handler, str(input_dir), __recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
    logger.info("Batch processor service stopped")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Batch processor service")
    parser.add_argument(
        "--input-dir", required=True, help="Directory to watch for input files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save output files"
    )
    parser.add_argument("--model-dir", help="Directory containing trained models")
    parser.add_argument("--config", help="Path to the configuration file")

    args = parser.parse_args()

    # Run the service
    run_batch_processor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        config_path=args.config,
    )
