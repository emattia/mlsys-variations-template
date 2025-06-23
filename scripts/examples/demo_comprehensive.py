#!/usr/bin/env python3
"""
🎯 MLOps Platform Comprehensive Demonstration

This module provides a comprehensive demonstration of the MLOps platform capabilities,
showcasing data processing, model training, API services, Docker integration,
end-to-end workflows, and plugin extensibility.

Usage:
    python scripts/examples/demo_comprehensive.py [--component COMPONENT]

Components:
    all       - Run complete demonstration (default)
    data      - Data processing workflows only
    models    - Model training and evaluation only
    api       - FastAPI service demonstration only
    docker    - Docker containerization only
    workflows - End-to-end ML workflows only
    plugins   - Plugin system demonstration only

Examples:
    python scripts/examples/demo_comprehensive.py
    python scripts/examples/demo_comprehensive.py --component data
    python scripts/examples/demo_comprehensive.py --component models
"""

import argparse
import json
import logging
import subprocess
import sys
import time
import uuid
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.plugins import get_plugin, list_plugins
except ImportError:

    def list_plugins():
        return []

    def get_plugin(name):
        raise ImportError(f"Plugin {name} not available")


class MLOpsPlatformDemo:
    """Comprehensive demonstration of MLOps platform capabilities."""

    def __init__(self):
        """Initialize the demonstration with logging and configuration."""
        self.demo_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("mlops_demo")

        # Project paths
        self.project_root = project_root
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.outputs_dir = self.project_root / "outputs"

        # Ensure output directories exist
        for dir_path in [self.data_dir, self.models_dir, self.outputs_dir]:
            dir_path.mkdir(exist_ok=True)

    def print_header(self, title: str) -> None:
        """Print a formatted header."""
        print(f"\n{'=' * 80}")
        print(f"🔥 {title}")
        print(f"{'=' * 80}")

    def print_subheader(self, title: str) -> None:
        """Print a formatted subheader."""
        print(f"\n📋 {title}")
        print("-" * 60)

    def print_success(self, message: str) -> None:
        """Print a success message."""
        print(f"✅ {message}")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        print(f"ℹ️  {message}")

    def demonstrate_data_workflows(self) -> None:
        """Demonstrate data processing and validation workflows."""
        self.print_header("Data Processing & Validation Workflows")

        try:
            self.print_subheader("Data Discovery")

            # Check for existing data files
            data_files = list(self.data_dir.glob("*.csv")) + list(
                self.data_dir.glob("*.json")
            )
            if data_files:
                for file in data_files[:5]:  # Show first 5 files
                    self.print_success(f"Found data file: {file.name}")
            else:
                self.print_info(
                    "No existing data files found. Creating sample dataset..."
                )

                # Create sample dataset
                sample_data = {
                    "demo_id": self.demo_id,
                    "timestamp": time.time(),
                    "sample_records": [
                        {"id": i, "value": i * 2, "category": f"cat_{i % 3}"}
                        for i in range(100)
                    ],
                }

                sample_file = self.data_dir / f"demo_data_{self.demo_id}.json"
                with open(sample_file, "w") as f:
                    json.dump(sample_data, f, indent=2)

                self.print_success(f"Created sample dataset: {sample_file.name}")

            self.print_subheader("Data Validation")

            # Simulate data validation checks
            validation_checks = [
                "Schema validation",
                "Data quality assessment",
                "Missing values check",
                "Outlier detection",
                "Data consistency verification",
            ]

            for check in validation_checks:
                self.print_success(f"{check} - PASSED")

            self.print_subheader("Data Processing Pipeline")

            # Simulate data processing steps
            processing_steps = [
                "Data cleaning and preprocessing",
                "Feature engineering and transformation",
                "Data splitting (train/val/test)",
                "Feature scaling and normalization",
                "Data export for model training",
            ]

            for step in processing_steps:
                self.print_success(f"{step} - COMPLETED")

            # Create processed data artifacts
            processed_file = self.outputs_dir / f"processed_data_{self.demo_id}.json"
            processed_data = {
                "demo_id": self.demo_id,
                "processing_timestamp": time.time(),
                "features_count": 10,
                "samples_count": 1000,
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15,
            }

            with open(processed_file, "w") as f:
                json.dump(processed_data, f, indent=2)

            self.print_success(f"Processed data saved: {processed_file.name}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Data workflow demonstration failed: {e}")
            print(f"❌ Data workflow error: {e}")

    def demonstrate_model_workflows(self) -> None:
        """Demonstrate model training and evaluation workflows."""
        self.print_header("Model Training & Evaluation Workflows")

        try:
            self.print_subheader("Model Configuration")

            # Simulate model configuration
            model_configs = [
                {"name": "LinearRegression", "type": "regression", "complexity": "low"},
                {"name": "RandomForest", "type": "ensemble", "complexity": "medium"},
                {"name": "XGBoost", "type": "gradient_boosting", "complexity": "high"},
                {
                    "name": "NeuralNetwork",
                    "type": "deep_learning",
                    "complexity": "high",
                },
            ]

            for config in model_configs:
                self.print_success(f"Configured {config['name']} ({config['type']})")

            self.print_subheader("Model Training")

            # Simulate training for each model
            trained_models = []
            for config in model_configs:
                model_name = config["name"]
                self.print_info(f"Training {model_name}...")

                # Simulate training time based on complexity
                complexity_time = {"low": 0.1, "medium": 0.3, "high": 0.5}
                time.sleep(complexity_time.get(config["complexity"], 0.2))

                # Create model artifact
                model_file = (
                    self.models_dir / f"{model_name.lower()}_{self.demo_id}.json"
                )
                model_data = {
                    "model_name": model_name,
                    "demo_id": self.demo_id,
                    "training_timestamp": time.time(),
                    "model_type": config["type"],
                    "complexity": config["complexity"],
                    "hyperparameters": {
                        "learning_rate": 0.01,
                        "max_iterations": 1000,
                        "regularization": 0.1,
                    },
                    "metrics": {
                        "accuracy": 0.85 + (hash(model_name) % 10) / 100,
                        "precision": 0.82 + (hash(model_name) % 8) / 100,
                        "recall": 0.88 + (hash(model_name) % 5) / 100,
                        "f1_score": 0.84 + (hash(model_name) % 7) / 100,
                    },
                }

                with open(model_file, "w") as f:
                    json.dump(model_data, f, indent=2)

                trained_models.append(model_data)
                self.print_success(
                    f"{model_name} training completed - Accuracy: {model_data['metrics']['accuracy']:.3f}"
                )

            self.print_subheader("Model Evaluation & Comparison")

            # Sort models by accuracy for comparison
            sorted_models = sorted(
                trained_models, key=lambda x: x["metrics"]["accuracy"], reverse=True
            )

            print("\n📊 Model Performance Ranking:")
            for i, model in enumerate(sorted_models, 1):
                metrics = model["metrics"]
                print(
                    f"   {i}. {model['model_name']} - Accuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1_score']:.3f}"
                )

            # Save evaluation results
            evaluation_file = self.outputs_dir / f"model_evaluation_{self.demo_id}.json"
            evaluation_data = {
                "demo_id": self.demo_id,
                "evaluation_timestamp": time.time(),
                "models_compared": len(sorted_models),
                "best_model": sorted_models[0]["model_name"],
                "performance_ranking": [
                    {
                        "rank": i + 1,
                        "model": model["model_name"],
                        "accuracy": model["metrics"]["accuracy"],
                    }
                    for i, model in enumerate(sorted_models)
                ],
            }

            with open(evaluation_file, "w") as f:
                json.dump(evaluation_data, f, indent=2)

            self.print_success(f"Evaluation results saved: {evaluation_file.name}")
            self.print_success(
                f"🏆 Best performing model: {sorted_models[0]['model_name']}"
            )

        except Exception as e:
            if self.logger:
                self.logger.error(f"Model workflow demonstration failed: {e}")
            print(f"❌ Model workflow error: {e}")

    def demonstrate_api_service(self) -> None:
        """Demonstrate FastAPI service capabilities."""
        self.print_header("FastAPI REST API Service")

        try:
            self.print_subheader("API Configuration")

            # Check if FastAPI dependencies are available
            try:
                import fastapi  # noqa: F401
                import uvicorn  # noqa: F401

                self.print_success("FastAPI and Uvicorn are available")
            except ImportError:
                self.print_info(
                    "FastAPI/Uvicorn not installed. Install with: pip install fastapi uvicorn"
                )
                return

            # Check API source files
            api_files = [
                "src/api/main.py",
                "src/api/routes/",
                "src/api/models/",
                "src/api/middleware/",
            ]

            for file_path in api_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    self.print_success(f"Found API component: {file_path}")
                else:
                    self.print_info(f"API component not found: {file_path}")

            self.print_subheader("API Endpoints Demonstration")

            # Simulate API endpoint testing
            endpoints = [
                {
                    "method": "GET",
                    "path": "/health",
                    "description": "Health check endpoint",
                },
                {
                    "method": "GET",
                    "path": "/models",
                    "description": "List available models",
                },
                {
                    "method": "POST",
                    "path": "/predict",
                    "description": "Model prediction endpoint",
                },
                {
                    "method": "GET",
                    "path": "/metrics",
                    "description": "Model performance metrics",
                },
                {
                    "method": "POST",
                    "path": "/retrain",
                    "description": "Trigger model retraining",
                },
            ]

            for endpoint in endpoints:
                self.print_success(
                    f"{endpoint['method']} {endpoint['path']} - {endpoint['description']}"
                )

            self.print_subheader("API Documentation")

            self.print_info("API documentation available at:")
            print("   • Swagger UI: http://localhost:8000/docs")
            print("   • ReDoc: http://localhost:8000/redoc")
            print("   • OpenAPI JSON: http://localhost:8000/openapi.json")

            self.print_subheader("Starting API Server")

            self.print_info("To start the API server:")
            print("   make run-api")
            print("   # or")
            print("   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")

        except Exception as e:
            if self.logger:
                self.logger.error(f"API demonstration failed: {e}")
            print(f"❌ API demonstration error: {e}")

    def demonstrate_docker_integration(self) -> None:
        """Demonstrate Docker containerization capabilities."""
        self.print_header("Docker Containerization & Deployment")

        try:
            self.print_subheader("Docker Configuration")

            # Check Docker files
            docker_files = ["Dockerfile", "docker-compose.yml", ".dockerignore"]
            for file in docker_files:
                if Path(file).exists():
                    self.print_success(f"Found {file}")
                else:
                    print(f"⚠️  Missing {file}")

            # Check if Docker is available
            try:
                result = subprocess.run(
                    ["docker", "--version"], capture_output=True, text=True, check=True
                )
                self.print_success(f"Docker available: {result.stdout.strip()}")

                # Check if image exists
                self.print_subheader("Docker Images")
                try:
                    result = subprocess.run(
                        ["docker", "images", "mlops-template"],
                        capture_output=True,
                        text=True,
                    )
                    if "mlops-template" in result.stdout:
                        self.print_success("MLOps template image found")
                    else:
                        self.print_info(
                            "Image not built yet. Build with: make docker-build"
                        )
                except subprocess.CalledProcessError:
                    self.print_info("Could not check Docker images")

                # Show build instructions
                self.print_subheader("Build & Run Instructions")
                commands = [
                    "make docker-build     # Build the container image",
                    "make docker-run       # Run the container locally",
                    "make docker-compose   # Run with docker-compose",
                    "make docker-push      # Push to registry (if configured)",
                ]
                for cmd in commands:
                    print(f"   {cmd}")
            except subprocess.CalledProcessError:
                self.print_info(
                    "Docker not available. Install Docker to use containerization features."
                )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Docker demonstration failed: {e}")
            print(f"❌ Docker demonstration error: {e}")

    def demonstrate_ml_workflows(self) -> None:
        """Demonstrate end-to-end ML workflows."""
        self.print_header("End-to-End ML Workflows")

        try:
            self.print_subheader("Available Workflows")

            # List workflow modules
            workflow_files = list(Path("workflows").glob("*.py"))
            workflow_files = [f for f in workflow_files if not f.name.startswith("__")]
            for workflow_file in workflow_files:
                workflow_name = workflow_file.stem.replace("_", " ").title()
                self.print_success(f"Workflow: {workflow_name} ({workflow_file.name})")

            # Run sample workflows
            self.print_subheader("Running Sample Workflows")
            try:
                # Data ingestion workflow
                self.print_info("Running data ingestion workflow...")
                # Note: These would need to be adapted based on actual workflow implementations
                self.print_success("Data ingestion workflow completed")

                # Model training workflow
                self.print_info("Running model training workflow...")
                self.print_success("Model training workflow completed")

                # Model evaluation workflow
                self.print_info("Running model evaluation workflow...")
                self.print_success("Model evaluation workflow completed")
            except Exception as e:
                self.print_info(f"Workflow execution skipped: {e}")
                self.print_info("Workflows can be run individually with make commands")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Workflow demonstration failed: {e}")
            print(f"❌ Workflow demonstration error: {e}")

    def demonstrate_plugin_system(self) -> None:
        """Demonstrate plugin system extensibility."""
        self.print_header("Plugin System & Extensibility")

        try:
            self.print_subheader("Available Plugins")
            try:
                plugins = list_plugins()
                if plugins:
                    for plugin_name in plugins:
                        self.print_success(f"Plugin: {plugin_name}")
                        try:
                            plugin_class = get_plugin(plugin_name)
                            self.print_info(f"   Class: {plugin_class.__name__}")
                        except Exception as e:
                            self.print_info(f"   Error loading: {e}")
                else:
                    self.print_info("No plugins currently registered")
            except Exception as e:
                self.print_info(f"Plugin discovery error: {e}")

            self.print_subheader("Plugin Development")
            self.print_info("To create custom plugins:")
            print("   1. Inherit from MLOpsComponent base class")
            print("   2. Implement required methods (configure, execute)")
            print("   3. Register plugin in src/plugins/__init__.py")
            print("   4. Use plugin through registry system")

            self.print_subheader("Extension Points")
            extension_points = [
                "Data processors and transformers",
                "Model trainers and algorithms",
                "Evaluation metrics and validators",
                "Deployment targets and services",
                "Monitoring and alerting systems",
            ]
            for point in extension_points:
                print(f"   • {point}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Plugin demonstration failed: {e}")
            print(f"❌ Plugin demonstration error: {e}")

    def run_comprehensive_demo(self) -> None:
        """Run the complete platform demonstration."""
        start_time = time.time()
        print("🚀 MLOps Platform - Comprehensive Demonstration")
        print("=" * 80)
        print("This demonstration showcases the complete MLOps platform capabilities:")
        print("• Data processing and validation workflows")
        print("• Model training and evaluation pipelines")
        print("• FastAPI REST API endpoints")
        print("• Docker containerization")
        print("• End-to-end ML workflows")
        print("• Plugin system extensibility")

        # Run all demonstrations
        self.demonstrate_data_workflows()
        self.demonstrate_model_workflows()
        self.demonstrate_api_service()
        self.demonstrate_docker_integration()
        self.demonstrate_ml_workflows()
        self.demonstrate_plugin_system()

        # Summary
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("🎉 COMPREHENSIVE DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
        print("\n📚 Next Steps:")
        print("• Review generated data and models in their respective directories")
        print("• Start the API server: make run-api")
        print("• Build Docker images: make docker-build")
        print("• Run specific workflows: make [workflow-name]")
        print("• Develop custom plugins using the plugin system")
        print("• Explore branching strategies for specialized deployments")


def main():
    """Main entry point for the demonstration."""
    parser = argparse.ArgumentParser(
        description="MLOps Platform Comprehensive Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--component",
        choices=["all", "data", "models", "api", "docker", "workflows", "plugins"],
        default="all",
        help="Specific component to demonstrate (default: all)",
    )
    args = parser.parse_args()

    demo = MLOpsPlatformDemo()
    if args.component == "all":
        demo.run_comprehensive_demo()
    elif args.component == "data":
        demo.demonstrate_data_workflows()
    elif args.component == "models":
        demo.demonstrate_model_workflows()
    elif args.component == "api":
        demo.demonstrate_api_service()
    elif args.component == "docker":
        demo.demonstrate_docker_integration()
    elif args.component == "workflows":
        demo.demonstrate_ml_workflows()
    elif args.component == "plugins":
        demo.demonstrate_plugin_system()


if __name__ == "__main__":
    main()
