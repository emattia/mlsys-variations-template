#!/usr/bin/env python3
"""
Phase 1 Implementation Demonstration Script

This script demonstrates the core improvements implemented in Phase 1:
1. Unified Configuration Management with Pydantic + Hydra
2. Plugin Architecture with Abstract Base Classes
3. Enhanced Testing Infrastructure
4. Type-safe Configuration Validation

Run with: python demo_phase1.py
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from src.config import AppConfig, ConfigManager, load_config
from src.plugins import (
    ExecutionContext,
    get_plugin,
    list_plugins,
)
from src.utils.common import create_run_id, setup_logging

# Import workflows to register plugins
try:
    import workflows.model_training  # This registers the sklearn_trainer plugin
except ImportError:
    pass  # Skip if dependencies are missing

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_configuration_system():
    """Demonstrate the unified configuration system."""
    print("\n" + "=" * 60)
    print("🔧 CONFIGURATION SYSTEM DEMONSTRATION")
    print("=" * 60)

    print("\n1. Creating default configuration...")
    config = AppConfig()
    print(f"   ✅ App Name: {config.app_name}")
    print(f"   ✅ Environment: {config.environment}")
    print(f"   ✅ Random Seed: {config.ml.random_seed}")
    print(f"   ✅ Model Type: {config.model.model_type}")

    print("\n2. Configuration validation in action...")
    try:
        # Valid configuration
        valid_config = AppConfig(
            environment="production", ml={"test_size": 0.3, "cv_folds": 10}
        )
        print(
            f"   ✅ Valid config: Environment={valid_config.environment}, Test Size={valid_config.ml.test_size}"
        )
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")

    try:
        # Invalid configuration
        invalid_config = AppConfig(
            environment="invalid_env",  # This should fail validation
        )
        print(f"   ❌ This shouldn't print: {invalid_config.environment}")
    except Exception as e:
        print(f"   ✅ Validation correctly caught invalid environment: {e}")

    print("\n3. Configuration overrides...")
    overrides = {
        "app_name": "demo-app",
        "ml": {"random_seed": 999, "hyperparameter_search": True},
        "model": {"model_type": "linear", "problem_type": "regression"},
    }

    config_with_overrides = load_config(overrides=overrides)
    print(f"   ✅ Overridden app name: {config_with_overrides.app_name}")
    print(f"   ✅ Overridden random seed: {config_with_overrides.ml.random_seed}")
    print(f"   ✅ Overridden model type: {config_with_overrides.model.model_type}")

    print("\n4. ConfigManager with Hydra integration...")
    with tempfile.TemporaryDirectory() as tmpdir:
        config_manager = ConfigManager(config_dir=Path(tmpdir) / "conf")
        config_manager.create_default_config_files()

        # Load config using Hydra
        hydra_config = config_manager.load_config()
        print(f"   ✅ Hydra config loaded: {hydra_config.app_name}")
        print(
            f"   ✅ Hydra-managed paths exist: {hydra_config.paths.data_raw.exists()}"
        )


def demo_plugin_architecture():
    """Demonstrate the plugin architecture."""
    print("\n" + "=" * 60)
    print("🔌 PLUGIN ARCHITECTURE DEMONSTRATION")
    print("=" * 60)

    print("\n1. Listing available plugins...")
    plugins = list_plugins()
    print(f"   ✅ Found {len(plugins)} registered plugins: {plugins}")

    print("\n2. Getting plugin information...")
    plugins_with_info = list_plugins(with_info=True)
    for name, info in plugins_with_info.items():
        print(f"   📦 {name}: {info['description']} (v{info['version']})")
        print(f"      Category: {info['category']}")
        if info["dependencies"]:
            print(f"      Dependencies: {info['dependencies']}")

    print("\n3. Creating execution context...")
    config = AppConfig(app_name="demo-plugin-test")
    run_id = create_run_id("demo")

    context = ExecutionContext(
        config=config,
        run_id=run_id,
        component_name="demo_component",
        input_data={"demo": True},
    )

    print(f"   ✅ Run ID: {context.run_id}")
    print(f"   ✅ Artifacts dir: {context.artifacts_dir}")
    print("   ✅ Context created successfully")

    print("\n4. Plugin execution example...")
    if "sklearn_trainer" in plugins:
        try:
            # Create some sample data
            np.random.seed(42)
            n_samples = 200

            data = {
                "feature_1": np.random.randn(n_samples),
                "feature_2": np.random.randn(n_samples),
                "feature_3": np.random.uniform(0, 1, n_samples),
                "target": np.random.choice([0, 1], n_samples),
            }

            df = pl.DataFrame(data)
            print(f"   ✅ Created sample dataset: {df.shape}")

            # Update context with data
            context.input_data = {"dataframe": df}

            # Get and execute plugin
            trainer = get_plugin("sklearn_trainer")
            trainer.initialize(context)
            print(f"   ✅ Initialized trainer: {trainer.name}")

            result = trainer.execute(context)
            print(f"   ✅ Training execution: {result.status.value}")

            if result.is_success():
                print(
                    f"      📊 CV Score: {result.metrics.get('cv_score_mean', 0):.4f}"
                )
                print(f"      ⏱️  Execution time: {result.execution_time:.2f}s")
                print(
                    f"      💾 Model saved to: {result.artifacts.get('model', 'N/A')}"
                )
            else:
                print(f"      ❌ Training failed: {result.error_message}")

        except Exception as e:
            print(f"   ⚠️  Plugin execution demo failed: {e}")
            print("      (This is expected if sklearn dependencies are missing)")
    else:
        print("   ⚠️  sklearn_trainer plugin not found (normal during development)")


def demo_type_safety():
    """Demonstrate type safety and validation."""
    print("\n" + "=" * 60)
    print("🛡️  TYPE SAFETY & VALIDATION DEMONSTRATION")
    print("=" * 60)

    print("\n1. Pydantic model validation...")

    # Test path validation
    try:
        from src.config.models import PathsConfig

        paths = PathsConfig(data_raw="/tmp/test")
        print(f"   ✅ Path validation: {paths.data_raw}")
    except Exception as e:
        print(f"   ❌ Path validation failed: {e}")

    # Test constraint validation
    try:
        from src.config.models import MLConfig

        ml_valid = MLConfig(test_size=0.3, cv_folds=5)
        print(
            f"   ✅ Constraint validation: test_size={ml_valid.test_size}, cv_folds={ml_valid.cv_folds}"
        )
    except Exception as e:
        print(f"   ❌ Constraint validation failed: {e}")

    # Test invalid constraints
    try:
        from src.config.models import MLConfig

        ml_invalid = MLConfig(test_size=1.5)  # Should fail: > 1.0
        print(f"   ❌ This shouldn't print: {ml_invalid.test_size}")
    except Exception as e:
        print(f"   ✅ Constraint validation correctly caught invalid test_size: {e}")

    print("\n2. Enum validation...")
    try:
        from src.plugins.base import ComponentStatus

        status = ComponentStatus.SUCCESS
        print(f"   ✅ Enum usage: {status.value}")

        # Status checking methods
        from src.plugins.base import ComponentResult

        result = ComponentResult(
            status=ComponentStatus.SUCCESS, component_name="test", execution_time=1.0
        )
        print(
            f"   ✅ Status checking: is_success()={result.is_success()}, is_failed()={result.is_failed()}"
        )

    except Exception as e:
        print(f"   ❌ Enum validation failed: {e}")


def demo_enhanced_testing():
    """Demonstrate enhanced testing capabilities."""
    print("\n" + "=" * 60)
    print("🧪 ENHANCED TESTING DEMONSTRATION")
    print("=" * 60)

    print("\n1. Test fixtures and utilities...")

    # Demonstrate what the test fixtures provide
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate test config creation
        test_config = AppConfig(
            app_name="test-app",
            environment="testing",
            paths={
                "data_raw": Path(tmpdir) / "data" / "raw",
                "models_trained": Path(tmpdir) / "models" / "trained",
                "logs_dir": Path(tmpdir) / "logs",
            },
        )

        test_config.create_directories()
        print(f"   ✅ Test directories created in: {tmpdir}")
        print(f"   ✅ Data directory exists: {test_config.paths.data_raw.exists()}")
        print(
            f"   ✅ Models directory exists: {test_config.paths.models_trained.exists()}"
        )

    print("\n2. Sample data generation...")
    # Simulate test data creation
    np.random.seed(42)
    n_samples = 100

    sample_data = {
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "target": np.random.choice([0, 1], n_samples),
    }

    df = pl.DataFrame(sample_data)
    print(f"   ✅ Sample classification data: {df.shape}")
    print(f"   ✅ Features: {[col for col in df.columns if col != 'target']}")
    print(f"   ✅ Target distribution: {df['target'].value_counts()}")

    print("\n3. Test execution context...")
    context = ExecutionContext(
        config=test_config,
        run_id="test-run-001",
        component_name="test-component",
        metadata={"test_mode": True},
    )

    print(f"   ✅ Test context created: {context.run_id}")
    print(f"   ✅ Test metadata: {context.metadata}")

    print("\n4. Plugin registry isolation...")
    from src.plugins.registry import get_registry

    registry = get_registry()
    initial_count = len(registry._plugins)

    # This would be done in test setup/teardown
    registry.clear()
    cleared_count = len(registry._plugins)

    print(
        f"   ✅ Registry isolation: {initial_count} plugins -> {cleared_count} plugins (cleared)"
    )


def demo_integration():
    """Demonstrate how all components work together."""
    print("\n" + "=" * 60)
    print("🎯 INTEGRATION DEMONSTRATION")
    print("=" * 60)

    print("\n1. End-to-end workflow simulation...")

    # Step 1: Load configuration with overrides
    config_overrides = {
        "app_name": "integration-demo",
        "ml": {
            "random_seed": 42,
            "test_size": 0.2,
            "cv_folds": 3,  # Faster for demo
        },
        "model": {
            "model_type": "random_forest",
            "problem_type": "classification",
            "target_column": "target",
        },
    }

    config = load_config(overrides=config_overrides)
    setup_logging(config)

    print(f"   ✅ Configuration loaded: {config.app_name}")

    # Step 2: Create execution context
    run_id = create_run_id("integration")
    context = ExecutionContext(
        config=config,
        run_id=run_id,
        component_name="integration_demo",
        metadata={"demo_type": "integration"},
    )

    print(f"   ✅ Execution context: {context.run_id}")

    # Step 3: Generate sample data
    np.random.seed(config.ml.random_seed)
    n_samples = 300

    # Create realistic classification data
    feature_1 = np.random.randn(n_samples)
    feature_2 = np.random.randn(n_samples)
    feature_3 = feature_1 * 0.5 + np.random.randn(n_samples) * 0.3

    # Create target with some signal
    target_prob = 1 / (1 + np.exp(-(feature_1 + feature_2 * 0.8 + feature_3 * 0.3)))
    target = np.random.binomial(1, target_prob, n_samples)

    data = {
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3,
        "noise_feature": np.random.randn(n_samples),  # Noise feature
        "target": target,
    }

    df = pl.DataFrame(data)
    print(f"   ✅ Generated realistic dataset: {df.shape}")
    print(f"   ✅ Target balance: {df['target'].sum()}/{len(df)} positive samples")

    # Step 4: Execute plugin-based workflow
    if "sklearn_trainer" in list_plugins():
        try:
            context.input_data = {"dataframe": df}

            trainer = get_plugin("sklearn_trainer")
            trainer.initialize(context)

            result = trainer.execute(context)

            if result.is_success():
                print("   ✅ Model training successful!")
                print(
                    f"      📊 CV Score: {result.metrics['cv_score_mean']:.4f} ± {result.metrics['cv_score_std']:.4f}"
                )
                print(
                    f"      🚀 Training samples: {result.output_data['train_samples']}"
                )
                print(f"      🎯 Test samples: {result.output_data['test_samples']}")
                print(f"      💾 Model saved: {result.artifacts['model'].name}")
                print(f"      ⏱️  Total time: {result.execution_time:.2f}s")

                # Demonstrate metadata tracking
                print("   ✅ Metadata tracking:")
                print(f"      🔧 Model type: {result.metadata['model_type']}")
                print(f"      🎲 Random seed: {result.metadata['random_seed']}")
                print(f"      📊 CV folds: {result.metadata['cv_folds']}")

            else:
                print(f"   ❌ Model training failed: {result.error_message}")

        except Exception as e:
            print(f"   ⚠️  Integration demo failed: {e}")
    else:
        print("   ⚠️  Sklearn trainer not available for integration demo")

    print("\n   ✅ Integration demo complete!")


def main():
    """Run all demonstrations."""
    print("🚀 MLOps Template Phase 1 Implementation Demo")
    print("=" * 60)
    print("This demonstration showcases the improvements implemented in Phase 1:")
    print("• Unified Configuration Management (Pydantic + Hydra)")
    print("• Plugin Architecture with Abstract Base Classes")
    print("• Enhanced Testing Infrastructure")
    print("• Type Safety and Validation")

    try:
        demo_configuration_system()
        demo_plugin_architecture()
        demo_type_safety()
        demo_enhanced_testing()
        demo_integration()

        print("\n" + "=" * 60)
        print("🎉 PHASE 1 DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("\nKey achievements:")
        print("✅ Configuration system consolidated and type-safe")
        print("✅ Plugin architecture provides extensibility")
        print("✅ Enhanced testing infrastructure ready")
        print("✅ All components integrate seamlessly")
        print("\nNext steps (Phase 2):")
        print("📋 Experiment tracking integration (MLflow)")
        print("🚀 Model serving with FastAPI")
        print("🐳 Containerization and deployment")
        print("🔄 CI/CD pipeline enhancements")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
