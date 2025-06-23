# workflows/validate_configs.py
import sys
from pathlib import Path

# Add the project root to the Python path
# This is necessary to ensure that the `src` module can be found
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.platform.config import ConfigManager  # noqa: E402


def validate_all_configs():
    """
    Validates all primary YAML configuration files in the 'conf' directory.

    This script attempts to load each configuration using the ConfigManager,
    which uses Hydra for composition and Pydantic for validation. It will
    print the status for each file.
    """
    print("Starting configuration validation...")

    # The root directory of the configuration files.
    config_dir = project_root / "conf"

    # List of configurations to test. These are the main entry points
    # that compose other files. Hydra needs the name of the file to load.
    # We will test the main 'config.yaml' but override the defaults
    # to simulate different environments.
    # This is a more robust test as it mirrors real-world usage.
    overrides_to_test = {
        "base_config": [],
        "production_api": ["api=production"],
        "development_api": ["api=development"],
    }

    all_valid = True
    for test_name, overrides in overrides_to_test.items():
        try:
            print(f"[*] Testing scenario: '{test_name}'...")
            # Always start with the main 'config.yaml'
            manager = ConfigManager(config_dir=str(config_dir), config_name="config")
            # Apply the overrides for the specific scenario
            manager.load_config(hydra_overrides=overrides)
            print(f"✅ SUCCESS: Scenario '{test_name}' is valid.\n")
        except Exception as e:
            print(f"❌ FAILED: Scenario '{test_name}' is invalid.")
            print(f"   Error: {e}\n")
            all_valid = False

    if all_valid:
        print("🎉 All tested configurations are valid!")
    else:
        print("🔥 Some configurations failed validation.")
        sys.exit(1)


if __name__ == "__main__":
    validate_all_configs()
