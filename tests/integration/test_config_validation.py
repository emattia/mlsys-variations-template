"""Integration tests for configuration validation."""

from pathlib import Path

import pytest

from src.platform.config import ConfigManager

# The root directory of the configuration files.
project_root = Path(__file__).parent.parent.parent
CONFIG_DIR = project_root / "conf"

# Define the different configuration scenarios to test.
# This approach mirrors real-world usage where the main 'config.yaml'
# is loaded with overrides for different environments.
scenarios = {
    "base_config": [],
    "production_config": ["environment=production"],
    "testing_config": ["environment=testing", "app.log_level=DEBUG"],
}


@pytest.mark.integration
@pytest.mark.parametrize("scenario_name, overrides", scenarios.items())
def test_config_scenario(scenario_name, overrides):
    """
    Tests that a given configuration scenario loads and validates successfully.

    This test parameterizes over different scenarios (e.g., base, production),
    loads the main 'config.yaml' with the respective overrides, and asserts
    that the configuration is valid. This ensures that all primary
    configuration combinations are correct and type-safe.

    Args:
        scenario_name (str): The name of the test scenario.
        overrides (list): A list of Hydra-style overrides for the scenario.
    """
    try:
        print(f"[*] Testing configuration scenario: '{scenario_name}'...")
        # Always start with the main 'config.yaml'
        manager = ConfigManager(config_dir=str(CONFIG_DIR), config_name="config")
        # Apply the overrides for the specific scenario
        config = manager.load_config(overrides=overrides)
        # The real test is whether the above line raises an exception.
        # If it runs without error, the configuration is valid.
        print(f"✅ SUCCESS: Scenario '{scenario_name}' is valid.")
        assert config is not None
    except Exception as e:
        pytest.fail(f"❌ FAILED: Scenario '{scenario_name}' is invalid.\n   Error: {e}")
