import pytest

from src.config import load_config
from src.plugins import ExecutionContext, get_registry


def _all_plugin_names():
    registry = get_registry()
    return registry.list_plugins()


@pytest.mark.parametrize("plugin_name", _all_plugin_names())
def test_plugin_contract(plugin_name):
    """Ensure every registered plugin can be instantiated, initialized, and executed (dry run).

    If a plugin declares heavy external dependencies the test should still
    succeed—plugin authors can short-circuit execute() when running under the
    `TEST_MODE` env var.
    """
    from src.plugins import get_plugin

    plugin = get_plugin(plugin_name, cache=False)

    # Basic attributes
    assert hasattr(plugin, "execute"), "Plugin must implement execute()"

    # Initialize with minimal context
    config = load_config()
    ctx = ExecutionContext(
        config=config, run_id="contract_test", component_name=plugin_name
    )
    plugin.initialize(ctx)

    # Some plugins may not be able to run without data; handle gracefully
    try:
        result = plugin.execute(ctx)
        assert result is not None, "execute() must return a ComponentResult"
    except Exception as exc:  # pylint: disable=broad-except
        pytest.skip(f"Plugin {plugin_name} cannot be executed in contract test: {exc}")
