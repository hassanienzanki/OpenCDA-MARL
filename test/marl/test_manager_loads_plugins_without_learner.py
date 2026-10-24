"""``MARL.plugins`` reaches a run that has no learner.

The manager loaded ``MARL.plugins`` only inside ``if algorithm is not
None``, so a run without a learner (rule-based, behavior, vanilla) never
imported the configured plugins: a planner, agent type or path provider
registered only through ``MARL.plugins`` was an unknown component there.
The manager now loads the list for every run.

The coordinator also loads ``MARL.plugins`` before either backend builds
anything, so the manager's call is the second one in a coordinated run.
Loading the same plugin again is a no-op: re-executing a ``.py`` plugin
would register its components a second time, fail on the duplicate key and
report a working plugin as failed.
"""
from __future__ import annotations

import sys
import textwrap

import pytest

from opencda_marl._kernel.policy.manager import MARLManager


@pytest.fixture
def restore_plugins():
    from opencda_marl.api import components as comp
    from opencda_marl.plugins import loader, registry

    snap = (
        {k: dict(v) for k, v in comp._COMPONENTS.items()},
        {k: dict(v) for k, v in comp._COMPONENT_SOURCES.items()},
        dict(loader._plugin_failures), loader._plugins_loaded,
        {k: list(v) for k, v in registry._JOURNAL.items()},
        set(loader._loaded_extras),
    )
    yield
    comp._COMPONENTS.clear(); comp._COMPONENTS.update(snap[0])
    comp._COMPONENT_SOURCES.clear(); comp._COMPONENT_SOURCES.update(snap[1])
    loader._plugin_failures.clear(); loader._plugin_failures.update(snap[2])
    loader._plugins_loaded = snap[3]
    registry._JOURNAL.clear(); registry._JOURNAL.update(snap[4])
    loader._loaded_extras.clear(); loader._loaded_extras.update(snap[5])
    for name in [m for m in sys.modules if m.startswith("opencda_marl_plugin_")]:
        sys.modules.pop(name, None)


def test_a_run_without_a_learner_loads_its_configured_plugins(restore_plugins, tmp_path):
    from opencda_marl.api.components import registered_components
    from opencda_marl.plugins.loader import plugin_failures

    plugin = tmp_path / "baseline_parts.py"
    plugin.write_text(textwrap.dedent("""
        from opencda_marl.api.components import register_component

        @register_component("planner", "baseline_probe_planner")
        class ProbePlanner:
            def __init__(self, world, config): ...
        """), encoding="utf-8")

    manager = MARLManager({"plugins": [str(plugin)]}, None)

    assert manager.algorithm is None
    assert str(plugin) not in plugin_failures()
    assert "baseline_probe_planner" in registered_components("planner")


def test_a_plugin_the_coordinator_loaded_is_not_loaded_again(restore_plugins, tmp_path):
    """Coordinator first, manager second: the plugin stays loaded, not failed."""
    from opencda_marl.api.components import registered_components
    from opencda_marl.plugins.loader import load_plugins, plugin_failures

    plugin = tmp_path / "twice_loaded_parts.py"
    plugin.write_text(textwrap.dedent("""
        from opencda_marl.api.components import register_component

        @register_component("planner", "twice_loaded_probe_planner")
        class ProbePlanner:
            def __init__(self, world, config): ...
        """), encoding="utf-8")

    load_plugins([str(plugin)])
    first_module = sys.modules["opencda_marl_plugin_twice_loaded_parts"]
    MARLManager({"plugins": [str(plugin)]}, None)

    assert str(plugin) not in plugin_failures()
    assert "twice_loaded_probe_planner" in registered_components("planner")
    # Not re-executed: the module the registrations came from is still the
    # one in sys.modules.
    assert sys.modules["opencda_marl_plugin_twice_loaded_parts"] is first_module
