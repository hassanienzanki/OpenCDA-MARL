"""The component registry: pluggable planners and agent types.

Same contract as the algorithm registry — register at import time through
the plugin door, resolve by key from configuration, fail loudly, never
name-dispatch inside the engine.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def components():
    from opencda_marl.api import components as comp

    before = ({k: dict(v) for k, v in comp._COMPONENTS.items()},
              {k: dict(v) for k, v in comp._COMPONENT_SOURCES.items()})
    yield comp
    comp._COMPONENTS.clear(); comp._COMPONENTS.update(
        {k: dict(v) for k, v in before[0].items()})
    comp._COMPONENT_SOURCES.clear(); comp._COMPONENT_SOURCES.update(
        {k: dict(v) for k, v in before[1].items()})


def test_register_resolve_and_override(components):
    @components.register_component("planner", "test_planner")
    class TestPlanner:
        def __init__(self, world, config):
            self.world, self.config = world, config

    assert components.resolve_component("planner", "test_planner") is TestPlanner
    with pytest.raises(ValueError, match="already registered"):
        components.register_component("planner", "test_planner")(object)
    components.register_component("planner", "test_planner", override=True)(dict)
    assert components.resolve_component("planner", "test_planner") is dict


def test_unknown_key_lists_what_exists(components):
    with pytest.raises(ValueError, match="Unknown planner component") as err:
        components.resolve_component("planner", "no_such_planner")
    assert "marl" in str(err.value)


def test_builtin_planner_is_registered(components):
    from opencda_marl.builtin.planners.marl_planner import MARLPlanner

    assert components.resolve_component("planner", "marl") is MARLPlanner
    assert "marl" in components.registered_components("planner")


def test_traffic_manager_resolves_planner_by_config(components):
    from opencda_marl._kernel.traffic.traffic_manager import _resolve_planner
    from opencda_marl.builtin.planners.marl_planner import MARLPlanner

    assert _resolve_planner({}) is MARLPlanner

    @components.register_component("planner", "custom_routes")
    class CustomPlanner:
        def __init__(self, world, config):
            pass

    assert _resolve_planner({"type": "custom_routes"}) is CustomPlanner


def test_builtin_agent_types_are_registered(components):
    from opencda_marl.builtin.agents.agent_factory import AgentFactory

    available = AgentFactory.get_available_types()
    assert {"behavior", "vanilla", "rule_based", "marl"} <= set(available)
    assert AgentFactory.validate_type("rule_based")
    assert not AgentFactory.validate_type("no_such_agent")


def test_custom_agent_type_constructs_through_the_factory(components):
    from opencda_marl.builtin.agents.agent_factory import AgentFactory

    built = {}

    @components.register_component("agent", "test_driver")
    def _build(vehicle, carla_map, config):
        built.update(vehicle=vehicle, carla_map=carla_map,
                     section=config.get("test_driver"))
        return "the-agent"

    agent = AgentFactory.get_agent("test_driver", "veh", "map",
                                   {"test_driver": {"aggressiveness": 0.7}})
    assert agent == "the-agent"
    assert built == {"vehicle": "veh", "carla_map": "map",
                     "section": {"aggressiveness": 0.7}}


def test_describe_components_reports_sources(components):
    rows = {(r["kind"], r["key"]): r["source"] for r in components.describe_components()}
    assert ("planner", "marl") in rows
    assert ("agent", "marl") in rows
