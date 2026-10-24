"""The public light-ignore setting must reach BasicAgent's red-light check."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch  # noqa: F401 - load Torch DLLs before the CARLA stack on Windows.
import carla
import pytest

from opencda_marl.builtin.agents import basic_agent
from opencda_marl.builtin.agents.vanilla_agent import VanillaAgent


@pytest.mark.parametrize("ignore_lights", [False, True])
def test_vanilla_light_setting_controls_cached_red_light(monkeypatch, ignore_lights):
    # Stub only expensive map/planner construction. The actual BasicAgent
    # constructor and traffic-light predicate execute unchanged.
    monkeypatch.setattr(basic_agent, "LocalPlanner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(basic_agent, "GlobalRoutePlannerDAO", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(basic_agent, "GlobalRoutePlanner", lambda *a, **kw: MagicMock())
    vehicle = MagicMock()
    vehicle.id = 7
    vehicle.get_world().get_actors().filter.return_value = []
    agent = VanillaAgent(vehicle, None, {
        "debug": False, "local_planner": {}, "ignore_traffic_light": ignore_lights,
        "max_speed": 45.0, "emergency_param": 0.4, "collision_time_ahead": 1.5,
    })
    red = SimpleNamespace(state=carla.TrafficLightState.Red)
    agent._last_traffic_light = red

    affected, light = agent._affected_by_traffic_light([red], max_distance=5.0)

    assert affected is (not ignore_lights)
    assert light is (None if ignore_lights else red)
