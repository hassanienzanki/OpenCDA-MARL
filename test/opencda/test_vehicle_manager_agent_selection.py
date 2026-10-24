"""VehicleManager builds the platooning agent for the ``platooning`` application.

The platooning scenarios pass ``['platooning']``; a check against any other
spelling leaves a platoon member on the plain behavior agent without an
error. The collaborating managers are stubbed, so nothing connects to a
simulator.
"""
from unittest.mock import MagicMock

import pytest

pytest.importorskip("carla", reason="the vendored vehicle manager imports carla")

from opencda.core.common import vehicle_manager as vm  # noqa: E402

_COLLABORATORS = (
    "V2XManager", "LocalizationManager", "PerceptionManager", "MapManager",
    "SafetyManager", "ControlManager", "PlatooningBehaviorAgent",
    "BehaviorAgent")

CONFIG = {
    "sensing": {"localization": {}, "perception": {}},
    "map_manager": {},
    "behavior": {},
    "controller": {},
    "v2x": {},
    "safety_manager": {},
    "platoon": {"max_capacity": 10},
}


@pytest.fixture
def stubs(monkeypatch):
    stubbed = {name: MagicMock(name=name) for name in _COLLABORATORS}
    for name, stub in stubbed.items():
        monkeypatch.setattr(vm, name, stub)
    return stubbed


def _build(application, **kwargs):
    return vm.VehicleManager(
        MagicMock(name="vehicle"), CONFIG, application,
        MagicMock(name="map"), MagicMock(name="world"), **kwargs)


def test_platooning_application_builds_the_platooning_agent(stubs):
    manager = _build(["platooning"])

    stubs["PlatooningBehaviorAgent"].assert_called_once()
    stubs["BehaviorAgent"].assert_not_called()
    assert stubs["PlatooningBehaviorAgent"].call_args.args[4] == CONFIG["platoon"]
    assert manager.agent is stubs["PlatooningBehaviorAgent"].return_value


def test_single_application_builds_the_plain_behavior_agent(stubs):
    manager = _build(["single"])

    stubs["BehaviorAgent"].assert_called_once()
    stubs["PlatooningBehaviorAgent"].assert_not_called()
    assert manager.agent is stubs["BehaviorAgent"].return_value


def test_a_passed_agent_takes_precedence_over_the_application(stubs):
    custom = MagicMock(name="custom_agent")

    manager = _build(["platooning"], agent=custom)

    assert manager.agent is custom
    stubs["PlatooningBehaviorAgent"].assert_not_called()
    stubs["BehaviorAgent"].assert_not_called()
