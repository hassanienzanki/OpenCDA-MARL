"""Path actions must not reroute vehicles outside the learning population."""
from collections import deque
from types import SimpleNamespace

import pytest

from opencda_marl._kernel import agent_manager as carla_module
from opencda_marl._kernel.agent_manager import MARLAgentManager
from opencda_marl._kernel.envs import sumo_marl_env as sumo_module
from opencda_marl._kernel.envs.episode_state import SumoEpisodeState
from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv
from opencda_marl._kernel.policy.decision_zone import DecisionZone


def _transform(x):
    return SimpleNamespace(location=SimpleNamespace(x=x, y=0.0, z=0.0),
                           rotation=SimpleNamespace(yaw=0.0))


def _event(population):
    return SimpleNamespace(vehicle_id="vehicle", blueprint=object(),
                           transform=_transform(0.0), destination=_transform(10.0),
                           target_speed=30.0, route_id=1, junction_id=4,
                           metadata={} if population is None else {"population": population})


class _Decision:
    admissible = True

    def __init__(self):
        self.destination = _transform(20.0)

    def route_update(self):
        return {"route_id": 2, "destination": self.destination}

    def as_metadata(self):
        return {"interface": "test", "route_changed": True}


def _provider(calls):
    def decide(context):
        calls.append(context)
        return _Decision()
    return decide


class _Adapter:
    def __init__(self, config, vehicle, carla_map, cav_world, agent_type, spawn_metadata):
        self.actor_id = vehicle.id
        self.vehicle = vehicle
        self.spawn_metadata = dict(spawn_metadata)
        self.vm = SimpleNamespace(update_info=lambda: None)
        self.destinations = []

    def set_destination(self, start, end, clean=False):
        self.destinations.append(end)

    def set_target_speed(self, speed):
        self.target_speed = speed

    def _calculate_distance_to_intersection(self):
        return 0.0


def _carla_manager(event, calls, trigger, monkeypatch):
    manager = object.__new__(MARLAgentManager)
    manager.agent_type = "learner"
    manager.config = {}
    manager.map = manager.cav_world = None
    manager.planner = object()
    manager.intent_provider = _provider(calls)
    manager._decision_zone = DecisionZone(trigger=trigger, distance_m=30.0)
    manager._failure_events = deque()
    manager._vehicle_adapters = []
    manager._spawn_event_by_actor_id = {}
    manager._spawn_metadata_by_actor_id = {}
    manager._spawn_retry_counts = {}
    manager._current_snapshot = None
    manager._junction_center = lambda actor_id: None
    manager.visualize_route = lambda *args: None
    vehicle = SimpleNamespace(id="vehicle", get_location=lambda: event.transform.location)
    manager.registry = SimpleNamespace(spawn_vehicle=lambda *args: vehicle,
                                       adopt_sensors=lambda *args: None,
                                       abort_spawn=lambda *args: pytest.fail("unexpected spawn failure"))
    monkeypatch.setattr(carla_module, "MARLVehicleAdapter", _Adapter)
    monkeypatch.setattr(carla_module.SensorRig, "from_vehicle_manager", lambda vm: SimpleNamespace(sensor_ids=lambda: []))
    return manager


@pytest.mark.parametrize("population,expected", [("background", False), ("learner", True), (None, True)])
@pytest.mark.parametrize("trigger", ["spawn", "approach"])
def test_carla_path_action_is_confined_to_its_population(population, expected, trigger, monkeypatch):
    event = _event(population)
    demand_destination = event.destination.location
    calls = []
    manager = _carla_manager(event, calls, trigger, monkeypatch)
    manager._spawn_vehicles([event])
    adapter = manager._vehicle_adapters[0]
    if trigger == "approach":
        # Exercise the gate even if a stale pending entry survives a reload.
        manager.decision_zone.register(adapter.actor_id)
        manager._process_decision_zone()
        manager._process_decision_zone()
    assert bool(calls) is expected
    assert len(calls) == int(expected), "one decision, with no background policy call"
    assert ("route_intent" in adapter.spawn_metadata) is expected
    assert adapter.destinations[-1].x == (20.0 if expected else 10.0)
    assert event.destination.location is demand_destination
    assert not manager.decision_zone.is_pending(adapter.actor_id)


def _sumo_env(event, calls, trigger, monkeypatch):
    env = object.__new__(SumoMARLEnv)
    env.config = {"agents": {"agent_type": "learner"}}
    env.intent_provider = _provider(calls)
    env.traffic_manager = SimpleNamespace(planner=object())
    env._sumo_intent_agent_manager = SimpleNamespace(_vehicle_adapters=[])
    env._build_active_route_adapters = lambda: []
    env._decision_zone = DecisionZone(trigger=trigger, distance_m=30.0)
    env.episode = SumoEpisodeState()
    env._distance_to_junction = lambda vehicle_id: 0.0
    env.retargets = []
    env.vehicle_spawner = SimpleNamespace(resolve_vehicle_id=lambda e: e.vehicle_id,
        get_spawn_event=lambda vid: event,
        retarget_vehicle=lambda vid, e: env.retargets.append(vid))
    monkeypatch.setattr(sumo_module, "traci", SimpleNamespace(
        vehicle=SimpleNamespace(getPosition=lambda vid: (0.0, 0.0),
                                getSpeed=lambda vid: 1.0, getAngle=lambda vid: 0.0),
        exceptions=SimpleNamespace(TraCIException=RuntimeError)))
    return env


@pytest.mark.parametrize("population,expected", [("background", False), ("learner", True), (None, True)])
@pytest.mark.parametrize("trigger", ["spawn", "approach"])
def test_sumo_path_action_is_confined_to_its_population(population, expected, trigger, monkeypatch):
    event = _event(population)
    original_destination = event.destination
    calls = []
    env = _sumo_env(event, calls, trigger, monkeypatch)
    env._apply_route_intent_to_spawn_events([event])
    if trigger == "approach":
        env.decision_zone.register(event.vehicle_id)
        env._process_decision_zone()
        env._process_decision_zone()
    assert bool(calls) is expected
    assert len(calls) == int(expected)
    assert ("route_intent" in event.metadata) is expected
    assert event.route_id == (2 if expected else 1)
    assert (event.destination is original_destination) is not expected
    assert len(env.retargets) == int(expected and trigger == "approach")
    assert not env.decision_zone.is_pending(event.vehicle_id)


def test_background_vehicles_remain_in_the_cav_route_context(monkeypatch):
    background, cav = _event("background"), _event("learner")
    background.vehicle_id, cav.vehicle_id = "bg", "cav"
    calls = []
    env = _sumo_env(cav, calls, "spawn", monkeypatch)
    env._apply_route_intent_to_spawn_events([background, cav])
    assert len(calls) == 1
    assert calls[0]["event"] is cav
    assert len(calls[0]["active_adapters"]) == 1
    visible = calls[0]["active_adapters"][0]
    assert visible.actor_id == "bg"
    assert visible.spawn_metadata["population"] == "background"
    assert background.route_id == 1 and "route_intent" not in background.metadata
