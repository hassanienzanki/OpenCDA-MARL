"""Mixed autonomy on SUMO: background vehicles are SUMO's own drivers, and
every shim position goes through one calibrated frame bridge."""
import json
from types import SimpleNamespace

from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv
from opencda_marl._kernel.traffic import sumo_adapter, sumo_spawner
from opencda_marl._kernel.traffic.coordinate_bridge import CoordinateBridge
from opencda_marl._kernel.traffic.sumo_spawner import (
    SUMO_DEFAULT_LANE_CHANGE_MODE,
    SUMO_RL_SPEED_CHECKS_MODE,
    SumoVehicleSpawner,
)


class _Recorder:
    """Stands in for ``traci.vehicle``: records every call it receives."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _record(*args):
            self.calls.append((name, args))
        return _record


def test_background_vehicles_are_left_to_sumo_with_their_recorded_driver(monkeypatch):
    vehicle = _Recorder()
    monkeypatch.setattr(sumo_spawner, "traci", SimpleNamespace(vehicle=vehicle))
    spawner = SumoVehicleSpawner(learning_agent_type="marl")

    background = SimpleNamespace(metadata={
        "population": "background",
        "driver": {"speed_factor": 0.9, "tau": 1.4, "min_gap": 2.0, "colour": "red"},
    })
    assert spawner._is_background(background)
    spawner._configure_background_vehicle_runtime("v1", background.metadata["driver"])
    assert ("setSpeedMode", ("v1", SUMO_RL_SPEED_CHECKS_MODE)) in vehicle.calls
    assert ("setLaneChangeMode", ("v1", SUMO_DEFAULT_LANE_CHANGE_MODE)) in vehicle.calls
    assert ("setSpeedFactor", ("v1", 0.9)) in vehicle.calls
    assert ("setTau", ("v1", 1.4)) in vehicle.calls
    assert ("setMinGap", ("v1", 2.0)) in vehicle.calls
    assert not any(name == "setColour" for name, _ in vehicle.calls)   # unknown keys are ignored
    assert vehicle.calls[-1] == ("setSpeed", ("v1", -1))                # control released to the model

    # The learner's own population, and homogeneous runs, are not background.
    assert not spawner._is_background(SimpleNamespace(metadata={"population": "marl"}))
    assert not spawner._is_background(SimpleNamespace(metadata={}))
    # Without a known learning population the spawner keeps its historical behaviour.
    assert not SumoVehicleSpawner()._is_background(background)


def test_shim_positions_go_through_the_published_bridge(monkeypatch):
    junction_api = SimpleNamespace(getPosition=lambda _id: (110.0, 120.0), getShape=lambda _id: [])
    monkeypatch.setattr(sumo_adapter, "traci", SimpleNamespace(junction=junction_api))
    try:
        sumo_adapter.set_coordinate_bridge(CoordinateBridge((10.0, 20.0), y_flip=True))
        waypoint = sumo_adapter.SumoWaypoint("e1", 0, (110.0, 120.0), road_id=1, lane_id=-1)
        assert waypoint.position == (100.0, -100.0)
        assert (waypoint.transform.location.x, waypoint.transform.location.y) == (100.0, -100.0)
        junction = sumo_adapter.SumoJunction("7")
        assert (junction.bounding_box.location.x, junction.bounding_box.location.y) == (100.0, -100.0)
        spawner = SumoVehicleSpawner(bridge=sumo_adapter.coordinate_bridge())
        assert spawner._carla_to_sumo(100.0, -100.0) == (110.0, 120.0)
        assert spawner._sumo_to_carla(110.0, 120.0) == (100.0, -100.0)
    finally:
        sumo_adapter.set_coordinate_bridge(None)
    # Without a published bridge the legacy constant offset applies unchanged.
    legacy = sumo_adapter.SumoWaypoint("e1", 0, (110.0, 120.0), road_id=1, lane_id=-1)
    assert legacy.position == (110.0 - 99.8, 120.0 - 100.0)


def _net(tmp_path, name="town.net.xml"):
    net = tmp_path / name
    net.write_text(
        '<?xml version="1.0"?>\n<net>\n'
        '  <location netOffset="100.00,100.00" convBoundary="0,0,1,1" origBoundary="0,0,1,1" projParameter="!"/>\n'
        '  <junction id="7" type="priority" x="150.00" y="40.00"/>\n'
        '  <junction id="12" type="priority" x="180.00" y="300.00"/>\n'
        '  <junction id="cluster_1" type="priority" x="1.0" y="2.0"/>\n'
        '</net>\n', encoding="utf-8")
    return net


def _env(runtime=None):
    env = object.__new__(SumoMARLEnv)
    env.coordinates = CoordinateBridge((100.0, 100.0))
    env.runtime_config = runtime or {}
    return env


# The mirrored convention lands the SUMO nodes on these CARLA centres.
_CENTRES = {"7": [50.0, 60.0], "12": [80.0, -200.0]}


def test_landmarks_come_from_the_configured_file(tmp_path):
    net = _net(tmp_path)
    (tmp_path / "elsewhere.json").write_text(json.dumps(_CENTRES), encoding="utf-8")
    env = _env({"coordinate_landmarks": str(tmp_path / "elsewhere.json")})
    try:
        SumoMARLEnv._calibrate_coordinates(env, str(net))
        assert env.coordinates.y_flip
        x, y = env.coordinates.sumo_to_carla(150.0, 40.0)
        assert abs(x - 50.0) < 1e-6 and abs(y - 60.0) < 1e-6
    finally:
        sumo_adapter.set_coordinate_bridge(None)


def test_an_adjacent_landmarks_file_is_found_without_configuration(tmp_path):
    net = _net(tmp_path)
    (tmp_path / "town_landmarks.json").write_text(json.dumps(_CENTRES), encoding="utf-8")
    env = _env()
    try:
        SumoMARLEnv._calibrate_coordinates(env, str(net))
        assert env.coordinates.y_flip
    finally:
        sumo_adapter.set_coordinate_bridge(None)


def test_a_missing_or_absent_landmark_file_keeps_the_default_convention(tmp_path):
    net = _net(tmp_path, name="intersection.net.xml")
    # Nothing adjacent, nothing configured: the bundled intersection's case.
    env = _env()
    try:
        SumoMARLEnv._calibrate_coordinates(env, str(net))
        assert not env.coordinates.y_flip
        assert sumo_adapter.coordinate_bridge() is env.coordinates
    finally:
        sumo_adapter.set_coordinate_bridge(None)
    # A configured file that does not exist warns but never fails the run.
    env = _env({"coordinate_landmarks": str(tmp_path / "missing.json")})
    try:
        SumoMARLEnv._calibrate_coordinates(env, str(net))
        assert not env.coordinates.y_flip
        assert sumo_adapter.coordinate_bridge() is env.coordinates
    finally:
        sumo_adapter.set_coordinate_bridge(None)
