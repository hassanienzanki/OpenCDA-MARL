"""SUMO path decisions score every vehicle in one coordinate frame.

A spawn-time ego is built from its spawn event, whose transform is in the
CARLA frame. The live neighbours were built from raw TraCI positions, in
SUMO's frame, so the nearest-distance and time-to-collision features of a
candidate path mixed two frames offset by the network's netOffset (and
mirrored on a y-flipped map). Every proxy the route provider sees is now in
the bridged (CARLA) frame, with a velocity that follows the bridged heading.
"""
import math
from types import SimpleNamespace

import pytest

from opencda_marl._kernel.traffic.coordinate_bridge import CoordinateBridge
from opencda_marl._kernel.traffic.sumo_route_intent import make_spawn_event_proxy
from opencda_marl.api.route_intent import RouteIntentCandidate, candidate_traffic_features


@pytest.fixture
def sumo_env():
    try:
        import opencda_marl._kernel.envs.sumo_marl_env as mod
        from opencda_marl._kernel.envs.episode_state import SumoEpisodeState
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"SUMO runtime unavailable: {exc}")
    return mod, SumoEpisodeState


class _Vehicles:
    """The TraCI vehicle domain the proxy builder reads: id -> (x, y, m/s, compass deg)."""

    def __init__(self, scene):
        self.scene = scene

    def getIDList(self):
        return tuple(self.scene)

    def getPosition(self, vid):
        return self.scene[vid][0], self.scene[vid][1]

    def getSpeed(self, vid):
        return self.scene[vid][2]

    def getAngle(self, vid):
        return self.scene[vid][3]


def _event(x, y, *, exit_direction="north", exit_lane_idx=0, route_id=7, yaw=0.0):
    return SimpleNamespace(
        vehicle_id="ego", route_id=route_id, lane_id=0, target_speed=36.0,
        transform=SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=0.0),
                                  rotation=SimpleNamespace(yaw=yaw)),
        metadata={"exit_direction": exit_direction, "exit_lane_idx": exit_lane_idx,
                  "route_id": route_id},
    )


def _env(mod, episode_cls, monkeypatch, scene, *, net_offset=(100.0, 200.0), y_flip=True):
    monkeypatch.setattr(mod.traci, "vehicle", _Vehicles(scene))
    env = object.__new__(mod.SumoMARLEnv)
    env.episode = episode_cls()
    env.coordinates = CoordinateBridge(net_offset, y_flip=y_flip)
    events = {vid: _event(0.0, 0.0) for vid in scene}
    env.vehicle_spawner = SimpleNamespace(get_spawn_event=events.get)
    return env


def test_neighbour_proxies_are_in_the_bridged_frame(sumo_env, monkeypatch):
    mod, episode_cls = sumo_env
    # SUMO (110, 220) is CARLA (10, -20) under offset (100, 200) with the y mirror;
    # compass 0 deg (SUMO north) is -y in the mirrored frame.
    env = _env(mod, episode_cls, monkeypatch, {"a": (110.0, 220.0, 10.0, 0.0)})

    (proxy,) = env._build_active_route_adapters()

    assert (proxy.location.x, proxy.location.y) == pytest.approx((10.0, -20.0))
    assert (proxy.velocity.x, proxy.velocity.y) == pytest.approx((0.0, -10.0), abs=1e-9)
    assert proxy.current_speed == pytest.approx(36.0)
    assert proxy.spawn_metadata["route_id"] == 7


def test_terminal_vehicles_are_not_neighbours(sumo_env, monkeypatch):
    mod, episode_cls = sumo_env
    env = _env(mod, episode_cls, monkeypatch, {"a": (110.0, 220.0, 0.0, 0.0),
                                               "done": (120.0, 220.0, 0.0, 0.0)})
    env.episode.terminal_agents.add("done")
    assert [p.actor_id for p in env._build_active_route_adapters()] == ["a"]


def test_a_spawn_ego_and_a_neighbour_at_the_same_point_are_zero_apart(sumo_env, monkeypatch):
    """The invariant the path features rely on: ego and neighbours share a frame."""
    mod, episode_cls = sumo_env
    # The neighbour sits at SUMO (110, 220) = CARLA (10, -20), exactly where
    # the ego's spawn transform (CARLA frame) puts the ego.
    env = _env(mod, episode_cls, monkeypatch, {"a": (110.0, 220.0, 0.0, 90.0)})
    neighbours = env._build_active_route_adapters()
    ego = make_spawn_event_proxy(_event(10.0, -20.0), actor_id="ego")
    candidate = RouteIntentCandidate(
        intent=None, route_id=7, junction_id=4, dest_tf=None,
        exit_direction="north", exit_lane_idx=0, is_original=True)

    features = candidate_traffic_features(candidate, neighbours, ego_adapter=ego)

    assert features.active_same_exit_lane_count == 1
    assert features.nearest_same_exit_lane_distance_m == pytest.approx(0.0, abs=1e-9)


def test_the_approach_mode_ego_uses_the_same_frame(sumo_env, monkeypatch):
    mod, episode_cls = sumo_env
    env = _env(mod, episode_cls, monkeypatch, {"a": (130.0, 190.0, 5.0, 90.0)})
    proxy = env._active_route_proxy("a", env.vehicle_spawner.get_spawn_event("a"))
    assert (proxy.location.x, proxy.location.y) == pytest.approx((30.0, 10.0))
    # SUMO east stays +x under the mirror.
    assert (proxy.velocity.x, proxy.velocity.y) == pytest.approx((5.0, 0.0), abs=1e-9)
    assert math.hypot(proxy.velocity.x, proxy.velocity.y) == pytest.approx(5.0)


def test_a_generated_spawn_ego_heads_along_its_lane(sumo_env, monkeypatch):
    """Live/record spawn transforms carry yaw 0; the ego takes its lane's heading."""
    mod, episode_cls = sumo_env
    env = _env(mod, episode_cls, monkeypatch, {})
    env.traffic_manager = SimpleNamespace(mode="live")
    monkeypatch.setattr(mod.traci, "simulation",
                        SimpleNamespace(convertRoad=lambda x, y: ("e", 0.0, 0)))
    # SUMO lane heading north: CARLA (30, -10) -> (30, -60) under the mirror.
    monkeypatch.setattr(mod.traci, "lane",
                        SimpleNamespace(getShape=lambda lane_id: [(130.0, 210.0), (130.0, 260.0)]))
    event = _event(30.0, -30.0)

    heading = env._spawn_lane_heading(event)
    ego = make_spawn_event_proxy(event, actor_id="ego", heading_rad=heading)

    # 36 km/h = 10 m/s along the lane, the same velocity a neighbour driving
    # SUMO-north on that lane gets (see test_neighbour_proxies_are_in_the_bridged_frame).
    assert (ego.velocity.x, ego.velocity.y) == pytest.approx((0.0, -10.0), abs=1e-9)


def test_a_replayed_spawn_keeps_its_recorded_yaw(sumo_env, monkeypatch):
    mod, episode_cls = sumo_env
    env = _env(mod, episode_cls, monkeypatch, {})
    env.traffic_manager = SimpleNamespace(mode="replay")
    assert env._spawn_lane_heading(_event(30.0, -30.0, yaw=90.0)) is None
