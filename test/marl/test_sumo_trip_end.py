"""Where a SUMO trip ends.

The recording puts a vehicle's destination a fixed offset past the junction
exit, and a path decision that changes the route moves it to the same offset
past the new exit. Three things made SUMO end trips elsewhere: the waypoint
walk stopped at a lane's end (so a route's destination sat at the next
junction), the destination edge was picked by vertex distance (so a long
edge lost to a short one nearby), and SUMO's arrival at the end of the
target edge decided the trip length.
"""
from types import SimpleNamespace

import pytest


@pytest.fixture
def sumo_adapter():
    try:
        import opencda_marl._kernel.traffic.sumo_adapter as mod
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"SUMO runtime unavailable: {exc}")
    return mod


def _fake_lanes(monkeypatch, mod, lanes, links):
    """lanes: lane_id -> shape (SUMO coordinates); links: lane_id -> successor lane id."""
    def length(lane_id):
        s = lanes[lane_id]
        return sum(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 for (x1, y1), (x2, y2) in zip(s, s[1:]))
    monkeypatch.setattr(mod.traci.lane, "getShape", lambda lane_id: lanes[lane_id])
    monkeypatch.setattr(mod.traci.lane, "getLength", length)
    monkeypatch.setattr(mod.traci.lane, "getLinks",
                        lambda lane_id: [(links[lane_id], True, True, False, "", "", "s", 0.0)] if lane_id in links else [])
    monkeypatch.setattr(mod, "_COORDINATE_BRIDGE", None)


def test_the_waypoint_walk_continues_onto_the_linked_lane(sumo_adapter, monkeypatch):
    mod = sumo_adapter
    # lane A_0 runs 10 m east, links to B_0 which runs 40 m further east
    _fake_lanes(monkeypatch, mod, {"A_0": [(0.0, 0.0), (10.0, 0.0)], "B_0": [(10.0, 0.0), (50.0, 0.0)]}, {"A_0": "B_0"})
    start = mod.SumoWaypoint("A", 0, (0.0, 0.0), road_id=1, lane_id=-1)
    (ahead,) = start.next(15.0)
    assert ahead.edge_id == "B" and ahead.lane_index == 0
    sumo_x, sumo_y = mod._carla_to_sumo_xy(*ahead.position)
    assert abs(sumo_x - 15.0) < 1e-6 and abs(sumo_y) < 1e-6
    # one-metre steps, as move_along_lane takes them, cross the boundary too
    step = start
    for _ in range(12):
        (step,) = step.next(1.0)
    assert step.edge_id == "B" and abs(mod._carla_to_sumo_xy(*step.position)[0] - 12.0) < 1e-6
    # a dead end stops at the lane's end instead of failing
    (end,) = mod.SumoWaypoint("B", 0, (40.0, 0.0), road_id=2, lane_id=-1).next(30.0)
    assert end.edge_id == "B" and abs(mod._carla_to_sumo_xy(*end.position)[0] - 50.0) < 1e-6


def test_the_walk_measures_progress_through_the_published_bridge(sumo_adapter, monkeypatch):
    mod = sumo_adapter
    _fake_lanes(monkeypatch, mod, {"A_0": [(0.0, 0.0), (10.0, 0.0)], "B_0": [(10.0, 0.0), (50.0, 0.0)]}, {"A_0": "B_0"})
    # a bridge that mirrors y and shifts x: positions stored in CARLA must come back exactly
    bridge = SimpleNamespace(sumo_to_carla=lambda x, y: (x + 100.0, -y), carla_to_sumo=lambda x, y: (x - 100.0, -y))
    monkeypatch.setattr(mod, "_COORDINATE_BRIDGE", bridge)
    start = mod.SumoWaypoint("A", 0, (4.0, 0.0), road_id=1, lane_id=-1)      # 4 m along A_0
    (ahead,) = start.next(20.0)
    assert ahead.edge_id == "B" and abs(mod._carla_to_sumo_xy(*ahead.position)[0] - 24.0) < 1e-6


def test_candidate_edges_rank_by_distance_to_the_lane_not_its_vertices(monkeypatch):
    try:
        import opencda_marl._kernel.traffic.sumo_spawner as mod
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"SUMO runtime unavailable: {exc}")
    shapes = {"long_0": [(0.0, 0.0), (100.0, 0.0)], "short_0": [(50.0, 8.0), (60.0, 8.0)]}
    monkeypatch.setattr(mod.traci.edge, "getIDList", lambda: ["long", "short", ":internal"])
    monkeypatch.setattr(mod.traci.edge, "getLaneNumber", lambda edge_id: 1)
    monkeypatch.setattr(mod.traci.lane, "getShape", lambda lane_id: shapes[lane_id])
    spawner = object.__new__(mod.SumoVehicleSpawner)
    ranked = mod.SumoVehicleSpawner._get_candidate_edges(spawner, 50.0, 1.0, max_distance=50.0)
    assert [edge for edge, _ in ranked] == ["long", "short"]
    assert abs(ranked[0][1] - 1.0) < 1e-9 and abs(ranked[1][1] - 7.0) < 1e-9


def test_a_sumo_trip_ends_at_its_destination_point(monkeypatch):
    try:
        import opencda_marl._kernel.envs.sumo_marl_env as mod
        from opencda_marl._kernel.envs.episode_state import SumoEpisodeState
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"SUMO runtime unavailable: {exc}")
    env = object.__new__(mod.SumoMARLEnv)
    env.episode = SumoEpisodeState()
    env.current_step = 700
    env.arrival_radius = 5.0
    env.coordinates = SimpleNamespace(sumo_to_carla=lambda x, y: (x, y))
    destination = SimpleNamespace(location=SimpleNamespace(x=50.0, y=0.0))
    events = {"near": SimpleNamespace(destination=destination), "far": SimpleNamespace(destination=destination)}
    removed, forgotten = [], []
    env.vehicle_spawner = SimpleNamespace(get_spawn_event=events.get, remove_vehicle=forgotten.append)
    env._queue_route_intent_terminal = lambda veh_id: None
    positions = {"near": (47.0, 1.0), "far": (10.0, 0.0)}
    monkeypatch.setattr(mod.traci.vehicle, "getIDList", lambda: ("near", "far"))
    monkeypatch.setattr(mod.traci.vehicle, "getPosition", lambda veh_id: positions[veh_id])
    monkeypatch.setattr(mod.traci.vehicle, "remove", lambda veh_id: removed.append(veh_id))
    monkeypatch.setattr(mod.traci.simulation, "getArrivedIDList", lambda: ())
    monkeypatch.setattr(mod.traci.simulation, "getCollisions", lambda: ())

    found = mod.SumoMARLEnv._check_events(env)

    assert [e.event_type for e in found] == ["success"] and found[0].vehicle_id == "near"
    assert removed == ["near"] and forgotten == ["near"]
    assert "near" in env.episode.departed_vehicles and "far" not in env.episode.departed_vehicles
