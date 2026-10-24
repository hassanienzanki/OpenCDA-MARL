"""SUMO's heading, velocity and neighbours are in the frame the learner sees.

SUMO reports a compass angle: degrees, clockwise from north. The observation
builder used it as a mathematical heading, so on SUMO the learner's heading
feature, the trace's heading and velocity, and every time-to-collision
derived from them were 90 degrees off and mirrored. Positions, meanwhile,
were in two frames at once: relative-to-junction bridged into CARLA's,
absolute in SUMO's own.

Now the bridge owns the convention, every SUMO kinematic quantity is in the
bridged frame, and the neighbour block and the min TTC come from the same
world-snapshot code CARLA uses, instead of a leader-only proxy.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

import opencda_marl._kernel.envs.sumo_marl_env as sumo_env_mod
from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv
from opencda_marl._kernel.traffic.coordinate_bridge import CoordinateBridge, sumo_angle_to_heading
from opencda_marl._kernel.traffic.sumo_route_intent import make_active_vehicle_proxy


# ─────────────────────────────────────────────────────────────────────────────
# The convention
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("angle_deg, unit", [
    (0.0, (0.0, 1.0)),      # north
    (90.0, (1.0, 0.0)),     # east
    (180.0, (0.0, -1.0)),   # south
    (270.0, (-1.0, 0.0)),   # west
])
def test_the_compass_angle_becomes_a_heading_in_sumos_own_frame(angle_deg, unit):
    heading = sumo_angle_to_heading(angle_deg)
    assert (math.cos(heading), math.sin(heading)) == pytest.approx(unit, abs=1e-9)
    assert -math.pi <= heading < math.pi


def test_a_flipped_frame_mirrors_the_heading_about_x():
    north = sumo_angle_to_heading(0.0, y_flip=True)
    assert (math.cos(north), math.sin(north)) == pytest.approx((0.0, -1.0), abs=1e-9)
    east = sumo_angle_to_heading(90.0, y_flip=True)
    assert (math.cos(east), math.sin(east)) == pytest.approx((1.0, 0.0), abs=1e-9)
    bridge = CoordinateBridge((0.0, 0.0), y_flip=True)
    assert bridge.heading_from_sumo_angle(0.0) == pytest.approx(north)
    assert CoordinateBridge((0.0, 0.0), y_flip=False).heading_from_sumo_angle(0.0) == pytest.approx(
        sumo_angle_to_heading(0.0))


def test_the_active_proxy_moves_the_way_it_points():
    east = make_active_vehicle_proxy(vehicle_id="v", spawn_event=None, position_xy=(0.0, 0.0),
                                     speed_kmh=36.0, angle_deg=90.0)
    assert (east.velocity.x, east.velocity.y) == pytest.approx((10.0, 0.0), abs=1e-9)
    north = make_active_vehicle_proxy(vehicle_id="v", spawn_event=None, position_xy=(0.0, 0.0),
                                      speed_kmh=36.0, angle_deg=0.0)
    assert (north.velocity.x, north.velocity.y) == pytest.approx((0.0, 10.0), abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# The observation builder
# ─────────────────────────────────────────────────────────────────────────────

class _Vehicles:
    """The TraCI vehicle domain the builder reads, for a scripted scene."""

    def __init__(self, scene):
        self.scene = scene           # id -> (x, y, speed_ms, angle_deg)

    def getIDList(self):
        return tuple(self.scene)

    def getPosition(self, vid):
        return self.scene[vid][0], self.scene[vid][1]

    def getSpeed(self, vid):
        return self.scene[vid][2]

    def getAngle(self, vid):
        return self.scene[vid][3]

    def getLeader(self, vid, dist=100.0):
        return None

    def getLaneID(self, vid):
        return "edge_0"

    def getLaneIndex(self, vid):
        return 0


def _env(monkeypatch, scene, *, y_flip, net_offset=(100.0, 200.0)):
    monkeypatch.setattr(sumo_env_mod.traci, "vehicle", _Vehicles(scene))
    env = object.__new__(SumoMARLEnv)
    env.episode = SimpleNamespace(terminal_agents=set(), active_vehicles={})
    env.coordinates = CoordinateBridge(net_offset, y_flip=y_flip)
    env.vehicle_spawner = None
    env.marl_config = {"algorithm": "pcmappo"}
    env.current_step = 0
    env._junction_centre_for = lambda vid: (0.0, 0.0)
    env._estimate_distance_to_destination = lambda vid, lane: 50.0
    env._population_of = lambda vid: ("marl", True)
    env._committed_exit_lanes = lambda vid, metadata: [0.0, 0.0, 0.0, 0.0]
    return env


def test_heading_velocity_and_location_share_the_bridged_frame(monkeypatch):
    # One vehicle at SUMO (110, 220), i.e. bridged (10, 20), driving east at 10 m/s.
    env = _env(monkeypatch, {"a": (110.0, 220.0, 10.0, 90.0)}, y_flip=True)
    obs = env._get_observations()["a"]
    assert (obs["location"]["x"], obs["location"]["y"]) == pytest.approx((10.0, -20.0))
    assert obs["heading_angle"] == pytest.approx(0.0, abs=1e-9)
    assert (obs["velocity"]["x"], obs["velocity"]["y"]) == pytest.approx((10.0, 0.0), abs=1e-9)
    assert obs["speed"] == pytest.approx(36.0)
    # Heading north in SUMO is -y in a flipped frame, and the velocity follows.
    env = _env(monkeypatch, {"a": (110.0, 220.0, 10.0, 0.0)}, y_flip=True)
    obs = env._get_observations()["a"]
    assert obs["heading_angle"] == pytest.approx(-math.pi / 2, abs=1e-9)
    assert (obs["velocity"]["x"], obs["velocity"]["y"]) == pytest.approx((0.0, -10.0), abs=1e-9)


def test_the_velocity_agrees_with_the_displacement_the_positions_imply(monkeypatch):
    """The property the old code violated: integrate the velocity for one
    step and land where the next bridged position is."""
    dt = 0.05
    first = {"a": (110.0, 220.0, 10.0, 45.0)}                         # north-east, 10 m/s
    step_east, step_north = 10.0 * dt * math.sin(math.radians(45)), 10.0 * dt * math.cos(math.radians(45))
    second = {"a": (110.0 + step_east, 220.0 + step_north, 10.0, 45.0)}
    for flip in (False, True):
        obs1 = _env(monkeypatch, first, y_flip=flip)._get_observations()["a"]
        obs2 = _env(monkeypatch, second, y_flip=flip)._get_observations()["a"]
        predicted = (obs1["location"]["x"] + obs1["velocity"]["x"] * dt,
                     obs1["location"]["y"] + obs1["velocity"]["y"] * dt)
        assert predicted == pytest.approx((obs2["location"]["x"], obs2["location"]["y"]), abs=1e-6), f"y_flip={flip}"


def test_neighbours_and_min_ttc_come_from_the_shared_snapshot(monkeypatch):
    # Two vehicles 20 m apart on the same line, closing head-on at 10 m/s each:
    # constant-velocity TTC to a 4 m collision radius is (20 - 4) / 20 = 0.8 s.
    scene = {"a": (100.0, 200.0, 10.0, 90.0),      # at bridged (0, 0), east
             "b": (120.0, 200.0, 10.0, 270.0)}     # at bridged (20, 0), west
    env = _env(monkeypatch, scene, y_flip=False)
    obs = env._get_observations()
    assert obs["a"]["min_ttc"] == pytest.approx(0.8, abs=1e-6)
    assert obs["b"]["min_ttc"] == pytest.approx(0.8, abs=1e-6)
    block = obs["a"]["nearby_vehicles"]
    assert len(block) == 5 * 7
    rel_x, rel_y, rel_vx, rel_vy, heading_diff, distance, ttc = block[:7]
    assert rel_x == pytest.approx(20.0 / 50.0) and rel_y == pytest.approx(0.0)
    assert rel_vx == pytest.approx(-20.0 / 40.0) and rel_vy == pytest.approx(0.0, abs=1e-9)
    assert abs(heading_diff) == pytest.approx(1.0)            # pi apart, normalised by pi
    assert distance == pytest.approx(20.0 / 50.0) and ttc == pytest.approx(0.8 / 10.0)
    assert block[7:] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0] * 4  # no other neighbours
    # A lone vehicle sees empty slots and no collision ahead.
    alone = _env(monkeypatch, {"a": scene["a"]}, y_flip=False)._get_observations()["a"]
    assert alone["min_ttc"] == float("inf")
    assert alone["nearby_vehicles"] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0] * 5


def test_the_neighbour_block_respects_the_configured_radius_and_clamps(monkeypatch):
    scene = {"a": (100.0, 200.0, 0.0, 90.0), "b": (130.0, 200.0, 0.0, 90.0)}   # 30 m apart
    env = _env(monkeypatch, scene, y_flip=False)
    env.marl_config = {"algorithm": "pcmappo", "pcmappo": {"nearby_vehicle_config": {"detection_radius": 25.0, "max_vehicles": 2}}}
    block = env._get_observations()["a"]["nearby_vehicles"]
    # The observation contract pads the block to its own slot count on both
    # backends; what the radius decides is that every slot stays empty.
    assert len(block) == 5 * 7 and block == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0] * 5, "30 m is outside a 25 m radius"
