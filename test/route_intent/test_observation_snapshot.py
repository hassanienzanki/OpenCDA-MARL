"""WorldSnapshot: one-round-trip observation source.

The load-bearing property is equivalence: the snapshot path must produce
the same neighbour feature vector and front-vehicle distance as the legacy
per-vehicle RPC path, so trained checkpoints read an unchanged observation.
"""
import math

import numpy as np
import pytest

from opencda_marl._kernel.adapter.world_snapshot import WorldSnapshot
from opencda_marl._kernel.adapter._nearby.nearby import compute_nearby_vehicle_features
from opencda_marl._kernel.adapter._spatial.spatial import calculate_distance_to_front_vehicle


# --------------------------------------------------------------------- #
# Fake CARLA geometry
# --------------------------------------------------------------------- #
class Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y, self.z - other.z)


class Rot:
    def __init__(self, yaw):
        self.yaw = float(yaw)


class Transform:
    def __init__(self, loc, yaw):
        self.location = loc
        self.rotation = Rot(yaw)

    def get_forward_vector(self):
        r = math.radians(self.rotation.yaw)
        return Vec(math.cos(r), math.sin(r), 0.0)


class FakeVehicle:
    _next = 100

    def __init__(self, x, y, vx=0.0, vy=0.0, yaw=0.0, lane=1, road=7):
        self.id = FakeVehicle._next
        FakeVehicle._next += 1
        self.type_id = "vehicle.fake"
        self.is_alive = True
        self._loc = Vec(x, y, 0.0)
        self._vel = Vec(vx, vy, 0.0)
        self._yaw = yaw
        self.lane, self.road = lane, road

    def get_location(self):
        return self._loc

    def get_velocity(self):
        return self._vel

    def get_transform(self):
        return Transform(self._loc, self._yaw)


class ActorList(list):
    def filter(self, pattern):
        return ActorList(a for a in self if a.type_id.startswith(pattern.rstrip("*").rstrip(".")))


class FakeWorld:
    def __init__(self, vehicles):
        self.vehicles = list(vehicles)
        self.get_actors_calls = 0

    def get_actors(self):
        self.get_actors_calls += 1
        return ActorList(self.vehicles)


class Waypoint:
    def __init__(self, lane, road):
        self.lane_id, self.road_id = lane, road


class FakeMap:
    """Waypoint from the vehicle whose location this is (by identity)."""

    def __init__(self, vehicles):
        self._by_loc = {id(v._loc): v for v in vehicles}

    def get_waypoint(self, loc):
        v = self._by_loc.get(id(loc))
        return Waypoint(v.lane, v.road) if v else None


def _scene():
    ego = FakeVehicle(0.0, 0.0, vx=10.0, vy=0.0, yaw=0.0, lane=1, road=7)
    ahead_same_lane = FakeVehicle(20.0, 0.0, vx=8.0, vy=0.0, yaw=0.0, lane=1, road=7)
    ahead_other_lane = FakeVehicle(12.0, 3.5, vx=9.0, vy=0.0, yaw=0.0, lane=2, road=7)
    behind = FakeVehicle(-15.0, 0.0, vx=11.0, vy=0.0, yaw=0.0, lane=1, road=7)
    crossing = FakeVehicle(30.0, -30.0, vx=0.0, vy=12.0, yaw=90.0, lane=1, road=9)
    far = FakeVehicle(300.0, 300.0, vx=0.0, vy=0.0, yaw=45.0, lane=1, road=3)
    vehicles = [ego, ahead_same_lane, ahead_other_lane, behind, crossing, far]
    return ego, vehicles, FakeWorld(vehicles), FakeMap(vehicles)


PARAMS = dict(nearby_detection_radius=50.0, max_nearby_vehicles=5,
              features_per_vehicle=7, max_relative_velocity=40.0, max_ttc=10.0)


def test_neighbour_features_match_legacy_path():
    ego, vehicles, world, cmap = _scene()
    legacy, legacy_min_ttc = compute_nearby_vehicle_features(ego, world, **PARAMS)

    snap = WorldSnapshot.capture(world)
    new, new_min_ttc = snap.neighbour_features(
        ego.id, radius=PARAMS["nearby_detection_radius"], k=PARAMS["max_nearby_vehicles"],
        max_relative_velocity=PARAMS["max_relative_velocity"], max_ttc=PARAMS["max_ttc"],
        features_per_vehicle=7)

    assert len(new) == len(legacy) == 35
    np.testing.assert_allclose(new, legacy, atol=1e-6)
    assert new_min_ttc == pytest.approx(legacy_min_ttc)


def test_front_vehicle_distance_matches_legacy_path():
    ego, vehicles, world, cmap = _scene()
    legacy = calculate_distance_to_front_vehicle(ego, world, cmap, ego.id)
    snap = WorldSnapshot.capture(world)
    new = snap.front_vehicle_distance(ego.id, cmap)
    assert new == pytest.approx(legacy)
    assert new == pytest.approx(20.0)  # same lane, ahead; other-lane car ignored


def test_snapshot_makes_one_actor_query_for_all_vehicles():
    ego, vehicles, world, cmap = _scene()
    world.get_actors_calls = 0
    snap = WorldSnapshot.capture(world)
    for v in vehicles:
        snap.neighbour_features(v.id, radius=50.0, k=5,
                                max_relative_velocity=40.0, max_ttc=10.0)
        snap.front_vehicle_distance(v.id, cmap)
    # 6 vehicles x (nearby + front) used to be 12 get_actors calls; now 1.
    assert world.get_actors_calls == 1


def test_dead_actors_are_skipped_at_capture():
    ego, vehicles, world, cmap = _scene()
    vehicles[1].is_alive = False
    snap = WorldSnapshot.capture(world)
    assert vehicles[1].id not in snap.by_id
    assert len(snap) == len(vehicles) - 1


def test_unknown_actor_yields_empty_slots():
    ego, vehicles, world, cmap = _scene()
    snap = WorldSnapshot.capture(world)
    feats, min_ttc = snap.neighbour_features(999999, radius=50.0, k=5,
                                             max_relative_velocity=40.0, max_ttc=10.0)
    assert feats == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0] * 5
    assert min_ttc == float("inf")
    assert snap.front_vehicle_distance(999999, cmap) == 999.0
