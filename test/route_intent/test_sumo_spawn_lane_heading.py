"""A SUMO spawn-time ego heads the way its spawn lane runs, in the bridged frame.

A spawn event generated on SUMO takes its transform from a lane waypoint
whose yaw is always 0, so the spawn-time ego built from it headed +x
whatever its lane, while the live neighbours carry their real headings
through the coordinate bridge. The proxy builder takes the heading from the
caller, and the lane helper computes it from the lane's SUMO polyline
through the same bridge as the positions: on a network with a netOffset and
a y mirror, a vehicle driving a lane and the lane's own heading agree.
"""
import math
from types import SimpleNamespace

import pytest

from opencda_marl._kernel.traffic.coordinate_bridge import CoordinateBridge
from opencda_marl._kernel.traffic.sumo_route_intent import (
    make_spawn_event_proxy,
    spawn_lane_heading_rad,
)


def _event(x, y, yaw=0.0, speed_kmh=36.0):
    return SimpleNamespace(
        vehicle_id="ego", route_id=3, target_speed=speed_kmh, metadata={},
        transform=SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=0.0),
                                  rotation=SimpleNamespace(yaw=yaw)))


@pytest.mark.parametrize("y_flip", [False, True])
def test_the_lane_heading_matches_a_vehicle_driving_the_lane(y_flip):
    bridge = CoordinateBridge((100.0, 200.0), y_flip=y_flip)
    # A lane running SUMO north (compass 0 deg), well away from the origin.
    shape = [(130.0, 210.0), (130.0, 260.0)]
    spawn = bridge.sumo_to_carla(130.0, 230.0)

    heading = spawn_lane_heading_rad(shape, spawn, bridge=bridge)

    assert heading == pytest.approx(bridge.heading_from_sumo_angle(0.0))
    assert heading == pytest.approx(-math.pi / 2 if y_flip else math.pi / 2)


def test_the_nearest_segment_of_a_bent_lane_gives_the_heading():
    bridge = CoordinateBridge((100.0, 200.0), y_flip=True)
    # SUMO east for 50 m, then SUMO north for 50 m.
    shape = [(100.0, 200.0), (150.0, 200.0), (150.0, 250.0)]
    on_first = bridge.sumo_to_carla(120.0, 200.5)
    on_second = bridge.sumo_to_carla(149.5, 240.0)

    assert spawn_lane_heading_rad(shape, on_first, bridge=bridge) == pytest.approx(0.0)
    assert spawn_lane_heading_rad(shape, on_second, bridge=bridge) == pytest.approx(-math.pi / 2)


def test_a_shape_without_two_distinct_points_has_no_heading():
    assert spawn_lane_heading_rad([], (0.0, 0.0)) is None
    assert spawn_lane_heading_rad([(5.0, 5.0), (5.0, 5.0)], (0.0, 0.0)) is None


def test_without_a_bridge_the_default_one_is_used():
    shape = [(10.0, 10.0), (10.0, 60.0)]
    default = CoordinateBridge()
    spawn = default.sumo_to_carla(10.0, 30.0)
    assert spawn_lane_heading_rad(shape, spawn) == pytest.approx(math.pi / 2)


def test_the_spawn_proxy_follows_a_given_heading_else_the_transform_yaw():
    bridge = CoordinateBridge((100.0, 200.0), y_flip=True)
    shape = [(130.0, 210.0), (130.0, 260.0)]
    x, y = bridge.sumo_to_carla(130.0, 230.0)
    event = _event(x, y, yaw=0.0, speed_kmh=36.0)          # a lane waypoint's yaw

    lane = make_spawn_event_proxy(
        event, heading_rad=spawn_lane_heading_rad(shape, (x, y), bridge=bridge))
    # The neighbour on the same lane at the same speed, as the SUMO environment
    # builds it: velocity from the bridged compass heading.
    neighbour_heading = bridge.heading_from_sumo_angle(0.0)
    assert (lane.velocity.x, lane.velocity.y) == pytest.approx(
        (10.0 * math.cos(neighbour_heading), 10.0 * math.sin(neighbour_heading)), abs=1e-9)
    assert (lane.location.x, lane.location.y) == pytest.approx((30.0, -30.0))

    # No heading given: the transform's yaw, unchanged for recorded events.
    recorded = make_spawn_event_proxy(_event(x, y, yaw=90.0, speed_kmh=36.0))
    assert (recorded.velocity.x, recorded.velocity.y) == pytest.approx((0.0, 10.0), abs=1e-9)
    assert make_spawn_event_proxy(event).velocity.x == pytest.approx(10.0)
