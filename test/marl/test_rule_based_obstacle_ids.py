"""The rule-based driver identifies neighbours by the id perception gives them.

Perception hands the driver ``ObstacleVehicle`` objects, which carry
``carla_id`` and no ``id``. The abreast guard read ``.id``, so every
neighbour had id 0: the tie-break between two exactly abreast vehicles made
both yield, and all neighbours shared one slot of the closing memory. These
tests build the neighbours the way perception does
(``ObstacleVehicle.from_snapshot``) rather than from stand-ins with an ``id``.
"""
import math
from types import SimpleNamespace

import pytest

carla = pytest.importorskip("carla", reason="the rule-based driver is a CARLA agent; no carla package here")

from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle  # noqa: E402
from opencda_marl.builtin.agents.rule_based_agent import RuleBasedAgent  # noqa: E402


def _obstacle(actor_id, x, y, yaw_deg=0.0, speed_kmh=0.0):
    location = carla.Location(x=float(x), y=float(y), z=0.0)
    speed = speed_kmh / 3.6
    yaw = math.radians(yaw_deg)
    return ObstacleVehicle.from_snapshot(SimpleNamespace(
        actor_id=actor_id, type_id="vehicle.test", location=location,
        transform=carla.Transform(location, carla.Rotation(yaw=float(yaw_deg))),
        velocity=carla.Vector3D(speed * math.cos(yaw), speed * math.sin(yaw), 0.0),
        bounding_box=carla.BoundingBox(carla.Location(), carla.Vector3D(2.4, 0.9, 0.75))))


def _guard(ego_id, neighbours, ego_speed=20.0):
    agent = object.__new__(RuleBasedAgent)
    agent._vehicle = SimpleNamespace(id=ego_id)
    agent._ego_pos = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation(yaw=0.0))
    agent._ego_speed = float(ego_speed)
    agent.obstacle_vehicles = list(neighbours)
    agent.same_lane_tolerance = math.radians(30)
    agent.abreast_lateral_m = 3.2
    agent.abreast_hard_m = 2.6
    agent.abreast_closing_m = 0.015
    agent.abreast_creep_kmh = 5.0
    agent.abreast_longitudinal_m = 6.0
    agent.abreast_drop_kmh = 8.0
    agent._abreast_memory = {}
    return agent


def test_obstacle_vehicles_carry_carla_id_and_no_id():
    neighbour = _obstacle(10, 0.0, 2.5)
    assert neighbour.carla_id == 10 and not hasattr(neighbour, "id")


def test_of_an_exactly_abreast_pair_only_the_larger_id_yields():
    # Both at 20 km/h, side by side in the hard zone, no longitudinal offset.
    larger = _guard(30, [_obstacle(10, 0.0, 2.5, speed_kmh=20.0)])
    smaller = _guard(10, [_obstacle(30, 0.0, 2.5, speed_kmh=20.0)])
    assert larger._abreast_cap() == pytest.approx(12.0), "the larger id drops 8 km/h below its neighbour"
    assert smaller._abreast_cap() is None, "the smaller id keeps going"


def test_each_neighbour_keeps_its_own_closing_memory():
    agent = _guard(30, [_obstacle(11, 1.0, 3.0, speed_kmh=20.0), _obstacle(12, -1.0, -3.0, speed_kmh=20.0)])
    agent._abreast_cap()
    assert agent._abreast_memory == pytest.approx({11: 3.0, 12: 3.0})
