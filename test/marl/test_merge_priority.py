"""Front-to-back clearance avoids a standing merge/following dependency."""

import math
from types import SimpleNamespace

import pytest

from opencda_marl.builtin.agents.merge_priority import standing_merge_precedes


@pytest.mark.parametrize('ego_id,other_id', [(165, 133), (133, 165)])
def test_front_vehicle_precedes_independently_of_actor_id(ego_id, other_id):
    args = (-2.9283, 4.4368, math.radians(-105.0913), math.radians(-90.4574), math.pi / 6)
    assert standing_merge_precedes(ego_id, other_id, *args)
    dx, dy, ego_yaw, other_yaw, tolerance = args
    assert not standing_merge_precedes(other_id, ego_id, -dx, -dy, other_yaw, ego_yaw, tolerance)


def test_side_by_side_tie_retains_deterministic_id_order():
    assert standing_merge_precedes(1, 2, 3.5, 0, -math.pi / 2, -math.pi / 2, math.pi / 6)
    assert not standing_merge_precedes(2, 1, -3.5, 0, -math.pi / 2, -math.pi / 2, math.pi / 6)


def test_opposing_or_non_finite_headings_preserve_legacy_order():
    assert standing_merge_precedes(1, 2, 0, 3, 0, math.pi, math.pi / 6)
    assert not standing_merge_precedes(2, 1, 0, -3, math.pi, 0, math.pi / 6)
    assert standing_merge_precedes(1, 2, math.nan, 0, 0, 0, math.pi / 6)


def _lane_hold_scene(enabled):
    pytest.importorskip('carla')
    from opencda_marl.builtin.agents.rule_based_agent import RuleBasedAgent

    agent = object.__new__(RuleBasedAgent)
    agent.lane_hold = True
    agent.lane_hold_geometry_priority = enabled
    agent._ego_speed = 0.0
    agent.same_lane_tolerance = math.pi / 6
    agent.lane_hold_standing_kmh = 5.0
    agent.lane_hold_overlap_m = 5.5
    agent.lane_hold_behind_m = 8.0
    agent.lane_hold_ahead_m = 12.0
    agent.lane_hold_drop_kmh = 8.0
    agent.lane_hold_lookahead_m = 8.0
    agent._vehicle = SimpleNamespace(id=165)
    agent._ego_pos = SimpleNamespace(
        location=SimpleNamespace(x=4.8558, y=17.3928),
        rotation=SimpleNamespace(yaw=-105.0913))
    ahead = SimpleNamespace(x=5.3, y=9.3928)
    waypoint = SimpleNamespace(
        is_junction=False,
        transform=SimpleNamespace(location=SimpleNamespace(x=5.3, y=17.3928),
                                  rotation=SimpleNamespace(yaw=-90.0)),
        next=lambda distance: [SimpleNamespace(transform=SimpleNamespace(location=ahead))])
    agent._map = SimpleNamespace(get_waypoint=lambda location: waypoint)
    other = SimpleNamespace(
        id=133,
        get_location=lambda: SimpleNamespace(x=1.9275, y=21.8296),
        get_transform=lambda: SimpleNamespace(rotation=SimpleNamespace(yaw=-90.4574)))
    agent.obstacle_vehicles = [other]
    agent._get_vehicle_speed = lambda vehicle: 0.0
    return agent, ahead, SimpleNamespace(x=1.8295, y=6.2745)


def test_recorded_merge_wait_can_clear_forward_in_its_current_lane():
    agent, ahead, target = _lane_hold_scene(True)
    location, cap = agent._hold_lane(target)
    assert location is ahead and cap is None
    assert agent.lane_hold_active
    assert agent.merge_priority_override


def test_disabled_option_preserves_the_original_block():
    agent, ahead, target = _lane_hold_scene(False)
    location, cap = agent._hold_lane(target)
    assert location is ahead and cap == 0.0 and agent.lane_hold_active


def test_merge_release_diagnostics_reset_when_no_hold_is_needed():
    agent, ahead, target = _lane_hold_scene(True)
    agent._hold_lane(target)
    assert agent.merge_priority_release and agent.merge_priority_override
    agent.lane_hold = False
    assert agent._hold_lane(target) == (target, None)
    assert not agent.merge_priority_release and not agent.merge_priority_override


# --------------------------------------------------------------------------- #
# Two agents: the merging vehicle and the standing vehicle behind it
# --------------------------------------------------------------------------- #

def _built_driver(monkeypatch, vehicle_id, config):
    """A RuleBasedAgent through its own __init__, with the CARLA-bound base
    constructor stubbed out, so the configuration defaults are the real ones."""
    pytest.importorskip('carla')
    from opencda_marl.builtin.agents.rule_based_agent import RuleBasedAgent
    from opencda_marl.builtin.agents.vanilla_agent import VanillaAgent

    monkeypatch.setattr(VanillaAgent, '__init__', lambda self, *args, **kwargs: None)
    agent = RuleBasedAgent(SimpleNamespace(id=vehicle_id), None, dict(config))
    agent._vehicle = SimpleNamespace(id=vehicle_id)
    agent.debug_rules = False
    return agent


def _obstacle(actor_id, x, y, yaw_deg=0.0, speed_kmh=0.0):
    """What perception hands a driver: an ObstacleVehicle built from a snapshot
    record (it carries carla_id, not id)."""
    carla = pytest.importorskip('carla')
    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle

    location = carla.Location(x=float(x), y=float(y), z=0.0)
    speed = speed_kmh / 3.6
    yaw = math.radians(yaw_deg)
    return ObstacleVehicle.from_snapshot(SimpleNamespace(
        actor_id=actor_id, type_id='vehicle.test', location=location,
        transform=carla.Transform(location, carla.Rotation(yaw=float(yaw_deg))),
        velocity=carla.Vector3D(speed * math.cos(yaw), speed * math.sin(yaw), 0.0),
        bounding_box=carla.BoundingBox(carla.Location(), carla.Vector3D(2.4, 0.9, 0.75))))


def _merge_standoff(monkeypatch, config):
    """The merging vehicle (id 20) stands 1.6 m off its lane centre, angled
    10 deg into the left lane; the vehicle with id 10 stands 4 m behind it in
    that lane. Returns (merging driver, its lane-ahead point, its target,
    follower driver)."""
    carla = pytest.importorskip('carla')
    merging = _built_driver(monkeypatch, 20, config)
    merging._ego_speed = 0.0
    merging._ego_pos = carla.Transform(carla.Location(x=0.0, y=1.6, z=0.0), carla.Rotation(yaw=10.0))
    ahead = SimpleNamespace(x=8.0, y=0.0)
    lane_centre = SimpleNamespace(
        is_junction=False,
        transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0), rotation=SimpleNamespace(yaw=0.0)),
        next=lambda distance: [SimpleNamespace(transform=SimpleNamespace(location=ahead))])
    merging._map = SimpleNamespace(get_waypoint=lambda location: lane_centre)
    merging.obstacle_vehicles = [_obstacle(10, -4.0, 3.5)]

    follower = _built_driver(monkeypatch, 10, config)
    follower._ego_speed = 0.0
    follower._ego_pos = carla.Transform(carla.Location(x=-4.0, y=3.5, z=0.0), carla.Rotation(yaw=0.0))
    follower.obstacle_vehicles = [_obstacle(20, 0.0, 1.6, yaw_deg=10.0)]
    return merging, ahead, SimpleNamespace(x=10.0, y=3.5), follower


def test_geometry_priority_is_on_with_the_lane_hold_by_default(monkeypatch):
    assert _built_driver(monkeypatch, 1, {}).lane_hold_geometry_priority is True
    assert _built_driver(monkeypatch, 1, {'lane_hold_geometry_priority': False}).lane_hold_geometry_priority is False


def test_the_default_releases_a_merging_vehicle_ahead_of_its_standing_follower(monkeypatch):
    merging, ahead, target, follower = _merge_standoff(monkeypatch, {})
    # The follower waits on the merging vehicle's body in its corridor ...
    assert follower._calculate_car_following_speed(30.0) == 0.0
    # ... so the merging vehicle, the front one, must be the one to go.
    location, cap = merging._hold_lane(target)
    assert location is ahead and cap is None
    assert merging.merge_priority_release and merging.merge_priority_override


def test_ordering_by_actor_id_alone_deadlocks_the_pair(monkeypatch):
    merging, ahead, target, follower = _merge_standoff(
        monkeypatch, {'lane_hold_geometry_priority': False})
    location, cap = merging._hold_lane(target)
    assert location is ahead and cap == 0.0, "the larger id holds at a stop"
    assert follower._calculate_car_following_speed(30.0) == 0.0, "and its follower waits on it"
