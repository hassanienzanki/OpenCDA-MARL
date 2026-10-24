"""The rule-based driver drops back when a same-direction neighbour alongside closes in laterally.

Two vehicles side by side in adjacent lanes, travelling the same way, can
collide without any longitudinal rule noticing: one drifts toward the other
until the bodies touch, and neither has a leader to follow. The abreast
guard is geometric and asymmetric: the vehicle that is behind in a closing
side-by-side pair caps its speed below the neighbour's until the pair is no
longer alongside.
"""
import math
from types import SimpleNamespace

import pytest

pytest.importorskip("carla", reason="the rule-based driver is a CARLA agent; no carla package here")

from opencda_marl.builtin.agents.rule_based_agent import RuleBasedAgent  # noqa: E402


class _Location:
    def __init__(self, x, y):
        self.x, self.y, self.z = float(x), float(y), 0.0

    def distance(self, other):
        return math.hypot(other.x - self.x, other.y - self.y)


def _vehicle(vid, x, y, yaw_deg=0.0, speed_kmh=0.0):
    yaw = math.radians(yaw_deg)
    v = speed_kmh / 3.6
    transform = SimpleNamespace(location=_Location(x, y), rotation=SimpleNamespace(yaw=yaw_deg))
    return SimpleNamespace(id=vid, get_location=lambda: transform.location, get_transform=lambda: transform,
                           get_velocity=lambda: SimpleNamespace(x=v * math.cos(yaw), y=v * math.sin(yaw), z=0.0),
                           bounding_box=SimpleNamespace(extent=SimpleNamespace(x=2.35, y=0.9, z=0.7)))


def _driver(others, *, ego_id=10, ego_speed=20.0, guard=True):
    agent = object.__new__(RuleBasedAgent)
    ego = _vehicle(ego_id, 0.0, 0.0, speed_kmh=ego_speed)
    agent._vehicle = ego
    agent._ego_pos = ego.get_transform()
    agent._ego_speed = float(ego_speed)
    agent.obstacle_vehicles = list(others)
    agent.same_lane_tolerance = math.radians(30)
    agent.abreast_guard = guard
    agent.abreast_lateral_m = 3.2
    agent.abreast_hard_m = 2.6
    agent.abreast_closing_m = 0.015
    agent.abreast_creep_kmh = 5.0
    agent._abreast_memory = {}
    agent.abreast_longitudinal_m = 6.0
    agent.abreast_drop_kmh = 8.0
    # the stages, wired to "cruising" so only the guard can bind
    agent.junction_speed_factor = 0.6
    agent.junction_min_reduction = 15.0
    agent.junction_stop_on_conflict = True
    agent.junction_stop_buffer = 2.0
    agent.junction_stop_decel = 1.5
    agent.max_speed = 65.0
    agent.cruising_speed_factor = 0.95
    agent.debug_rules = False
    agent.current_stage = 0
    agent._junction_distance = float("inf")
    agent.abreast_active = False
    agent._should_apply_junction_management = lambda: False
    agent._calculate_car_following_speed = lambda vanilla_speed: None
    return agent


def test_the_vehicle_behind_in_a_closing_pair_drops_below_the_neighbour():
    # A neighbour 2.5 m to the left (closer than a lane apart), 2 m ahead, at 20 km/h: the ego is behind and yields.
    neighbour = _vehicle(3, 2.0, 2.5, speed_kmh=20.0)
    agent = _driver([neighbour])
    assert agent._abreast_cap() == pytest.approx(12.0)
    assert agent._calculate_rule_based_speed(40.0) == pytest.approx(12.0)
    assert agent.abreast_active and agent.current_stage == 2, "the guard reports as a following rule (stage 2)"
    # A stopped neighbour in the hard zone while the ego still moves: the yielder stops rather than going backwards.
    agent = _driver([_vehicle(3, 2.0, 2.5, speed_kmh=0.0)])
    assert agent._calculate_rule_based_speed(40.0) == 0.0
    # Both at rest and not closing: a stationary pair cannot collide, the rear one creeps ahead to end it.
    agent = _driver([_vehicle(3, 2.0, 2.5, speed_kmh=0.0)], ego_speed=0.0)
    assert agent._calculate_rule_based_speed(40.0) == pytest.approx(5.0)
    # ...unless the neighbour is still sliding closer.
    agent = _driver([_vehicle(3, 2.0, 2.5, speed_kmh=0.0)], ego_speed=0.0)
    agent._abreast_memory = {3: 2.6}
    assert agent._calculate_rule_based_speed(40.0) == 0.0


def test_the_vehicle_ahead_carries_on_and_ties_go_by_actor_id():
    # The same neighbour 2 m behind: the ego is ahead, no cap.
    agent = _driver([_vehicle(3, -2.0, 2.5, speed_kmh=20.0)])
    assert agent._abreast_cap() is None and agent._calculate_rule_based_speed(40.0) == pytest.approx(38.0)
    assert not agent.abreast_active and agent.current_stage == 3
    # Exactly abreast: the larger actor id yields.
    assert _driver([_vehicle(3, 0.0, 2.5, speed_kmh=20.0)], ego_id=10)._abreast_cap() == pytest.approx(12.0)
    assert _driver([_vehicle(30, 0.0, 2.5, speed_kmh=20.0)], ego_id=10)._abreast_cap() is None


def test_a_lane_apart_or_far_along_or_the_other_way_does_not_count():
    assert _driver([_vehicle(3, 2.0, 3.5, speed_kmh=20.0)])._abreast_cap() is None, "adjacent lanes at 3.5 m are not closing"
    assert _driver([_vehicle(3, 8.0, 2.5, speed_kmh=20.0)])._abreast_cap() is None, "8 m ahead is a leader's business"
    assert _driver([_vehicle(3, 2.0, 2.5, yaw_deg=180.0, speed_kmh=20.0)])._abreast_cap() is None, "oncoming traffic"
    assert _driver([_vehicle(3, 2.0, 2.5, yaw_deg=70.0, speed_kmh=20.0)])._abreast_cap() is None, "crossing traffic"


def test_the_guard_is_one_flag_away_and_takes_the_tightest_neighbour():
    agent = _driver([_vehicle(3, 2.0, 2.5, speed_kmh=20.0)], guard=False)
    assert agent._calculate_rule_based_speed(40.0) == pytest.approx(38.0) and not agent.abreast_active
    agent = _driver([_vehicle(3, 2.0, 2.5, speed_kmh=30.0), _vehicle(4, 1.0, -2.4, speed_kmh=12.0)])   # both in the hard zone
    assert agent._abreast_cap() == pytest.approx(4.0)


def test_in_the_soft_zone_only_a_closing_moving_neighbour_binds():
    # 2.9 m: closer than a lane but outside the hard zone. Steady -> no cap, even though it is ahead.
    steady = _vehicle(3, 2.0, 2.9, speed_kmh=20.0)
    agent = _driver([steady])
    assert agent._abreast_cap() is None
    assert agent._abreast_cap() is None, "a second step at the same lateral distance is still steady"
    # Now it has come 3 cm closer since the last step (the drifts in the traces run at ~2.5 cm per step): closing -> the ego (behind) drops back.
    agent.obstacle_vehicles = [_vehicle(3, 2.0, 2.87, speed_kmh=20.0)]
    assert agent._abreast_cap() == pytest.approx(12.0)
    # A stopped neighbour in the soft zone never binds, closing or not: parallel queues must not lock.
    agent = _driver([_vehicle(3, 2.0, 2.9, speed_kmh=0.0)])
    agent._abreast_memory = {3: 3.1}
    assert agent._abreast_cap() is None
    # Inside the hard zone a stopped neighbour does bind while the ego moves: the bodies nearly touch.
    agent = _driver([_vehicle(3, 2.0, 2.4, speed_kmh=0.0)])
    assert agent._abreast_cap() == 0.0
