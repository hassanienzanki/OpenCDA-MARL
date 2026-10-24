"""The rule-based driver keeps its lane while the lane its planner steers into is occupied alongside.

The route follower's next waypoint can lie on the adjacent lane (the recorded
exit lane, or the lane a turn needs), and the vehicle steered into it without
looking, drifting into a vehicle travelling alongside. SUMO's lane-change
model waits for a gap; so does this hold. While a vehicle occupies the
target lane within the window, the driver is steered to a point ahead on its
own lane and capped below the occupant's speed, so it drops back and the gap
opens.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("carla", reason="the rule-based driver is a CARLA agent; no carla package here")

from opencda_marl.builtin.agents.rule_based_agent import RuleBasedAgent  # noqa: E402


def _loc(x, y):
    return SimpleNamespace(x=float(x), y=float(y), z=0.0)


def _wp(x, y, yaw=0.0, in_junction=False):
    w = SimpleNamespace(transform=SimpleNamespace(location=_loc(x, y), rotation=SimpleNamespace(yaw=float(yaw))),
                        is_junction=in_junction)
    w.next = lambda d: [_wp(x + d, y, yaw, in_junction)]
    return w


def _driver(occupants=(), *, in_junction=False, hold=True, ego_speed=20.0, ego_id=5):
    """The ego drives +x along y = 0; its lane centre waypoint sits at the origin.
    Occupants are (x, y, speed km/h[, actor id]) tuples."""
    agent = object.__new__(RuleBasedAgent)
    agent._vehicle = SimpleNamespace(id=ego_id)
    agent._ego_speed = float(ego_speed)
    agent.lane_hold = hold
    agent.lane_hold_behind_m = 8.0
    agent.lane_hold_ahead_m = 12.0
    agent.lane_hold_drop_kmh = 8.0
    agent.lane_hold_lookahead_m = 8.0
    agent.lane_hold_standing_kmh = 5.0
    agent.lane_hold_overlap_m = 5.5
    agent.lane_hold_active = False
    agent._ego_pos = SimpleNamespace(location=_loc(0.0, 0.0), rotation=SimpleNamespace(yaw=0.0))
    agent._map = SimpleNamespace(get_waypoint=lambda location: _wp(0.0, 0.0, 0.0, in_junction))
    agent.obstacle_vehicles = [
        SimpleNamespace(id=(occ[3] if len(occ) > 3 else 9),
                        get_location=lambda _x=occ[0], _y=occ[1]: _loc(_x, _y),
                        get_velocity=lambda _v=occ[2]: SimpleNamespace(x=_v / 3.6, y=0.0, z=0.0))
        for occ in occupants]
    return agent


def test_a_target_on_the_own_lane_passes_through():
    agent = _driver(occupants=[(2.0, 3.5, 25.0)])
    target = _loc(10.0, 0.3)
    assert agent._hold_lane(target) == (target, None) and agent.lane_hold_active is False


def test_the_hold_binds_only_while_the_target_lane_is_occupied_alongside():
    target = _loc(10.0, 3.5)                       # the next lane to the left
    agent = _driver(occupants=[(2.0, 3.5, 25.0)])   # a neighbour 2 m ahead in that lane at 25 km/h
    location, cap = agent._hold_lane(target)
    assert (location.x, location.y) == (8.0, 0.0), "steered to a point ahead on the own lane"
    assert cap == pytest.approx(17.0), "held 8 km/h below the occupant"
    assert agent.lane_hold_active is True
    agent = _driver(occupants=[(20.0, 3.5, 25.0)])  # the same neighbour 20 m ahead: outside the window
    assert agent._hold_lane(target) == (target, None)
    agent = _driver(occupants=[(2.0, -3.5, 25.0)])  # a neighbour on the other side
    assert agent._hold_lane(target) == (target, None)
    agent = _driver(occupants=[(2.0, 7.5, 25.0)])   # two lanes over
    assert agent._hold_lane(target) == (target, None)


def test_the_slowest_occupant_sets_the_cap_and_the_hold_never_asks_for_reverse():
    target = _loc(10.0, 3.5)
    agent = _driver(occupants=[(-3.0, 3.5, 30.0), (5.0, 3.5, 4.0)])
    _location, cap = agent._hold_lane(target)
    assert cap == pytest.approx(0.0)


def test_a_standing_queue_in_the_target_lane_blocks_only_where_the_bodies_would_overlap():
    target = _loc(10.0, 3.5)
    # A standing car 8 m ahead in the target lane leaves room to slip in beside it: no hold.
    agent = _driver(occupants=[(8.0, 3.5, 0.0)])
    assert agent._hold_lane(target) == (target, None)
    # The same car 4 m ahead overlaps the ego's body: hold, at a stop.
    agent = _driver(occupants=[(4.0, 3.5, 0.0)])
    location, cap = agent._hold_lane(target)
    assert (location.x, location.y) == (8.0, 0.0) and cap == pytest.approx(0.0)
    # A moving car 8 m ahead is inside the moving window: hold.
    agent = _driver(occupants=[(8.0, 3.5, 20.0)])
    assert agent._hold_lane(target)[1] == pytest.approx(12.0)


def test_two_standing_vehicles_beside_each_other_do_not_wait_for_ever():
    target = _loc(10.0, 3.5)
    # Both standing, abreast, each wanting the other's lane: the lower id pulls straight ahead, uncapped.
    agent = _driver(occupants=[(0.5, 3.5, 0.0, 9)], ego_speed=0.0, ego_id=5)
    location, cap = agent._hold_lane(target)
    assert (location.x, location.y) == (8.0, 0.0) and cap is None and agent.lane_hold_active is True
    # The higher id holds at a stop and lets the other go first.
    agent = _driver(occupants=[(0.5, 3.5, 0.0, 2)], ego_speed=0.0, ego_id=5)
    location, cap = agent._hold_lane(target)
    assert (location.x, location.y) == (8.0, 0.0) and cap == pytest.approx(0.0)
    # A moving ego beside a standing car is not the standoff: it holds and stops.
    agent = _driver(occupants=[(0.5, 3.5, 0.0, 9)], ego_speed=20.0, ego_id=5)
    assert agent._hold_lane(target)[1] == pytest.approx(0.0)


def test_inside_a_junction_and_with_the_hold_off_the_planner_is_never_overridden():
    target = _loc(10.0, 3.5)
    agent = _driver(occupants=[(2.0, 3.5, 25.0)], in_junction=True)
    assert agent._hold_lane(target) == (target, None)
    agent = _driver(occupants=[(2.0, 3.5, 25.0)], hold=False)
    assert agent._hold_lane(target) == (target, None)
