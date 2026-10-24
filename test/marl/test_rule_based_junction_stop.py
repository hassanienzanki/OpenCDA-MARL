"""The rule-based driver yields at a junction by stopping before the entry, not by slowing to 60 %.

Stage 1 only reduced the speed for a priority conflict and never stopped,
so a yielding vehicle could still enter the junction across a conflicting
one at crossing speed -- unlike SUMO, whose right-of-way does stop a
yielding vehicle. The yielder now brakes on a curve to a stop line before
the entry and holds there until the conflict clears; inside the junction
the slow-down rule still applies so the box is never blocked.
"""
import math

import pytest

pytest.importorskip("carla", reason="the rule-based driver is a CARLA agent; no carla package here")

from opencda_marl.builtin.agents.rule_based_agent import RuleBasedAgent  # noqa: E402


def _driver(junction_distance, *, conflict=True, stop=True, vanilla=40.0):
    agent = object.__new__(RuleBasedAgent)
    agent.junction_speed_factor = 0.6
    agent.junction_min_reduction = 15.0
    agent.junction_stop_on_conflict = stop
    agent.junction_stop_buffer = 2.0
    agent.junction_stop_decel = 1.5
    agent.abreast_guard = False               # orthogonal here; it needs an ego pose and an obstacle list
    agent.max_speed = 65.0
    agent.cruising_speed_factor = 0.95
    agent.debug_rules = False
    agent.current_stage = 0
    agent._junction_distance = float("inf")

    def should_apply():
        agent._junction_distance = junction_distance     # what _should_apply_junction_management records
        return conflict

    agent._should_apply_junction_management = should_apply
    agent._calculate_car_following_speed = lambda vanilla_speed: None
    return agent, vanilla


def test_a_yielding_driver_brakes_to_a_stop_line_before_the_entry():
    # 20 m out: the braking curve (1.5 m/s^2 to 18 m) still allows ~26 km/h, below the 60 % rule's 24 km/h? No: min of the two.
    agent, vanilla = _driver(20.0)
    curve = math.sqrt(2.0 * 1.5 * 18.0) * 3.6
    assert agent._calculate_rule_based_speed(vanilla) == pytest.approx(min(max(0.6 * vanilla, vanilla - 15.0), curve))
    assert agent.current_stage == 1
    # 6 m out: 4 m to the stop line -> ~12.5 km/h, well under the old 25 km/h.
    agent, vanilla = _driver(6.0)
    assert agent._calculate_rule_based_speed(vanilla) == pytest.approx(math.sqrt(2.0 * 1.5 * 4.0) * 3.6)
    # At the stop line and inside the buffer: a stop, held while the conflict lasts.
    for d in (2.0, 1.0, 0.5):
        agent, vanilla = _driver(d)
        assert agent._calculate_rule_based_speed(vanilla) == 0.0


def test_inside_the_junction_the_driver_keeps_moving_on_the_slow_down_rule():
    agent, vanilla = _driver(0.0)                 # the current waypoint is a junction waypoint
    assert agent._calculate_rule_based_speed(vanilla) == pytest.approx(max(0.6 * vanilla, vanilla - 15.0))


def test_the_old_behaviour_is_one_flag_away_and_no_conflict_means_no_stage_one():
    agent, vanilla = _driver(2.0, stop=False)
    assert agent._calculate_rule_based_speed(vanilla) == pytest.approx(max(0.6 * vanilla, vanilla - 15.0))
    agent, vanilla = _driver(2.0, conflict=False)
    assert agent._calculate_rule_based_speed(vanilla) == pytest.approx(vanilla * 0.95) and agent.current_stage == 3
