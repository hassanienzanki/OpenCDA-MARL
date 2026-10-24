"""Yielding at a junction must still respect a stopped leading vehicle."""
from types import SimpleNamespace

import pytest

from opencda_marl.builtin.agents.rule_based_agent import RuleBasedAgent


def _driver(junction, following_speed):
    agent = object.__new__(RuleBasedAgent)
    agent._vehicle = SimpleNamespace(id=1)
    agent.junction_speed_factor = 0.6
    agent.junction_min_reduction = 15.0
    agent.junction_stop_on_conflict = False
    agent.cruising_speed_factor = 0.9
    agent.abreast_guard = False
    agent.max_speed = 45.0
    agent.debug_rules = False
    agent._should_apply_junction_management = lambda: junction
    agent._calculate_car_following_speed = lambda _: following_speed
    return agent


def test_yielding_does_not_command_motion_into_a_stopped_leader():
    agent = _driver(junction=True, following_speed=0.0)
    assert agent._calculate_rule_based_speed(30.0) == 0.0
    assert agent.current_stage == 1


@pytest.mark.parametrize('following_speed, expected', [(4.0, 4.0), (30.0, 18.0), (None, 18.0)])
def test_the_stricter_junction_or_following_cap_wins(following_speed, expected):
    assert _driver(True, following_speed)._calculate_rule_based_speed(30.0) == pytest.approx(expected)


def test_following_and_cruising_outside_a_junction_keep_their_contract():
    assert _driver(False, 4.0)._calculate_rule_based_speed(30.0) == 4.0
    assert _driver(False, None)._calculate_rule_based_speed(30.0) == pytest.approx(27.0)
