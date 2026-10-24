"""The recorded driver profile reaches the CARLA driver the way it reaches SUMO's.

Every recorded vehicle carries a ``driver`` block -- speed_factor, tau,
min_gap, imperfection, max_speed_kmh -- and the SUMO spawner hands all of it
to the car-following model (``sumo_spawner.BACKGROUND_DRIVER_SETTERS``). The
CARLA side mapped four of the five and dropped ``imperfection``, so its
background traffic drove without the Krauss dawdling that makes SUMO traffic
break down, and applied speed_factor only while cruising. Both are the
driver's own parameters and belong in every regime.
"""
import math
from types import SimpleNamespace

import pytest

pytest.importorskip("carla", reason="the rule-based driver is a CARLA agent; no carla package here")

from opencda_marl.builtin.agents.rule_based_agent import RuleBasedAgent  # noqa: E402


def _driver(**config):
    """A rule-based agent with only the speed path configured."""
    agent = object.__new__(RuleBasedAgent)
    agent.time_headway = config.get('time_headway', 2.0)
    agent.minimum_distance_buffer = config.get('minimum_distance_buffer', 3.0)
    agent.max_speed = config.get('max_speed', 65.0)
    agent.cruising_speed_factor = config.get('cruising_speed_factor', 0.95)
    agent.desired_speed_factor = config.get('desired_speed_factor', 1.0)
    agent.driver_imperfection = config.get('driver_imperfection', 0.0)
    agent.driver_imperfection_accel = config.get('driver_imperfection_accel', 2.6)
    agent.driver_imperfection_hold_steps = config.get('driver_imperfection_hold_steps', 0)
    agent.driver_imperfection_scale = config.get('driver_imperfection_scale', 1.0)
    agent._imperfection_rng = __import__('random').Random(
        RuleBasedAgent._imperfection_seed(agent, config))
    agent._imperfection_draw = 0.0
    agent._imperfection_age = 0
    agent.abreast_guard = False
    agent.debug_rules = False
    agent.current_stage = 0
    agent._junction_distance = float('inf')
    agent.abreast_active = False
    # No junction ahead and no leader: the speed path reaches stage 3.
    agent._should_apply_junction_management = lambda: False
    agent._calculate_car_following_speed = lambda vanilla: None
    return agent


def test_the_speed_factor_scales_the_desired_speed_not_only_free_flow():
    # SUMO's speedFactor multiplies the desired speed whatever the vehicle is
    # doing. A factor above one is a faster-than-limit driver, not an error.
    slow = _driver(desired_speed_factor=0.85, cruising_speed_factor=1.0)
    fast = _driver(desired_speed_factor=1.15, cruising_speed_factor=1.0)
    assert slow._calculate_rule_based_speed(40.0) == pytest.approx(34.0)
    assert fast._calculate_rule_based_speed(40.0) == pytest.approx(46.0)


def test_the_speed_factor_composes_with_the_cruising_factor():
    agent = _driver(desired_speed_factor=0.9, cruising_speed_factor=0.95)
    assert agent._calculate_rule_based_speed(40.0) == pytest.approx(40.0 * 0.9 * 0.95)


def test_a_driver_without_imperfection_is_unchanged():
    agent = _driver(cruising_speed_factor=1.0)
    assert agent._calculate_rule_based_speed(40.0) == pytest.approx(40.0)


def test_imperfection_only_ever_lowers_the_target():
    # sigma * accel * U(0,1), with U in [0, 1): the drop is bounded by
    # sigma * accel and never negative, so no rule above can be undone by it.
    agent = _driver(driver_imperfection=0.5, driver_imperfection_accel=2.6,
                    cruising_speed_factor=1.0)
    ceiling = 0.5 * 2.6 * 3.6
    for _ in range(200):
        speed = agent._calculate_rule_based_speed(40.0)
        assert 40.0 - ceiling <= speed <= 40.0


def test_imperfection_never_drives_the_target_below_a_stop():
    agent = _driver(driver_imperfection=0.5, cruising_speed_factor=1.0)
    for _ in range(50):
        assert agent._calculate_rule_based_speed(0.0) == 0.0


def test_the_drop_is_redrawn_every_step_by_default():
    agent = _driver(driver_imperfection=0.5, cruising_speed_factor=1.0)
    drops = {round(agent._imperfection_drop_kmh(), 9) for _ in range(50)}
    assert len(drops) > 40, "a hold of zero is SUMO's rule: a fresh draw each step"


def test_a_hold_keeps_one_draw_for_its_window():
    # 10 steps is 0.5 s at a 0.05 s delta.
    agent = _driver(driver_imperfection=0.5, driver_imperfection_hold_steps=10,
                    cruising_speed_factor=1.0)
    first = [agent._imperfection_drop_kmh() for _ in range(10)]
    second = [agent._imperfection_drop_kmh() for _ in range(10)]
    assert len(set(first)) == 1 and len(set(second)) == 1
    assert first[0] != second[0]


def test_two_vehicles_with_the_same_draw_share_a_noise_stream():
    # The seed is the vehicle's own recorded driver draw, so a replay of the
    # same recording reproduces the same noise however actor ids come out.
    profile = dict(driver_imperfection=0.4, time_headway=1.481,
                   minimum_distance_buffer=2.958, max_speed=46.829,
                   desired_speed_factor=0.851, cruising_speed_factor=1.0)
    a, b = _driver(**profile), _driver(**profile)
    assert [a._imperfection_drop_kmh() for _ in range(20)] == \
           [b._imperfection_drop_kmh() for _ in range(20)]


def test_a_different_driver_draw_gives_a_different_noise_stream():
    base = dict(driver_imperfection=0.4, time_headway=1.481,
                minimum_distance_buffer=2.958, max_speed=46.829,
                desired_speed_factor=0.851, cruising_speed_factor=1.0)
    other = dict(base, time_headway=1.902)
    a, b = _driver(**base), _driver(**other)
    assert [a._imperfection_drop_kmh() for _ in range(20)] != \
           [b._imperfection_drop_kmh() for _ in range(20)]


def test_the_mean_drop_matches_the_krauss_term():
    # E[sigma * accel * U(0,1)] = sigma * accel / 2, in m/s.
    sigma, accel = 0.3, 2.6
    agent = _driver(driver_imperfection=sigma, driver_imperfection_accel=accel,
                    cruising_speed_factor=1.0)
    drops = [agent._imperfection_drop_kmh() / 3.6 for _ in range(20000)]
    assert sum(drops) / len(drops) == pytest.approx(sigma * accel / 2.0, abs=0.02)


def test_the_scale_turns_dawdling_off_without_touching_the_rest():
    # A recording states sigma per vehicle, so a profile cannot restate it.
    # Scale 0 answers "was it the dawdling?" while every other recorded
    # parameter still lands.
    agent = _driver(driver_imperfection=0.5, driver_imperfection_scale=0.0,
                    desired_speed_factor=0.9, cruising_speed_factor=1.0)
    assert agent._imperfection_drop_kmh() == 0.0
    assert agent._calculate_rule_based_speed(40.0) == pytest.approx(36.0)


def test_the_scale_multiplies_the_recorded_sigma():
    sigma, accel, scale = 0.4, 2.6, 0.5
    agent = _driver(driver_imperfection=sigma, driver_imperfection_accel=accel,
                    driver_imperfection_scale=scale, cruising_speed_factor=1.0)
    drops = [agent._imperfection_drop_kmh() / 3.6 for _ in range(20000)]
    assert sum(drops) / len(drops) == pytest.approx(sigma * scale * accel / 2.0, abs=0.02)


def test_a_retired_follower_key_is_reported_once_not_ignored(monkeypatch):
    # The gap-closing follower law has no speed buffer; a profile that still
    # sets following_speed_buffer is told so, once per process.
    monkeypatch.setattr(RuleBasedAgent, '_reported_retired_keys', set())
    assert RuleBasedAgent._report_retired_keys({'following_speed_buffer': 5.0}) == ['following_speed_buffer']
    assert RuleBasedAgent._report_retired_keys({'following_speed_buffer': 5.0}) == []
    assert RuleBasedAgent._report_retired_keys({'time_headway': 2.0}) == []
