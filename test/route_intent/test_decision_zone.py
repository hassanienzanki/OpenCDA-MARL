"""The decision zone: when the one-shot route decision is taken."""
import pytest

from opencda_marl._kernel.policy.action_space import (
    derive_route_intent_config,
    resolve_action_space_config,
)
from opencda_marl._kernel.policy.decision_zone import (
    DecisionZone,
    build_decision_zone,
)


def test_spawn_is_the_default_and_never_holds_a_vehicle_pending():
    zone = DecisionZone()
    assert zone.decides_at_spawn
    zone.register("v1")
    assert zone.pending == set()
    assert not zone.due("v1", 1.0)


def test_approach_holds_until_the_vehicle_reaches_the_zone():
    zone = DecisionZone(trigger="approach", distance_m=30.0)
    zone.register("v1")
    assert zone.is_pending("v1")
    assert not zone.due("v1", 80.0)          # still far away
    assert not zone.due("v1", None)          # distance unknown: not yet
    assert zone.due("v1", 30.0)              # on the boundary
    assert zone.due("v1", 12.0)


def test_the_decision_is_one_shot():
    zone = DecisionZone(trigger="approach", distance_m=30.0)
    zone.register("v1")
    assert zone.due("v1", 10.0)
    zone.mark_decided("v1")
    assert not zone.due("v1", 5.0)
    zone.register("v1")                      # re-registering must not reopen it
    assert not zone.is_pending("v1")


def test_a_departed_vehicle_is_forgotten():
    zone = DecisionZone(trigger="approach")
    zone.register("v1")
    zone.forget("v1")
    assert not zone.is_pending("v1")
    zone.register("v1")                      # a fresh vehicle may reuse the id
    assert zone.is_pending("v1")


def test_episode_reset_clears_pending_and_decided():
    zone = DecisionZone(trigger="approach")
    zone.register("v1")
    zone.mark_decided("v1")
    zone.reset_episode()
    zone.register("v1")
    assert zone.is_pending("v1")


def test_info_exposes_what_an_external_loop_needs():
    zone = DecisionZone(trigger="approach", distance_m=25.0)
    zone.register("v2")
    zone.register("v1")
    info = zone.as_info()
    assert info["trigger"] == "approach"
    assert info["distance_m"] == 25.0
    assert info["decision_pending"] == ["v1", "v2"]


def test_unknown_trigger_is_rejected():
    with pytest.raises(ValueError, match="unknown route-intent trigger"):
        DecisionZone(trigger="mid_junction")


def test_builder_defaults_to_spawn_and_reads_the_block():
    assert build_decision_zone(None).decides_at_spawn
    assert build_decision_zone({}).decides_at_spawn
    zone = build_decision_zone(
        {"decision_zone": {"trigger": "approach", "distance_m": 40}})
    assert zone.trigger == "approach" and zone.distance_m == 40.0


def test_action_space_config_carries_the_trigger_into_the_route_block():
    marl_cfg = {"action_space": {
        "mode": "path_speed",
        "path": {"selector": "learned",
                 "decision_zone": {"trigger": "approach", "distance_m": 35}},
    }}
    resolved = resolve_action_space_config(marl_cfg)
    assert resolved.path.decision_trigger == "approach"
    assert resolved.path.decision_distance_m == 35.0

    route_cfg = derive_route_intent_config({}, marl_cfg)
    assert route_cfg["decision_zone"] == {"trigger": "approach", "distance_m": 35.0}
    assert build_decision_zone(route_cfg).trigger == "approach"


def test_speed_only_runs_keep_the_spawn_default():
    route_cfg = derive_route_intent_config({}, {"action_space": {"mode": "speed_only"}})
    assert route_cfg["mode"] == "disabled"
    assert build_decision_zone(route_cfg).decides_at_spawn
