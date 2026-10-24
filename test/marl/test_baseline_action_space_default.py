"""Baseline scenarios need no MARL.action_space.

``rule_based`` / ``vanilla`` / ``behavior`` configure no learner, so the route
hook is off by construction. MARLManager always defaulted that case to
``speed_only``; the scenario manager -- built first -- did not, and every
baseline scenario (the documented quick-start included) died with
"MARL.action_space is required". Learner configs keep the strict contract.
"""
import pytest

from opencda_marl._kernel.policy.action_space import (
    derive_route_intent_config,
    resolve_action_space_config,
    with_baseline_default,
)


def test_baseline_marl_block_defaults_to_speed_only():
    cfg = with_baseline_default({})
    assert cfg["action_space"] == {"mode": "speed_only"}
    assert resolve_action_space_config(cfg).mode == "speed_only"
    route = derive_route_intent_config({}, {})
    assert route["mode"] == "disabled"


def test_baseline_default_does_not_override_an_explicit_block():
    explicit = {"action_space": {"mode": "path_speed", "path": {"selector": "rule_based"}}}
    assert with_baseline_default(explicit) is explicit
    assert derive_route_intent_config({}, explicit)["mode"] == "rule_based_path"


def test_learner_without_action_space_is_still_an_error():
    with pytest.raises(ValueError, match="MARL.action_space is required"):
        derive_route_intent_config({}, {"algorithm": "masac"})


def test_legacy_intent_policy_is_still_rejected():
    with pytest.raises(ValueError, match="intent_policy"):
        derive_route_intent_config({}, {"intent_policy": {"mode": "x"}})
