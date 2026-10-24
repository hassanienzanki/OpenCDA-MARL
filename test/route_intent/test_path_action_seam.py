"""A plugin owns its path semantics without editing the engine.

The engine runs the lifecycle around a one-shot path decision; which
alternatives exist, how wide the head is, and what makes an outcome legal
belong to the provider. These tests register a provider that shares nothing
with the built-ins and check that the engine honours all three.
"""
from types import SimpleNamespace

import pytest

from opencda_marl._kernel.policy.action_space import (
    derive_route_intent_config,
    resolve_action_space_config,
)
from opencda_marl.api import register_component
from opencda_marl.api.path_action import (
    HeadSpec,
    decision_destination,
    decision_is_admissible,
    resolve_head_spec,
)


class _Decision:
    """A decision with semantics of its own: no direction, no candidates."""

    def __init__(self, *, admissible=True, destination=None, label="corner"):
        self._admissible = admissible
        self._destination = destination
        self.label = label

    @property
    def admissible(self):
        return self._admissible

    @property
    def destination(self):
        return self._destination

    def as_metadata(self):
        return {"label": self.label, "admissible": self._admissible}


def _tf(x=7.0, y=8.0):
    return SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=0.0))


# ─────────────────────────────────────────────────────────────────────────────
# The decision contract
# ─────────────────────────────────────────────────────────────────────────────

def test_the_engine_reads_only_admissible_and_destination():
    ok = _Decision(destination=_tf())
    assert decision_is_admissible(ok) and decision_destination(ok).location.x == 7.0
    rejected = _Decision(admissible=False, destination=_tf())
    assert not decision_is_admissible(rejected)
    # A decision that declares no invariant is admissible: a provider that
    # never rejects anything writes nothing.
    assert decision_is_admissible(SimpleNamespace(as_metadata=dict))
    # A provider built on the framework's candidate helpers is read too.
    candidate = SimpleNamespace(dest_tf=_tf(1.0, 2.0))
    assert decision_destination(
        SimpleNamespace(selected_candidate=candidate)).location.x == 1.0
    assert decision_destination(SimpleNamespace()) is None


def test_the_builtin_decision_still_declares_its_own_invariant():
    from opencda_marl.api.route_intent import RouteIntent, RouteIntentCandidate, build_decision

    candidate = RouteIntentCandidate(
        intent=RouteIntent.KEEP, route_id=11, junction_id=4, dest_tf=_tf(),
        exit_direction="south", exit_lane_idx=0, is_original=True, route_type="straight")
    kept = build_decision(requested_intent=RouteIntent.KEEP, candidate=candidate,
                          mask={RouteIntent.KEEP: True}, original_route_id=11,
                          diagnostics={"original_exit_direction": "south"})
    assert kept.admissible and kept.direction_preserved
    assert decision_destination(kept) is candidate.dest_tf
    assert kept.as_metadata()["admissible"] is True
    # Changing the demand exit direction is what the built-ins call illegal.
    turned = RouteIntentCandidate(
        intent=RouteIntent.LEFT, route_id=12, junction_id=4, dest_tf=_tf(),
        exit_direction="west", exit_lane_idx=0, is_original=False, route_type="left")
    changed = build_decision(requested_intent=RouteIntent.LEFT, candidate=turned,
                             mask={RouteIntent.LEFT: True}, original_route_id=11,
                             diagnostics={"original_exit_direction": "south"})
    assert not changed.admissible and not decision_is_admissible(changed)


# ─────────────────────────────────────────────────────────────────────────────
# The head shape
# ─────────────────────────────────────────────────────────────────────────────

def test_a_provider_declares_how_wide_its_head_must_be():
    def build(cfg, planner, *, marl_manager=None, training=True, previous=None):
        return lambda context: None

    build.head_spec = lambda cfg: HeadSpec(
        observation_dim=12 * int(cfg.get("n_max", 4)), num_actions=int(cfg.get("n_max", 4)))
    register_component("intent_provider", "seam_test", override=True)(build)

    spec = resolve_head_spec("seam_test", {"n_max": 3})
    assert spec == HeadSpec(observation_dim=36, num_actions=3)
    # A provider that declares nothing leaves the engine's default in place.
    def plain(cfg, planner, **kw):
        return lambda context: None
    register_component("intent_provider", "seam_plain", override=True)(plain)
    assert resolve_head_spec("seam_plain", {}) is None
    assert resolve_head_spec("no_such_provider", {}) is None
    # A constant spec works as well as a callable.
    plain.head_spec = HeadSpec(observation_dim=5, num_actions=2)
    assert resolve_head_spec("seam_plain", {}).observation_dim == 5
    # A nonsense spec fails loudly rather than sizing a head to zero.
    with pytest.raises(ValueError):
        HeadSpec(observation_dim=0, num_actions=1)
    plain.head_spec = lambda cfg: "not a spec"
    with pytest.raises(TypeError):
        resolve_head_spec("seam_plain", {})


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def test_a_registered_provider_is_selectable_and_gets_its_whole_block():
    cfg = resolve_action_space_config({"action_space": {"mode": "path_speed", "path": {
        "provider": "seam_test", "n_max": 3, "custom_knob": 0.7,
        "reward_dim": 2, "preference": {"dim": 2}}}})
    assert cfg.path.provider == "seam_test"
    # A custom provider drives the learned lifecycle: it owns the head.
    assert cfg.uses_learned_path
    # Its own keys reach it untouched; the engine reads none of them.
    assert cfg.path.options["n_max"] == 3
    assert cfg.path.options["custom_knob"] == 0.7

    derived = derive_route_intent_config({}, {"action_space": {"mode": "path_speed", "path": {
        "provider": "seam_test", "n_max": 3, "custom_knob": 0.7}}})
    assert derived["mode"] == "seam_test"
    assert derived["n_max"] == 3 and derived["custom_knob"] == 0.7


def test_the_historical_selectors_still_name_the_builtin_providers():
    for selector, provider in (("keep_current", "tao_fixed"),
                               ("random_legal", "naive_random"),
                               ("rule_based", "rule_based_path"),
                               ("learned", "marl")):
        cfg = resolve_action_space_config({"action_space": {
            "mode": "path_speed", "path": {"selector": selector}}})
        assert cfg.path.provider == provider
        assert derive_route_intent_config({}, {"action_space": {
            "mode": "path_speed", "path": {"selector": selector}}})["mode"] == (
            "marl" if selector == "learned" else provider)
    # speed_only has no path action at all.
    speed_only = resolve_action_space_config({"action_space": {"mode": "speed_only"}})
    assert not speed_only.uses_path and speed_only.path.provider == "tao_fixed"
    assert derive_route_intent_config({}, {"action_space": {"mode": "speed_only"}})["mode"] == "disabled"
    # An unknown selector with no provider is still rejected loudly.
    with pytest.raises(ValueError):
        resolve_action_space_config({"action_space": {
            "mode": "path_speed", "path": {"selector": "teleport"}}})
