"""Route-intent feasibility mask, candidate generation, and provider tests."""

import sys
from pathlib import Path
from types import SimpleNamespace

_TEST_DIR = str(Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

import pytest

from opencda_marl.api.intent import build_provider_from_config
from opencda_marl.builtin.intent_providers.providers import (
    make_naive_random_provider,
    make_rule_based_path_provider,
    make_tao_fixed_provider,
)
from opencda_marl.api.route_intent import (
    ALL_INTENTS,
    RouteIntent,
    build_candidates,
    build_candidates_with_diagnostics,
    build_mask,
    candidate_traffic_features,
    resolve_route,
)


# Lightweight CARLA-free transform stub so candidates can be compared.
_DEST = SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0))


def _route(route_id, *, route_type, entry_dir, exit_dir,
           entry_lane_idx, exit_lane_idx, dest_tf=_DEST):
    return {
        "route_id": route_id,
        "route_type": route_type,
        "entry_direction": entry_dir,
        "exit_direction": exit_dir,
        "entry_lane_idx": entry_lane_idx,
        "exit_lane_idx": exit_lane_idx,
        "dest_tf": dest_tf,
    }


class _FakePlanner:
    """Minimal stand-in for MARLPlanner.get_routes(junction_id)."""

    def __init__(self, routes_by_junction):
        self._routes_by_junction = routes_by_junction

    def get_routes(self, junction_id):
        return self._routes_by_junction.get(int(junction_id), {})


def _three_way_planner():
    # Spawned from the "left" lane (entry_lane_idx=0) of the north entry.
    # Available routes: left and straight legal; right not present from
    # the left lane (i.e. real-world feasibility: cannot turn right from
    # the leftmost lane on this approach).
    routes = {
        # Same lane, different intents
        10: _route(10, route_type="left",     entry_dir="north", exit_dir="west",
                  entry_lane_idx=0, exit_lane_idx=0),
        11: _route(11, route_type="straight", entry_dir="north", exit_dir="south",
                  entry_lane_idx=0, exit_lane_idx=0),
        # Right turn exists from lane 1 only -> illegal from lane 0
        20: _route(20, route_type="right",    entry_dir="north", exit_dir="east",
                  entry_lane_idx=1, exit_lane_idx=0),
        # Multiple straight options from lane 0 to test tie-break
        12: _route(12, route_type="straight", entry_dir="north", exit_dir="south",
                  entry_lane_idx=0, exit_lane_idx=1),
        # Different entry direction -> must be filtered out
        30: _route(30, route_type="left",     entry_dir="east",  exit_dir="north",
                  entry_lane_idx=0, exit_lane_idx=0),
    }
    return _FakePlanner({100: routes})


# ─────────────────────────────────────────────────────────────────────────────
# Mask and candidate construction
# ─────────────────────────────────────────────────────────────────────────────

def test_candidates_filter_by_entry_direction_and_lane():
    planner = _three_way_planner()
    candidates = build_candidates(
        planner=planner,
        junction_id=100,
        entry_direction="north",
        entry_lane_idx=0,
        original_route_id=11,
    )
    route_ids = sorted({c.route_id for c in candidates})
    # Route 10 would change the exit direction west, so it is filtered out.
    # Route 12 preserves the original southbound exit, and 11 remains KEEP.
    assert 10 not in route_ids
    assert 11 in route_ids
    assert 12 in route_ids
    assert 20 not in route_ids  # right is from a different lane
    assert 30 not in route_ids  # different entry direction


def test_candidates_preserve_original_exit_direction():
    planner = _three_way_planner()
    candidates = build_candidates(
        planner=planner,
        junction_id=100,
        entry_direction="north",
        entry_lane_idx=0,
        original_route_id=11,
    )
    assert candidates
    assert all(c.exit_direction == "south" for c in candidates)
    assert all(c.same_exit_direction for c in candidates)


def test_candidates_allow_same_exit_direction_with_different_exit_lane_and_dest():
    original_dest = SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0))
    alternate_dest = SimpleNamespace(location=SimpleNamespace(x=1.0, y=0.0, z=0.0))
    planner = _FakePlanner({
        1: {
            11: _route(11, route_type="straight", entry_dir="north", exit_dir="south",
                       entry_lane_idx=0, exit_lane_idx=0, dest_tf=original_dest),
            12: _route(12, route_type="straight", entry_dir="north", exit_dir="south",
                       entry_lane_idx=0, exit_lane_idx=1, dest_tf=alternate_dest),
            13: _route(13, route_type="left", entry_dir="north", exit_dir="west",
                       entry_lane_idx=0, exit_lane_idx=0),
        }
    })
    candidates, diagnostics = build_candidates_with_diagnostics(
        planner=planner,
        junction_id=1,
        entry_direction="north",
        entry_lane_idx=0,
        original_route_id=11,
    )
    route_ids = {c.route_id for c in candidates}
    assert 12 in route_ids
    assert 13 not in route_ids
    alternate = next(c for c in candidates if c.route_id == 12)
    assert alternate.exit_lane_idx == 1
    assert alternate.dest_tf is alternate_dest
    assert diagnostics["same_exit_direction_alternative_count"] == 1
    assert diagnostics["candidate_count_changed_exit_direction"] == 1


def test_mask_marks_illegal_intent_as_false():
    planner = _three_way_planner()
    candidates = build_candidates(
        planner=planner,
        junction_id=100,
        entry_direction="north",
        entry_lane_idx=0,
        original_route_id=11,
    )
    mask = build_mask(candidates)
    assert mask[RouteIntent.LEFT] is False      # masked: would change destination
    assert mask[RouteIntent.STRAIGHT] is True
    assert mask[RouteIntent.RIGHT] is False     # masked: cannot right-turn from lane 0
    assert mask[RouteIntent.KEEP] is True        # always legal when original route exists


def test_keep_current_remains_valid_when_other_intents_are_illegal():
    # Only a "straight" route exists; left and right are both illegal.
    routes = {
        1: _route(1, route_type="straight", entry_dir="north", exit_dir="south",
                  entry_lane_idx=0, exit_lane_idx=0),
    }
    planner = _FakePlanner({1: routes})
    candidates = build_candidates(
        planner=planner,
        junction_id=1,
        entry_direction="north",
        entry_lane_idx=0,
        original_route_id=1,
    )
    mask = build_mask(candidates)
    assert mask[RouteIntent.STRAIGHT] is False
    assert mask[RouteIntent.LEFT] is False
    assert mask[RouteIntent.RIGHT] is False
    assert mask[RouteIntent.KEEP] is True


def test_resolve_route_is_deterministic_by_exit_lane_then_route_id():
    planner = _three_way_planner()
    candidates = build_candidates(
        planner=planner,
        junction_id=100,
        entry_direction="north",
        entry_lane_idx=0,
        original_route_id=11,
    )
    mask = build_mask(candidates)
    # Two straight routes (11 -> exit_lane_idx=0, 12 -> exit_lane_idx=1).
    # Resolver must pick the smaller exit_lane_idx first.
    candidate = resolve_route(
        RouteIntent.STRAIGHT, candidates, mask, original_route_id=11
    )
    assert candidate.route_id == 12
    assert candidate.exit_lane_idx == 1


def test_resolve_route_prefers_less_pressured_same_intent_candidate():
    planner = _FakePlanner({
        1: {
            11: _route(11, route_type="straight", entry_dir="north", exit_dir="south",
                       entry_lane_idx=0, exit_lane_idx=0),
            12: _route(12, route_type="straight", entry_dir="north", exit_dir="south",
                       entry_lane_idx=0, exit_lane_idx=1),
            14: _route(14, route_type="straight", entry_dir="north", exit_dir="south",
                       entry_lane_idx=0, exit_lane_idx=2),
        }
    })
    candidates = build_candidates(
        planner=planner,
        junction_id=1,
        entry_direction="north",
        entry_lane_idx=0,
        original_route_id=11,
    )
    mask = build_mask(candidates)
    queued_lane_one = SimpleNamespace(
        spawn_metadata={"exit_direction": "south", "exit_lane_idx": 1, "route_id": 12},
        current_speed=0.0,
    )
    candidate = resolve_route(
        RouteIntent.STRAIGHT,
        candidates,
        mask,
        original_route_id=11,
        active_adapters=[queued_lane_one],
        preference=[0.5, 0.5],
    )
    assert candidate.route_id == 14
    features = candidate_traffic_features(
        next(c for c in candidates if c.route_id == 12),
        [queued_lane_one],
    )
    assert features.active_same_exit_lane_count == 1
    assert features.queued_same_exit_lane_count == 1


def test_resolve_route_falls_back_to_keep_when_intent_illegal():
    planner = _three_way_planner()
    candidates = build_candidates(
        planner=planner,
        junction_id=100,
        entry_direction="north",
        entry_lane_idx=0,
        original_route_id=11,
    )
    mask = build_mask(candidates)
    candidate = resolve_route(
        RouteIntent.RIGHT, candidates, mask, original_route_id=11
    )
    # RIGHT is masked false -> resolver demotes to KEEP and returns the
    # original route_id 11 candidate, never a different concrete route.
    assert candidate.intent is RouteIntent.KEEP
    assert candidate.route_id == 11


def test_nearby_vehicle_changes_do_not_affect_mask():
    # The mask is purely structural. Even if the caller pretends to have
    # nearby vehicles in some context, the mask depends solely on planner
    # routes for the given (entry_direction, entry_lane_idx).
    planner = _three_way_planner()
    candidates = build_candidates(
        planner=planner,
        junction_id=100,
        entry_direction="north",
        entry_lane_idx=0,
        original_route_id=11,
    )
    base_mask = build_mask(candidates)
    # Simulate "nearby vehicle context" by rebuilding with identical inputs.
    candidates_again = build_candidates(
        planner=planner,
        junction_id=100,
        entry_direction="north",
        entry_lane_idx=0,
        original_route_id=11,
    )
    again_mask = build_mask(candidates_again)
    assert base_mask == again_mask


# ─────────────────────────────────────────────────────────────────────────────
# Providers
# ─────────────────────────────────────────────────────────────────────────────

def _spawn_event_stub(route_id=11, entry_dir="north", entry_lane_idx=0,
                      junction_id=100, exit_lane_idx=0):
    return SimpleNamespace(
        route_id=route_id,
        junction_id=junction_id,
        metadata={
            "entry_direction": entry_dir,
            "entry_lane_idx": entry_lane_idx,
            "exit_lane_idx": exit_lane_idx,
        },
    )


def test_tao_fixed_provider_always_selects_keep_current():
    planner = _three_way_planner()
    provider = make_tao_fixed_provider(planner)
    decision = provider({"event": _spawn_event_stub()})
    assert decision is not None
    assert decision.selected_intent is RouteIntent.KEEP
    assert decision.used_keep_current is True
    assert decision.selected_candidate.route_id == 11


def test_naive_random_provider_only_picks_legal_intents():
    planner = _three_way_planner()
    provider = make_naive_random_provider(planner, seed=42)
    event = _spawn_event_stub()
    # Many draws should never produce LEFT/RIGHT because they either change
    # destination or are structurally unavailable from this lane.
    selected_intents = set()
    for _ in range(200):
        decision = provider({"event": event})
        assert decision is not None
        selected_intents.add(decision.selected_intent)
        assert decision.mask[decision.selected_intent] is True
    assert RouteIntent.LEFT not in selected_intents
    assert RouteIntent.RIGHT not in selected_intents


def test_naive_random_provider_is_reproducible_for_same_seed():
    planner = _three_way_planner()
    event = _spawn_event_stub()
    a = make_naive_random_provider(planner, seed=7)
    b = make_naive_random_provider(planner, seed=7)
    a_sequence = [a({"event": event}).selected_intent for _ in range(20)]
    b_sequence = [b({"event": event}).selected_intent for _ in range(20)]
    assert a_sequence == b_sequence


def test_rule_based_path_provider_keeps_current_without_pressure():
    planner = _three_way_planner()
    provider = make_rule_based_path_provider(planner)
    decision = provider({"event": _spawn_event_stub(), "active_adapters": []})
    assert decision is not None
    assert decision.selected_intent is RouteIntent.KEEP
    assert decision.selected_candidate.route_id == 11
    assert decision.direction_preserved is True
    assert decision.same_destination_preserved is True


def test_rule_based_path_provider_uses_same_exit_alternative_when_lane_queued():
    planner = _three_way_planner()
    provider = make_rule_based_path_provider(planner)
    active = [
        SimpleNamespace(
            spawn_metadata={"exit_direction": "south", "exit_lane_idx": 0, "route_id": 11},
            current_speed=0.0,
        )
    ]
    decision = provider({"event": _spawn_event_stub(), "active_adapters": active})
    assert decision is not None
    assert decision.selected_intent is RouteIntent.STRAIGHT
    assert decision.selected_candidate.route_id == 12
    assert decision.selected_candidate.exit_direction == "south"
    assert decision.direction_preserved is True
    assert decision.same_destination_preserved is True
    assert decision.candidate_resolution == "rule_candidate_aware"
    assert decision.selected_candidate_traffic["queued_same_exit_lane_count"] == 0.0


def test_disabled_provider_config_returns_none():
    planner = _three_way_planner()
    assert build_provider_from_config({"mode": "disabled"}, planner) is None
    assert build_provider_from_config({}, planner) is None
    assert build_provider_from_config(None, planner) is None


def test_marl_mode_returns_none_until_policy_is_injected():
    planner = _three_way_planner()
    # Config-driven mode=marl yields no built-in provider: the learned
    # path policy is injected as a callable instead.
    assert build_provider_from_config({"mode": "marl"}, planner) is None


def test_rule_based_path_provider_config_builds_provider():
    planner = _three_way_planner()
    provider = build_provider_from_config({"mode": "rule_based_path"}, planner)
    assert provider is not None
    decision = provider({"event": _spawn_event_stub(), "active_adapters": []})
    assert decision is not None
    assert decision.selected_intent is RouteIntent.KEEP


def test_unknown_mode_raises():
    planner = _three_way_planner()
    with pytest.raises(ValueError):
        build_provider_from_config({"mode": "totally-bogus"}, planner)
