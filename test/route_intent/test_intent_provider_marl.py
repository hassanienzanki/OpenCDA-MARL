"""MARL intent provider tests (no CARLA, no real algorithm training)."""
# Import torch before numpy to avoid a Windows DLL load conflict in this pixi env
# (numpy's loaded MKL DLLs shadow ones torch needs from torch/lib/shm.dll).
import torch  # noqa: F401, I001  (import order matters)
from types import SimpleNamespace

import numpy as np

from opencda_marl.builtin.algorithms.intent.intent_algorithm import IntentMAPPOAlgorithm
from opencda_marl.api.intent_observation import SPAWN_OBS_DIM_NO_PREF
from opencda_marl._kernel.policy.intent_bridge import make_marl_intent_provider
from opencda_marl.api.route_intent import RouteIntent


class _Planner:
    def __init__(self, routes):
        self._routes = routes

    def get_routes(self, j):
        return self._routes


def _routes():
    return {
        10: {"route_type": "left",     "entry_direction": "north", "entry_lane_idx": 0, "exit_direction": "west",  "exit_lane_idx": 0, "dest_tf": object()},
        11: {"route_type": "straight", "entry_direction": "north", "entry_lane_idx": 0, "exit_direction": "south", "exit_lane_idx": 0, "dest_tf": object()},
        12: {"route_type": "straight", "entry_direction": "north", "entry_lane_idx": 0, "exit_direction": "south", "exit_lane_idx": 1, "dest_tf": object()},
        14: {"route_type": "straight", "entry_direction": "north", "entry_lane_idx": 0, "exit_direction": "south", "exit_lane_idx": 2, "dest_tf": object()},
        13: {"route_type": "right",    "entry_direction": "north", "entry_lane_idx": 0, "exit_direction": "east",  "exit_lane_idx": 0, "dest_tf": object()},
    }


def _event():
    return SimpleNamespace(
        route_id=11,
        junction_id=100,
        metadata={"entry_direction": "north", "entry_lane_idx": 0,
                  "route_type": "straight"},
    )


def _replay_event():
    return SimpleNamespace(
        route_id=11,
        junction_id=100,
        metadata={"entry_direction": "north", "lane_id": 0,
                  "route_type": "straight"},
    )


def _adapter(actor_id=1):
    return SimpleNamespace(actor_id=actor_id, spawn_metadata={},
                           current_speed=0.0)


def test_provider_returns_decision_with_mask_invariant():
    planner = _Planner(_routes())
    algo = IntentMAPPOAlgorithm(
        config={"hidden_dims": [8, 8], "device": "cpu",
                "tensorboard": {"enabled": False}},
        state_dim=SPAWN_OBS_DIM_NO_PREF, num_actions=4,
    )
    provider = make_marl_intent_provider(algorithm=algo,
                                         get_active_adapters=lambda: [])
    decision = provider({"event": _event(), "adapter": _adapter(),
                         "planner": planner})
    assert decision is not None
    # LEFT/RIGHT are present in the route table but would change the exit
    # direction, so the learned provider must not expose them.
    assert decision.mask[RouteIntent.LEFT] is False
    assert decision.mask[RouteIntent.RIGHT] is False
    assert decision.mask[RouteIntent.STRAIGHT] is True
    assert decision.mask[decision.selected_intent] is True
    assert decision.direction_preserved is True


def test_provider_calls_store_pending_once_per_call():
    planner = _Planner(_routes())
    algo = IntentMAPPOAlgorithm(
        config={"hidden_dims": [8, 8], "device": "cpu",
                "tensorboard": {"enabled": False}},
        state_dim=SPAWN_OBS_DIM_NO_PREF, num_actions=4,
    )
    provider = make_marl_intent_provider(algorithm=algo,
                                         get_active_adapters=lambda: [])
    provider({"event": _event(), "adapter": _adapter(actor_id=42),
              "planner": planner})
    assert algo.has_pending(42)


def test_provider_uses_lane_id_fallback_for_replay_events():
    planner = _Planner(_routes())
    algo = IntentMAPPOAlgorithm(
        config={"hidden_dims": [8, 8], "device": "cpu",
                "tensorboard": {"enabled": False}},
        state_dim=SPAWN_OBS_DIM_NO_PREF, num_actions=4,
    )
    provider = make_marl_intent_provider(algorithm=algo,
                                         get_active_adapters=lambda: [])
    decision = provider({"event": _replay_event(), "adapter": _adapter(actor_id=7),
                         "planner": planner})
    assert decision is not None
    assert algo.has_pending(7)


def test_provider_resolves_forced_intent_to_less_pressured_candidate():
    planner = _Planner(_routes())
    queued_lane_one = SimpleNamespace(
        spawn_metadata={"exit_direction": "south", "exit_lane_idx": 1, "route_id": 12},
        current_speed=0.0,
    )
    algo = IntentMAPPOAlgorithm(
        config={"hidden_dims": [8, 8], "device": "cpu",
                "tensorboard": {"enabled": False}},
        state_dim=SPAWN_OBS_DIM_NO_PREF, num_actions=4,
    )
    provider = make_marl_intent_provider(
        algorithm=algo,
        get_active_adapters=lambda: [queued_lane_one],
        force_intent="straight",
    )
    decision = provider({"event": _event(), "adapter": _adapter(actor_id=9),
                         "planner": planner})
    assert decision is not None
    assert decision.selected_intent is RouteIntent.STRAIGHT
    assert decision.selected_candidate.route_id == 14
    assert decision.candidate_resolution == "candidate_aware"
    assert decision.selected_candidate_traffic["queued_same_exit_lane_count"] == 0.0
