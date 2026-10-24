"""End-to-end smoke for IntentPCMAPPO without CARLA."""
import torch  # noqa: F401  (Windows pixi torch-before-numpy DLL ordering)
from types import SimpleNamespace

import numpy as np

from opencda_marl.builtin.algorithms.intent.intent_algorithm import IntentPCMAPPOAlgorithm
from opencda_marl.api.intent_observation import SPAWN_OBS_DIM_NO_PREF
from opencda_marl._kernel.policy.intent_bridge import make_marl_intent_provider


class _Planner:
    def __init__(self):
        self._routes = {
            1: {"route_type": "left",     "entry_direction": "north", "entry_lane_idx": 0,
                "exit_direction": "west",  "exit_lane_idx": 0, "dest_tf": object()},
            2: {"route_type": "straight", "entry_direction": "north", "entry_lane_idx": 0,
                "exit_direction": "south", "exit_lane_idx": 0, "dest_tf": object()},
            3: {"route_type": "right",    "entry_direction": "north", "entry_lane_idx": 0,
                "exit_direction": "east",  "exit_lane_idx": 0, "dest_tf": object()},
        }

    def get_routes(self, _j):
        return self._routes


def _event(actor_id, preference):
    return SimpleNamespace(
        route_id=2, junction_id=1,
        metadata={"entry_direction": "north", "entry_lane_idx": 0,
                  "route_type": "straight", "preference": preference},
    )


def test_ten_spawn_then_terminal_cycles_complete():
    rng = np.random.default_rng(0)
    pref_dim = reward_dim = 3
    algo = IntentPCMAPPOAlgorithm(
        config={"hidden_dims": [16, 16], "batch_size": 4, "ppo_epochs": 2,
                "device": "cpu", "tensorboard": {"enabled": False}},
        state_dim=SPAWN_OBS_DIM_NO_PREF + pref_dim, num_actions=4,
        pref_dim=pref_dim, reward_dim=reward_dim,
    )
    planner = _Planner()
    provider = make_marl_intent_provider(algorithm=algo,
                                         get_active_adapters=lambda: [])
    for i in range(10):
        pref = rng.dirichlet(np.ones(pref_dim)).astype(np.float32)
        adapter = SimpleNamespace(actor_id=i, spawn_metadata={},
                                  current_speed=0.0)
        decision = provider({"event": _event(i, pref),
                             "adapter": adapter,
                             "planner": planner})
        assert decision is not None
        terminal_vec = rng.standard_normal(reward_dim).astype(np.float32)
        algo.finalize_terminal(actor_id=i, terminal_reward=terminal_vec)
    metrics = algo.update()
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["buffer_size"] >= 0.0
