"""The smallest complete learner, as a contrib drop-in.

This module is the reference for both ways of plugging an algorithm into
the framework lifecycle without touching ``opencda_marl/``:

- **You cloned the repository**: copy this file next to itself in
  ``contrib/``, rename the key, done — the package auto-imports every
  module here, so your ``register_algorithm`` call runs at start-up and the
  name works everywhere ``MARL.algorithm`` is read (launcher, evaluation,
  monitor).
- **You installed the framework**: put the same code in your own package
  and advertise it with an ``opencda_marl.plugins`` entry point (or name
  the module in ``MARL.plugins`` / ``OPENCDA_MARL_PLUGINS``). See
  ``docs/marl/custom_algorithms.md``.

The contract (``opencda_marl.api.policy``): ``select_action`` /
``store_transition`` / ``update`` / ``reset_episode`` /
``get_training_info`` / ``save`` / ``load`` plus the ``max_action`` and
``supports_batched_actions`` attributes. Subclassing ``BaseAlgorithm`` is
optional; this one does not.

The policy itself is deliberately trivial -- a target speed drawn uniformly
per decision. Everything visible in the monitor for it is framework
plumbing: transitions stored, updates called, episodes reset, checkpoints
saved and loaded.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np

from opencda_marl.api.algorithms import register_algorithm

__all__ = ["RandomSpeedPolicy", "build_random_speed"]

#: Policy-contract version this module targets (checked by the engine).
OPENCDA_MARL_POLICY_API = 1


class RandomSpeedPolicy:
    """Uniform random target speed in ``[min_action, max_action]`` km/h."""

    supports_batched_actions = False
    supports_vector_rewards = False

    def __init__(self, config: Dict[str, Any], state_dim: int, action_dim: int):
        self.config = dict(config)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.min_action = float(config.get("min_action", 0.0))
        self.max_action = float(config.get("max_action", 50.0))
        self._rng = np.random.default_rng(int(config.get("seed", 0)))
        self.transitions = 0
        self.updates = 0
        self.episodes = 0
        self._episode_reward = 0.0

    # --- acting -----------------------------------------------------------
    def select_action(self, multi_agent_obs: Dict[str, np.ndarray], ego_agent_id: Any,
                      training: bool = True) -> Optional[float]:
        return float(self._rng.uniform(self.min_action, self.max_action))

    # --- learning ---------------------------------------------------------
    def store_transition(self, multi_agent_obs, ego_agent_id, action, reward,
                         next_multi_agent_obs, done) -> None:
        self.transitions += 1
        self._episode_reward += float(np.sum(reward)) if reward is not None else 0.0

    def update(self) -> Dict[str, float]:
        self.updates += 1
        return {"random_speed/transitions": float(self.transitions),
                "random_speed/updates": float(self.updates)}

    def reset_episode(self) -> None:
        self.episodes += 1
        self._episode_reward = 0.0

    # --- framework plumbing -----------------------------------------------
    def get_training_info(self) -> Dict[str, Any]:
        return {"algorithm_type": "random_speed", "transitions": self.transitions,
                "updates": self.updates, "episodes": self.episodes,
                "training_mode": bool(self.config.get("training_mode", True))}

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"transitions": self.transitions, "updates": self.updates,
                       "episodes": self.episodes, "rng": self._rng.bit_generator.state}, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.transitions = int(state.get("transitions", 0))
        self.updates = int(state.get("updates", 0))
        self.episodes = int(state.get("episodes", 0))
        if "rng" in state:
            self._rng.bit_generator.state = state["rng"]

    def close(self) -> None:           # optional: called at shutdown when present
        pass


@register_algorithm("random_speed", label="RandomSpeed",
                    multi_agent=True, reward_aggregation="per_agent", supports_intent=False)
def build_random_speed(config: Dict[str, Any], *, state_dim: int, action_dim: int):
    return RandomSpeedPolicy(config, state_dim, action_dim)
