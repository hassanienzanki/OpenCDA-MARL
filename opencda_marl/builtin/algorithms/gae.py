"""Generalized Advantage Estimation over agent-interleaved rollouts.

On-policy learners here store one flat list of transitions per rollout,
appended agent by agent at every step (A_t, B_t, C_t, A_t+1, ...). Running
the GAE recursion straight over that list bootstraps agent A's TD target
from agent B's value and chains advantages across agents. This helper
groups the flat rollout by *trajectory* (one agent's run within one
episode) and runs the recursion within each group, in the original order,
so the maths is the textbook one:

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

with ``V(s_{t+1})`` the *same agent's* next stored value, and — for the last
stored transition of a trajectory that did not terminate (rollout cut,
episode time-limit) — the caller-supplied bootstrap ``last_values[traj]``
(zero when absent, i.e. treated as terminal).
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def group_by_trajectory(trajectory_ids: Sequence[Hashable]) -> Dict[Hashable, List[int]]:
    """Indices of each trajectory in order of appearance."""
    groups: Dict[Hashable, List[int]] = {}
    for index, trajectory in enumerate(trajectory_ids):
        groups.setdefault(trajectory, []).append(index)
    return groups


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    trajectory_ids: Optional[Sequence[Hashable]] = None,
    last_values: Optional[Mapping[Hashable, Any]] = None,
    last_value: Any = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(advantages, returns)`` for a rollout.

    ``rewards``/``values`` are ``[T]`` (scalar critic) or ``[T, R]`` (vector
    critic); ``dones`` is ``[T]``. With ``trajectory_ids`` (length ``T``)
    the recursion runs per trajectory and the bootstrap for each open
    trajectory comes from ``last_values``. Without ids the whole rollout is
    one trajectory bootstrapped by ``last_value`` (legacy single-agent use).
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    T = rewards.shape[0]
    if T == 0:
        return np.zeros_like(rewards), np.zeros_like(rewards)
    if values.shape != rewards.shape:
        raise ValueError(f"values {values.shape} must match rewards {rewards.shape}")
    vector = rewards.ndim == 2
    zero = np.zeros(rewards.shape[1], dtype=np.float32) if vector else np.float32(0.0)

    if trajectory_ids is None:
        groups = {None: list(range(T))}
        bootstrap = {None: last_value}
    else:
        if len(trajectory_ids) != T:
            raise ValueError("trajectory_ids must have one entry per transition")
        groups = group_by_trajectory(trajectory_ids)
        bootstrap = dict(last_values or {})

    advantages = np.zeros_like(rewards)
    for trajectory, indices in groups.items():
        tail = bootstrap.get(trajectory, zero)
        next_value = np.asarray(tail, dtype=np.float32) if vector else np.float32(tail)
        gae = zero.copy() if vector else np.float32(0.0)
        for position in range(len(indices) - 1, -1, -1):
            t = indices[position]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * non_terminal - values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]
    returns = advantages + values
    return advantages, returns


__all__ = ["compute_gae", "group_by_trajectory"]
