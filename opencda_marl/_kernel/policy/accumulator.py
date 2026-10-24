"""Per-actor terminal reward accumulator (scalar or vector)."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


class PerActorRewardAccumulator:
    """Element-wise accumulation of step rewards keyed by actor id.

    Use ``reward_dim=0`` for scalar accumulation, ``reward_dim>=1`` for
    vector accumulation. ``finalize`` returns the running sum and deletes
    the entry; subsequent ``finalize`` on the same actor yields zero.
    """

    def __init__(self, *, reward_dim: int):
        if int(reward_dim) < 0:
            raise ValueError("reward_dim must be >= 0")
        self.reward_dim = int(reward_dim)
        self._scalar: Dict[Any, float] = {}
        self._vector: Dict[Any, np.ndarray] = {}

    def add(self, actor_id: Any, reward: Any) -> None:
        if self.reward_dim == 0:
            self._scalar[actor_id] = self._scalar.get(actor_id, 0.0) + float(reward)
            return
        arr = np.asarray(reward, dtype=np.float32)
        if arr.shape != (self.reward_dim,):
            raise ValueError(
                f"reward shape {arr.shape} does not match reward_dim={self.reward_dim}"
            )
        current = self._vector.get(actor_id)
        if current is None:
            self._vector[actor_id] = arr.copy()
        else:
            self._vector[actor_id] = current + arr

    def finalize(self, actor_id: Any):
        if self.reward_dim == 0:
            return self._scalar.pop(actor_id, 0.0)
        return self._vector.pop(actor_id, np.zeros(self.reward_dim, dtype=np.float32))

    def clear(self) -> None:
        self._scalar.clear()
        self._vector.clear()
