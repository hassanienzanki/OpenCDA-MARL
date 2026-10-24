"""Small dependency-free space classes for the standalone MOMARL subset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Discrete:
    """Minimal discrete space compatible with simple sampling and validation."""

    n: int

    def __post_init__(self):
        if int(self.n) <= 0:
            raise ValueError("Discrete space size must be positive.")

    def sample(self, rng: np.random.Generator | None = None) -> int:
        generator = rng or np.random.default_rng()
        return int(generator.integers(low=0, high=int(self.n)))

    def contains(self, value: object) -> bool:
        if not isinstance(value, (int, np.integer)):
            return False
        return 0 <= int(value) < int(self.n)


@dataclass(frozen=True)
class Box:
    """Minimal box space for shape metadata and validation."""

    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]
    dtype: np.dtype

    def __init__(self, low, high, shape: tuple[int, ...], dtype=np.float32):
        low_arr = np.broadcast_to(np.asarray(low, dtype=dtype), shape).copy()
        high_arr = np.broadcast_to(np.asarray(high, dtype=dtype), shape).copy()
        object.__setattr__(self, "low", low_arr)
        object.__setattr__(self, "high", high_arr)
        object.__setattr__(self, "shape", tuple(shape))
        object.__setattr__(self, "dtype", np.dtype(dtype))

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        generator = rng or np.random.default_rng()
        return generator.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, value: object) -> bool:
        arr = np.asarray(value, dtype=self.dtype)
        return (
            arr.shape == self.shape
            and np.all(arr >= self.low)
            and np.all(arr <= self.high)
        )
