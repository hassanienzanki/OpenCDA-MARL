"""Preference-regime selector.

Two preference regimes:

- ALIGNED:    every vehicle samples from Dirichlet([1,1,1,1]) (symmetric prior).
- CONFLICTING: antipodal Dirichlet split.
  Even ``vehicle_idx`` samples from Dirichlet([3,1,1,1]) (safety-dominant);
  odd ``vehicle_idx`` samples from Dirichlet([1,3,1,1]) (throughput-dominant).

The split-by-vehicle-idx convention lets the upstream flow generator
interleave the two groups by entry direction so safety- and
throughput-weighted vehicles arrive at the intersection simultaneously.
"""
from __future__ import annotations

from enum import Enum
from typing import Union

import numpy as np


class PreferenceRegime(str, Enum):
    ALIGNED = "aligned"
    CONFLICTING = "conflicting"


_ALIGNED_ALPHA = np.array([1.0, 1.0, 1.0, 1.0])
_SAFETY_DOMINANT_ALPHA = np.array([3.0, 1.0, 1.0, 1.0])
_THROUGHPUT_DOMINANT_ALPHA = np.array([1.0, 3.0, 1.0, 1.0])


def sample_preference(
    regime: Union[PreferenceRegime, str],
    vehicle_idx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a 4-dim preference vector under the requested regime.

    Args:
        regime: ``PreferenceRegime`` value or string ("aligned"/"conflicting").
        vehicle_idx: integer position used to split the conflicting regime.
        rng: a numpy ``Generator`` for reproducibility.

    Returns:
        ``np.ndarray`` of shape ``(4,)`` summing to 1.
    """
    if isinstance(regime, str):
        try:
            regime = PreferenceRegime(regime)
        except ValueError as exc:
            raise ValueError(f"unknown regime: {regime!r}") from exc

    if regime is PreferenceRegime.ALIGNED:
        return rng.dirichlet(_ALIGNED_ALPHA).astype(np.float32)
    alpha = _SAFETY_DOMINANT_ALPHA if vehicle_idx % 2 == 0 else _THROUGHPUT_DOMINANT_ALPHA
    return rng.dirichlet(alpha).astype(np.float32)
