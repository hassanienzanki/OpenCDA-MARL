"""Per-vehicle and per-episode preference vector samplers.

The built-in profiles use a shared 2D objective basis:
``[efficiency, interaction_risk]``.  The legacy 4D sampler is retained for
backward compatibility, but new training/eval runs should route 2D metadata to
both the speed policy and any route-intent policy.
"""
from __future__ import annotations

import numpy as np

from opencda_marl._kernel.traffic.regime import (
    PreferenceRegime,
    sample_preference,
)


class PreferenceSampler:
    """Sample preference vectors on the simplex.

    Modes:
        ``per_vehicle`` - sample a fresh vector every ``sample()`` call.
        ``per_episode`` - sample once per episode; cache until ``reset_episode``.
        ``disabled``    - return an empty ``np.ndarray`` (used by scalar MARL).

    Built-in 2D regimes:
        ``aligned``     - every vehicle receives ``[0.5, 0.5]``.
        ``conflicting`` - parity split by ``vehicle_idx``:
                          even -> ``[0.75, 0.25]`` and
                          odd -> ``[0.25, 0.75]``.

    Legacy 4D regimes:
        ``aligned``     - symmetric Dirichlet(alpha=[1,1,1,1]).
        ``conflicting`` - antipodal split keyed off ``vehicle_idx`` (even =
                          safety-dominant, odd = throughput-dominant).
    """

    def __init__(self, *, mode: str, sampler: str, dim: int, seed: int,
                 alpha: float = 1.0, regime: str = "aligned"):
        self.mode = str(mode).lower()
        self.sampler = str(sampler).lower()
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.regime = str(regime).lower()
        self._rng = np.random.default_rng(int(seed))
        self._cached: np.ndarray | None = None
        if self.mode not in {"per_vehicle", "per_episode", "disabled"}:
            raise ValueError(f"Unknown preference mode: {mode!r}")
        if self.sampler not in {"dirichlet"}:
            raise ValueError(f"Unsupported sampler: {sampler!r}")
        if self.regime not in {"aligned", "conflicting"}:
            raise ValueError(f"Unknown regime: {regime!r}")

    def sample(self, vehicle_idx: int = 0) -> np.ndarray:
        if self.mode == "disabled":
            return np.zeros(0, dtype=np.float32)
        if self.mode == "per_episode" and self._cached is not None:
            return self._cached
        if self.dim == 2:
            if self.regime == "aligned":
                pref = np.array([0.5, 0.5], dtype=np.float32)
            else:
                pref = (
                    np.array([0.75, 0.25], dtype=np.float32)
                    if int(vehicle_idx) % 2 == 0
                    else np.array([0.25, 0.75], dtype=np.float32)
                )
        if self.dim == 4:
            regime_enum = (
                PreferenceRegime.CONFLICTING
                if self.regime == "conflicting"
                else PreferenceRegime.ALIGNED
            )
            pref = sample_preference(regime_enum, int(vehicle_idx), self._rng)
        elif self.dim != 2:
            # Non-4-dim configs keep the legacy alpha-uniform path.
            alpha_vec = np.full(self.dim, self.alpha, dtype=np.float64)
            pref = self._rng.dirichlet(alpha_vec).astype(np.float32)
        if self.mode == "per_episode":
            self._cached = pref
        return pref

    def reset_episode(self) -> None:
        self._cached = None
