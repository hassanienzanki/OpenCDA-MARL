"""Per-vehicle population assignment: who is a CAV and who is background.

A mixed-autonomy run needs each spawned vehicle to carry its own agent
type — the learner drives the CAVs, everything else is background traffic
under its own model — and the mix to be *exact* and *reproducible*, so
that two runs whose configured shares differ meet otherwise identical
traffic.

Assignment is therefore a deterministic pattern, not per-vehicle coin
flips: shares become integer quotas over a fixed-length pattern (largest
remainder), the pattern is shuffled once from the seed, and draws take
consecutive slots, wrapping around. The shares are exact over every window
of ``resolution`` consecutive draws, and the same seed reproduces the same
assignment draw for draw. A shorter prefix only approximates the ratio
(the shuffle may cluster one type), and the guarantee is over the order of
draws: live flows draw flow by flow, so the share within one flow is not
exact either.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
from loguru import logger

#: Pattern length: 100 slots express whole-percent mixes exactly.
DEFAULT_RESOLUTION = 100


def is_learning_population(metadata: Mapping | None, agent_type: str | None) -> bool:
    """Whether a vehicle may receive the configured population's actions.

    Missing population metadata keeps homogeneous runs unchanged. Both speed
    and path actions belong to the learning population; background vehicles
    remain visible as traffic without receiving policy decisions.
    """
    if not agent_type:
        return True
    return str((metadata or {}).get("population") or agent_type) == str(agent_type)


class PopulationSampler:
    """Assign an agent type per spawned vehicle from a share mapping."""

    def __init__(self, mix: Mapping[str, float], *, seed: int = 0,
                 resolution: int = DEFAULT_RESOLUTION):
        shares: Dict[str, float] = {}
        for key, value in (mix or {}).items():
            share = float(value)
            if share < 0:
                raise ValueError(f"population share for {key!r} must be >= 0")
            if share > 0:
                shares[str(key)] = share
        if not shares:
            raise ValueError("population mix must name at least one type "
                             "with a positive share")
        total = sum(shares.values())
        self.shares = {k: v / total for k, v in shares.items()}
        self.resolution = max(1, int(resolution))
        self._pattern = self._build_pattern(int(seed))
        self._counter = 0

    def _build_pattern(self, seed: int) -> list:
        """Quota-fill the pattern (largest remainder), then shuffle once."""
        exact = {k: v * self.resolution for k, v in self.shares.items()}
        counts = {k: int(np.floor(v)) for k, v in exact.items()}
        remainder = self.resolution - sum(counts.values())
        if remainder > 0:
            order = sorted(exact, key=lambda k: (-(exact[k] - counts[k]), k))
            for key in order[:remainder]:
                counts[key] += 1
        pattern: list = []
        for key in sorted(counts):
            pattern.extend([key] * counts[key])
        rng = np.random.default_rng(int(seed))
        rng.shuffle(pattern)
        return pattern

    # ------------------------------------------------------------------ #
    def assign(self, vehicle_idx: int | None = None) -> str:
        """Next agent type. ``vehicle_idx`` pins the slot when given."""
        if vehicle_idx is None:
            slot = self._counter
            self._counter += 1
        else:
            slot = int(vehicle_idx)
        return self._pattern[slot % len(self._pattern)]

    def reset_episode(self) -> None:
        """Restart the pattern so every episode draws the same population."""
        self._counter = 0

    def describe(self) -> Dict[str, float]:
        """Realised shares of the pattern (what the run actually gets)."""
        total = len(self._pattern)
        return {key: self._pattern.count(key) / total
                for key in sorted(set(self._pattern))}


def build_population_sampler(population_cfg: Mapping | None):
    """Build a sampler from an ``agents.population`` block, or None.

    ``{"mix": {"marl": 0.5, "rule_based": 0.5}, "seed": 7}``; a missing,
    empty or ``enabled: false`` block means one homogeneous population and
    the caller keeps using ``agents.agent_type`` for every vehicle.
    """
    if not population_cfg:
        return None
    if not population_cfg.get("enabled", True):
        return None
    mix = population_cfg.get("mix") or {}
    if not mix:
        return None
    return PopulationSampler(
        mix,
        seed=int(population_cfg.get("seed", 0)),
        resolution=int(population_cfg.get("resolution", DEFAULT_RESOLUTION)),
    )


def validate_population_mix(population_cfg: Mapping | None,
                            learning_agent_type: str | None, *,
                            known_agent_types: Optional[Iterable[str]] = None):
    """Reject an ``agents.population`` block that cannot run as configured.

    The mix must name the learning population (``agents.agent_type``):
    without it every vehicle is background and the learner silently gets no
    vehicle at all. With ``known_agent_types`` (CARLA builds every
    population's agent through the agent registry) every population with a
    positive share must be one of them, so an unknown name fails here
    instead of at its first spawn. On SUMO any other name is background
    traffic driven by SUMO itself, and only the learner check applies.

    Returns the sampler the block builds, or None for a homogeneous run.
    """
    sampler = build_population_sampler(population_cfg)
    if sampler is None:
        return None
    if learning_agent_type:
        learner = str(learning_agent_type)
        named = {str(key) for key in (population_cfg.get("mix") or {})}
        if learner not in named:
            raise ValueError(
                f"agents.population.mix {sorted(named)} names no learning "
                f"population: agents.agent_type is {learner!r}, so no vehicle "
                f"would be driven by the learner. Add {learner!r} to the mix.")
        if learner not in sampler.shares:
            logger.warning(
                f"agents.population.mix gives the learning population "
                f"{learner!r} a zero share: every vehicle is background.")
    if known_agent_types is not None:
        # Exact, like the spawn-time lookup (resolve_component) the name
        # goes through: a label that only matches with its case changed
        # would pass here and fail at its first spawn.
        known = {str(key) for key in known_agent_types}
        unknown = sorted(key for key in sampler.shares if key not in known)
        if unknown:
            raise ValueError(
                f"agents.population.mix names unregistered agent type(s) "
                f"{unknown}; registered: {sorted(known)}.")
    return sampler
