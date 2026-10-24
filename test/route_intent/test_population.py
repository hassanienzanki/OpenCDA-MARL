"""Mixed autonomy: per-vehicle population and the learner's gate.

Runs that differ only in the share of learner-controlled vehicles must
differ in nothing else, so the assignment must be exact and reproducible,
and the learner must command — and learn from — its own population only.
"""
import pytest

from opencda_marl._kernel.policy.manager import MARLManager
from opencda_marl._kernel.traffic.population import (
    PopulationSampler,
    build_population_sampler,
)


def _assign(sampler, n):
    return [sampler.assign() for _ in range(n)]


def test_shares_are_exact_over_the_pattern():
    sampler = PopulationSampler({"marl": 0.5, "background": 0.5}, seed=7)
    got = _assign(sampler, 100)
    assert got.count("marl") == 50 and got.count("background") == 50


def test_uneven_shares_use_largest_remainder():
    sampler = PopulationSampler({"marl": 0.25, "background": 0.75}, seed=1)
    got = _assign(sampler, 100)
    assert got.count("marl") == 25 and got.count("background") == 75
    assert sampler.describe() == {"background": 0.75, "marl": 0.25}


def test_same_seed_reproduces_the_assignment_vehicle_for_vehicle():
    a = _assign(PopulationSampler({"marl": 0.5, "bg": 0.5}, seed=3), 40)
    b = _assign(PopulationSampler({"marl": 0.5, "bg": 0.5}, seed=3), 40)
    c = _assign(PopulationSampler({"marl": 0.5, "bg": 0.5}, seed=4), 40)
    assert a == b
    assert a != c  # a different seed is a different (still exact) draw


def test_episode_reset_replays_the_same_population():
    sampler = PopulationSampler({"marl": 0.5, "bg": 0.5}, seed=5)
    first = _assign(sampler, 20)
    sampler.reset_episode()
    assert _assign(sampler, 20) == first


def test_full_penetration_is_one_population():
    sampler = PopulationSampler({"marl": 1.0, "bg": 0.0}, seed=0)
    assert set(_assign(sampler, 30)) == {"marl"}


def test_builder_returns_none_when_not_configured():
    assert build_population_sampler(None) is None
    assert build_population_sampler({}) is None
    assert build_population_sampler({"mix": {"marl": 1.0}, "enabled": False}) is None
    assert build_population_sampler({"mix": {"marl": 0.5, "bg": 0.5}}) is not None


def test_empty_mix_is_rejected_loudly():
    with pytest.raises(ValueError, match="at least one type"):
        PopulationSampler({"marl": 0.0})


def test_learner_gate_keeps_only_its_own_population():
    observations = {
        "cav_1": {"speed": 1.0, "population": "marl", "learning": True},
        "bg_1": {"speed": 2.0, "population": "background", "learning": False},
        "cav_2": {"speed": 3.0, "population": "marl", "learning": True},
    }
    kept = MARLManager._learning_observations(observations)
    assert set(kept) == {"cav_1", "cav_2"}


def test_homogeneous_runs_pass_through_unchanged():
    # No population flags at all: single-population behaviour is untouched.
    observations = {"a": {"speed": 1.0}, "b": {"speed": 2.0}}
    assert MARLManager._learning_observations(observations) is observations
    assert MARLManager._learning_observations({}) == {}
