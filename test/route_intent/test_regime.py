"""Preference-regime selector contract tests."""
import numpy as np
import pytest

from opencda_marl._kernel.traffic.regime import (
    PreferenceRegime,
    sample_preference,
)


def test_aligned_regime_returns_symmetric_dirichlet():
    rng = np.random.default_rng(42)
    samples = np.stack([
        sample_preference(PreferenceRegime.ALIGNED, vehicle_idx=i, rng=rng)
        for i in range(2000)
    ])
    means = samples.mean(axis=0)
    assert np.all(np.abs(means - 0.25) < 0.02), f"expected ~0.25, got {means}"


def test_conflicting_regime_splits_by_vehicle_idx():
    rng = np.random.default_rng(7)
    even_idx = np.stack([
        sample_preference(PreferenceRegime.CONFLICTING, vehicle_idx=2 * i, rng=rng)
        for i in range(1000)
    ])
    odd_idx = np.stack([
        sample_preference(PreferenceRegime.CONFLICTING, vehicle_idx=2 * i + 1, rng=rng)
        for i in range(1000)
    ])
    # Safety-dominant cohort (even idx): dim 0 averages noticeably > 0.25.
    assert even_idx[:, 0].mean() > 0.40
    # Throughput-dominant cohort (odd idx): dim 1 averages noticeably > 0.25.
    assert odd_idx[:, 1].mean() > 0.40


def test_unknown_regime_string_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_preference("not-a-regime", vehicle_idx=0, rng=rng)


def test_preference_vector_sums_to_one_and_is_nonneg():
    rng = np.random.default_rng(13)
    for regime in (PreferenceRegime.ALIGNED, PreferenceRegime.CONFLICTING):
        for i in range(20):
            p = sample_preference(regime, vehicle_idx=i, rng=rng)
            assert abs(p.sum() - 1.0) < 1e-6, f"regime {regime} sum {p.sum()}"
            assert np.all(p >= 0)
            assert p.shape == (4,)


def test_string_regime_alias_accepted():
    rng = np.random.default_rng(99)
    p = sample_preference("aligned", vehicle_idx=0, rng=rng)
    assert p.shape == (4,)
    p2 = sample_preference("conflicting", vehicle_idx=0, rng=rng)
    assert p2.shape == (4,)
