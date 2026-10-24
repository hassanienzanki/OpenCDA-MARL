"""PreferenceSampler tests."""
import numpy as np
import pytest

from opencda_marl._kernel.traffic.preference_sampler import PreferenceSampler


def test_dirichlet_sample_lies_on_simplex():
    sampler = PreferenceSampler(mode="per_vehicle", sampler="dirichlet",
                                dim=4, seed=7, alpha=1.0)
    for _ in range(50):
        p = sampler.sample()
        assert p.shape == (4,)
        assert p.dtype == np.float32
        assert np.all(p >= 0.0)
        assert pytest.approx(p.sum(), abs=1e-5) == 1.0


def test_2d_aligned_regime_is_balanced():
    sampler = PreferenceSampler(mode="per_vehicle", sampler="dirichlet",
                                dim=2, seed=7, regime="aligned")
    assert np.allclose(sampler.sample(vehicle_idx=0), [0.5, 0.5])
    assert np.allclose(sampler.sample(vehicle_idx=13), [0.5, 0.5])


def test_2d_conflicting_regime_splits_by_vehicle_index():
    sampler = PreferenceSampler(mode="per_vehicle", sampler="dirichlet",
                                dim=2, seed=7, regime="conflicting")
    assert np.allclose(sampler.sample(vehicle_idx=0), [0.75, 0.25])
    assert np.allclose(sampler.sample(vehicle_idx=1), [0.25, 0.75])


def test_sampler_is_reproducible_under_seed():
    a = PreferenceSampler(mode="per_vehicle", sampler="dirichlet", dim=4, seed=42)
    b = PreferenceSampler(mode="per_vehicle", sampler="dirichlet", dim=4, seed=42)
    seq_a = [a.sample().tolist() for _ in range(10)]
    seq_b = [b.sample().tolist() for _ in range(10)]
    assert seq_a == seq_b


def test_per_episode_mode_returns_same_until_reset():
    sampler = PreferenceSampler(mode="per_episode", sampler="dirichlet",
                                dim=3, seed=1)
    p1 = sampler.sample()
    p2 = sampler.sample()
    assert np.allclose(p1, p2)
    sampler.reset_episode()
    p3 = sampler.sample()
    assert not np.allclose(p1, p3)


def test_disabled_returns_empty_array():
    sampler = PreferenceSampler(mode="disabled", sampler="dirichlet",
                                dim=4, seed=0)
    p = sampler.sample()
    assert p.shape == (0,) and p.dtype == np.float32
