"""PerActorRewardAccumulator tests."""
import numpy as np
import pytest

from opencda_marl._kernel.policy.accumulator import PerActorRewardAccumulator


def test_scalar_accumulates_and_finalizes():
    acc = PerActorRewardAccumulator(reward_dim=0)
    acc.add("v1", 0.5)
    acc.add("v1", 0.25)
    acc.add("v2", 1.0)
    assert acc.finalize("v1") == pytest.approx(0.75)
    assert acc.finalize("v2") == pytest.approx(1.0)
    # finalized actors are gone
    assert acc.finalize("v1") == 0.0


def test_vector_accumulates_elementwise():
    acc = PerActorRewardAccumulator(reward_dim=3)
    acc.add("v1", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    acc.add("v1", np.array([0.5, -0.5, 1.5], dtype=np.float32))
    out = acc.finalize("v1")
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert out.dtype == np.float32
    assert np.allclose(out, [1.5, 1.5, 4.5])


def test_finalize_unknown_actor_returns_zero():
    scalar = PerActorRewardAccumulator(reward_dim=0)
    assert scalar.finalize("ghost") == 0.0
    vector = PerActorRewardAccumulator(reward_dim=2)
    assert np.allclose(vector.finalize("ghost"), np.zeros(2, dtype=np.float32))


def test_add_validates_shape_for_vector():
    acc = PerActorRewardAccumulator(reward_dim=2)
    with pytest.raises(ValueError):
        acc.add("v1", np.array([1.0, 2.0, 3.0], dtype=np.float32))
