"""seed_everything makes the learner reproducible, not just the traffic."""
import numpy as np
import torch

from opencda_marl.utils.seeding import seed_everything


def _fresh_algorithm():
    from opencda_marl.builtin.algorithms.sac import SACAlgorithm

    cfg = {"warmup_steps": 0, "memory_size": 16, "batch_size": 4,
           "actor_hidden_dims": [8, 8], "critic_hidden_dims": [8, 8],
           "lstm_hidden_size": 4, "tensorboard": {"enabled": False}}
    return SACAlgorithm(cfg, 3, 1)


def _first_weight(algo):
    return next(algo.actor.parameters()).detach().cpu().numpy().copy()


def test_same_seed_same_network_init():
    seed_everything(123)
    w1 = _first_weight(_fresh_algorithm())
    seed_everything(123)
    w2 = _first_weight(_fresh_algorithm())
    np.testing.assert_array_equal(w1, w2)


def test_different_seed_different_network_init():
    seed_everything(123)
    w1 = _first_weight(_fresh_algorithm())
    seed_everything(456)
    w2 = _first_weight(_fresh_algorithm())
    assert not np.array_equal(w1, w2)


def test_seeds_python_and_numpy_too():
    import random

    seed_everything(7)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(7)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_none_disables_seeding():
    assert seed_everything(None) is None
