"""Opt-in value normalisation for the PPO family.

The critic learns normalised returns; every value that reaches GAE or a
bootstrap is in reward units; statistics persist through checkpoints.
"""
import numpy as np
import pytest
import torch

from opencda_marl.builtin.algorithms.mappo import MAPPOAlgorithm
from opencda_marl.builtin.algorithms.pcmappo import PCMAPPOAlgorithm
from opencda_marl.builtin.algorithms.value_norm import ValueNormalizer


def test_value_normalizer_tracks_mean_and_std_and_round_trips():
    vn = ValueNormalizer(2, torch.device("cpu"))
    x = torch.tensor([[1.0, 100.0], [3.0, 300.0], [5.0, 500.0], [7.0, 700.0]])
    vn.update(x)                                   # de-biased: one batch -> its own stats
    mean, var = vn.mean_var()
    torch.testing.assert_close(mean, torch.tensor([4.0, 400.0]), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(var, torch.tensor([5.0, 50000.0]), rtol=1e-3, atol=1e-2)
    z = vn.normalize(x)
    torch.testing.assert_close(z.mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(vn.denormalize(z), x, atol=1e-4, rtol=1e-5)
    other = ValueNormalizer(2, torch.device("cpu"))
    other.load_state_dict(vn.state_dict())
    torch.testing.assert_close(other.normalize(x), z)


def _mappo(**o):
    cfg = {"warmup_steps": 0, "n_steps": 3, "batch_size": 4, "num_mini_batch": 1, "ppo_epochs": 1,
           "lstm_hidden_size": 8, "actor_hidden_dims": [16, 16], "critic_hidden_dims": [16, 16],
           "tensorboard": {"enabled": False}}
    cfg.update(o)
    return MAPPOAlgorithm(cfg, 5, 1)


def _pcmappo(**o):
    cfg = {"warmup_steps": 0, "n_steps": 3, "batch_size": 4, "num_mini_batch": 1, "ppo_epochs": 1,
           "lstm_hidden_size": 8, "actor_hidden_dims": [16, 16], "critic_hidden_dims": [16, 16],
           "pref_dim": 2, "tensorboard": {"enabled": False}}
    cfg.update(o)
    return PCMAPPOAlgorithm(cfg, 5, 1)


def _obs(rng, n):
    return {f"v{i}": rng.normal(size=5).astype(np.float32) for i in range(n)}


def _drive(algo, rng, steps, agents=3, reward_scale=1000.0):
    for _ in range(steps):
        obs, nxt = _obs(rng, agents), _obs(rng, agents)
        actions = algo.select_actions(obs, list(obs), training=True)
        for aid in obs:
            algo.store_transition(obs, aid, actions[aid], reward_scale * float(rng.normal()), nxt, False)
        algo.update()


@pytest.mark.parametrize("make", [_mappo, _pcmappo], ids=["mappo", "pcmappo"])
def test_off_by_default_and_on_by_flag(make):
    assert make().value_normalizer is None
    algo = make(value_normalization=True)
    assert algo.value_normalizer is not None


@pytest.mark.parametrize("make", [_mappo, _pcmappo], ids=["mappo", "pcmappo"])
def test_values_are_in_reward_units_and_targets_are_normalized(make):
    algo = make(value_normalization=True)
    rng = np.random.default_rng(0)
    _drive(algo, rng, 6, reward_scale=1000.0)      # two rollouts of large returns
    mean, var = algo.value_normalizer.mean_var()
    assert float(var.sqrt()) > 50.0, "statistics tracked returns of order 1e3"
    # Acting values are de-normalised: stash a value and compare with the raw
    # critic output scaled back by the normaliser.
    obs = _obs(rng, 3)
    algo.select_actions(obs, list(obs), training=True)
    stashed = algo.last_values["v0"] if hasattr(algo, "last_values") else algo.last_scalar_values["v0"]
    # The batch inputs live on the acting device (a CPU replica when
    # training on a GPU), so the raw value comes from the acting critic and
    # the normaliser mirror that serves it.
    ego_states, ctx, _ = algo._prepare_batch_inputs(obs, ["v0"])
    critic = algo._acting_module("critic")
    with torch.no_grad():
        if isinstance(algo, MAPPOAlgorithm):
            raw = critic(torch.cat([ego_states, ctx], dim=1))
        else:
            raw = critic(ego_states, ctx)[0]
        expected = float(algo._acting_normalizer("value", raw).denormalize(raw).reshape(-1)[0])
    assert stashed == pytest.approx(expected, rel=1e-4, abs=1e-4)
    # Bootstrap values are de-normalised too.
    algo.store_transition(obs, "v0", 30.0, 1.0, _obs(rng, 3), False)
    boots = algo._bootstrap_values()
    if isinstance(boots, tuple):
        boots = boots[0]
    assert (algo.episode_count, "v0") in boots
    # The critic itself outputs normalised values (order 1, not 1e3).
    assert abs(float(raw.reshape(-1)[0])) < 50.0


@pytest.mark.parametrize("make", [_mappo, _pcmappo], ids=["mappo", "pcmappo"])
def test_normalizer_state_round_trips_through_checkpoints(make, tmp_path):
    algo = make(value_normalization=True)
    rng = np.random.default_rng(1)
    _drive(algo, rng, 3)
    algo.save(str(tmp_path / "c.pth"))
    fresh = make(value_normalization=True)
    fresh.load(str(tmp_path / "c.pth"))
    a, b = algo.value_normalizer.state_dict(), fresh.value_normalizer.state_dict()
    for k in a:
        torch.testing.assert_close(a[k], b[k])
