"""PPO family: a loaded checkpoint skips the random warmup, and updates advance
``training_step``.

``steps_collected`` is per-process and starts at 0, so gating the uniform-random
warmup on ``steps_collected < warmup_steps`` alone made every resumed or
warm-started run throw away its first ``warmup_steps`` agent-steps on random
speeds (and discard those transitions). SAC/TD3 already honoured
``_pretrained``. Separately, neither PPO class ever incremented
``training_step``, which is the global step every ``category='losses'`` scalar
is logged at — the whole loss curve collapsed onto step 0.
"""
import numpy as np
import pytest

from opencda_marl.builtin.algorithms.mappo import MAPPOAlgorithm
from opencda_marl.builtin.algorithms.pcmappo import PCMAPPOAlgorithm


def _cfg(**overrides):
    cfg = {"warmup_steps": 12, "n_steps": 3, "batch_size": 4, "num_mini_batch": 1,
           "ppo_epochs": 1, "lstm_hidden_size": 8, "actor_hidden_dims": [16, 16],
           "critic_hidden_dims": [16, 16], "tensorboard": {"enabled": False}}
    cfg.update(overrides)
    return cfg


def _mappo(**o):
    return MAPPOAlgorithm(_cfg(**o), 5, 1)


def _pcmappo(**o):
    return PCMAPPOAlgorithm(_cfg(pref_dim=2, **o), 5, 1)


def _obs(rng, n=3):
    return {f"v{i}": rng.normal(size=5).astype(np.float32) for i in range(n)}


def _drive(algo, rng, steps):
    for _ in range(steps):
        obs, nxt = _obs(rng), _obs(rng)
        actions = algo.select_actions(obs, list(obs), training=True)
        for aid in obs:
            algo.store_transition(obs, aid, actions[aid], float(rng.normal()), nxt, False)
        algo.update()


@pytest.mark.parametrize("make", [_mappo, _pcmappo], ids=["mappo", "pcmappo"])
def test_warmup_runs_once_and_is_skipped_after_a_load(make, tmp_path):
    algo = make()
    rng = np.random.default_rng(0)
    assert algo._ppo_in_warmup() is True
    _drive(algo, rng, 6)                      # 18 agent-steps > warmup_steps=12
    assert algo._ppo_in_warmup() is False, "warmup finished in this process"
    algo.save(str(tmp_path / "c.pth"))

    fresh = make()
    assert fresh._ppo_in_warmup() is True, "a fresh learner warms up"
    fresh.load(str(tmp_path / "c.pth"))
    assert fresh._pretrained is True
    assert fresh._ppo_in_warmup() is False, (
        "a loaded checkpoint must not redo the random warmup")

    # The first post-load step is a real policy action, and it is stored.
    obs, nxt = _obs(rng), _obs(rng)
    actions = fresh.select_actions(obs, list(obs), training=True)
    assert all(a is not None for a in actions.values())
    before = len(fresh.rollout_buffer)
    for aid in obs:
        fresh.store_transition(obs, aid, actions[aid], 1.0, nxt, False)
    assert len(fresh.rollout_buffer) > before, "post-load transitions enter the rollout"


@pytest.mark.parametrize("make", [_mappo, _pcmappo], ids=["mappo", "pcmappo"])
def test_updates_advance_training_step(make):
    algo = make(warmup_steps=0)
    rng = np.random.default_rng(1)
    assert algo.training_step == 0
    _drive(algo, rng, 9)
    assert algo.updates_performed > 0, "at least one PPO update ran"
    assert algo.training_step == algo.updates_performed, (
        "training_step is the global step for loss scalars")


@pytest.mark.parametrize("make", [_mappo, _pcmappo], ids=["mappo", "pcmappo"])
def test_steps_collected_round_trips_through_a_checkpoint(make, tmp_path):
    algo = make()
    _drive(algo, np.random.default_rng(2), 4)
    collected = algo.steps_collected
    assert collected > 0
    algo.save(str(tmp_path / "c.pth"))
    fresh = make()
    fresh.load(str(tmp_path / "c.pth"))
    assert fresh.steps_collected == collected
