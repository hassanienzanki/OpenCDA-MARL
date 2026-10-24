"""Checkpoints written under one train_context_encoder setting must load
under the other: networks always apply, optimizer moments are best-effort,
and the loaded learner is marked pretrained (no second warmup)."""
import numpy as np
import pytest
import torch

from opencda_marl.builtin.algorithms.sac import SACAlgorithm
from opencda_marl.builtin.algorithms.td3 import TD3Algorithm


def _cfg(**overrides):
    cfg = {"warmup_steps": 0, "memory_size": 256, "batch_size": 8, "policy_freq": 1,
           "actor_hidden_dims": [16, 16], "critic_hidden_dims": [16, 16],
           "lstm_hidden_size": 8, "tensorboard": {"enabled": False}}
    cfg.update(overrides)
    return cfg


def _obs(rng, n):
    return {f"v{i}": rng.normal(size=5).astype(np.float32) for i in range(n)}


def _train_a_bit(algo, rng, n=24):
    for _ in range(n):
        algo.store_transition(_obs(rng, 3), "v0", 30.0, 1.0, _obs(rng, 3), False)
    for _ in range(3):
        algo.update()


@pytest.mark.parametrize("cls", [SACAlgorithm, TD3Algorithm], ids=["sac", "td3"])
@pytest.mark.parametrize("saved_flag,loaded_flag", [(False, True), (True, False), (False, False)])
def test_checkpoint_round_trips_across_train_context_encoder(tmp_path, cls, saved_flag, loaded_flag):
    rng = np.random.default_rng(0)
    src = cls(_cfg(train_context_encoder=saved_flag), 5, 1)
    _train_a_bit(src, rng)
    path = str(tmp_path / "ckpt.pth")
    src.save(path)

    dst = cls(_cfg(train_context_encoder=loaded_flag), 5, 1)
    dst.load(path)

    # Networks always come across.
    for name in ("actor", "critic", "lstm_encoder"):
        for p_src, p_dst in zip(getattr(src, name).parameters(), getattr(dst, name).parameters()):
            assert torch.equal(p_src.detach().cpu(), p_dst.detach().cpu()), name
    # The loaded learner skips warmup and its optimizer set matches ITS flag.
    assert dst._pretrained is True
    opt_params = {id(p) for g in dst.critic_optimizer.param_groups for p in g["params"]}
    assert all((id(p) in opt_params) == loaded_flag for p in dst.lstm_encoder.parameters())
    # And it can keep training right away.
    _train_a_bit(dst, rng, n=16)
    step = dst.training_step
    assert step > 0
