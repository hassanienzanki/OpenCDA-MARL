"""TD3 stores raw neighbour sequences and encodes them at update time — the
same contract as SAC (see test_sac_batched_context.py), on both replay
buffers TD3 can use (PER by default, SmartReplayBuffer otherwise)."""
import numpy as np
import pytest
import torch

from opencda_marl.builtin.algorithms.td3 import TD3Algorithm


def _algo(**overrides):
    cfg = {"warmup_steps": 0, "memory_size": 512, "batch_size": 8,
           "actor_hidden_dims": [16, 16], "critic_hidden_dims": [16, 16],
           "lstm_hidden_size": 8, "tensorboard": {"enabled": False},
           "policy_freq": 1}
    cfg.update(overrides)
    return TD3Algorithm(cfg, 5, 1)


def _obs(rng, n):
    return {f"v{i}": rng.normal(size=5).astype(np.float32) for i in range(n)}


def _fill(algo, rng, n=40, agents=(2, 3, 5)):
    for i in range(n):
        k = agents[i % len(agents)]
        algo.store_transition(_obs(rng, k), "v0", 30.0, 1.0, _obs(rng, k), i % 7 == 0)


@pytest.mark.parametrize("use_per", [True, False], ids=["per", "smart"])
def test_layout_growth_and_update_on_both_buffers(use_per):
    algo = _algo(use_per=use_per)
    rng = np.random.default_rng(0)
    _fill(algo, rng)
    assert algo._seq_width == 5                       # widest set seen
    tr = (algo.memory.sample(1)[0] if use_per else algo.memory.sample(1))[0]
    assert len(tr) == 9
    assert tr[0].shape == (5,) and tr[1].shape == (5, 5) and tr[6].shape == (5, 5)
    assert 1 <= int(tr[2]) <= 5 and 1 <= int(tr[7]) <= 5
    # TD3.update() swallows exceptions and returns stale metrics, so assert
    # on evidence that the update ran end to end: the step counter advanced
    # AND the critic loss is a real (non-zero) number.
    metrics = algo.update()
    assert algo.training_step == 1
    assert np.isfinite(metrics["critic_loss"]) and metrics["critic_loss"] != 0.0
    # A wider set later grows the stored width without touching old rows.
    algo.store_transition(_obs(rng, 9), "v0", 30.0, 1.0, _obs(rng, 9), False)
    assert algo._seq_width == 9
    tr = (algo.memory.sample(1)[0] if use_per else algo.memory.sample(1))[0]
    assert tr[1].shape == (9, 5)
    metrics = algo.update()
    assert algo.training_step == 2 and np.isfinite(metrics["critic_loss"])
    assert metrics["target_updates"] == 2            # policy_freq=1 -> actor+targets every step


def test_update_does_not_swallow_an_exception_silently(monkeypatch):
    """Guard for the failure mode that hid the layout bug: an exception inside
    update() must at least be logged at ERROR level and leave training_step
    untouched, so a probe reading the metrics/log can see it."""
    from loguru import logger as loguru_logger
    algo = _algo()
    rng = np.random.default_rng(4)
    _fill(algo, rng, n=16)
    records = []
    sink = loguru_logger.add(lambda m: records.append(m), level="ERROR")
    try:
        monkeypatch.setattr(algo, "_encode_context_batch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        algo.update()
    finally:
        loguru_logger.remove(sink)
    assert algo.training_step == 0
    assert any("Error in TD3 update" in str(r) for r in records)


def test_store_transition_runs_no_lstm_forward():
    algo = _algo()
    calls = {"n": 0}
    real = algo.lstm_encoder.forward

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    algo.lstm_encoder.forward = spy
    rng = np.random.default_rng(1)
    algo.store_transition(_obs(rng, 4), "v0", 30.0, 1.0, _obs(rng, 4), False)
    assert calls["n"] == 0


def test_encoder_frozen_by_default_and_trainable_with_flag():
    for flag in (False, True):
        algo = _algo(train_context_encoder=flag)
        rng = np.random.default_rng(2)
        before = [p.detach().clone() for p in algo.lstm_encoder.parameters()]
        _fill(algo, rng, n=32)
        for _ in range(3):
            algo.update()
        changed = any(not torch.equal(b, a) for b, a in zip(before, algo.lstm_encoder.parameters()))
        assert changed == flag, f"train_context_encoder={flag} -> changed={changed}"
        opt = {id(p) for g in algo.critic_optimizer.param_groups for p in g["params"]}
        assert all((id(p) in opt) == flag for p in algo.lstm_encoder.parameters())


def test_acting_and_stored_sequences_agree():
    """_prepare_inputs (per-agent acting), _prepare_batch_inputs (batched
    acting) and the stored sequence share one ordering."""
    algo = _algo()
    algo.device = torch.device("cpu")
    algo.lstm_encoder = algo.lstm_encoder.cpu()
    rng = np.random.default_rng(3)
    obs = _obs(rng, 6)
    seq, n = algo._context_sequence(obs, "v2")
    with torch.no_grad():
        ego, ctx = algo._prepare_inputs(obs, "v2")
        ref = algo.lstm_encoder(torch.as_tensor(seq[None]))
    np.testing.assert_array_equal(ego.numpy()[0], obs["v2"])
    torch.testing.assert_close(ctx, ref, rtol=1e-6, atol=1e-6)
