"""SAC update() internals rewritten for fewer device syncs / launches.

Each rewrite is checked against a straightforward reference implementation
of the same quantity, so a speed change cannot hide a semantic one.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from opencda_marl.builtin.algorithms.sac import SACAlgorithm
from opencda_marl.builtin.algorithms.smart_replay_buffer import SmartReplayBuffer


def _algo(**overrides):
    cfg = {"warmup_steps": 0, "memory_size": 512, "batch_size": 16,
           "actor_hidden_dims": [16, 16], "critic_hidden_dims": [16, 16],
           "lstm_hidden_size": 8, "tensorboard": {"enabled": False}}
    cfg.update(overrides)
    return SACAlgorithm(cfg, 5, 1)


def _obs(rng, n):
    return {f"v{i}": rng.normal(size=5).astype(np.float32) for i in range(n)}


def test_sample_fields_draws_the_same_batch_as_sample():
    buf = SmartReplayBuffer(capacity=200, recency_ratio=0.5)
    rng = np.random.default_rng(0)
    for i in range(150):
        buf.push(rng.normal(size=3).astype(np.float32), i, float(i), bool(i % 2),
                 rng.normal(size=(4, 3)).astype(np.float32))
    np.random.seed(123)
    tuples = buf.sample(32)
    np.random.seed(123)
    fields = buf.sample_fields(32)
    assert len(fields) == 5
    for f, arr in enumerate(fields):
        np.testing.assert_array_equal(arr, np.stack([t[f] for t in tuples]))
    assert fields[0].shape == (32, 3) and fields[4].shape == (32, 4, 3)
    assert fields[1].dtype == np.int64 and fields[3].dtype == np.bool_


def test_soft_update_is_polyak_averaging():
    algo = _algo(tau=0.1)
    with torch.no_grad():
        for p in algo.critic.parameters():
            p.add_(torch.randn_like(p))
    before = [p.detach().clone() for p in algo.critic_target.parameters()]
    source = [p.detach().clone() for p in algo.critic.parameters()]
    algo._soft_update_target()
    for tgt, b, s in zip(algo.critic_target.parameters(), before, source):
        torch.testing.assert_close(tgt.detach(), 0.9 * b + 0.1 * s, rtol=1e-6, atol=1e-7)


def _reference_pref_sep(raw, eff, margin):
    """The original boolean-gather implementation."""
    labels = torch.argmax(raw, dim=-1)
    dist = torch.cdist(eff, eff, p=2)
    mask = torch.triu(labels.unsqueeze(0) != labels.unsqueeze(1), diagonal=1)
    active = dist[mask]
    if active.numel() == 0:
        return torch.tensor(0.0), torch.tensor(0.0)
    return F.relu(margin - active).mean(), torch.tensor(float(active.numel()))


def test_pref_sep_loss_matches_boolean_gather_reference():
    algo = _algo(pref_sep_margin=0.25)
    algo.device = torch.device("cpu")
    g = torch.Generator().manual_seed(3)
    raw = torch.rand(24, 2, generator=g)
    eff = torch.rand(24, 2, generator=g)
    loss, pairs = algo._compute_pref_sep_loss(raw, eff)
    ref_loss, ref_pairs = _reference_pref_sep(raw, eff, 0.25)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-6, atol=1e-7)
    assert float(pairs) == float(ref_pairs) > 0
    # No active pair (everyone in the same preference group) -> (0, 0).
    same = torch.tensor([[0.9, 0.1]] * 6)
    loss0, pairs0 = algo._compute_pref_sep_loss(same, eff[:6])
    assert float(loss0) == 0.0 and float(pairs0) == 0.0
    # Degenerate inputs keep the old contract.
    z1, z2 = algo._compute_pref_sep_loss(raw[:1], eff[:1])
    assert float(z1) == 0.0 and float(z2) == 0.0
    z1, z2 = algo._compute_pref_sep_loss(raw, None)
    assert float(z1) == 0.0 and float(z2) == 0.0


def test_encode_context_batch_zeroes_ego_only_rows_and_packs_the_rest():
    algo = _algo()
    algo.device = torch.device("cpu")
    algo.lstm_encoder = algo.lstm_encoder.cpu()
    rng = np.random.default_rng(5)
    seqs = torch.as_tensor(rng.normal(size=(6, 4, 5)).astype(np.float32))
    lengths = torch.tensor([1, 4, 2, 1, 3, 4])
    with torch.no_grad():
        ctx = algo._encode_context_batch(seqs, lengths)
        for i, n in enumerate(lengths.tolist()):
            if n == 1:
                assert not ctx[i].any()
            else:
                ref = algo.lstm_encoder(seqs[i:i + 1, :n])[0]
                torch.testing.assert_close(ctx[i], ref, rtol=1e-6, atol=1e-6)
    # Lengths tensor is not mutated by the encoder's clamp.
    assert lengths.tolist() == [1, 4, 2, 1, 3, 4]


def test_update_metrics_are_plain_floats_and_alpha_tracks_log_alpha():
    algo = _algo(auto_entropy_tuning=True)
    rng = np.random.default_rng(7)
    for _ in range(40):
        algo.store_transition(_obs(rng, 4), "v0", 30.0, 1.0, _obs(rng, 4), False)
    metrics = algo.update()
    for key in ("actor_loss", "critic_loss", "alpha_loss", "alpha", "q1_mean", "q2_mean",
                "entropy", "memory_size", "pref_shift_l2", "pref_sep_loss"):
        assert key in metrics, key
        assert isinstance(metrics[key], (float, int)), key
    assert metrics["alpha"] == pytest.approx(float(algo.log_alpha.detach().exp()), rel=1e-6)
    assert algo.alpha == metrics["alpha"]
    assert metrics["memory_size"] == 40


def test_update_runs_without_periodic_gc_or_cache_flush(monkeypatch):
    """The per-100-step gc.collect()/empty_cache() pair is gone from update();
    episode-level cleanup remains in reset_episode."""
    import gc
    calls = {"gc": 0}
    real = gc.collect

    def counting():
        calls["gc"] += 1
        return real()

    monkeypatch.setattr(gc, "collect", counting)
    algo = _algo()
    rng = np.random.default_rng(8)
    for _ in range(20):
        algo.store_transition(_obs(rng, 3), "v0", 30.0, 1.0, _obs(rng, 3), False)
    algo.training_step = 99            # would have triggered the old cleanup on step 100
    algo.update()
    assert algo.training_step == 100 and calls["gc"] == 0
