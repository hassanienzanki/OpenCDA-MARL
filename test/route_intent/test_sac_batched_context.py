"""SAC batched update-time context encoding.

Two contracts:
1. Numerical equivalence with the legacy per-transition path (frozen
   encoder): the packed batch encoding of a stored raw sequence must equal
   ``_prepare_inputs`` on the original observation dict.
2. ``train_context_encoder=True`` actually trains the encoder (parameters
   change under update) while the default leaves it frozen.
"""
import numpy as np
import pytest
import torch

from opencda_marl.builtin.algorithms.sac import SACAlgorithm


def _algo(**overrides):
    cfg = {"warmup_steps": 0, "memory_size": 256, "batch_size": 8,
           "actor_hidden_dims": [16, 16], "critic_hidden_dims": [16, 16],
           "lstm_hidden_size": 8, "tensorboard": {"enabled": False}}
    cfg.update(overrides)
    return SACAlgorithm(cfg, 5, 1)


def _obs(rng, n):
    return {f"v{i}": rng.normal(size=5).astype(np.float32) for i in range(n)}


def test_context_sequence_matches_prepare_inputs_ordering():
    algo = _algo()
    rng = np.random.default_rng(0)
    obs = _obs(rng, 4)
    seq, n = algo._context_sequence(obs, "v2")
    assert n == 4 and seq.shape == (4, 5)
    # Row 0 is the ego.
    np.testing.assert_array_equal(seq[0], obs["v2"])
    # Rows 1..n-1 are the others sorted by planar distance (nearest first).
    others = [k for k in obs if k != "v2"]
    others.sort(key=lambda k: float(obs[k][0]) ** 2 + float(obs[k][1]) ** 2)
    for row, k in enumerate(others, start=1):
        np.testing.assert_array_equal(seq[row], obs[k])


def test_context_sequence_ties_are_broken_by_table_order():
    """Equal distances keep dict order (stable sort), as the legacy
    per-ego list.sort did — so identical positions never reorder."""
    algo = _algo()
    obs = {"a": np.array([1, 1, 0, 0, 0], np.float32),
           "b": np.array([1, 1, 1, 0, 0], np.float32),
           "c": np.array([1, 1, 2, 0, 0], np.float32),
           "d": np.array([0, 0, 3, 0, 0], np.float32)}
    seq, n = algo._context_sequence(obs, "b")
    assert n == 4
    np.testing.assert_array_equal(seq[1], obs["d"])   # nearest
    np.testing.assert_array_equal(seq[2], obs["a"])   # tie: table order a, c
    np.testing.assert_array_equal(seq[3], obs["c"])


def test_sorted_table_is_memoised_per_table_object():
    algo = _algo()
    rng = np.random.default_rng(9)
    obs = _obs(rng, 5)
    same_values = dict(obs)             # equal content, different object
    e1 = algo._sorted_context_table(obs)
    e2 = algo._sorted_context_table(obs)
    e3 = algo._sorted_context_table(same_values)
    assert e1 is e2, "same table object -> cached entry"
    assert e1 is not e3, "a different object is never served from the cache"
    np.testing.assert_array_equal(e1[1], e3[1])
    np.testing.assert_array_equal(e1[2], e3[2])
    # Cache is bounded.
    for _ in range(10):
        algo._sorted_context_table(_obs(rng, 3))
    assert len(algo._context_table_cache) <= algo._CONTEXT_TABLE_CACHE_SLOTS


def test_acting_and_stored_sequences_agree_with_and_without_cap():
    """The context the actor encodes must be the context the learner stores."""
    rng = np.random.default_rng(11)
    obs = _obs(rng, 9)
    for cap in (None, 4):
        algo = _algo(max_context_agents=cap)
        algo.device = torch.device("cpu")
        algo.lstm_encoder = algo.lstm_encoder.cpu()
        ids = list(obs.keys())
        # Acting path: rebuild the [B, L, D] sequences exactly as
        # _prepare_batch_inputs does, then compare per ego with the stored one.
        seen = {}
        original_encoder = algo.lstm_encoder

        class _Spy(torch.nn.Module):
            def forward(self, x):
                seen["seqs"] = x.detach().cpu().numpy().copy()
                return original_encoder(x)

        algo.lstm_encoder = _Spy()
        ego_states, _, resolved = algo._prepare_batch_inputs(obs, ids)
        algo.lstm_encoder = original_encoder
        for i, aid in enumerate(resolved):
            stored, n = algo._context_sequence(obs, aid)
            expected_len = len(obs) if cap is None else cap
            assert n == expected_len
            np.testing.assert_array_equal(seen["seqs"][i], stored)
            np.testing.assert_array_equal(ego_states[i].numpy(), obs[aid])


def test_batched_encoding_equals_legacy_per_transition_encoding():
    algo = _algo()
    rng = np.random.default_rng(1)
    dicts = [_obs(rng, n) for n in (1, 2, 4, 7)]   # incl. ego-only and long
    seqs, lens, expected = [], [], []
    for obs in dicts:
        seq, n = algo._context_sequence(obs, "v0")
        seqs.append(seq)
        lens.append(n)
        with torch.no_grad():
            _, ctx = algo._prepare_inputs(obs, "v0")
        expected.append(ctx.cpu().numpy()[0])
    seqs_t = torch.as_tensor(_pad(seqs), device=algo.device)
    lens_t = torch.as_tensor(lens, device=algo.device)
    with torch.no_grad():
        batched = algo._encode_context_batch(seqs_t, lens_t).cpu().numpy()
    # Mathematically identical; on CPU they agree to float32 eps (1e-7).
    # On CUDA the packed-sequence cuDNN kernel differs from the unpacked
    # one at ~1e-5, so the tolerance reflects the device, not the math.
    atol = 1e-6 if algo.device.type == "cpu" else 1e-4
    np.testing.assert_allclose(batched, np.stack(expected), atol=atol)


def test_batched_encoding_is_exact_on_cpu():
    """The device-independent statement of equivalence."""
    algo = _algo()
    algo.device = torch.device("cpu")
    algo.lstm_encoder = algo.lstm_encoder.cpu()
    rng = np.random.default_rng(4)
    dicts = [_obs(rng, n) for n in (1, 3, 5, 8)]
    seqs, lens, expected = [], [], []
    for obs in dicts:
        seq, n = algo._context_sequence(obs, "v0")
        seqs.append(seq)
        lens.append(n)
        with torch.no_grad():
            _, ctx = algo._prepare_inputs(obs, "v0")
        expected.append(ctx.numpy()[0])
    with torch.no_grad():
        batched = algo._encode_context_batch(
            torch.as_tensor(_pad(seqs)), torch.as_tensor(lens)).numpy()
    np.testing.assert_allclose(batched, np.stack(expected), atol=1e-6)


def _pad(seqs):
    width = max(s.shape[0] for s in seqs)
    out = np.zeros((len(seqs), width, seqs[0].shape[1]), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, :s.shape[0]] = s
    return out


def test_truncation_keeps_the_nearest_neighbours():
    algo = _algo(max_context_agents=3)
    rng = np.random.default_rng(2)
    obs = _obs(rng, 6)
    seq, n = algo._context_sequence(obs, "v0")
    assert n == 3 and seq.shape == (3, 5)
    others = sorted((k for k in obs if k != "v0"),
                    key=lambda k: float(obs[k][0]) ** 2 + float(obs[k][1]) ** 2)
    np.testing.assert_array_equal(seq[1], obs[others[0]])
    np.testing.assert_array_equal(seq[2], obs[others[1]])


def test_uncapped_by_default_and_zero_or_none_means_uncapped():
    assert _algo().max_context_agents is None
    assert _algo(max_context_agents=0).max_context_agents is None
    assert _algo(max_context_agents=None).max_context_agents is None
    assert _algo(max_context_agents=5).max_context_agents == 5
    with pytest.raises(ValueError):
        _algo(max_context_agents=-1)


def test_stored_width_grows_with_the_widest_sequence_and_keeps_old_rows():
    algo = _algo()
    rng = np.random.default_rng(5)
    algo.store_transition(_obs(rng, 3), "v0", 30.0, 1.0, _obs(rng, 3), False)
    assert algo._seq_width == 3
    first = algo.memory._indices_to_transitions(np.array([0]))[0]
    seq_before = first[1].copy()
    # A wider neighbour set arrives: width grows, the old row is intact.
    algo.store_transition(_obs(rng, 7), "v0", 30.0, 1.0, _obs(rng, 2), False)
    assert algo._seq_width == 7
    rows = algo.memory._indices_to_transitions(np.array([0, 1]))
    assert rows[0][1].shape == (7, 5) and rows[1][1].shape == (7, 5)
    np.testing.assert_array_equal(rows[0][1][:3], seq_before)
    assert not rows[0][1][3:].any()
    assert int(rows[0][2]) == 3 and int(rows[1][2]) == 7 and int(rows[1][7]) == 2
    # Narrower transitions after growth are padded, not rejected.
    algo.store_transition(_obs(rng, 1), "v0", 30.0, 1.0, _obs(rng, 1), True)
    assert len(algo.memory) == 3
    # And update() runs on the mixed-width buffer.
    for _ in range(12):
        algo.store_transition(_obs(rng, 4), "v0", 30.0, 1.0, _obs(rng, 4), False)
    assert "critic_loss" in algo.update()


def _fill_and_update(algo, steps=40):
    rng = np.random.default_rng(3)
    for _ in range(steps):
        o, n = _obs(rng, 4), _obs(rng, 4)
        algo.store_transition(o, "v0", 30.0, 1.0, n, False)
    return algo.update()


def test_frozen_encoder_by_default():
    algo = _algo()
    before = [p.detach().clone() for p in algo.lstm_encoder.parameters()]
    metrics = _fill_and_update(algo)
    assert "critic_loss" in metrics
    for b, a in zip(before, algo.lstm_encoder.parameters()):
        assert torch.equal(b, a), "encoder must not change when train_context_encoder is off"
    # Encoder params are not in the critic optimizer.
    opt_params = {id(p) for g in algo.critic_optimizer.param_groups for p in g["params"]}
    assert not any(id(p) in opt_params for p in algo.lstm_encoder.parameters())


def test_train_context_encoder_updates_encoder():
    algo = _algo(train_context_encoder=True)
    before = [p.detach().clone() for p in algo.lstm_encoder.parameters()]
    _fill_and_update(algo)
    changed = any(not torch.equal(b, a)
                  for b, a in zip(before, algo.lstm_encoder.parameters()))
    assert changed, "encoder should learn from the critic loss"
    opt_params = {id(p) for g in algo.critic_optimizer.param_groups for p in g["params"]}
    assert all(id(p) in opt_params for p in algo.lstm_encoder.parameters())


def test_buffer_handed_to_a_fresh_learner_keeps_its_width():
    """CARLA world reloads rebuild the learner and hand it the OLD replay
    buffer (TrainingStateSnapshot). The new instance must adopt the buffer's
    real sequence width instead of its own initial 1 — otherwise every push
    after a reload is rejected (narrower rows cannot be broadcast, and a
    'widen' to a smaller width raises)."""
    rng = np.random.default_rng(12)
    old = _algo()
    old.store_transition(_obs(rng, 6), "v0", 30.0, 1.0, _obs(rng, 6), False)
    assert old._seq_width == 6

    fresh = _algo()
    fresh.memory = old.memory                  # what the snapshot does
    fresh.store_transition(_obs(rng, 3), "v0", 30.0, 1.0, _obs(rng, 3), False)   # narrower
    assert len(fresh.memory) == 2 and fresh._seq_width == 6
    rows = fresh.memory._indices_to_transitions(np.array([1]))
    assert rows[0][1].shape == (6, 5) and int(rows[0][2]) == 3
    fresh.store_transition(_obs(rng, 8), "v0", 30.0, 1.0, _obs(rng, 8), False)   # wider
    assert fresh._seq_width == 8 and len(fresh.memory) == 3
    # The old handle catches up on its next push, too.
    old.store_transition(_obs(rng, 2), "v0", 30.0, 1.0, _obs(rng, 2), False)
    assert old._seq_width == 8 and len(old.memory) == 4


def test_stored_layout_round_trips_through_update():
    algo = _algo()
    _fill_and_update(algo)
    tr = algo.memory.sample(1)[0]
    assert len(tr) == 9
    assert tr[0].shape == (5,)              # ego
    assert tr[1].shape == (algo._seq_width, 5)   # raw sequence, buffer width
    assert algo._seq_width == 4             # widest sequence seen so far
    assert int(tr[2]) >= 1                  # length
    assert tr[6].shape == (algo._seq_width, 5)
