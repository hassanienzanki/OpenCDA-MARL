"""On-policy family (MAPPO / PCMAPPO): per-trajectory GAE, rollout cadence,
critic bootstrap at truncation, and update-time context encoding.

Before this rework the rollout buffers ran GAE over the flat agent-interleaved
list (agent A's TD target bootstrapped from agent B's value), updated as soon
as ``batch_size`` transitions existed (every ~2 env steps), treated every
rollout cut as terminal, and never sent a gradient to the LSTM encoder (PCMAPPO
stored detached contexts, MAPPO fed zeros to the critic at update).
"""
import numpy as np
import pytest
import torch

from opencda_marl.builtin.algorithms.gae import compute_gae
from opencda_marl.builtin.algorithms.mappo import MAPPOAlgorithm
from opencda_marl.builtin.algorithms.pcmappo import PCMAPPOAlgorithm

GAMMA, LAM = 0.99, 0.95


def _flat_reference(rewards, values, dones, last_value):
    """The textbook single-trajectory recursion (what the buffers used to run
    over the whole interleaved list)."""
    adv = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nv = last_value if t == len(rewards) - 1 else values[t + 1]
        nn_ = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * nv * nn_ - values[t]
        gae = delta + GAMMA * LAM * nn_ * gae
        adv[t] = gae
    return adv, adv + values


def test_gae_runs_within_each_trajectory_with_its_own_bootstrap():
    rng = np.random.default_rng(0)
    # Interleaved rollout: A, B, A, B, A, B, B  (A ends with done at its 3rd step)
    ids = ["A", "B", "A", "B", "A", "B", "B"]
    rewards = rng.normal(size=7).astype(np.float32)
    values = rng.normal(size=7).astype(np.float32)
    dones = np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.float32)
    last_values = {"B": 0.7}          # B is open -> bootstrap; A is done -> none
    adv, ret = compute_gae(rewards, values, dones, gamma=GAMMA, gae_lambda=LAM,
                           trajectory_ids=ids, last_values=last_values)
    for agent, boot in (("A", 0.0), ("B", 0.7)):
        idx = [i for i, a in enumerate(ids) if a == agent]
        ref_adv, ref_ret = _flat_reference(rewards[idx], values[idx], dones[idx], boot)
        np.testing.assert_allclose(adv[idx], ref_adv, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(ret[idx], ref_ret, rtol=1e-6, atol=1e-6)
    # And it is NOT what the flat recursion over the interleaved list gives.
    flat_adv, _ = _flat_reference(rewards, values, dones, 0.0)
    assert not np.allclose(adv, flat_adv)


def test_gae_without_ids_is_the_legacy_single_trajectory_recursion():
    rng = np.random.default_rng(1)
    rewards = rng.normal(size=9).astype(np.float32)
    values = rng.normal(size=9).astype(np.float32)
    dones = (rng.random(9) < 0.2).astype(np.float32)
    adv, ret = compute_gae(rewards, values, dones, gamma=GAMMA, gae_lambda=LAM, last_value=0.3)
    ref_adv, ref_ret = _flat_reference(rewards, values, dones, 0.3)
    np.testing.assert_allclose(adv, ref_adv, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ret, ref_ret, rtol=1e-6, atol=1e-6)


def test_gae_vector_form_matches_per_component_scalar_runs():
    rng = np.random.default_rng(2)
    ids = ["A", "B"] * 4
    rewards = rng.normal(size=(8, 2)).astype(np.float32)
    values = rng.normal(size=(8, 2)).astype(np.float32)
    dones = np.zeros(8, dtype=np.float32)
    last = {"A": np.array([0.1, -0.2], np.float32), "B": np.array([0.5, 0.5], np.float32)}
    adv, ret = compute_gae(rewards, values, dones, gamma=GAMMA, gae_lambda=LAM,
                           trajectory_ids=ids, last_values=last)
    for c in range(2):
        adv_c, ret_c = compute_gae(rewards[:, c], values[:, c], dones, gamma=GAMMA, gae_lambda=LAM,
                                   trajectory_ids=ids, last_values={k: v[c] for k, v in last.items()})
        np.testing.assert_allclose(adv[:, c], adv_c, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(ret[:, c], ret_c, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# Algorithms
# --------------------------------------------------------------------------- #

def _mappo(**overrides):
    cfg = {"warmup_steps": 0, "n_steps": 4, "batch_size": 4, "num_mini_batch": 1,
           "ppo_epochs": 1, "lstm_hidden_size": 8, "actor_hidden_dims": [16, 16],
           "critic_hidden_dims": [16, 16], "tensorboard": {"enabled": False}}
    cfg.update(overrides)
    return MAPPOAlgorithm(cfg, 5, 1)


def _pcmappo(**overrides):
    cfg = {"warmup_steps": 0, "n_steps": 4, "batch_size": 4, "num_mini_batch": 1,
           "ppo_epochs": 1, "lstm_hidden_size": 8, "actor_hidden_dims": [16, 16],
           "critic_hidden_dims": [16, 16], "pref_dim": 2, "tensorboard": {"enabled": False}}
    cfg.update(overrides)
    return PCMAPPOAlgorithm(cfg, 5, 1)


def _obs(rng, n):
    return {f"v{i}": rng.normal(size=5).astype(np.float32) for i in range(n)}


def _drive(algo, rng, steps, agents=3, done_agent=None):
    """Act, then store one transition per agent per step (as the strategy
    does); ``done_agent`` terminates at the last step."""
    updates_before = algo.updates_performed
    for step in range(steps):
        obs, nxt = _obs(rng, agents), _obs(rng, agents)
        actions = algo.select_actions(obs, list(obs.keys()), training=True)
        for aid in obs:
            done = (aid == done_agent and step == steps - 1)
            next_table = {k: v for k, v in nxt.items() if not (done and k == aid)}
            algo.store_transition(obs, aid, actions[aid], 1.0, next_table, done)
        algo.update()
    return algo.updates_performed - updates_before


@pytest.mark.parametrize("make", [_mappo, _pcmappo], ids=["mappo", "pcmappo"])
def test_update_fires_once_per_n_steps_and_clears(make):
    algo = make(n_steps=4)
    rng = np.random.default_rng(3)
    assert _drive(algo, rng, 3) == 0            # 3 < n_steps: no update yet
    assert len(algo.rollout_buffer) == 9
    assert _drive(algo, rng, 1) == 1            # 4th data-bearing step -> update
    assert len(algo.rollout_buffer) == 0 and algo._steps_since_update == 0
    assert _drive(algo, rng, 8) == 2            # two more rollouts of 4


@pytest.mark.parametrize("make", [_mappo, _pcmappo], ids=["mappo", "pcmappo"])
def test_open_trajectories_bootstrap_from_the_critic_and_done_ones_do_not(make):
    algo = make(n_steps=100)                    # never fires during _drive
    rng = np.random.default_rng(4)
    _drive(algo, rng, 3, agents=3, done_agent="v2")
    values = algo._bootstrap_values()
    if isinstance(values, tuple):
        values = values[0]
    open_ids = {(0, "v0"), (0, "v1")}
    assert set(values) == open_ids, values
    assert all(np.isfinite(v) for v in values.values())
    # done agent's trajectory is not in the open map any more
    assert (0, "v2") not in algo._open_next_obs
    # trajectory ids carry the episode: after reset a new run starts
    algo.reset_episode()
    _drive(algo, rng, 1, agents=2)
    assert (1, "v0") in algo._open_next_obs and (0, "v0") in algo._open_next_obs
    assert set(algo.rollout_buffer.trajectory_ids) >= {(0, "v0"), (1, "v0")}


@pytest.mark.parametrize("make", [_mappo, _pcmappo], ids=["mappo", "pcmappo"])
def test_store_transition_runs_no_lstm_forward(make):
    algo = make()
    calls = {"n": 0}
    real = algo.lstm_encoder.forward

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    algo.lstm_encoder.forward = spy
    rng = np.random.default_rng(5)
    obs, nxt = _obs(rng, 3), _obs(rng, 3)
    algo.select_actions(obs, list(obs), training=True)
    acting_calls = calls["n"]
    for aid in obs:
        algo.store_transition(obs, aid, 30.0, 1.0, nxt, False)
    assert calls["n"] == acting_calls, "storage must not encode"
    assert len(algo.rollout_buffer.sequences) == 3
    assert algo.rollout_buffer.sequences[0].shape == (3, 5)


@pytest.mark.parametrize("make", [_mappo, _pcmappo], ids=["mappo", "pcmappo"])
def test_encoder_frozen_by_default_and_trained_when_flag_on(make):
    for flag in (False, True):
        algo = make(train_context_encoder=flag, n_steps=2)
        rng = np.random.default_rng(6)
        before = [p.detach().clone() for p in algo.lstm_encoder.parameters()]
        assert _drive(algo, rng, 2, agents=4) == 1
        changed = any(not torch.equal(b, a) for b, a in zip(before, algo.lstm_encoder.parameters()))
        assert changed == flag, f"train_context_encoder={flag} -> encoder changed={changed}"


@pytest.mark.parametrize("make", [_mappo, _pcmappo], ids=["mappo", "pcmappo"])
def test_stored_action_is_the_raw_sample_so_the_ratio_starts_at_one(make):
    """The rollout stores the unclipped Gaussian sample the log-prob refers
    to (the environment gets the clipped speed), so re-scoring it under the
    unchanged policy reproduces old_log_prob exactly."""
    # Tight bounds and a wide initial std guarantee clipped samples.
    algo = make(n_steps=100, min_action=0.0, max_action=1.0, init_std=2.0)
    torch.manual_seed(8)
    rng = np.random.default_rng(8)
    _drive(algo, rng, 3, agents=8)
    buf = algo.rollout_buffer
    obs = torch.as_tensor(np.stack(buf.observations), device=algo.device)
    if isinstance(algo, PCMAPPOAlgorithm):
        actions = torch.as_tensor(np.stack(buf.actions), device=algo.device)
        new_log_probs = algo.actor.evaluate_actions(obs, actions)[0]
    else:
        actions = torch.as_tensor(np.asarray(buf.actions, dtype=np.float32), device=algo.device).unsqueeze(-1)
        new_log_probs = algo.actor.evaluate_actions(obs, actions)[0]
    old = torch.as_tensor(np.asarray(buf.log_probs, dtype=np.float32), device=algo.device)
    torch.testing.assert_close(new_log_probs.detach(), old, rtol=1e-4, atol=1e-4)
    stored = np.asarray([float(np.asarray(a).reshape(-1)[0]) for a in buf.actions])
    assert (stored < 0.0).any() or (stored > 1.0).any(), "test needs clipped samples to be meaningful"


@pytest.mark.parametrize("make", [_mappo, _pcmappo], ids=["mappo", "pcmappo"])
def test_warmup_transitions_stay_out_of_the_on_policy_rollout(make):
    algo = make(warmup_steps=6, n_steps=100)
    rng = np.random.default_rng(9)
    _drive(algo, rng, 2, agents=3)             # 6 warmup transitions
    assert algo.steps_collected == 6 and len(algo.rollout_buffer) == 0
    _drive(algo, rng, 1, agents=3)             # policy actions now
    assert len(algo.rollout_buffer) == 3
    assert all(lp != 0.0 for lp in algo.rollout_buffer.log_probs)


@pytest.mark.parametrize("make", [_mappo, _pcmappo], ids=["mappo", "pcmappo"])
def test_initial_policy_mean_is_mid_range_not_zero(make):
    algo = make(min_action=0.0, max_action=45.0)
    obs = torch.as_tensor(np.random.default_rng(10).normal(size=(64, 5)).astype(np.float32),
                          device=algo.device)
    with torch.no_grad():
        mean = algo.actor.forward(obs)[0]
    assert 15.0 < float(mean.mean()) < 30.0, float(mean.mean())


@pytest.mark.parametrize("make", [_mappo, _pcmappo], ids=["mappo", "pcmappo"])
def test_actor_and_critic_gradients_are_clipped_separately(make, monkeypatch):
    """A huge value loss must not scale the policy gradient away: the actor's
    parameters are clipped on their own, the critic(+encoder) on their own."""
    import torch.nn as nn_mod
    algo = make(n_steps=2)
    calls = []
    real = nn_mod.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **k):
        params = list(params)
        calls.append((len(params), float(max_norm)))
        return real(params, max_norm, *a, **k)

    monkeypatch.setattr(nn_mod.utils, "clip_grad_norm_", spy)
    rng = np.random.default_rng(11)
    assert _drive(algo, rng, 2, agents=4) == 1
    n_actor = len(list(algo.actor.parameters()))
    n_rest = len(list(algo.critic.parameters())) + len(list(algo.lstm_encoder.parameters()))
    sizes = {n for n, _ in calls}
    assert n_actor in sizes and n_rest in sizes
    assert (n_actor + n_rest) not in sizes, "joint actor+critic clip is gone"


def test_mappo_critic_trains_on_the_encoded_context_not_zeros():
    # The spy below wraps the live critic through a closure the CPU acting
    # replica cannot copy across devices, so this test acts on the training
    # device; the update-time inputs it inspects are unaffected.
    algo = _mappo(n_steps=2, acting_device="train")
    seen = {}
    real_critic = algo.critic

    class _Spy(torch.nn.Module):
        def forward(self, x):
            seen.setdefault("inputs", []).append(x.detach().clone())
            return real_critic(x)

    algo.critic = _Spy()
    rng = np.random.default_rng(7)
    assert _drive(algo, rng, 2, agents=4) == 1
    # The update-time critic input's context slice (last lstm_hidden_size
    # columns) is non-zero for multi-agent rows.
    update_inputs = [x for x in seen["inputs"] if x.requires_grad or x.shape[0] >= 8]
    assert update_inputs, "critic was not called during update"
    ctx = update_inputs[-1][:, -algo.lstm_hidden_size:]
    assert float(ctx.abs().sum()) > 0.0
