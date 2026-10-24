"""The learner crosses the device boundary once per acting step and uploads a rollout once per update.

The per-agent work of the PPO families was already batched: one context
table sort, one LSTM encode, one actor pass and one critic pass per step.
What the profile still showed at the boundary were two host-to-device
copies of the same table rows, three or four device-to-host syncs for the
step's outputs, and a rollout that was re-stacked and re-uploaded for every
PPO epoch. Each of those is a launch and a sync on a GPU. Now the ego rows
are read from the sequence tensor already on the device, the outputs come
back in one transfer, and the rollout's device tensors are built once per
fill and dropped by ``add`` and ``clear``. Numerics are unchanged: the same
float32 values travel, in fewer trips.
"""
import numpy as np
import pytest
import torch

from opencda_marl.builtin.algorithms.pcmappo import PCMAPPOAlgorithm, _to_host

STATE_DIM = 11
CFG = {
    "actor_hidden_dims": [32, 32], "critic_hidden_dims": [32, 32], "lstm_hidden_size": 16,
    "lstm_num_layers": 1, "pref_dim": 2, "reward_dim": 2, "max_action": 45.0, "min_action": 0.0,
    "warmup_steps": 0, "batch_size": 4, "share_policy": True, "n_steps": 1,
}


def _obs(n_agents, seed=0):
    rng = np.random.default_rng(seed)
    obs = {}
    for i in range(n_agents):
        vec = rng.standard_normal(STATE_DIM).astype(np.float32)
        vec[-2:] = [0.5, 0.5]                       # the preference tail
        obs[str(i)] = vec
    return obs


def _algo():
    torch.manual_seed(0)
    return PCMAPPOAlgorithm(dict(CFG), state_dim=STATE_DIM, action_dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Host-to-device: the ego rows ride in the sequence tensor
# ─────────────────────────────────────────────────────────────────────────────

def test_ego_states_are_the_sequence_tensors_first_slot_and_the_table_is_copied_once(monkeypatch):
    algo = _algo()
    obs = _obs(5)
    copies = []
    real = torch.as_tensor

    def counting(*args, **kwargs):
        copies.append(kwargs.get("device"))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", counting)
    ego_states, context, ids = algo._prepare_batch_inputs(obs, list(obs))
    assert len(copies) == 1, "one host-to-device copy for the whole step"
    expected = np.stack([obs[i] for i in ids])
    np.testing.assert_array_equal(ego_states.detach().cpu().numpy(), expected)
    assert ego_states.shape == (5, STATE_DIM) and context.shape == (5, CFG["lstm_hidden_size"])


def test_a_lone_agent_still_gets_its_ego_state_and_a_zero_context():
    algo = _algo()
    obs = _obs(1)
    ego_states, context, ids = algo._prepare_batch_inputs(obs, ["0"])
    np.testing.assert_array_equal(ego_states.detach().cpu().numpy(), obs["0"][None])
    assert torch.count_nonzero(context) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Device-to-host: one transfer for the step's outputs
# ─────────────────────────────────────────────────────────────────────────────

def test_to_host_returns_the_same_values_as_separate_transfers():
    n = 6
    actions = torch.randn(n, 1)
    log_probs = torch.randn(n)
    scalar = torch.randn(n, 1)
    vector = torch.randn(n, 2)
    a, lp, sv, vv = _to_host(actions, log_probs, scalar, vector)
    np.testing.assert_array_equal(a, actions.numpy())
    np.testing.assert_array_equal(lp, log_probs.numpy())
    np.testing.assert_array_equal(sv, scalar.reshape(n).numpy())
    np.testing.assert_array_equal(vv, vector.numpy())
    a2, lp2, sv2, vv2 = _to_host(actions, log_probs, scalar)
    assert vv2 is None and a2.shape == (n, 1) and lp2.shape == (n,) and sv2.shape == (n,)


def test_the_batched_forward_syncs_the_device_once_and_stashes_the_same_numbers(monkeypatch):
    algo = _algo()
    obs = _obs(4)
    ego_states, context, ids = algo._prepare_batch_inputs(obs, list(obs))
    syncs = []
    real_cpu = torch.Tensor.cpu

    def counting(self, *args, **kwargs):
        syncs.append(tuple(self.shape))
        return real_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counting)
    results = algo._batched_forward(ego_states, context, ids, training=False)
    monkeypatch.undo()
    assert len(syncs) == 1, f"one device-to-host transfer per step, got {syncs}"
    with torch.no_grad():                                   # the acting modules: a CPU replica when training on a GPU
        mean, _std, _aux = algo._acting_module("actor").forward(ego_states)
        scalar, _vector = algo._critic_values(ego_states, context)
    for i, agent_id in enumerate(ids):
        raw = float(mean[i].flatten()[0])
        assert algo.last_raw_actions[agent_id] == pytest.approx(raw, abs=1e-6)
        assert results[agent_id] == pytest.approx(float(np.clip(raw, 0.0, 45.0)), abs=1e-6)
        assert algo.last_scalar_values[agent_id] == pytest.approx(float(scalar[i].flatten()[0]), abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# The rollout: stacked and uploaded once per fill
# ─────────────────────────────────────────────────────────────────────────────

def _fill(buffer, n, seq_len=3, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        seq = rng.standard_normal((seq_len, STATE_DIM)).astype(np.float32)
        buffer.add(observation=seq[0], sequence=seq, action=np.asarray([float(i)], dtype=np.float32),
                   scalar_reward=float(i), scalar_value=0.1 * i,
                   raw_preference=np.asarray([0.5, 0.5], dtype=np.float32),
                   log_prob=-0.5, done=(i == n - 1), trajectory_id=("ep", "a"))
    buffer.compute_returns_and_advantages(last_scalar_values={}, normalize_advantages=False)


def _batches(buffer, batch_size):
    return [{k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v) for k, v in b.items()}
            for b in buffer.get_batches(batch_size=batch_size, shuffle=False)]


def test_the_rollout_is_stacked_once_for_all_epochs_and_the_batches_are_identical(monkeypatch):
    algo = _algo()
    buffer = algo.rollout_buffer
    _fill(buffer, 6)
    stacks = []
    real = np.stack

    def counting(*args, **kwargs):
        stacks.append(len(args[0]))
        return real(*args, **kwargs)

    monkeypatch.setattr(np, "stack", counting)
    epochs = [_batches(buffer, 4) for _ in range(3)]
    assert stacks == [6, 6, 6], "observations, actions and preferences stacked once, not once per epoch"
    for later in epochs[1:]:
        assert len(later) == len(epochs[0]) == 2
        for a, b in zip(epochs[0], later):
            assert a.keys() == b.keys()
            for key in a:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    first = epochs[0][0]
    assert first["observations"].shape == (4, STATE_DIM) and first["sequences"].shape == (4, 3, STATE_DIM)
    assert first["lengths"].tolist() == [3, 3, 3, 3]


def test_add_and_clear_drop_the_cached_tensors():
    algo = _algo()
    buffer = algo.rollout_buffer
    _fill(buffer, 3)
    list(buffer.get_batches(batch_size=3, shuffle=False))
    assert buffer._device_cache is not None and buffer._device_cache[0] == 3
    _fill(buffer, 1, seed=1)                                   # add() drops it, the next read rebuilds it
    assert buffer._device_cache is None
    (batch,) = list(buffer.get_batches(batch_size=8, shuffle=False))
    assert batch["observations"].shape[0] == 4
    buffer.clear()
    assert buffer._device_cache is None and len(buffer) == 0
