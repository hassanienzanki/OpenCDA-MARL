"""Action selection runs on a CPU replica of the acting modules; training stays on the GPU.

At the small batch sizes of a traffic step the GPU is launch-bound: acting
on the CPU is faster than on CUDA at these small shapes, and
refreshing every acting weight through one pinned flat copy is cheap. So
the learners now act on CPU copies of their actor, critic
and context encoder, refreshed lazily after any weight change -- an
optimiser step, a checkpoint load -- so the values and log-probabilities
stashed while acting are always those of the current weights. The value
normalisers are mirrored with the replica. ``acting_device: train`` keeps
acting on the training device.
"""
import numpy as np
import pytest
import torch

from opencda_marl.builtin.algorithms.base_algorithm import ActingReplica
from opencda_marl.builtin.algorithms.pcmappo import PCMAPPOAlgorithm

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the replica exists to avoid a GPU")

STATE_DIM = 11
CFG = {
    "actor_hidden_dims": [32, 32], "critic_hidden_dims": [32, 32], "lstm_hidden_size": 16,
    "lstm_num_layers": 1, "pref_dim": 2, "reward_dim": 2, "max_action": 45.0, "min_action": 0.0,
    "warmup_steps": 0, "batch_size": 4, "share_policy": True, "n_steps": 4, "ppo_epochs": 1,
    "value_normalization": True,
}


def _obs(n_agents, seed=0):
    rng = np.random.default_rng(seed)
    obs = {}
    for i in range(n_agents):
        vec = rng.standard_normal(STATE_DIM).astype(np.float32)
        vec[-2:] = [0.5, 0.5]
        obs[str(i)] = vec
    return obs


def _pcmappo(**overrides):
    torch.manual_seed(0)
    return PCMAPPOAlgorithm({**CFG, **overrides}, state_dim=STATE_DIM, action_dim=1)


def _weights_equal(replica, live):
    return all(torch.equal(r.detach().cpu(), l.detach().cpu()) for r, l in zip(replica.parameters(), live.parameters()))


# ─────────────────────────────────────────────────────────────────────────────
# Where acting runs
# ─────────────────────────────────────────────────────────────────────────────

@needs_cuda
def test_acting_runs_on_a_cpu_replica_while_the_live_modules_stay_on_the_gpu():
    algo = _pcmappo()
    assert algo.device.type == "cuda"
    replica = algo._acting()
    assert isinstance(replica, ActingReplica) and replica.device.type == "cpu"
    assert set(replica.modules) == {"actor", "critic", "lstm_encoder"}
    assert next(algo.actor.parameters()).device.type == "cuda"
    assert next(replica.modules["actor"].parameters()).device.type == "cpu"
    obs = _obs(5)
    ego_states, context, ids = algo._prepare_batch_inputs(obs, list(obs))
    assert ego_states.device.type == "cpu" and context.device.type == "cpu"
    results = algo.select_actions(obs, list(obs), training=False)
    assert set(results) == set(obs) and all(0.0 <= v <= 45.0 for v in results.values())
    assert replica.refreshes == 1, "one refresh at first use, none while the weights stand"


@needs_cuda
def test_the_replica_answers_like_the_live_modules_within_float_noise():
    algo = _pcmappo()
    obs = _obs(6, seed=3)
    ego_states, context, ids = algo._prepare_batch_inputs(obs, list(obs))
    algo._batched_forward(ego_states, context, ids, training=False)
    with torch.no_grad():
        gpu_mean, _, _ = algo.actor.forward(ego_states.cuda())
        gpu_scalar, _ = algo._critic_values(ego_states.cuda(), context.cuda())
    for i, agent_id in enumerate(ids):
        assert algo.last_raw_actions[agent_id] == pytest.approx(float(gpu_mean[i, 0]), abs=1e-4)
        assert algo.last_scalar_values[agent_id] == pytest.approx(float(gpu_scalar.reshape(-1)[i]), abs=1e-3)


def test_training_on_the_cpu_or_asking_for_the_training_device_means_no_replica():
    class _Stub:
        config = {"acting_device": "auto"}
        device = torch.device("cpu")

        def _acting_modules(self):
            return {"actor": torch.nn.Linear(2, 1)}

        def _acting_normalizers(self):
            return {}

    stub = _Stub()
    assert PCMAPPOAlgorithm._build_acting_replica(stub) is False, "training on the CPU already"
    stub.config = {"acting_device": "train"}
    stub.device = torch.device("cuda")
    assert PCMAPPOAlgorithm._build_acting_replica(stub) is False, "acting on the training device by request"


@needs_cuda
def test_acting_device_train_keeps_the_gpu_path():
    algo = _pcmappo(acting_device="train")
    assert algo._acting() is None
    obs = _obs(3)
    ego_states, context, _ = algo._prepare_batch_inputs(obs, list(obs))
    assert ego_states.device.type == "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# Staying current
# ─────────────────────────────────────────────────────────────────────────────

@needs_cuda
def test_a_weight_change_reaches_the_replica_once_marked_and_not_before():
    algo = _pcmappo()
    replica = algo._acting()
    assert _weights_equal(replica.modules["actor"], algo.actor)
    with torch.no_grad():
        for parameter in algo.actor.parameters():
            parameter.add_(0.25)
    assert not _weights_equal(replica.modules["actor"], algo.actor)
    assert algo._acting() is replica and not _weights_equal(replica.modules["actor"], algo.actor), "unmarked: stale"
    algo._mark_acting_dirty()
    algo._acting()
    assert _weights_equal(replica.modules["actor"], algo.actor)
    assert _weights_equal(replica.modules["critic"], algo.critic)
    assert _weights_equal(replica.modules["lstm_encoder"], algo.lstm_encoder)
    assert replica.refreshes == 2


@needs_cuda
def test_a_real_update_refreshes_the_replica_and_the_normaliser_mirror():
    algo = _pcmappo()
    rng = np.random.default_rng(5)
    replica = algo._acting()
    before = replica.refreshes
    obs = _obs(3)
    for step in range(12):
        actions = algo.select_actions(obs, list(obs), training=True)
        next_obs = {k: (v + rng.standard_normal(STATE_DIM).astype(np.float32) * 0.1) for k, v in obs.items()}
        for agent_id in obs:
            algo.store_transition(obs, agent_id, actions[agent_id], 1000.0 * rng.random(), next_obs, False)
        algo.update()
        obs = next_obs
    assert algo.updates_performed >= 1
    assert replica.refreshes > before
    assert replica.dirty, "the last update left the replica marked; it refreshes on the next acting call"
    algo._acting()
    assert _weights_equal(replica.modules["actor"], algo.actor)
    live_mean, live_var = algo.value_normalizer.mean_var()
    mirror_mean, mirror_var = replica.normalizers["value"].mean_var()
    assert torch.allclose(mirror_mean.cpu(), live_mean.cpu()) and torch.allclose(mirror_var.cpu(), live_var.cpu())
    assert float(live_var.sqrt()) > 1.0, "the normaliser saw returns of order 1e3"
    # And the acting critic de-normalises with the mirrored statistics.
    ego_states, context, ids = algo._prepare_batch_inputs(obs, list(obs))
    with torch.no_grad():
        acting_scalar, _ = algo._critic_values(ego_states, context)
        live_scalar, _ = algo._critic_values(ego_states.cuda(), context.cuda())
    assert torch.allclose(acting_scalar.cpu(), live_scalar.cpu(), atol=1e-2)


@needs_cuda
def test_loading_a_checkpoint_refreshes_the_replica(tmp_path):
    source = _pcmappo()
    with torch.no_grad():
        for parameter in source.actor.parameters():
            parameter.fill_(0.123)
    path = tmp_path / "ckpt.pth"
    source.save(str(path))
    target = _pcmappo()
    replica = target._acting()
    assert not _weights_equal(replica.modules["actor"], source.actor)
    target.load(str(path))
    target._acting()
    assert _weights_equal(replica.modules["actor"], target.actor) and _weights_equal(replica.modules["actor"], source.actor)


# ─────────────────────────────────────────────────────────────────────────────
# The other learners opt in the same way
# ─────────────────────────────────────────────────────────────────────────────

@needs_cuda
def test_mappo_sac_and_td3_act_on_a_replica_too():
    from opencda_marl.builtin.algorithms.mappo import MAPPOAlgorithm
    from opencda_marl.builtin.algorithms.sac import SACAlgorithm
    from opencda_marl.builtin.algorithms.td3 import TD3Algorithm
    common = {"lstm_hidden_size": 32, "lstm_num_layers": 1, "critic_hidden_dims": [64, 64], "max_action": 45.0,
              "min_action": 0.0, "warmup_steps": 0, "memory_size": 1000, "batch_size": 32}
    cases = [
        (MAPPOAlgorithm({**common, "actor_hidden_dims": [64, 64], "batch_size": 8, "share_policy": True}, 9, 1), 9,
         {"actor", "critic", "lstm_encoder"}),
        (SACAlgorithm({**common, "actor_hidden_dims": [64, 64], "pref_dim": 2, "reward_dim": 2}, 11, 1), 11,
         {"actor", "lstm_encoder"}),
        (TD3Algorithm(dict(common), 8, 1), 8, {"actor", "lstm_encoder"}),
    ]
    for algo, state_dim, modules in cases:
        rng = np.random.default_rng(1)
        obs = {str(i): rng.standard_normal(state_dim).astype(np.float32) for i in range(4)}
        for vec in obs.values():
            vec[6] = 5.0
        replica = algo._acting()
        assert isinstance(replica, ActingReplica) and set(replica.modules) == modules, type(algo).__name__
        results = algo.select_actions(obs, list(obs), training=False)
        assert set(results) == set(obs), type(algo).__name__
        assert all(v is None or np.isfinite(v) for v in results.values()), type(algo).__name__


@needs_cuda
def test_the_replica_caps_the_cpu_thread_pool_for_its_small_batches():
    """A tiny op pays to wake every thread of the default pool, so the
    replica caps it for its small batches. ``acting_threads`` sets the cap
    explicitly; otherwise it is at most 8."""
    before = torch.get_num_threads()
    try:
        algo = _pcmappo(acting_threads=3)
        assert algo._acting() is not None
        assert torch.get_num_threads() == 3
        algo = _pcmappo()                                    # auto: at most 8
        assert algo._acting() is not None
        assert torch.get_num_threads() <= 8
    finally:
        torch.set_num_threads(before)
