"""TrainingStateSnapshot: SAC state must round-trip through world reloads."""
from types import SimpleNamespace

import torch
import pytest

from opencda_marl._kernel.lifecycle.world_manager import TrainingStateSnapshot


class _FakeSAC:
    """Shape-compatible stand-in with the attributes capture() reads."""

    def __init__(self):
        self.actor = torch.nn.Linear(2, 2)
        self.critic = torch.nn.Linear(2, 2)
        self.critic_target = torch.nn.Linear(2, 2)
        self.lstm_encoder = torch.nn.Linear(2, 2)
        self.log_alpha = torch.tensor(0.5, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        # BaseAlgorithm bookkeeping that capture() reads unconditionally.
        self.training_step = 7
        self.episode_count = 3
        self.is_converged = False
        self.convergence_episode = None
        self.writer = None
        self.tb_enabled = False


def _fake_env(algorithm):
    return SimpleNamespace(
        marl_manager=SimpleNamespace(algorithm=algorithm),
        metrics=object(),
        checkpoint_manager=object(),
        training_config={},
    )


def test_sac_snapshot_round_trips_temperature_and_targets():
    algo = _FakeSAC()
    env = _fake_env(algo)

    snapshot = TrainingStateSnapshot.capture(env)
    # The SAC branch must have fired — these were silently lost before.
    assert 'critic_target' in snapshot.model_state_dicts
    assert 'log_alpha' in snapshot.model_state_dicts
    assert 'lstm_encoder' in snapshot.model_state_dicts
    assert 'alpha_optimizer' in snapshot.model_state_dicts

    original_log_alpha_tensor = algo.log_alpha
    original_target = {
        k: v.clone() for k, v in algo.critic_target.state_dict().items()
    }

    # Simulate post-capture drift.
    with torch.no_grad():
        algo.log_alpha.fill_(-2.0)
        for parameter in algo.critic_target.parameters():
            parameter.add_(1.0)

    snapshot.restore(env)

    # Value restored AND tensor identity preserved (the alpha optimizer holds
    # a reference to this exact tensor — rebinding would detach them).
    assert algo.log_alpha is original_log_alpha_tensor
    assert algo.log_alpha.item() == pytest.approx(0.5)
    for key, value in algo.critic_target.state_dict().items():
        assert torch.allclose(value, original_target[key])


def test_sac_snapshot_without_lstm_is_fine():
    algo = _FakeSAC()
    del algo.lstm_encoder
    env = _fake_env(algo)
    snapshot = TrainingStateSnapshot.capture(env)
    assert 'lstm_encoder' not in snapshot.model_state_dicts
    snapshot.restore(env)  # must not raise
    assert algo.log_alpha.item() == pytest.approx(0.5)


class _FakeTD3:
    def __init__(self):
        self.actor = torch.nn.Linear(2, 2)
        self.actor_target = torch.nn.Linear(2, 2)
        self.critic = torch.nn.Linear(2, 2)
        self.critic_target = torch.nn.Linear(2, 2)
        self.lstm_encoder = torch.nn.Linear(2, 2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.memory = ["a", "b"]
        self.training_step = 1
        self.episode_count = 1
        self.is_converged = False
        self.convergence_episode = None
        self.writer = None
        self.tb_enabled = False


class _FakePPO:
    def __init__(self):
        self.actor = torch.nn.Linear(2, 2)
        self.critic = torch.nn.Linear(2, 2)
        self.lstm_encoder = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters())
            + list(self.lstm_encoder.parameters()), lr=1e-3)
        self.rollout_buffer = ["partial", "rollout"]
        self._open_next_obs = {(0, "v1"): "table"}
        self.steps_collected = 640
        self.updates_performed = 4
        self._steps_since_update = 37
        self.training_step = 0
        self.episode_count = 2
        self.is_converged = False
        self.convergence_episode = None
        self.writer = None
        self.tb_enabled = False


def _drift(module):
    with torch.no_grad():
        for p in module.parameters():
            p.add_(1.0)


def test_td3_snapshot_keeps_encoder_and_optimizers():
    algo = _FakeTD3()
    snapshot = TrainingStateSnapshot.capture(_fake_env(algo))
    assert 'lstm_encoder' in snapshot.model_state_dicts
    assert 'actor_optimizer' in snapshot.model_state_dicts
    assert 'critic_optimizer' in snapshot.model_state_dicts
    original = {k: v.clone() for k, v in algo.lstm_encoder.state_dict().items()}
    fresh = _FakeTD3()                      # a rebuilt learner after reload
    _drift(fresh.lstm_encoder)
    snapshot.restore(_fake_env(fresh))
    for k, v in fresh.lstm_encoder.state_dict().items():
        assert torch.allclose(v, original[k])
    assert fresh.memory is algo.memory      # buffer handed over by reference


def test_ppo_snapshot_keeps_encoder_rollout_and_counters():
    algo = _FakePPO()
    snapshot = TrainingStateSnapshot.capture(_fake_env(algo))
    assert 'lstm_encoder' in snapshot.model_state_dicts
    original = {k: v.clone() for k, v in algo.lstm_encoder.state_dict().items()}
    fresh = _FakePPO()
    _drift(fresh.lstm_encoder)
    snapshot.restore(_fake_env(fresh))
    for k, v in fresh.lstm_encoder.state_dict().items():
        assert torch.allclose(v, original[k])
    assert fresh.rollout_buffer is algo.rollout_buffer
    assert fresh._open_next_obs is algo._open_next_obs
    assert (fresh.steps_collected, fresh.updates_performed, fresh._steps_since_update) == (640, 4, 37)


def test_merge_shared_config_does_not_mutate_source():
    from opencda_marl._kernel.policy.manager import MARLManager

    manager = object.__new__(MARLManager)
    manager.config = {
        "training": {"training_mode": False},  # eval mode disables tensorboard
        "outputs": {},
    }
    source = {"tensorboard": {"enabled": True}, "nested": {"lr": 1e-3}}
    merged = manager._merge_shared_config(source)

    assert merged["tensorboard"]["enabled"] is False
    # The caller's config must be untouched (shallow copy used to share the
    # nested dicts, so this write leaked back into the live config).
    assert source["tensorboard"]["enabled"] is True
    merged["nested"]["lr"] = 999
    assert source["nested"]["lr"] == 1e-3
