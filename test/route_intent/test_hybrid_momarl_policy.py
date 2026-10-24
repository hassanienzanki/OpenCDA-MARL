"""Hybrid MOMARL policy wiring tests."""
# Import torch before numpy to avoid a Windows DLL load conflict in this pixi env
# (numpy's loaded MKL DLLs shadow ones torch needs from torch/lib/shm.dll).
import torch  # noqa: I001
import numpy as np
import pytest

from opencda_marl.builtin.algorithms.pcmappo import PCMAPPOAlgorithm
from opencda_marl._kernel.policy.checkpoint import CheckpointManager
from opencda_marl._kernel.policy.manager import MARLManager


def _pcmappo_config():
    return {
        "tensorboard": {"enabled": False},
        "pref_mode": "direct",
        "pref_dim": 2,
        "actor_hidden_dims": [16, 16],
        "critic_hidden_dims": [16, 16],
        "lstm_hidden_size": 8,
        "lstm_num_layers": 1,
        "init_std": 0.3,
        "clip_param": 0.1,
        "ppo_epochs": 1,
        "num_mini_batch": 1,
        "entropy_coef": 0.01,
        "value_loss_coef": 0.5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "n_steps": 8,
        "batch_size": 4,
        "warmup_steps": 0,
        "max_action": 45.0,
        "min_action": 0.0,
    }


def _route_config():
    return {
        "hidden_dims": [16, 16],
        "clip_param": 0.2,
        "ppo_epochs": 1,
        "batch_size": 4,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "tensorboard": {"enabled": False},
    }


def _learned_action_space():
    return {
        "mode": "path_speed",
        "path": {
            "selector": "learned",
            "reward_dim": 2,
            "preference": {"dim": 2},
            "head": _route_config(),
        },
    }


def test_pcmappo_owns_route_head_and_speed_head():
    algo = PCMAPPOAlgorithm(_pcmappo_config(), state_dim=5, action_dim=1)
    algo.enable_route_intent(
        _route_config(),
        state_dim=7,
        num_actions=4,
        pref_dim=2,
        reward_dim=2,
    )

    speed = algo.select_action(
        {"ego": np.zeros(5, dtype=np.float32)},
        "ego",
        training=False,
    )
    assert isinstance(float(speed), float)

    rng = np.random.default_rng(0)
    for i in range(8):
        obs = rng.standard_normal(7).astype(np.float32)
        mask = np.array([True, False, True, True])
        pref = np.array([0.6, 0.4], dtype=np.float32)
        action_index, log_prob, value = algo.select_intent(
            obs, mask, pref, training=True
        )
        assert mask[int(action_index)]
        algo.store_pending(
            actor_id=i,
            observation=obs,
            action_index=action_index,
            action_mask=mask,
            log_prob=log_prob,
            value=value,
            preference=pref,
        )
        algo.finalize_terminal(
            actor_id=i,
            terminal_reward=rng.standard_normal(2).astype(np.float32),
        )

    metrics = algo.update_route_intent()
    assert np.isfinite(list(metrics.values())).all()
    info = algo.get_training_info()
    assert info["hybrid_route_intent_enabled"] is True
    assert info["route_updates"] == 1.0


def test_marl_manager_hybrid_mode_uses_base_algorithm_as_intent_algorithm():
    cfg = {
        "algorithm": "pcmappo",
        "state_dim": 5,
        "action_dim": 1,
        "action_space": _learned_action_space(),
        "preference": {"profile": "balanced", "dim": 2},
        "pcmappo": _pcmappo_config(),
        "training": {"training_mode": True},
        "outputs": {"tensorboard": {"enabled": False}},
    }
    manager = MARLManager(cfg, "pcmappo")
    assert manager.intent_algorithm is manager.algorithm
    assert manager.intent_reward_accumulator is not None
    assert hasattr(manager.algorithm, "route_intent_head")
    assert manager.algorithm.route_intent_head is not None


def test_marl_manager_updates_route_head_at_ppo_half_batch_threshold():
    cfg = {
        "algorithm": "pcmappo",
        "state_dim": 5,
        "action_dim": 1,
        "action_space": _learned_action_space(),
        "preference": {"profile": "balanced", "dim": 2},
        "pcmappo": _pcmappo_config(),
        "training": {"training_mode": True},
        "outputs": {"tensorboard": {"enabled": False}},
    }
    manager = MARLManager(cfg, "pcmappo")
    rng = np.random.default_rng(11)
    route_state_dim = manager.algorithm.route_intent_head.state_dim
    for actor_id in range(2):
        obs = rng.standard_normal(route_state_dim).astype(np.float32)
        mask = np.array([True, False, True, True])
        pref = np.array([0.5, 0.5], dtype=np.float32)
        action_index, log_prob, value = manager.algorithm.select_intent(
            obs, mask, pref, training=True
        )
        manager.algorithm.store_pending(
            actor_id=actor_id,
            observation=obs,
            action_index=action_index,
            action_mask=mask,
            log_prob=log_prob,
            value=value,
            preference=pref,
        )
        manager.algorithm.finalize_terminal(
            actor_id=actor_id,
            terminal_reward=rng.standard_normal(2).astype(np.float32),
        )

    assert manager.get_training_metrics()["intent_buffer_size"] == 2.0
    manager.update(rewards={}, observations={}, next_observations={}, objective_outputs={})
    assert manager.algorithm.route_intent_head.diagnostics["updates"] == 1.0


def test_hybrid_checkpoint_embeds_route_head_without_intent_sibling(tmp_path):
    algo = PCMAPPOAlgorithm(_pcmappo_config(), state_dim=5, action_dim=1)
    algo.enable_route_intent(
        _route_config(),
        state_dim=7,
        num_actions=4,
        pref_dim=2,
        reward_dim=2,
    )
    before = {
        name: param.detach().clone()
        for name, param in algo.route_intent_head.actor.named_parameters()
    }

    manager = CheckpointManager(str(tmp_path), "pcmappo")
    manager.save_checkpoint(
        algo,
        episode=1,
        metrics={"total_reward": 1.0},
        intent_algorithm=algo,
    )
    latest = tmp_path / "pcmappo" / "latest_checkpoint.pth"
    assert latest.exists()
    assert not (tmp_path / "pcmappo" / "latest_checkpoint.pth.intent").exists()

    for param in algo.route_intent_head.actor.parameters():
        param.data.zero_()
    algo.load(str(latest))
    for name, param in algo.route_intent_head.actor.named_parameters():
        assert torch.allclose(param.detach(), before[name])


def test_hybrid_checkpoint_does_not_restore_route_runtime_counters(tmp_path):
    algo = PCMAPPOAlgorithm(_pcmappo_config(), state_dim=5, action_dim=1)
    algo.enable_route_intent(
        _route_config(),
        state_dim=7,
        num_actions=4,
        pref_dim=2,
        reward_dim=2,
    )
    algo.route_intent_head.diagnostics["decisions"] = 12.0
    algo.route_intent_head.diagnostics["updates"] = 3.0
    algo.route_intent_head.training_metrics["policy_loss"] = 9.0

    manager = CheckpointManager(str(tmp_path), "pcmappo")
    manager.save_checkpoint(
        algo,
        episode=1,
        metrics={"total_reward": 1.0},
        intent_algorithm=algo,
    )
    latest = tmp_path / "pcmappo" / "latest_checkpoint.pth"

    restored = PCMAPPOAlgorithm(_pcmappo_config(), state_dim=5, action_dim=1)
    restored.enable_route_intent(
        _route_config(),
        state_dim=7,
        num_actions=4,
        pref_dim=2,
        reward_dim=2,
    )
    restored.load(str(latest))

    assert restored.route_intent_head.diagnostics["decisions"] == 0.0
    assert restored.route_intent_head.diagnostics["updates"] == 0.0
    assert restored.route_intent_head.training_metrics["policy_loss"] == 0.0


def test_marl_manager_hybrid_mode_supports_sac_owned_route_head():
    cfg = {
        "algorithm": "pcmasac",
        "state_dim": 5,
        "action_dim": 1,
        "action_space": _learned_action_space(),
        "preference": {"profile": "balanced", "dim": 2},
        "pcmasac": {
            "tensorboard": {"enabled": False},
            "pref_mode": "direct",
            "pref_dim": 2,
            "actor_hidden_dims": [16, 16],
            "critic_hidden_dims": [16, 16],
            "lstm_hidden_size": 8,
            "lstm_num_layers": 1,
            "memory_size": 16,
            "batch_size": 4,
            "warmup_steps": 0,
            "max_action": 45.0,
            "min_action": 0.0,
        },
        "training": {"training_mode": True},
        "outputs": {"tensorboard": {"enabled": False}},
    }
    manager = MARLManager(cfg, "pcmasac")
    assert manager.intent_algorithm is manager.algorithm
    assert manager.algorithm.route_intent_head is not None

    rng = np.random.default_rng(3)
    route_state_dim = manager.algorithm.route_intent_head.state_dim
    for i in range(8):
        obs = rng.standard_normal(route_state_dim).astype(np.float32)
        mask = np.array([True, True, False, True])
        pref = np.array([0.4, 0.6], dtype=np.float32)
        action_index, log_prob, value = manager.algorithm.select_intent(
            obs, mask, pref, training=True
        )
        assert mask[int(action_index)]
        manager.algorithm.store_pending(
            actor_id=i,
            observation=obs,
            action_index=action_index,
            action_mask=mask,
            log_prob=log_prob,
            value=value,
            preference=pref,
        )
        manager.algorithm.finalize_terminal(
            actor_id=i,
            terminal_reward=rng.standard_normal(2).astype(np.float32),
        )
    metrics = manager.algorithm.update_route_intent()
    assert np.isfinite(list(metrics.values())).all()
    assert manager.algorithm.route_intent_head.diagnostics["updates"] == 1.0


def test_hybrid_mode_rejects_td3_until_route_score_variant_exists():
    cfg = {
        "algorithm": "pcmatd3",
        "state_dim": 5,
        "action_dim": 1,
        "action_space": _learned_action_space(),
        "preference": {"profile": "balanced", "dim": 2},
        "pcmatd3": {
            "tensorboard": {"enabled": False},
            "lstm_hidden_size": 8,
            "lstm_num_layers": 1,
            "actor_hidden_dims": [16, 16],
            "critic_hidden_dims": [16, 16],
            "memory_size": 16,
            "batch_size": 4,
            "warmup_steps": 0,
            "max_action": 45.0,
            "min_action": 0.0,
        },
        "training": {"training_mode": True},
        "outputs": {"tensorboard": {"enabled": False}},
    }
    with pytest.raises(NotImplementedError, match="Learned path is not implemented for PCMATD3"):
        MARLManager(cfg, "pcmatd3")


def test_speed_only_masac_creates_no_path_provider():
    cfg = {
        "algorithm": "masac",
        "state_dim": 5,
        "action_dim": 1,
        "action_space": {"mode": "speed_only"},
        "masac": {
            "tensorboard": {"enabled": False},
            "pref_mode": "direct",
            "pref_dim": 0,
            "actor_hidden_dims": [16, 16],
            "critic_hidden_dims": [16, 16],
            "lstm_hidden_size": 8,
            "lstm_num_layers": 1,
            "memory_size": 16,
            "batch_size": 4,
            "warmup_steps": 0,
            "max_action": 45.0,
            "min_action": 0.0,
        },
        "training": {"training_mode": True},
        "outputs": {"tensorboard": {"enabled": False}},
    }
    manager = MARLManager(cfg, "masac")
    assert manager.intent_algorithm is None
    assert manager.get_intent_provider() is None


def test_rule_path_uses_provider_config_without_learned_head():
    cfg = {
        "algorithm": "pcmappo",
        "state_dim": 5,
        "action_dim": 1,
        "action_space": {
            "mode": "path_speed",
            "path": {"selector": "rule_based"},
        },
        "preference": {"profile": "balanced", "dim": 2},
        "pcmappo": _pcmappo_config(),
        "training": {"training_mode": True},
        "outputs": {"tensorboard": {"enabled": False}},
    }
    manager = MARLManager(cfg, "pcmappo")
    assert manager.intent_algorithm is None
    assert manager.get_intent_provider() is None


def test_legacy_intent_policy_schema_is_rejected():
    cfg = {
        "algorithm": "pcmappo",
        "state_dim": 5,
        "action_dim": 1,
        "action_space": {"mode": "speed_only"},
        "intent_policy": {"algorithm": "mappo"},
        "pcmappo": _pcmappo_config(),
    }
    with pytest.raises(ValueError, match="MARL.intent_policy"):
        MARLManager(cfg, "pcmappo")


def test_baseline_manager_does_not_require_action_space():
    manager = MARLManager({}, None)
    assert manager.algorithm is None
    assert manager.compute_actions({}, training=False) == {}
