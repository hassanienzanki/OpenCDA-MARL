"""IntentMAPPO and IntentPCMAPPO smoke tests."""
# Import torch before numpy to avoid a Windows DLL load conflict in this pixi env
# (numpy's loaded MKL DLLs shadow ones torch needs from torch/lib/shm.dll).
import torch  # noqa: I001  (import order matters)
import numpy as np

from opencda_marl.builtin.algorithms.intent.intent_algorithm import (
    IntentMAPPOAlgorithm,
    IntentPCMAPPOAlgorithm,
)


def _build_scalar(state_dim=21, num_actions=4):
    return IntentMAPPOAlgorithm(
        config={"hidden_dims": [16, 16], "learning_rate": 3e-4,
                "clip_param": 0.2, "ppo_epochs": 2, "batch_size": 4,
                "entropy_coef": 0.01, "device": "cpu", "tensorboard": {"enabled": False}},
        state_dim=state_dim, num_actions=num_actions,
    )


def _build_vector(state_dim=21, num_actions=4, pref_dim=3, reward_dim=3):
    return IntentPCMAPPOAlgorithm(
        config={"hidden_dims": [16, 16], "learning_rate": 3e-4,
                "clip_param": 0.2, "ppo_epochs": 2, "batch_size": 4,
                "entropy_coef": 0.01, "device": "cpu", "tensorboard": {"enabled": False}},
        state_dim=state_dim, num_actions=num_actions,
        pref_dim=pref_dim, reward_dim=reward_dim,
    )


def test_scalar_select_intent_returns_legal_action():
    algo = _build_scalar()
    state = np.zeros(21, dtype=np.float32)
    mask = np.array([True, False, True, True])
    pref = np.zeros(0, dtype=np.float32)
    action_index, log_prob, value = algo.select_intent(state, mask, pref,
                                                       training=True)
    assert 0 <= int(action_index) < 4
    assert mask[int(action_index)]
    assert isinstance(float(log_prob), float)
    assert isinstance(float(value), float)


def test_scalar_finalize_then_update_returns_finite_metrics():
    algo = _build_scalar()
    rng = np.random.default_rng(0)
    for i in range(10):
        state = rng.standard_normal(21).astype(np.float32)
        mask = np.array([True, True, False, True])
        pref = np.zeros(0, dtype=np.float32)
        action_index, log_prob, value = algo.select_intent(state, mask, pref,
                                                           training=True)
        algo.store_pending(actor_id=i, observation=state,
                           action_index=action_index, action_mask=mask,
                           log_prob=log_prob, value=value, preference=pref)
        algo.finalize_terminal(actor_id=i, terminal_reward=float(rng.random()))
    metrics = algo.update()
    assert all(np.isfinite(v) for v in metrics.values())


def test_vector_pipeline_uses_preference_scalarization():
    algo = _build_vector()
    rng = np.random.default_rng(1)
    for i in range(8):
        state = rng.standard_normal(21).astype(np.float32)
        mask = np.array([True, True, True, True])
        pref = np.array([0.2, 0.5, 0.3], dtype=np.float32)
        action_index, log_prob, value = algo.select_intent(state, mask, pref,
                                                           training=True)
        algo.store_pending(actor_id=i, observation=state,
                           action_index=action_index, action_mask=mask,
                           log_prob=log_prob, value=value, preference=pref)
        terminal_vec = rng.standard_normal(3).astype(np.float32)
        algo.finalize_terminal(actor_id=i, terminal_reward=terminal_vec)
    metrics = algo.update()
    assert all(np.isfinite(v) for v in metrics.values())


def test_pending_finalize_invariant():
    algo = _build_scalar()
    state = np.zeros(21, dtype=np.float32)
    mask = np.array([True, True, True, True])
    pref = np.zeros(0, dtype=np.float32)
    a, lp, v = algo.select_intent(state, mask, pref, training=True)
    algo.store_pending(actor_id="x", observation=state,
                       action_index=a, action_mask=mask,
                       log_prob=lp, value=v, preference=pref)
    import pytest
    with pytest.raises(KeyError):
        algo.finalize_terminal(actor_id="unknown", terminal_reward=0.0)
    algo.finalize_terminal(actor_id="x", terminal_reward=1.0)
    with pytest.raises(KeyError):
        algo.finalize_terminal(actor_id="x", terminal_reward=1.0)


def test_save_load_roundtrip_preserves_actor_weights(tmp_path):
    """Regression test: marl_manager.save_checkpoint relies on this to persist
    the intent head alongside the base controller. If actor weights do not
    survive save/load, the eval pass would silently use random intent weights.
    """
    algo = _build_vector()
    # Stash a snapshot of every actor parameter before save.
    before = {n: p.detach().clone() for n, p in algo.actor.named_parameters()}
    path = tmp_path / "intent.pth.intent"
    algo.save(str(path))
    assert path.exists() and path.stat().st_size > 0
    # Zero the actor; load; verify weights restored exactly.
    for p in algo.actor.parameters():
        p.data.zero_()
    algo.load(str(path))
    for n, p in algo.actor.named_parameters():
        assert torch.allclose(p.detach(), before[n]), \
            f"actor parameter {n!r} did not round-trip through save/load"
