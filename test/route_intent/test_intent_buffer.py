"""OneShotRolloutBuffer tests."""
# Import torch before numpy to avoid a Windows DLL load conflict in this pixi env
# (numpy's loaded MKL DLLs shadow ones torch needs from torch/lib/shm.dll).
import torch  # noqa: I001  (import order matters)
import numpy as np
import pytest

from opencda_marl.builtin.algorithms.intent.buffer import OneShotRolloutBuffer


def test_scalar_buffer_round_trip():
    buf = OneShotRolloutBuffer(state_dim=6, pref_dim=0, reward_dim=0,
                               device=torch.device("cpu"))
    buf.add(
        observation=np.zeros(6, dtype=np.float32),
        action_index=2,
        action_mask=np.array([True, True, False, True]),
        log_prob=-0.5,
        value=0.1,
        preference=np.zeros(0, dtype=np.float32),
        terminal_reward=1.0,
    )
    assert len(buf) == 1
    batch = next(buf.get_batches(batch_size=1))
    assert batch["observations"].shape == (1, 6)
    assert batch["actions"].tolist() == [2]
    assert batch["action_masks"].tolist() == [[True, True, False, True]]
    assert batch["old_log_probs"].dtype == torch.float32
    # advantage = return - value for one-shot PPO
    assert batch["returns"].tolist() == [1.0]
    assert pytest.approx(batch["advantages"].item(), abs=1e-5) == (1.0 - 0.1)


def test_vector_buffer_uses_vector_values_and_returns():
    buf = OneShotRolloutBuffer(state_dim=6, pref_dim=2, reward_dim=3,
                               device=torch.device("cpu"))
    buf.add(
        observation=np.zeros(6, dtype=np.float32),
        action_index=0,
        action_mask=np.array([True, False, False, True]),
        log_prob=-1.0,
        value=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        preference=np.array([0.5, 0.5], dtype=np.float32),
        terminal_reward=np.array([1.0, -0.5, 0.25], dtype=np.float32),
    )
    batch = next(buf.get_batches(batch_size=1))
    assert batch["vector_values"].shape == (1, 3)
    assert batch["vector_returns"].shape == (1, 3)
    assert batch["preferences"].shape == (1, 2)


def test_clear_empties_buffer():
    buf = OneShotRolloutBuffer(state_dim=3, pref_dim=0, reward_dim=0,
                               device=torch.device("cpu"))
    buf.add(
        observation=np.zeros(3, dtype=np.float32),
        action_index=0,
        action_mask=np.array([True, True, True, True]),
        log_prob=0.0,
        value=0.0,
        preference=np.zeros(0, dtype=np.float32),
        terminal_reward=0.0,
    )
    buf.clear()
    assert len(buf) == 0
    with pytest.raises(StopIteration):
        next(buf.get_batches(batch_size=1))


def _add_scalar(buf, reward, obs_dim=2):
    buf.add(
        observation=np.full(obs_dim, reward, dtype=np.float32),
        action_index=0,
        action_mask=np.array([True, True, True, True]),
        log_prob=0.0,
        value=0.0,
        preference=np.zeros(0, dtype=np.float32),
        terminal_reward=reward,
    )


def test_full_buffer_overwrites_oldest_fifo():
    buf = OneShotRolloutBuffer(state_dim=2, pref_dim=0, reward_dim=0,
                               device=torch.device("cpu"), buffer_size=3)
    for r in [0.0, 1.0, 2.0, 3.0, 4.0]:
        _add_scalar(buf, r)
    # Capacity respected; the two oldest samples (0.0, 1.0) were replaced.
    assert len(buf) == 3
    assert buf.total_overwrites == 2
    batch = next(buf.get_batches(batch_size=3, normalize_advantages=False))
    assert sorted(batch["returns"].tolist()) == [2.0, 3.0, 4.0]


def test_ring_keeps_parallel_lists_consistent_for_vector_rewards():
    buf = OneShotRolloutBuffer(state_dim=2, pref_dim=2, reward_dim=2,
                               device=torch.device("cpu"), buffer_size=2)
    for r in [0.0, 1.0, 2.0]:
        buf.add(
            observation=np.zeros(2, dtype=np.float32),
            action_index=int(r),
            action_mask=np.array([True, True, True, True]),
            log_prob=0.0,
            value=np.array([r, r], dtype=np.float32),
            preference=np.array([0.5, 0.5], dtype=np.float32),
            terminal_reward=np.array([r, r], dtype=np.float32),
        )
    assert len(buf) == 2
    assert len(buf.vector_values) == len(buf.vector_returns) == len(buf.actions) == 2
    batch = next(buf.get_batches(batch_size=2, normalize_advantages=False))
    assert batch["vector_returns"].shape == (2, 2)
    assert sorted(batch["actions"].tolist()) == [1, 2]


def test_clear_resets_ring_state():
    buf = OneShotRolloutBuffer(state_dim=2, pref_dim=0, reward_dim=0,
                               device=torch.device("cpu"), buffer_size=2)
    for r in [0.0, 1.0, 2.0]:
        _add_scalar(buf, r)
    buf.clear()
    assert len(buf) == 0
    assert buf.total_overwrites == 0
    _add_scalar(buf, 7.0)
    assert len(buf) == 1
    batch = next(buf.get_batches(batch_size=1, normalize_advantages=False))
    assert batch["returns"].tolist() == [7.0]


def test_advantage_normalization_is_zero_when_one_sample():
    buf = OneShotRolloutBuffer(state_dim=2, pref_dim=0, reward_dim=0,
                               device=torch.device("cpu"))
    buf.add(
        observation=np.zeros(2, dtype=np.float32),
        action_index=0,
        action_mask=np.array([True, True, True, True]),
        log_prob=0.0,
        value=0.4,
        preference=np.zeros(0, dtype=np.float32),
        terminal_reward=1.0,
    )
    batch = next(buf.get_batches(batch_size=1, normalize_advantages=True))
    # With a single sample std == 0; normalization must not produce NaN.
    assert torch.isfinite(batch["advantages"]).all()
