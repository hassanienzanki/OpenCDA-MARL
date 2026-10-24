"""SmartReplayBuffer / PrioritizedReplayBuffer window semantics.

The buffers are circular. ``clear_old`` must keep the *newest* entries and
drop the oldest — and after it runs, sampling must still see exactly the
kept entries and nothing that was dropped.
"""
import numpy as np
import pytest

from opencda_marl.builtin.algorithms.smart_replay_buffer import (
    PrioritizedReplayBuffer,
    SmartReplayBuffer,
)


def _fill(buf, n, start=0, **push_kwargs):
    for i in range(start, start + n):
        buf.push(np.full(3, i, dtype=np.float32), i, float(i), False, **push_kwargs)


def _all_stamps(buf):
    """Every entry the buffer currently exposes, via its own sampler."""
    return sorted(int(t[1]) for t in buf._indices_to_transitions(np.arange(len(buf))))


def test_smart_clear_old_keeps_the_newest_entries():
    buf = SmartReplayBuffer(capacity=2000, recency_ratio=0.5)
    _fill(buf, 1800)                    # 90% full, never wrapped
    buf.clear_old(keep_ratio=0.7)
    assert len(buf) == 1260
    stamps = _all_stamps(buf)
    assert stamps == list(range(540, 1800)), "must keep the newest 70%, not the oldest"
    # Sampling after the clear draws only from kept entries.
    for _ in range(20):
        batch = buf.sample(64)
        assert all(540 <= int(t[1]) < 1800 for t in batch)
    # New pushes land after the kept block and stay visible.
    _fill(buf, 10, start=1800)
    assert max(_all_stamps(buf)) == 1809
    assert len(buf) == 1270


def test_smart_clear_old_after_wraparound():
    buf = SmartReplayBuffer(capacity=1000, recency_ratio=0.5)
    _fill(buf, 1500)                    # wrapped: valid = 500..1499
    assert len(buf) == 1000
    buf.clear_old(keep_ratio=0.7)
    assert _all_stamps(buf) == list(range(800, 1500))
    for _ in range(20):
        assert all(800 <= int(t[1]) < 1500 for t in buf.sample(64))


def test_smart_recent_window_after_clear_is_the_newest_slice():
    """Recency-biased half of the batch must come from the newest 20%."""
    buf = SmartReplayBuffer(capacity=2000, recency_ratio=1.0)   # all samples 'recent'
    _fill(buf, 1800)
    buf.clear_old(keep_ratio=0.7)       # keeps 540..1799 (1260 entries)
    recent_start = 1800 - int(1260 * 0.2)
    for _ in range(20):
        assert all(int(t[1]) >= recent_start for t in buf.sample(64))


def test_smart_clear_old_below_threshold_is_noop():
    buf = SmartReplayBuffer(capacity=100)
    _fill(buf, 50)
    buf.clear_old(0.5)
    assert len(buf) == 50


def test_per_clear_old_keeps_the_newest_entries_and_priorities():
    buf = PrioritizedReplayBuffer(capacity=10000, alpha=0.6, beta=0.4)
    for i in range(9000):
        buf.push(np.full(3, i, dtype=np.float32), i, float(i), False, td_error=float(i))
    buf.clear_old(keep_ratio=0.5)
    assert len(buf) == 4500
    assert _all_stamps(buf) == list(range(4500, 9000))
    # Priorities travel with their entries: entry i was pushed with td_error=i.
    expected = (np.abs(np.arange(4500, 9000, dtype=np.float32)) + buf.epsilon) ** buf.alpha
    np.testing.assert_allclose(buf.priorities[:4500], expected, rtol=1e-5)
    batch, indices, weights = buf.sample(32)
    assert all(4500 <= int(t[1]) < 9000 for t in batch)
    assert indices.max() < 4500 and weights.shape == (32,)


def test_widen_field_preserves_existing_rows_and_accepts_wider_pushes():
    buf = SmartReplayBuffer(capacity=8)
    for i in range(3):
        buf.push(np.full((2, 4), i, dtype=np.float32), i)
    buf.widen_field(0, (5, 4))
    for t in buf._indices_to_transitions(np.arange(len(buf))):
        assert t[0].shape == (5, 4)
        assert not t[0][2:].any()                       # zero-filled tail
        assert (t[0][:2] == int(t[1])).all()            # original block intact
    buf.push(np.full((5, 4), 7, dtype=np.float32), 7)   # wider row now fits
    assert len(buf) == 4
    with pytest.raises(ValueError):
        buf.widen_field(0, (3, 4))                      # never shrinks
