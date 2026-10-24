"""A reopened external-policy store continues its counters.

For a policy that does not subclass ``BaseAlgorithm`` the manager keeps the
metrics store, the update counter (the scalar step) and the episode counter.
A resumed run reopens the store of the same run name; counting from 0 again
overwrote its episode rows, interleaved its update scalars with the earlier
ones, and left a reader that polls for steps past the last one it saw with
nothing new until the counters passed the old maximum. The counters now
continue from the rows the store already holds.
"""
from __future__ import annotations

from opencda_marl._kernel.policy.manager import MARLManager
from opencda_marl.telemetry import MetricsStore


def _cfg(tmp_path):
    return {
        "algorithm": "random_speed",
        "action_space": {"mode": "speed_only"},
        "reward_profile": "competitive_intersection_v2",
        "state_dim": 3, "action_dim": 1,
        "random_speed": {"features": {"rel_x": 1, "rel_y": 1, "speed": 1},
                         "min_action": 5.0, "max_action": 9.0, "seed": 3},
        "outputs": {"metrics_store": {"enabled": True, "log_dir": str(tmp_path),
                                      "run_name": "resumed", "log_frequency": 1}},
    }


def _episode(reward):
    return {"total_reward": reward, "step_length": 10, "success_rate": 0.0,
            "collision_rate": 0.0}


def test_a_fresh_store_starts_at_zero(tmp_path):
    manager = MARLManager(_cfg(tmp_path), "random_speed")
    try:
        assert manager._external_store is not None
        assert manager._update_calls == 0 and manager._episodes_logged == 0
    finally:
        manager.close()


def test_a_reopened_store_continues_its_steps_and_episodes(tmp_path):
    first = MARLManager(_cfg(tmp_path), "random_speed")
    for _ in range(3):
        first.record_update_metrics(first.algorithm.update())
    first.reset_episode(episode_metrics=_episode(1.0))
    first.close()

    second = MARLManager(_cfg(tmp_path), "random_speed")
    assert second._update_calls == 3 and second._episodes_logged == 1
    second.record_update_metrics(second.algorithm.update())
    second.reset_episode(episode_metrics=_episode(2.0))
    second.close()

    reader = MetricsStore(tmp_path / "random_speed" / "resumed", read_only=True)
    try:
        episodes = reader.episodes()
        assert [e["episode"] for e in episodes] == [0, 1]
        assert [e["reward"] for e in episodes] == [1.0, 2.0]
        steps = [row[0] for row in reader.scalars("random_speed/updates")]
        assert steps == [1, 2, 3, 4]
    finally:
        reader.close()
