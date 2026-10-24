"""What a learner's metrics store is called and what its episode rows carry.

* The run directory and the ``algorithm`` label are the registered key:
  masac and pcmasac share one learner class (``SACAlgorithm``), and a name
  derived from the class stored both under ``sac``. A learner built without
  the manager keeps the class-derived name, so stores of older runs and
  hand-built learners read as before.
* Plain MASAC does not log the ``SAC/pref_*`` loss terms: they exist only
  under an internal preference mode (shift) or ``internalpref_sep``
  (separation), and are constant zeros everywhere else.
* Built-in learners log the same outcome columns as the external-policy
  path (``timeout_rate``, ``successes``, ``collisions``, ``timeouts``), so
  the monitor's episode table does not depend on the learner's base class.
"""
from __future__ import annotations

import pytest

from opencda_marl._kernel.policy.manager import MARLManager
from opencda_marl.telemetry import MetricsStore

_FEATURES = {"rel_x": 1, "rel_y": 1, "speed": 1, "heading_angle": 1,
             "dist_to_intersection": 1, "dist_to_front_vehicle": 1,
             "lane_position": 1, "waypoint_buffer": 1}


def _sac_section(**overrides):
    section = {"features": dict(_FEATURES), "warmup_steps": 0, "batch_size": 8,
               "memory_size": 64, "lstm_hidden_size": 4, "actor_hidden_dims": [8, 8],
               "critic_hidden_dims": [8, 8], "tensorboard": {"enabled": False}}
    section.update(overrides)
    return section


def _manager(tmp_path, algorithm, run_name="r", **section):
    cfg = {"algorithm": algorithm, "action_space": {"mode": "speed_only"},
           "reward_profile": "simple_progress_v1", "state_dim": 8, "action_dim": 1,
           algorithm: _sac_section(**section), "training": {"training_mode": True},
           "outputs": {"tensorboard": {"enabled": False},
                       "metrics_store": {"enabled": True, "log_dir": str(tmp_path),
                                         "run_name": run_name, "log_frequency": 1}}}
    return MARLManager(cfg, algorithm)


def _read(path):
    reader = MetricsStore(path, read_only=True)
    try:
        return reader.meta(), {t["tag"] for t in reader.tags()}, reader.episodes()
    finally:
        reader.close()


@pytest.mark.parametrize("algorithm", ["masac", "pcmasac"])
def test_the_store_is_named_by_the_registered_key(tmp_path, algorithm):
    manager = _manager(tmp_path, algorithm)
    try:
        store = manager.run_store
        assert store is manager.algorithm.metrics_store
        assert store.path.parent == tmp_path / algorithm / "r"
    finally:
        manager.close()
    meta, _tags, _eps = _read(tmp_path / algorithm / "r")
    assert meta["algorithm"] == algorithm
    assert not (tmp_path / "sac").exists(), "the learner class no longer names the run"


def test_a_learner_built_without_the_manager_keeps_the_class_name(tmp_path):
    from opencda_marl.builtin.algorithms.sac import SACAlgorithm

    cfg = _sac_section(metrics_store={"enabled": True, "log_dir": str(tmp_path),
                                      "run_name": "r", "log_frequency": 1})
    algo = SACAlgorithm(cfg, 5, 1)
    algo.close()
    meta, _tags, _eps = _read(tmp_path / "sac" / "r")
    assert meta["algorithm"] == "sac"


def _logged_pref_tags(tmp_path, pref_mode):
    from opencda_marl.builtin.algorithms.sac import SACAlgorithm

    cfg = _sac_section(pref_mode=pref_mode, pref_dim=2,
                       metrics_store={"enabled": True, "log_dir": str(tmp_path / pref_mode),
                                      "run_name": "r", "log_frequency": 1})
    algo = SACAlgorithm(cfg, 7, 1)
    algo.training_step = 0
    algo._log_update_metrics()
    algo.close()
    _meta, tags, _eps = _read(tmp_path / pref_mode / "sac" / "r")
    assert "SAC/alpha" in tags, "the update scalars were logged at all"
    return {t for t in tags if t.startswith("SAC/pref_")}


def test_plain_sac_logs_no_preference_terms(tmp_path):
    assert _logged_pref_tags(tmp_path, "direct") == set()


def test_internal_preference_modes_log_their_own_terms(tmp_path):
    shift = {"SAC/pref_shift_l2", "SAC/pref_shift_norm", "SAC/pref_shift_abs_max"}
    separation = {"SAC/pref_sep_loss", "SAC/pref_sep_active_pairs"}
    assert _logged_pref_tags(tmp_path, "internalpref") == shift
    assert _logged_pref_tags(tmp_path, "internalpref_sep") == shift | separation


def _episode_metrics():
    return {"total_reward": 3.0, "episode_length": 100, "success_rate": 50.0,
            "collision_rate": 25.0, "timeout_rate": 25.0, "successes": 2,
            "collisions": 1, "timeouts": 1, "near_miss_count": 4,
            "ttc_violation_rate": 0.1, "throughput": 720.0}


def test_builtin_learners_log_the_outcome_columns(tmp_path):
    manager = _manager(tmp_path, "masac")
    try:
        manager.reset_episode(episode_metrics=_episode_metrics())
    finally:
        manager.close()
    _meta, tags, episodes = _read(tmp_path / "masac" / "r")
    (row,) = episodes
    assert (row["successes"], row["collisions"], row["timeouts"]) == (2.0, 1.0, 1.0)
    assert row["timeout_rate"] == pytest.approx(25.0)
    assert row["success_rate"] == pytest.approx(50.0) and row["collision_rate"] == pytest.approx(25.0)
    assert row["traffic/throughput"] == pytest.approx(720.0)
    assert {"Episode/timeout_rate", "Episode/timeouts"} <= tags


def test_both_paths_log_the_same_outcome_columns(tmp_path):
    """The external-policy path (manager-owned store) carries the same keys."""
    cfg = {"algorithm": "random_speed", "action_space": {"mode": "speed_only"},
           "reward_profile": "competitive_intersection_v2", "state_dim": 3, "action_dim": 1,
           "random_speed": {"features": {"rel_x": 1, "rel_y": 1, "speed": 1},
                            "min_action": 5.0, "max_action": 9.0, "seed": 3},
           "outputs": {"metrics_store": {"enabled": True, "log_dir": str(tmp_path),
                                         "run_name": "ext", "log_frequency": 1}}}
    external = MARLManager(cfg, "random_speed")
    try:
        assert external.run_store is external._external_store
        external.reset_episode(episode_metrics=_episode_metrics())
    finally:
        external.close()
    _meta, _tags, (ext_row,) = _read(tmp_path / "random_speed" / "ext")

    builtin = _manager(tmp_path, "masac", run_name="int")
    try:
        builtin.reset_episode(episode_metrics=_episode_metrics())
    finally:
        builtin.close()
    _meta, _tags, (int_row,) = _read(tmp_path / "masac" / "int")
    for key in MARLManager.OUTCOME_SUMMARY_KEYS:
        assert key in ext_row and key in int_row, key
        assert ext_row[key] == pytest.approx(int_row[key])
