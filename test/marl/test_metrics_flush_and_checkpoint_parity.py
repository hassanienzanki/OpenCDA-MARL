"""Run-end flush and CARLA/SUMO checkpoint parity.

Two boundary defects this pins down:

* ``Metrics.export_history`` only ran every ``export_interval`` episodes, so
  the trailing episodes of any run whose length is not a multiple of it (and
  every interrupted run) were lost — while the launcher's resume, which counts
  exported pkls, treated them as never run and replayed them on top of a
  checkpoint that had already trained on them.
* SUMO gated the *save* on ``save_freq`` while CARLA gates only the cleanup,
  so the last ``save_freq - 1`` episodes of a SUMO run were never persisted
  and ``best_checkpoint.pth`` could only ever be one of the sampled episodes.
"""
import inspect
from types import SimpleNamespace

import pytest

from opencda_marl.evaluation.training_metrics import TrainingMetrics as Metrics
from opencda_marl._kernel.lifecycle.coordinator import MARLCoordinator


def _episode(metrics, reward):
    metrics.update_step({"v0": reward}, {})
    metrics.finish_episode({"step": 1, "collision": 0, "success": 0,
                            "active_agents": 0, "fixed_dt": 0.05})


def test_close_flushes_the_trailing_episodes(tmp_path):
    metrics = Metrics(export_interval=5, export_dir=str(tmp_path))
    for reward in (1.0, 2.0, 3.0):                 # fewer than one interval
        _episode(metrics, reward)
    assert list(tmp_path.glob("*.pkl")) == [], "nothing exported yet"

    coordinator = object.__new__(MARLCoordinator)
    coordinator.trace_writer = None
    coordinator.marl_env = SimpleNamespace(metrics=metrics)   # no close()
    coordinator.marl_manager = None
    coordinator.scenario_manager = None
    coordinator.close()

    exported = list(tmp_path.glob("*.pkl"))
    assert len(exported) == 1, exported
    import pickle
    history = pickle.loads(exported[0].read_bytes())
    assert len(history["rewards"]) == 3


def test_close_is_a_no_op_when_archiving_is_disabled(tmp_path):
    metrics = Metrics(export_interval=0, export_dir=str(tmp_path))
    _episode(metrics, 1.0)
    coordinator = object.__new__(MARLCoordinator)
    coordinator.trace_writer = None
    coordinator.marl_env = SimpleNamespace(metrics=metrics)
    coordinator.marl_manager = None
    coordinator.scenario_manager = None
    coordinator.close()
    assert list(tmp_path.glob("*.pkl")) == []


def test_double_flush_does_not_duplicate_episodes(tmp_path):
    """export_history clears its buffer, so the close-time flush after a
    regular interval export must write nothing."""
    metrics = Metrics(export_interval=1, export_dir=str(tmp_path))
    _episode(metrics, 1.0)
    before = sorted(p.name for p in tmp_path.glob("*.pkl"))
    metrics.export_history()
    assert sorted(p.name for p in tmp_path.glob("*.pkl")) == before


def test_sumo_saves_every_episode_and_prunes_periodically():
    """Parity with CARLA: save every episode, cleanup on save_freq."""
    from opencda_marl.envs import sumo_marl_env as mod

    reset_src = inspect.getsource(mod.SumoMARLEnv.reset)
    save_src = inspect.getsource(mod.SumoMARLEnv._save_checkpoint)
    assert "self._save_checkpoint(episode_metrics)" in reset_src
    assert "current_episode % save_freq" not in reset_src, (
        "save must not be gated on save_freq -- the tail of the run is lost")
    assert "current_episode % save_freq" in save_src, (
        "cleanup_old_checkpoints stays periodic")
    assert "cleanup_old_checkpoints" in save_src
