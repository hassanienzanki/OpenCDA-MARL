"""The coordinator hands the run's metrics store to the evaluation manager.

``EvaluationManager.attach_run_store`` keeps one run as one monitor entry,
but nothing called it: every learner run still produced a second monitor
run under ``metrics/evaluation/``, labelled with the agent type (``marl``)
and ``eval`` even while the learner trained. ``initialize()`` now attaches
the learner's (or the external policy's) store and always passes the
registered learner key and the run's ``training_mode``, so a learner whose
own store is disabled still gets a separate run labelled with its key.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencda_marl._kernel.lifecycle.coordinator import MARLCoordinator
from opencda_marl.evaluation.manager import EvaluationManager
from opencda_marl.telemetry import MetricsStore, discover_runs


def _evaluation(tmp_path):
    cfg = {"mode": "summary", "root_dir": str(tmp_path / "evaluation"),
           "metrics_root": str(tmp_path / "metrics")}
    return EvaluationManager(config=cfg, scenario_name="intersection", agent_name="marl")


def _one_episode(evaluation):
    evaluation.update_step({"step": 400, "episode": 0, "fixed_dt": 0.05, "success": 3,
                            "collision": 1, "active_agents": 0, "pending_spawns": 0}, {"v0": 1.0})
    evaluation.end_episode(final=True)


def _coordinator(tmp_path, manager, *, on_env=False, training_mode=True):
    coord = object.__new__(MARLCoordinator)
    coord.config = {"MARL": {"training": {"training_mode": training_mode}}}
    coord.evaluation_manager = _evaluation(tmp_path)
    if on_env:              # SUMO: the env builds and holds the manager
        coord.marl_env = SimpleNamespace(marl_manager=manager)
    else:                   # CARLA: the coordinator holds it
        coord.marl_manager = manager
        coord.marl_env = None
    return coord


@pytest.mark.parametrize("on_env", [False, True], ids=["carla", "sumo"])
def test_a_learner_run_is_one_monitor_run(tmp_path, on_env):
    learner_dir = tmp_path / "metrics" / "masac" / "r"
    store = MetricsStore(learner_dir, meta={"algorithm": "masac", "mode": "train"})
    coord = _coordinator(tmp_path, SimpleNamespace(run_store=store, algorithm_name="masac"),
                         on_env=on_env)

    coord._attach_evaluation_to_run_store()
    assert coord.evaluation_manager.store_dir == learner_dir
    _one_episode(coord.evaluation_manager)
    store.close()

    assert not (tmp_path / "metrics" / "evaluation").exists()
    assert [r.run_id for r in discover_runs(tmp_path / "metrics")] == ["masac/r"]


@pytest.mark.parametrize("training_mode, mode", [(True, "train"), (False, "eval")])
def test_a_learner_without_a_store_labels_the_separate_run(tmp_path, training_mode, mode):
    coord = _coordinator(tmp_path, SimpleNamespace(run_store=None, algorithm_name="masac"),
                         training_mode=training_mode)

    coord._attach_evaluation_to_run_store()
    _one_episode(coord.evaluation_manager)

    (run,) = discover_runs(tmp_path / "metrics")
    assert run.run_id.startswith("evaluation/")
    assert (run.meta["algorithm"], run.meta["mode"]) == ("masac", mode)


def test_a_baseline_run_keeps_the_agent_label(tmp_path):
    coord = _coordinator(tmp_path, SimpleNamespace(run_store=None, algorithm_name=None))

    coord._attach_evaluation_to_run_store()
    _one_episode(coord.evaluation_manager)

    (run,) = discover_runs(tmp_path / "metrics")
    assert (run.meta["algorithm"], run.meta["mode"]) == ("marl", "eval")


def test_nothing_to_attach_is_not_an_error(tmp_path):
    coord = object.__new__(MARLCoordinator)
    coord.config = {}
    coord.marl_env = None
    coord._attach_evaluation_to_run_store()                 # no evaluation manager
    coord.evaluation_manager = _evaluation(tmp_path)
    coord._attach_evaluation_to_run_store()                 # no manager either
    _one_episode(coord.evaluation_manager)
    (run,) = discover_runs(tmp_path / "metrics")
    assert run.meta["algorithm"] == "marl"


def test_initialize_attaches_the_learner_store(tmp_path, monkeypatch):
    """The SUMO branch end to end: the method above must also be called."""
    import opencda_marl.plugins as plugins
    import opencda_marl.plugins.loader as loader
    from opencda_marl._kernel.envs import sumo_marl_env

    learner_dir = tmp_path / "metrics" / "masac" / "r"
    store = MetricsStore(learner_dir)
    manager = SimpleNamespace(run_store=store, algorithm_name="masac")
    monkeypatch.setattr(sumo_marl_env, "SumoMARLEnv",
                        lambda config: SimpleNamespace(marl_manager=manager))
    monkeypatch.setattr(loader, "load_plugins", lambda extra=None, **_: {})
    monkeypatch.setattr(plugins, "freeze", lambda: None)
    coord = MARLCoordinator({
        "runtime": {"backend": "sumo"},
        "scenario": {"type": "intersection"},
        "agents": {"agent_type": "marl"},
        "outputs": {"root_dir": str(tmp_path), "evaluation": {"mode": "summary", "dir": "evaluation"},
                    "monitor": False, "traffic_view": False},
        "MARL": {"training": {"training_mode": True}},
    })
    try:
        coord.initialize()
        assert coord.evaluation_manager.store_dir == learner_dir
    finally:
        store.close()
