"""Where the evaluation manager's metrics-store mirror goes.

One run is one monitor entry. With the run's own store attached (the
learner's), the ``Eval/*`` scalars, one ``evaluation_episode`` event per
episode and the final summary go into it; its episode rows stay the
learner's, and the store stays open for its owner. Only a run without a
store of its own gets a separate evaluation run, labelled with a mode (the
run list showed ``?`` without it: ``train`` while a named learner trains,
``eval`` otherwise), the algorithm or agent, the scenario and the agent.
The console summary is ASCII ('+/-', not a plus-minus sign a non-UTF-8
Windows console garbles).
"""
from __future__ import annotations

import pytest

from opencda_marl.evaluation import manager as eval_mod
from opencda_marl.evaluation.manager import EvaluationManager
from opencda_marl.telemetry import MetricsStore, discover_runs


def _evaluation(tmp_path, **kw):
    cfg = {"mode": "summary", "root_dir": str(tmp_path / "evaluation"),
           "metrics_root": str(tmp_path / "metrics")}
    return EvaluationManager(config=cfg, scenario_name="intersection", agent_name="marl", **kw)


def _metrics(episode, success=3, collision=1, active=1):
    return {"step": 400, "episode": episode, "fixed_dt": 0.05, "success": success,
            "collision": collision, "active_agents": active, "pending_spawns": 0}


def _run_two_episodes(em):
    for episode in range(2):
        em.update_step(_metrics(episode), {"v0": 1.0})
        em.end_episode(final=(episode == 1))


def _read(path):
    reader = MetricsStore(path, read_only=True)
    try:
        return (reader.meta(), {t["tag"] for t in reader.tags()},
                reader.episodes(), reader.events())
    finally:
        reader.close()


def test_a_run_without_a_store_gets_a_labelled_evaluation_run(tmp_path):
    em = _evaluation(tmp_path)
    _run_two_episodes(em)
    (run,) = discover_runs(tmp_path / "metrics")
    assert run.run_id.startswith("evaluation/")
    assert run.meta["mode"] == "eval"
    assert run.meta["algorithm"] == "marl", "no learner: labelled with the agent type"
    assert (run.meta["scenario"], run.meta["agent"], run.meta["kind"]) == ("intersection", "marl", "evaluation")
    meta, tags, episodes, _events = _read(run.path)
    assert [e["episode"] for e in episodes] == [0, 1]
    assert "Eval/success_rate" in tags and "final_summary" in meta


def test_a_separate_run_carries_the_algorithm_when_known(tmp_path):
    em = _evaluation(tmp_path, algorithm_name="rule_based")
    _run_two_episodes(em)
    (run,) = discover_runs(tmp_path / "metrics")
    assert run.meta["algorithm"] == "rule_based"


@pytest.mark.parametrize("algorithm, training, mode", [
    ("masac", True, "train"),      # a learner whose own store is disabled still trains
    ("masac", False, "eval"),      # a frozen learner
    (None, True, "eval"),          # no learner: nothing trains, whatever training_mode says
])
def test_a_separate_run_carries_the_run_mode(tmp_path, algorithm, training, mode):
    em = _evaluation(tmp_path)
    assert not em.attach_run_store(None, algorithm_name=algorithm, training=training)
    _run_two_episodes(em)
    (run,) = discover_runs(tmp_path / "metrics")
    assert run.meta["mode"] == mode
    assert run.meta["algorithm"] == (algorithm or "marl")


def test_an_attached_run_store_receives_the_evaluation(tmp_path):
    learner_dir = tmp_path / "metrics" / "masac" / "cell"
    learner = MetricsStore(learner_dir, meta={"algorithm": "masac", "mode": "eval"})
    learner.log_episode(0, {"reward": 5.0, "success_rate": 60.0})
    learner.log_episode(1, {"reward": 6.0, "success_rate": 70.0})
    learner.flush()

    em = _evaluation(tmp_path)
    assert em.attach_run_store(learner, algorithm_name="masac")
    assert em.store_dir == learner_dir, "the traffic view lands in the run the monitor shows"
    _run_two_episodes(em)

    assert not learner._closed, "the borrowed store stays open for its owner"
    learner.log_scalar("Loss/critic", 1.0, step=1)          # its owner keeps writing
    learner.close()

    runs = discover_runs(tmp_path / "metrics")
    assert [r.run_id for r in runs] == ["masac/cell"], "one evaluation is one monitor run"
    meta, tags, episodes, events = _read(learner_dir)
    assert {"Eval/success_rate", "Eval/timeout_rate", "Eval/throughput"} <= tags
    assert [e["reward"] for e in episodes] == [5.0, 6.0], "the learner's episode rows are untouched"
    assert [e["payload"]["episode"] for e in events if e["kind"] == "evaluation_episode"] == [0, 1]
    assert meta["final_summary"]["total_episodes"] == 2
    assert meta["algorithm"] == "masac" and meta["mode"] == "eval"
    assert not (tmp_path / "metrics" / "evaluation").exists()


def test_a_closed_borrowed_store_stops_the_mirror_not_the_run(tmp_path):
    learner = MetricsStore(tmp_path / "metrics" / "masac" / "cell")
    em = _evaluation(tmp_path)
    em.attach_run_store(learner)
    learner.close()                                     # its owner closed it early
    _run_two_episodes(em)                               # no exception
    assert len(em.episode_history) == 2
    assert (em.data_dir / "final_summary.json").is_file()


def test_attaching_after_a_separate_run_started_keeps_one_run(tmp_path):
    em = _evaluation(tmp_path)
    em.update_step(_metrics(0), {"v0": 1.0})
    em.end_episode()                                    # the separate run exists now
    learner = MetricsStore(tmp_path / "metrics" / "masac" / "cell")
    try:
        assert not em.attach_run_store(learner)
    finally:
        learner.close()
    em.update_step(_metrics(1), {"v0": 1.0})
    em.end_episode(final=True)
    evaluation_runs = [r for r in discover_runs(tmp_path / "metrics") if r.run_id.startswith("evaluation/")]
    (run,) = evaluation_runs
    assert run.counts["episodes"] == 2


def test_disabled_evaluation_attaches_nothing(tmp_path):
    cfg = {"mode": "disabled", "metrics_root": str(tmp_path / "metrics")}
    em = EvaluationManager(config=cfg, scenario_name="s", agent_name="marl")
    learner = MetricsStore(tmp_path / "metrics" / "masac" / "cell")
    try:
        assert not em.attach_run_store(learner)
    finally:
        learner.close()


def test_the_console_summary_is_ascii(tmp_path):
    messages = []
    sink = eval_mod.logger.add(lambda m: messages.append(m.record["message"]), level="INFO")
    try:
        em = _evaluation(tmp_path)
        _run_two_episodes(em)
    finally:
        eval_mod.logger.remove(sink)
    summary = [m for m in messages if "Rate:" in m or "Throughput:" in m]
    assert len(summary) == 4
    assert all("+/-" in m for m in summary)
    for message in messages:
        if str(tmp_path) not in message:               # a path may be anything
            assert message.isascii(), message
