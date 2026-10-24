"""The evaluation manager closes an episode when the coordinator says so.

It used to infer the end from ``step == max_steps - 1``: one step early on
CARLA (the step counter is incremented before the metrics snapshot), never on
SUMO (whose metrics carried no ``step``/``max_steps``), and never for an
episode the environment ended early. ``coordinator.reset_episode`` now calls
``EvaluationManager.end_episode`` explicitly on both backends.
"""
from types import SimpleNamespace

import pytest

from opencda_marl.evaluation.manager import EvaluationManager
from opencda_marl._kernel.lifecycle.coordinator import MARLCoordinator


def _manager(tmp_path, mode="summary"):
    # ``root_dir`` is the key the manager reads; the old ``output_dir`` was
    # ignored, so every suite run wrote CSVs into the repository root.
    cfg = {"mode": mode, "root_dir": str(tmp_path), "plot_interval": 100}
    return EvaluationManager(config=cfg, scenario_name="t", agent_name="marl")


def _metrics(step, max_steps=10, episode=0, max_episodes=2, **extra):
    m = {"step": step, "max_steps": max_steps, "episode": episode,
         "max_episodes": max_episodes, "fixed_dt": 0.05, "success": 1,
         "collision": 0, "active_agents": 2, "pending_spawns": 0}
    m.update(extra)
    return m


def test_update_step_never_closes_an_episode_by_itself(tmp_path):
    em = _manager(tmp_path)
    if not em.is_enabled():
        pytest.skip("evaluation manager disabled in this configuration")
    for step in range(1, 11):                     # includes step == max_steps - 1 and == max_steps
        em.update_step(_metrics(step), {"v0": 1.0})
    assert em.episode_history == [], "no summary until end_episode()"


def test_end_episode_closes_exactly_once_and_final_writes_summary(tmp_path, monkeypatch):
    em = _manager(tmp_path)
    if not em.is_enabled():
        pytest.skip("evaluation manager disabled in this configuration")
    finals = []
    monkeypatch.setattr(em, "summary_evaluation", lambda: finals.append(True))
    for step in range(1, 11):
        em.update_step(_metrics(step), {"v0": 1.0})
    em.end_episode()                               # from the last snapshot
    assert len(em.episode_history) == 1
    assert finals == []
    for step in range(1, 5):                       # a shorter (early-ended) episode
        em.update_step(_metrics(step, episode=1), {"v0": 1.0})
    em.end_episode(final=True)
    assert len(em.episode_history) == 2
    assert finals == [True]


def test_end_episode_without_any_step_is_a_no_op(tmp_path):
    em = _manager(tmp_path)
    if not em.is_enabled():
        pytest.skip("evaluation manager disabled in this configuration")
    em.end_episode()
    assert em.episode_history == []


def _coordinator(backend, env, states):
    coord = object.__new__(MARLCoordinator)
    coord.config = {"runtime": {"backend": backend}}
    coord.states = states
    coord.marl_env = env
    coord.scenario_manager = None
    return coord


def test_sumo_metrics_carry_progress_keys():
    """SUMO get_metrics now exposes the same progress keys CARLA does.

    ``active_vehicles`` records every vehicle observed this episode and is
    never pruned, so finished vehicles appear in it as well: the active
    count is what has neither arrived nor collided, plus the vehicles still
    in SUMO's insertion queue (spawned, never entered: unfinished routes).
    ``pending_spawns`` is only the spawn events not yet due.
    """
    env = SimpleNamespace(
        current_step=7, current_episode=1,
        episode=SimpleNamespace(departed_vehicles={"a"}, collided_vehicles={"d"},
                                active_vehicles={"a": 1, "b": 1, "c": 1, "d": 1}),
        get_episode_metrics=lambda: {"total_reward": 3.0, "fixed_dt": 0.05},
        pending_spawn_count=lambda: 4,
        queued_insertions=lambda: ["q"],
    )
    coord = _coordinator("sumo", env, {"max_steps": 400, "max_episodes": 3, "fixed_dt": 0.05})
    m = coord.get_metrics()
    assert (m["step"], m["episode"], m["max_steps"], m["max_episodes"]) == (7, 1, 400, 3)
    assert (m["success"], m["collision"], m["active_agents"]) == (1, 1, 3)   # b, c + queued q
    assert m["pending_spawns"] == 4
    assert m["total_reward"] == 3.0 and m["fixed_dt"] == 0.05


def test_sumo_evaluation_reads_an_all_success_episode_as_all_success(tmp_path):
    """20 vehicles, all arrived: 100 % success, 0 % timeout -- not 50/50."""
    vehicles = [f"v{i}" for i in range(20)]
    env = SimpleNamespace(
        current_step=900, current_episode=0,
        episode=SimpleNamespace(departed_vehicles=set(vehicles), collided_vehicles=set(),
                                active_vehicles={v: {} for v in vehicles}),
        get_episode_metrics=lambda: {"fixed_dt": 0.05},
        pending_spawn_count=lambda: 0,
    )
    coord = _coordinator("sumo", env, {"max_steps": 2400, "max_episodes": 1, "fixed_dt": 0.05})
    em = _manager(tmp_path)
    if not em.is_enabled():
        pytest.skip("evaluation manager disabled in this configuration")

    em.update_step(coord.get_metrics(), {})
    em.end_episode()

    (stat,) = em.episode_history
    assert (stat["success"], stat["active_agents"], stat["total_vehicles"]) == (20, 0, 20)
    assert stat["success_rate"] == pytest.approx(100.0)
    assert stat["timeout_rate"] == pytest.approx(0.0)


def test_sumo_evaluation_counts_a_vehicle_left_in_the_insertion_queue(tmp_path):
    """19 arrived, 1 still queued at the end: 95 % success, 5 % timeout.

    The queued vehicle is an unfinished route, as the training summary and
    CARLA's never-entered accounting count it; it must neither drop out of
    the total (100 % success of 19) nor let the episode read as complete.
    """
    vehicles = [f"v{i}" for i in range(19)]
    env = SimpleNamespace(
        current_step=2399, current_episode=0,
        episode=SimpleNamespace(departed_vehicles=set(vehicles), collided_vehicles=set(),
                                active_vehicles={v: {} for v in vehicles}),
        get_episode_metrics=lambda: {"fixed_dt": 0.05},
        pending_spawn_count=lambda: 0,
        queued_insertions=lambda: ["queued"],
    )
    coord = _coordinator("sumo", env, {"max_steps": 2400, "max_episodes": 1, "fixed_dt": 0.05})
    em = _manager(tmp_path)
    if not em.is_enabled():
        pytest.skip("evaluation manager disabled in this configuration")

    metrics = coord.get_metrics()
    assert (metrics["success"], metrics["active_agents"], metrics["pending_spawns"]) == (19, 1, 0)
    em.update_step(metrics, {})
    em._step_statistics(metrics, {})
    assert em.last_completion_step is None, "an episode with a queued vehicle is not complete"
    em.end_episode()

    (stat,) = em.episode_history
    assert (stat["success"], stat["active_agents"], stat["total_vehicles"]) == (19, 1, 20)
    assert stat["success_rate"] == pytest.approx(95.0)
    assert stat["timeout_rate"] == pytest.approx(5.0)


def test_sumo_queued_insertions_degrade_to_none_when_unavailable():
    def broken():
        raise RuntimeError("connection closed")

    episode = SimpleNamespace(departed_vehicles=set(), collided_vehicles=set(),
                              active_vehicles={"a": 1})
    for env in (
        SimpleNamespace(current_step=1, current_episode=0, episode=episode,
                        get_episode_metrics=lambda: {}),                   # no queue reader
        SimpleNamespace(current_step=1, current_episode=0, episode=episode,
                        get_episode_metrics=lambda: {}, queued_insertions=broken),
    ):
        coord = _coordinator("sumo", env, {"max_steps": 10, "max_episodes": 1, "fixed_dt": 0.05})
        assert coord.get_metrics()["active_agents"] == 1


def test_sumo_pending_spawns_degrade_to_zero_when_unavailable():
    def broken():
        raise RuntimeError("connection closed")

    episode = SimpleNamespace(departed_vehicles=set(), collided_vehicles=set(), active_vehicles={})
    for env in (
        SimpleNamespace(current_step=1, current_episode=0, episode=episode,
                        get_episode_metrics=lambda: {}),                   # no counter at all
        SimpleNamespace(current_step=1, current_episode=0, episode=episode,
                        get_episode_metrics=lambda: {}, pending_spawn_count=broken),
    ):
        coord = _coordinator("sumo", env, {"max_steps": 10, "max_episodes": 1, "fixed_dt": 0.05})
        assert coord.get_metrics()["pending_spawns"] == 0


def _sumo_env_with_one_queued_vehicle(monkeypatch):
    try:
        import opencda_marl._kernel.envs.sumo_marl_env as mod
        from opencda_marl._kernel.envs.episode_state import SumoEpisodeState
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"SUMO runtime unavailable: {exc}")
    env = object.__new__(mod.SumoMARLEnv)
    env.episode = SumoEpisodeState()
    env.episode.departed_vehicles.add("arrived")
    env.vehicle_spawner = SimpleNamespace(
        spawned_vehicles={"queued": object(), "driving": object(), "arrived": object()})
    env.use_traffic_manager = True
    env.traffic_manager = SimpleNamespace(total_events=5)
    monkeypatch.setattr(mod.traci.vehicle, "getIDList", lambda: ("driving",))
    monkeypatch.delattr(mod.traci.simulation, "getPendingVehicles", raising=False)
    return env


def test_the_sumo_env_lists_vehicles_waiting_for_insertion(monkeypatch):
    env = _sumo_env_with_one_queued_vehicle(monkeypatch)
    assert env.queued_insertions() == ["queued"]       # spawned, not in the network, not finished


def test_the_sumo_env_counts_only_not_yet_due_spawns_as_pending(monkeypatch):
    env = _sumo_env_with_one_queued_vehicle(monkeypatch)
    assert env.pending_spawn_count() == 5              # the queued vehicle is not pending
    env.use_traffic_manager = False
    assert env.pending_spawn_count() == 0


def test_the_coordinator_reads_the_sumo_env_queue_as_active(monkeypatch):
    """The coordinator and the real SUMO env agree on the method names."""
    env = _sumo_env_with_one_queued_vehicle(monkeypatch)
    env.episode.active_vehicles["driving"] = {}
    env.episode.active_vehicles["arrived"] = {}
    env.current_step, env.current_episode = 3, 0
    env.get_episode_metrics = lambda: {}
    coord = _coordinator("sumo", env, {"max_steps": 10, "max_episodes": 1, "fixed_dt": 0.05})
    m = coord.get_metrics()
    assert (m["success"], m["active_agents"], m["pending_spawns"]) == (1, 2, 5)   # driving + queued


@pytest.mark.parametrize("backend", ["carla", "sumo"])
def test_coordinator_closes_the_evaluation_episode_on_reset(backend):
    calls = []
    manager = SimpleNamespace(
        is_enabled=lambda: True,
        update_step=lambda m, r: None,
        end_episode=lambda m, r, final=False: calls.append((m["episode"], final)),
    )
    env = SimpleNamespace(
        current_step=10, current_episode=1,
        episode=SimpleNamespace(departed_vehicles=set(), collided_vehicles=set(), active_vehicles={}),
        get_episode_metrics=lambda: {},
        get_current_step_rewards=lambda: {},
    )
    states = {"max_steps": 10, "max_episodes": 2, "episode": 1, "fixed_dt": 0.05,
              "success": 0, "collision": 0, "active_agents": 0}
    coord = _coordinator(backend, env, states)
    coord.evaluation_manager = manager
    coord._end_evaluation_episode()
    assert calls == [(1, True)], calls           # last episode -> final summary
