"""SumoMARLEnv.reset(): every episode that ran is finished exactly once.

reset() is only called at episode end, so the first episode of a run ends
with current_episode == 0. The finish path (metrics.finish_episode, the
learner's reset_episode hook, checkpointing) must fire for it; the pre-run
case (no step advanced) must not finish anything.
"""
from types import SimpleNamespace

import pytest


def _env(sumo_env_mod, SumoMARLEnv, *, current_step, current_episode):
    from opencda_marl._kernel.envs.episode_state import SumoEpisodeState

    env = object.__new__(SumoMARLEnv)
    env.current_step = current_step
    env.current_episode = current_episode
    env.step_length = 0.05
    env.episode = SumoEpisodeState()
    env.previous_observations = {}
    env.episode_events = []
    env.current_step_rewards = {}
    env.current_step_reward_components = {}
    env.current_step_reward_vectors = {}
    env.current_step_objective_components = {}
    env.current_step_objective_vectors = {}
    env.current_step_agent_preferences = {}
    env.competition_agent_metadata = {}
    env.competition_terminal_records = {}
    env._sumo_intent_agent_manager = SimpleNamespace(_vehicle_adapters=["stale"])
    env.ttc_metrics = SimpleNamespace(near_miss_count=0, check_count=0, violation_count=0,
                                      reset=lambda: None)
    env.vehicle_spawner = None
    env.use_traffic_manager = False
    env.traffic_manager = None
    env.is_training_mode = True
    env.training_config = {"save_checkpoints": False}
    env.checkpoint_manager = None
    env.last_episode_metrics = None
    env._build_timeout_competition_records = lambda ids, step: []
    env._notify_route_intent_terminal = lambda vid: None

    calls = []
    env.metrics = SimpleNamespace(
        finish_episode=lambda states: calls.append(("finish", dict(states))) or {"total_reward": 1.0},
    )
    env.marl_manager = SimpleNamespace(
        get_intent_diagnostics=lambda: {},
        reset_episode=lambda episode_metrics=None: calls.append(("reset_episode", episode_metrics)),
    )
    return env, calls


@pytest.fixture
def sumo_env():
    try:
        import opencda_marl._kernel.envs.sumo_marl_env as sumo_env_mod
        from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"SUMO runtime unavailable: {exc}")
    return sumo_env_mod, SumoMARLEnv


def test_first_episode_is_finished_even_though_its_number_is_zero(sumo_env, monkeypatch):
    sumo_env_mod, SumoMARLEnv = sumo_env
    env, calls = _env(sumo_env_mod, SumoMARLEnv, current_step=2075, current_episode=0)
    monkeypatch.setattr(sumo_env_mod.traci.vehicle, "getIDList", lambda: ())
    monkeypatch.setattr(sumo_env_mod.traci, "simulationStep", lambda: None)

    SumoMARLEnv.reset(env, prepare_next_episode=True)

    kinds = [c[0] for c in calls]
    assert kinds == ["finish", "reset_episode"], kinds
    assert calls[0][1]["step"] == 2075                 # the episode that ran
    assert env.last_episode_metrics == {"total_reward": 1.0}
    assert env.current_step == 0 and env.current_episode == 1
    assert env._sumo_intent_agent_manager._vehicle_adapters == []


def test_vehicles_still_queued_for_insertion_count_as_unfinished_and_are_removed(sumo_env, monkeypatch):
    """SUMO queues a vehicle whose departure lane has no room; getIDList does not
    list it, so it used to survive the reset, enter the next episode with this
    episode's route and block its own id there ('to add already exists')."""
    sumo_env_mod, SumoMARLEnv = sumo_env
    env, calls = _env(sumo_env_mod, SumoMARLEnv, current_step=3600, current_episode=4)
    env.vehicle_spawner = SimpleNamespace(
        spawned_vehicles={"j4_north_N_0_6": object(), "j4_south_S_0_2": object()},
        reset_episode_registries=lambda: None)
    env.episode.active_vehicles["j4_south_S_0_2"] = {}          # in the network, unfinished the usual way
    removed = []
    monkeypatch.setattr(sumo_env_mod.traci.vehicle, "getIDList", lambda: ("j4_south_S_0_2",))
    monkeypatch.setattr(sumo_env_mod.traci.simulation, "getPendingVehicles", lambda: ("j4_north_N_0_6",))
    monkeypatch.setattr(sumo_env_mod.traci.vehicle, "remove", lambda veh_id: removed.append(veh_id))
    monkeypatch.setattr(sumo_env_mod.traci, "simulationStep", lambda: None)

    SumoMARLEnv.reset(env, prepare_next_episode=False)

    assert calls[0][1]["active_agents"] == 2 and calls[0][1]["success"] == 0
    assert sorted(removed) == ["j4_north_N_0_6", "j4_south_S_0_2"]


def test_the_insertion_queue_is_read_from_the_spawner_when_sumo_has_no_pending_api(sumo_env, monkeypatch):
    sumo_env_mod, SumoMARLEnv = sumo_env
    env, calls = _env(sumo_env_mod, SumoMARLEnv, current_step=3600, current_episode=4)
    env.vehicle_spawner = SimpleNamespace(
        spawned_vehicles={"j4_north_N_0_6": object(), "arrived": object()},
        reset_episode_registries=lambda: None)
    env.episode.departed_vehicles.add("arrived")
    removed = []
    monkeypatch.setattr(sumo_env_mod.traci.vehicle, "getIDList", lambda: ())
    monkeypatch.delattr(sumo_env_mod.traci.simulation, "getPendingVehicles", raising=False)
    monkeypatch.setattr(sumo_env_mod.traci.vehicle, "remove", lambda veh_id: removed.append(veh_id))
    monkeypatch.setattr(sumo_env_mod.traci, "simulationStep", lambda: None)

    SumoMARLEnv.reset(env, prepare_next_episode=False)

    assert calls[0][1]["active_agents"] == 1 and calls[0][1]["success"] == 1
    assert removed == ["j4_north_N_0_6"]


def test_reset_before_any_step_finishes_nothing(sumo_env, monkeypatch):
    sumo_env_mod, SumoMARLEnv = sumo_env
    env, calls = _env(sumo_env_mod, SumoMARLEnv, current_step=0, current_episode=0)
    monkeypatch.setattr(sumo_env_mod.traci.vehicle, "getIDList", lambda: ())
    monkeypatch.setattr(sumo_env_mod.traci, "simulationStep", lambda: None)

    SumoMARLEnv.reset(env, prepare_next_episode=False)

    assert calls == []
    assert env.last_episode_metrics is None
    assert env.current_episode == 0                    # no episode consumed
