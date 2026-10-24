"""The traffic view's ``role`` separates learning agents from background traffic.

The role is the schematic view's only semantic channel ("learning agents in
blue"). SUMO marked every live vehicle ``agent`` and CARLA every vehicle with
an adapter, so in a population mix the background vehicles were drawn as
learners on both backends. ``agent`` now means a live vehicle of the learning
population.
"""
from types import SimpleNamespace

import pytest


class _Vehicles:
    """The TraCI vehicle domain the SUMO view reads: id -> (x, y)."""

    def __init__(self, scene):
        self.scene = scene

    def getIDList(self):
        return tuple(self.scene)

    def getPosition(self, vid):
        return self.scene[vid]

    def getAngle(self, vid):
        return 90.0

    def getSpeed(self, vid):
        return 5.0


def test_sumo_marks_only_live_learners_as_agents(monkeypatch):
    try:
        import opencda_marl._kernel.envs.sumo_marl_env as mod
        from opencda_marl._kernel.envs.episode_state import SumoEpisodeState
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"SUMO runtime unavailable: {exc}")
    scene = {"learner": (0.0, 0.0), "background": (5.0, 0.0), "finished": (9.0, 0.0)}
    monkeypatch.setattr(mod.traci, "vehicle", _Vehicles(scene))
    env = object.__new__(mod.SumoMARLEnv)
    env.episode = SumoEpisodeState()
    env.episode.terminal_agents.add("finished")
    env.current_episode, env.current_step = 2, 40
    env.config = {"agents": {"agent_type": "marl"}}
    populations = {"learner": "marl", "background": "behavior", "finished": "marl"}
    events = {vid: SimpleNamespace(metadata={"population": pop}) for vid, pop in populations.items()}
    env.vehicle_spawner = SimpleNamespace(get_spawn_event=events.get)

    vehicles, episode, step = env.collect_traffic_view()

    roles = {v["id"]: v["role"] for v in vehicles}
    assert roles == {"learner": "agent", "background": "traffic", "finished": "traffic"}
    assert (episode, step) == (2, 40)


def test_sumo_without_a_population_mix_keeps_every_live_vehicle_an_agent(monkeypatch):
    try:
        import opencda_marl._kernel.envs.sumo_marl_env as mod
        from opencda_marl._kernel.envs.episode_state import SumoEpisodeState
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"SUMO runtime unavailable: {exc}")
    monkeypatch.setattr(mod.traci, "vehicle", _Vehicles({"a": (0.0, 0.0), "b": (5.0, 0.0)}))
    env = object.__new__(mod.SumoMARLEnv)
    env.episode = SumoEpisodeState()
    env.current_episode, env.current_step = 0, 1
    env.config = {"agents": {"agent_type": "marl"}}
    env.vehicle_spawner = SimpleNamespace(get_spawn_event=lambda vid: SimpleNamespace(metadata={}))

    vehicles, _, _ = env.collect_traffic_view()

    assert {v["role"] for v in vehicles} == {"agent"}


def test_carla_marks_only_learning_adapters_as_agents():
    pytest.importorskip("carla", reason="MARLEnv imports the CARLA scenario stack")
    from opencda_marl._kernel.envs.marl_env import MARLEnv

    def adapter(actor_id, population=None):
        metadata = {"population": population} if population else {}
        return SimpleNamespace(actor_id=actor_id, spawn_metadata=metadata)

    def kinematics(actor_id):
        return SimpleNamespace(actor_id=actor_id, x=1.0, y=2.0, vx=3.0, vy=4.0, yaw_rad=0.5)

    agent_manager = SimpleNamespace(
        agent_type="marl",
        _vehicle_adapters=[adapter(1, "marl"), adapter(2, "behavior"), adapter(3)],
        _current_snapshot=SimpleNamespace(vehicles=[kinematics(i) for i in (1, 2, 3, 4)]),
    )
    env = object.__new__(MARLEnv)
    env.sm = SimpleNamespace(agent_manager=agent_manager, states={"episode": 1, "step": 7})

    vehicles, episode, step = env.collect_traffic_view()

    roles = {v["id"]: v["role"] for v in vehicles}
    # 1: learning population; 2: background population; 3: no population
    # metadata (a homogeneous run); 4: an actor without an adapter.
    assert roles == {"1": "agent", "2": "traffic", "3": "agent", "4": "traffic"}
    assert (episode, step) == (1, 7)
    assert vehicles[0]["speed"] == pytest.approx(5.0)
