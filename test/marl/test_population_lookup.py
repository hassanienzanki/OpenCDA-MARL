"""A SUMO vehicle's population is resolved once per episode, not once per step.

The observation builder asked, for every vehicle on every step, which
population it belongs to: a config read through the resolver plus a spawn
event lookup, for an answer fixed by the spawn event for the vehicle's
whole life. The episode state now remembers it; a new episode state
forgets it; and a vehicle seen before its spawn event is registered is
asked again rather than remembered with the default.
"""
from types import SimpleNamespace

from opencda_marl._kernel.envs.episode_state import SumoEpisodeState
from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv


class _CountingConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reads = 0

    def get(self, key, default=None):
        self.reads += 1
        return super().get(key, default)


class _Spawner:
    def __init__(self, events):
        self.events = dict(events)
        self.lookups = 0

    def get_spawn_event(self, vid):
        self.lookups += 1
        return self.events.get(vid)


def _env(events, agent_type="marl"):
    env = object.__new__(SumoMARLEnv)
    env.episode = SumoEpisodeState()
    env.config = _CountingConfig({"agents": {"agent_type": agent_type}})
    env.vehicle_spawner = _Spawner(events)
    return env


def _event(population=None):
    return SimpleNamespace(metadata={"population": population} if population else {})


def test_population_and_learning_flag_follow_the_spawn_event():
    env = _env({"cav": _event("marl"), "hv": _event("human"), "plain": _event()})
    assert env._population_of("cav") == ("marl", True)
    assert env._population_of("hv") == ("human", False)
    assert env._population_of("plain") == ("marl", True), "no population in the event: the run's agent type"


def test_each_vehicle_is_resolved_once_per_episode():
    env = _env({"cav": _event("marl"), "hv": _event("human")})
    for _step in range(50):
        for vid in ("cav", "hv"):
            env._population_of(vid)
    assert env.vehicle_spawner.lookups == 2
    assert env.config.reads == 2
    assert env.episode.populations == {"cav": ("marl", True), "hv": ("human", False)}


def test_a_new_episode_state_forgets_the_populations():
    env = _env({"cav": _event("human")})
    assert env._population_of("cav") == ("human", False)
    env.vehicle_spawner.events["cav"] = _event("marl")           # the next episode's replay may differ
    assert env._population_of("cav") == ("human", False), "still this episode: remembered"
    env.episode = SumoEpisodeState()
    assert env._population_of("cav") == ("marl", True)


def test_a_vehicle_without_a_spawn_event_yet_is_asked_again():
    env = _env({})
    assert env._population_of("late") == ("marl", True)
    assert "late" not in env.episode.populations
    env.vehicle_spawner.events["late"] = _event("human")
    assert env._population_of("late") == ("human", False)
    assert env.vehicle_spawner.lookups == 2


def test_without_a_spawner_the_default_is_remembered():
    env = _env({})
    env.vehicle_spawner = None
    assert env._population_of("v") == ("marl", True)
    assert env._population_of("v") == ("marl", True)
    assert env.config.reads == 1
