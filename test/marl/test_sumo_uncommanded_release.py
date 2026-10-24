"""A SUMO vehicle the policy stops sending speeds for is driven by SUMO again.

TraCI's ``setSpeed`` persists until it is released with ``-1``, so a learning
vehicle that was commanded once and then received no action (a warmup or
failed step, or an external loop that skipped it through the ParallelEnv)
kept its last commanded speed -- under the default ``unshielded`` runtime
mode, without SUMO's safe-speed checks -- instead of the fallback driver the ParallelEnv
promised. ``_apply_actions`` now hands such a vehicle back to SUMO's
car-following model on the step it stops being commanded.

Only that transition releases: a learning vehicle that has never been
commanded keeps the departure speed the spawner set, and no release call is
made for it.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from opencda_marl.envs.parallel_env import ExternalPolicy, OpenCDAMARLParallelEnv


@pytest.fixture
def sumo_env():
    try:
        import opencda_marl._kernel.envs.sumo_marl_env as mod
        from opencda_marl._kernel.envs.episode_state import SumoEpisodeState
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"SUMO runtime unavailable: {exc}")
    return mod, SumoEpisodeState


class _SpeedLog:
    """The TraCI vehicle domain _apply_actions writes to."""

    def __init__(self):
        self.calls = []

    def setSpeed(self, veh_id, speed):
        self.calls.append((veh_id, speed))


def _env(mod, episode_cls, monkeypatch):
    log = _SpeedLog()
    monkeypatch.setattr(mod.traci, "vehicle", log)
    env = object.__new__(mod.SumoMARLEnv)
    env.episode = episode_cls()
    return env, log


def test_a_vehicle_that_stops_being_commanded_is_released_once(sumo_env, monkeypatch):
    mod, episode_cls = sumo_env
    env, log = _env(mod, episode_cls, monkeypatch)

    env._apply_actions({"a": 36.0, "b": 18.0})
    assert sorted(log.calls) == [("a", 10.0), ("b", 5.0)]

    log.calls.clear()
    env._apply_actions({"a": 36.0})
    assert sorted(log.calls) == [("a", 10.0), ("b", -1)], "b goes back to SUMO's driver"

    log.calls.clear()
    env._apply_actions({"a": 36.0})
    assert log.calls == [("a", 10.0)], "an already released vehicle costs no further call"

    log.calls.clear()
    env._apply_actions({})
    assert log.calls == [("a", -1)]


def test_a_vehicle_never_commanded_is_left_alone(sumo_env, monkeypatch):
    mod, episode_cls = sumo_env
    env, log = _env(mod, episode_cls, monkeypatch)
    env._apply_actions({})
    env._apply_actions({})
    assert log.calls == []


def test_a_finished_vehicle_is_not_released(sumo_env, monkeypatch):
    mod, episode_cls = sumo_env
    env, log = _env(mod, episode_cls, monkeypatch)
    env._apply_actions({"a": 36.0})
    env.episode.terminal_agents.add("a")
    log.calls.clear()
    env._apply_actions({})
    assert log.calls == []


class _FakeSumoInner:
    """A SUMO env step reduced to its action path: the manager asks the policy
    for every observed vehicle, drops ``None`` (as MARLManager does), and the
    real ``SumoMARLEnv._apply_actions`` applies the rest."""

    def __init__(self, env, policy, agents):
        self.env, self.policy, self.agents = env, policy, list(agents)

    def reset(self):
        pass

    def _get_current_observations(self):
        return {a: {"speed": 20.0} for a in self.agents}

    def step(self):
        raw = self._get_current_observations()
        commands = {a: self.policy.select_action(raw, a) for a in raw}
        self.env._apply_actions({a: s for a, s in commands.items() if s is not None})
        return {"observations": raw, "rewards": {a: 0.0 for a in raw}, "events": [], "done": False}


class _Manager:
    def __init__(self):
        self.algorithm = ExternalPolicy({"max_action": 45.0}, 1, 1)

    def resolve_state_dim(self):
        return 1

    def _extract_states(self, raw):
        return {str(a): {"ego_state": np.asarray([o["speed"]], dtype=np.float32), "all_states": {}}
                for a, o in raw.items()}


def test_an_agent_the_external_loop_skips_is_driven_by_sumo(sumo_env, monkeypatch):
    mod, episode_cls = sumo_env
    env, log = _env(mod, episode_cls, monkeypatch)
    manager = _Manager()
    parallel = OpenCDAMARLParallelEnv(_FakeSumoInner(env, manager.algorithm, ["a", "b"]), manager)
    parallel.reset()

    parallel.step({"a": 36.0, "b": 36.0})
    assert sorted(log.calls) == [("a", 10.0), ("b", 10.0)]

    log.calls.clear()
    parallel.step({"a": 36.0})                       # no action for b this step
    assert ("b", -1) in log.calls and ("b", 10.0) not in log.calls
