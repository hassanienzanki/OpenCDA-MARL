"""ParallelEnv facade: the bring-your-own-loop boundary.

Uses a fake inner environment that emulates the real contract (inner asks
the manager's policy for actions), so the mailbox round-trip is exercised
with the real ExternalPolicy.
"""
import numpy as np
import pytest

from opencda_marl.envs.parallel_env import ExternalPolicy, OpenCDAMARLParallelEnv
from opencda_marl.api.events import StepEvent

try:
    import gymnasium  # noqa: F401
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False


class FakeManager:
    def __init__(self, state_dim=3):
        self._state_dim = state_dim
        self.algorithm = ExternalPolicy({"max_action": 45.0}, state_dim, 1)

    def resolve_state_dim(self):
        return self._state_dim

    def _extract_states(self, raw):
        states = {}
        for agent, obs in raw.items():
            vector = np.asarray(
                [obs.get("speed", 0.0)] * self._state_dim, dtype=np.float32)
            states[str(agent)] = {"ego_state": vector, "all_states": {}}
        return states


class FakeInner:
    """Emulates the inner env: reads commands through the policy, like the
    real manager loop does."""

    def __init__(self, manager):
        self.manager = manager
        self.commands_seen = []
        self.script = []  # list of result dicts to return per step
        self.reset_calls = 0
        self.closed = False
        self._current = {}

    def reset(self):
        self.reset_calls += 1

    def _get_current_observations(self):
        return dict(self._current)

    def step(self):
        result = self.script.pop(0)
        raw = result.get("observations") or {}
        # The real inner loop asks the policy for each observed agent.
        self.commands_seen.append({
            agent: self.manager.algorithm.select_action(raw, agent)
            for agent in raw
        })
        return result

    def close(self):
        self.closed = True


def _env(script, initial=None, state_dim=3):
    manager = FakeManager(state_dim=state_dim)
    inner = FakeInner(manager)
    inner.script = list(script)
    inner._current = dict(initial or {})
    return OpenCDAMARLParallelEnv(inner, manager), inner


def _obs(speed):
    return {"speed": speed}


def test_requires_external_policy():
    manager = FakeManager()
    manager.algorithm = object()
    with pytest.raises(TypeError, match="external"):
        OpenCDAMARLParallelEnv(FakeInner(manager), manager)


def test_reset_returns_initial_observations_and_infos():
    env, inner = _env([], initial={"veh_a": _obs(10.0)})
    observations, infos = env.reset()
    assert inner.reset_calls == 1
    assert list(observations) == ["veh_a"]
    assert observations["veh_a"].shape == (3,)
    assert infos["veh_a"]["raw_observation"] == {"speed": 10.0}
    assert env.agents == ["veh_a"]


def test_step_routes_actions_through_the_mailbox():
    env, inner = _env([
        {
            "observations": {"veh_a": _obs(11.0), "veh_b": _obs(12.0)},
            "rewards": {"veh_a": 1.0, "veh_b": -0.5},
            "events": [],
            "done": False,
        },
    ])
    env.reset()
    observations, rewards, terminations, truncations, infos = env.step(
        {"veh_a": 30.0})  # veh_b gets no command
    assert inner.commands_seen[-1] == {"veh_a": 30.0, "veh_b": None}
    assert rewards == {"veh_a": 1.0, "veh_b": -0.5}
    assert terminations == {"veh_a": False, "veh_b": False}
    assert truncations == {"veh_a": False, "veh_b": False}
    assert sorted(env.agents) == ["veh_a", "veh_b"]
    assert sorted(env.possible_agents) == ["veh_a", "veh_b"]


def test_terminal_agent_appears_once_more_then_leaves():
    env, _ = _env([
        {
            "observations": {"veh_a": _obs(11.0), "veh_b": _obs(12.0)},
            "rewards": {"veh_a": 0.0, "veh_b": 0.0},
            "events": [],
            "done": False,
        },
        {
            # veh_b reached its destination: gone from observations,
            # success event emitted.
            "observations": {"veh_a": _obs(13.0)},
            "rewards": {"veh_a": 0.0, "veh_b": 100.0},
            "events": [StepEvent(step=1, event_id="s", vehicle_id="veh_b",
                                 event_type="success")],
            "done": False,
        },
    ])
    env.reset()
    env.step({})
    observations, rewards, terminations, truncations, infos = env.step({})
    assert terminations["veh_b"] is True
    assert rewards["veh_b"] == 100.0
    assert infos["veh_b"]["event"] == "success"
    # Final appearance carries the last known vector.
    assert observations["veh_b"] == pytest.approx([12.0] * 3)
    assert env.agents == ["veh_a"]


def test_vanished_agent_without_event_is_terminated():
    env, _ = _env([
        {"observations": {"veh_a": _obs(1.0)}, "rewards": {}, "events": [],
         "done": False},
        {"observations": {}, "rewards": {}, "events": [], "done": False},
    ])
    env.reset()
    env.step({})
    observations, rewards, terminations, truncations, infos = env.step({})
    assert terminations["veh_a"] is True
    assert infos["veh_a"]["event"] == "removed"
    assert env.agents == []


def test_episode_done_truncates_survivors():
    env, _ = _env([
        {"observations": {"veh_a": _obs(1.0)}, "rewards": {"veh_a": 0.0},
         "events": [], "done": True},
    ])
    env.reset()
    observations, rewards, terminations, truncations, infos = env.step({})
    assert terminations["veh_a"] is False
    assert truncations["veh_a"] is True
    assert env.agents == []


def test_infos_carry_reward_breakdown_and_preferences():
    env, _ = _env([
        {
            "observations": {"veh_a": _obs(1.0)},
            "rewards": {"veh_a": 2.0},
            "reward_components": {"veh_a": {"progress": 2.0}},
            "reward_vectors": {"veh_a": [2.0, 0.0]},
            "objective_vectors": {"veh_a": [1.0, 1.0]},
            "agent_preferences": {"veh_a": [0.7, 0.3]},
            "events": [],
            "done": False,
        },
    ])
    env.reset()
    _, _, _, _, infos = env.step({"veh_a": 20.0})
    info = infos["veh_a"]
    assert info["reward_components"] == {"progress": 2.0}
    assert info["reward_vector"] == [2.0, 0.0]
    assert info["objective_vector"] == [1.0, 1.0]
    assert info["preference"] == [0.7, 0.3]


def test_close_propagates():
    env, inner = _env([])
    env.close()
    assert inner.closed is True


# --------------------------------------------------------------------- #
# Action coercion
# --------------------------------------------------------------------- #
def test_coerce_speed_accepts_common_shapes():
    policy = ExternalPolicy({}, 3, 1)
    policy.set_actions({
        "a": 30.0,
        "b": np.float32(25.0),
        "c": np.array([20.0], dtype=np.float32),
        "d": {"target_speed": 15.0},
        "e": None,
    })
    obs = {}
    assert policy.select_action(obs, "a") == 30.0
    assert policy.select_action(obs, "b") == 25.0
    assert policy.select_action(obs, "c") == 20.0
    assert policy.select_action(obs, "d") == 15.0
    assert policy.select_action(obs, "e") is None
    assert policy.select_action(obs, "unknown") is None


def test_coerce_speed_rejects_vectors():
    policy = ExternalPolicy({}, 3, 1)
    with pytest.raises(ValueError, match="scalar target speed"):
        policy.set_actions({"a": np.array([1.0, 2.0])})


# --------------------------------------------------------------------- #
# Spaces (only when gymnasium is installed)
# --------------------------------------------------------------------- #
@pytest.mark.skipif(not HAS_GYMNASIUM, reason="gymnasium not installed")
def test_spaces_reflect_benchmark_contract():
    env, _ = _env([], state_dim=5)
    obs_space = env.observation_space("any")
    act_space = env.action_space("any")
    assert obs_space.shape == (5,)
    assert act_space.shape == ()
    assert float(act_space.high) == 45.0
    # Identical objects per agent (PettingZoo recommendation).
    assert env.observation_space("a") is env.observation_space("b")


@pytest.mark.skipif(HAS_GYMNASIUM, reason="covers the no-gymnasium path")
def test_spaces_error_is_actionable_without_gymnasium():
    env, _ = _env([])
    with pytest.raises(ImportError, match="pixi add gymnasium"):
        env.observation_space("any")
