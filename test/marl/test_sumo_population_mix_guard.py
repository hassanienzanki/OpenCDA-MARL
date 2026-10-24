"""A SUMO population mix that names no learning population fails at start.

Without the learning population in ``agents.population.mix`` every vehicle
is background traffic and the learner silently gets no vehicle at all, so
the environment refuses the configuration before it builds the traffic
manager. Any other name is background traffic SUMO drives itself.
"""
from types import SimpleNamespace

import pytest


@pytest.fixture
def sumo_env_module():
    try:
        import opencda_marl._kernel.envs.sumo_marl_env as mod
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"SUMO runtime unavailable: {exc}")
    return mod


def _env(mod, monkeypatch, mix):
    built = []
    monkeypatch.setattr(mod, "SumoWorld", lambda: SimpleNamespace())

    def _manager(**kwargs):
        built.append(kwargs)
        raise RuntimeError("stop after construction")

    monkeypatch.setattr(mod, "MARLTrafficManager", _manager)
    env = object.__new__(mod.SumoMARLEnv)
    env.config = {"agents": {"agent_type": "marl", "population": {"mix": mix}}}
    env.scenario_config = {"traffic": {}}
    env.step_length = 0.05
    env.intent_cfg = {}
    return env, built


def test_a_mix_without_the_learner_is_refused(sumo_env_module, monkeypatch):
    env, built = _env(sumo_env_module, monkeypatch, {"background": 1.0})
    with pytest.raises(ValueError, match="names no learning population"):
        env._initialize_traffic_manager()
    assert built == []


def test_a_mix_with_the_learner_reaches_the_traffic_manager(sumo_env_module, monkeypatch):
    env, built = _env(sumo_env_module, monkeypatch, {"marl": 0.5, "background": 0.5})
    with pytest.raises(RuntimeError, match="stop after construction"):
        env._initialize_traffic_manager()
    assert len(built) == 1
