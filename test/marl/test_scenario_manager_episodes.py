"""CARLA scenario manager: episode horizon for baseline agents, failure text.

A baseline agent (``behavior`` / ``vanilla`` / ``rule_based``) used to run
exactly one episode on CARLA whatever ``scenario.simulation.max_episodes``
said, while SUMO honoured the configured count. One episode is now only the
default when the config sets none.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("carla", reason="the CARLA scenario manager imports the CARLA stack")

from opencda.scenario_testing.utils import sim_api  # noqa: E402
from opencda_marl._kernel.lifecycle import scenario_manager as sm_module  # noqa: E402


class _FakeAgentManager:
    def __init__(self, agents_cfg, *_args, **_kwargs):
        self.agent_type = agents_cfg.get("agent_type", "marl")
        self.intent_provider = None
        self.marl_manager = None


@pytest.fixture
def stub_world(monkeypatch):
    """Build MARLScenarioManager without a CARLA server: the base manager,
    the traffic and agent managers and the provider factory are stand-ins."""
    def _base_init(self, scenario_params, apply_ml, town=None, xodr_path=None, cav_world=None):
        self.world = SimpleNamespace(tick=lambda: None)
        self.client = object()
        self.cav_world = cav_world
        self.fixed_dt = 0.05

    monkeypatch.setattr(sim_api.ScenarioManager, "__init__", _base_init)
    monkeypatch.setattr(sm_module, "MARLTrafficManager",
                        lambda *a, **k: SimpleNamespace(planner=None))
    monkeypatch.setattr(sm_module, "MARLAgentManager", _FakeAgentManager)
    monkeypatch.setattr(sm_module, "build_provider_from_config", lambda *a, **k: None)
    monkeypatch.setattr(sm_module, "derive_route_intent_config", lambda *a, **k: {})


def _build(agent_type, simulation):
    params = {"world": {}, "agents": {"agent_type": agent_type},
              "scenario": {"simulation": simulation}, "MARL": {}}
    return sm_module.MARLScenarioManager(params, apply_ml=False)


@pytest.mark.parametrize("agent_type", ["rule_based", "behavior", "vanilla"])
def test_a_baseline_honours_the_configured_episode_count(stub_world, agent_type):
    assert _build(agent_type, {"max_episodes": 7}).states["max_episodes"] == 7


@pytest.mark.parametrize("agent_type", ["rule_based", "behavior", "vanilla"])
def test_a_baseline_without_a_count_runs_one_episode(stub_world, agent_type):
    assert _build(agent_type, {"max_steps": 100}).states["max_episodes"] == 1


def test_a_learner_keeps_its_configured_count(stub_world):
    assert _build("marl", {"max_episodes": 7}).states["max_episodes"] == 7


def test_a_world_build_timeout_names_the_setting_to_raise(monkeypatch):
    def _timed_out(self, *args, **kwargs):
        raise RuntimeError("time-out of 40000ms while waiting for the simulator")

    monkeypatch.setattr(sim_api.ScenarioManager, "__init__", _timed_out)
    with pytest.raises(RuntimeError, match="world.timeout"):
        sm_module.MARLScenarioManager({"world": {}}, apply_ml=False, xodr_path="x.xodr")


def test_other_construction_errors_carry_no_timeout_hint(monkeypatch):
    def _broken(self, *args, **kwargs):
        raise RuntimeError("map not found")

    monkeypatch.setattr(sim_api.ScenarioManager, "__init__", _broken)
    with pytest.raises(RuntimeError) as excinfo:
        sm_module.MARLScenarioManager({"world": {}}, apply_ml=False, town="Town05")
    assert "map not found" in str(excinfo.value)
    assert "world.timeout" not in str(excinfo.value)
