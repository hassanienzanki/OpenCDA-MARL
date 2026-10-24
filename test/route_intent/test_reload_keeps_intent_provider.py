"""World reload must rebuild the built-in route-intent providers.

The rule-based / random / fixed providers close over the traffic planner,
which is rebuilt together with the CARLA world. ``rebuild_after_world_reload``
used to hand the new agent manager ``intent_provider=None`` and rely on
``wire_intent_provider``, which only knows the *learned* provider -- so every
``rule_path_*`` / ``random_legal_path_*`` / ``keep_current_path_*`` cell
silently degraded to keep-current after its first periodic reload.
"""
import random
from types import SimpleNamespace

import pytest

import opencda_marl._kernel.lifecycle.scenario_manager as sm_mod
from opencda_marl._kernel.lifecycle.scenario_manager import MARLScenarioManager
from opencda_marl.api import intent as ip


class _Planner:
    def __init__(self, tag):
        self.tag = tag


class _FakeTrafficManager:
    def __init__(self, world, traffic_cfg, states, fixed_dt, intent_cfg=None,
                 population_cfg=None, outputs_cfg=None):
        self.planner = _Planner(tag=f"planner-{id(self)}")
        self.intent_cfg = intent_cfg
        self.population_cfg = population_cfg
        self.outputs_cfg = outputs_cfg


class _FakeAgentManager:
    def __init__(self, agents_cfg, states, world, cav_world, client=None,
                 planner=None, intent_provider=None, marl_manager=None):
        self.planner = planner
        self.intent_provider = intent_provider
        self.marl_manager = marl_manager


def _scenario_manager(mode, monkeypatch, marl_manager=None):
    monkeypatch.setattr(sm_mod, "MARLTrafficManager", _FakeTrafficManager)
    monkeypatch.setattr(sm_mod, "MARLAgentManager", _FakeAgentManager)
    # rebuild_after_world_reload imports these names locally too.
    import opencda_marl._kernel.traffic.traffic_manager as traffic_mod
    import opencda_marl._kernel.agent_manager as agent_manager_mod
    monkeypatch.setattr(traffic_mod, "MARLTrafficManager", _FakeTrafficManager)
    monkeypatch.setattr(agent_manager_mod, "MARLAgentManager", _FakeAgentManager)

    sm = object.__new__(MARLScenarioManager)
    sm.params = {"traffic": {}}
    sm.states = {}
    sm.world = object()
    sm.cav_world = object()
    sm.fixed_dt = 0.05
    sm.client = None
    sm.marl_manager = marl_manager
    return sm


_SELECTOR = {"rule_based_path": "rule_based", "tao_fixed": "keep_current",
             "naive_random": "random_legal", "marl": "learned"}


def _builder(mode):
    """Config as the launcher writes it: MARL.action_space drives the mode."""
    return SimpleNamespace(config={
        "agents": {"route_intent": {"seed": 3}},
        "world": {}, "scenario": {},
        "MARL": {"training": {"training_mode": True},
                 "action_space": {"mode": "path_speed", "path": {"selector": _SELECTOR[mode]}}},
    })


@pytest.mark.parametrize("mode", ["rule_based_path", "tao_fixed"])
def test_reload_rebuilds_builtin_provider_over_the_new_planner(mode, monkeypatch):
    captured = {}

    def fake_build(intent_cfg, planner, marl_manager=None, training=True, previous=None):
        captured.update(intent_cfg=intent_cfg, planner=planner, previous=previous, training=training)
        return lambda context: ("decision", planner.tag)

    monkeypatch.setattr(sm_mod, "build_provider_from_config", fake_build)
    sm = _scenario_manager(mode, monkeypatch)
    sm.agent_manager = SimpleNamespace(intent_provider=lambda context: "old")

    sm.rebuild_after_world_reload(_builder(mode))

    assert captured["intent_cfg"]["mode"] == mode
    assert captured["planner"] is sm.traffic_manager.planner, "rebuilt over the NEW planner"
    assert captured["previous"] is not None, "previous provider handed over"
    assert sm.agent_manager.intent_provider is not None
    assert sm.agent_manager.intent_provider({})[1] == sm.traffic_manager.planner.tag


def test_reload_keeps_the_random_provider_draw_sequence(monkeypatch):
    """naive_random: the RNG survives the rebuild instead of restarting at seed."""
    from opencda_marl.builtin.intent_providers.providers import make_naive_random_provider
    old = make_naive_random_provider(_Planner("old"), seed=3)
    reference = random.Random(3)
    assert old.rng.getstate() == reference.getstate()
    old.rng.random()                                   # consume one draw

    rebuilt = ip.build_provider_from_config(
        {"mode": "naive_random", "seed": 3}, _Planner("new"), previous=old)
    assert rebuilt.rng is old.rng, "same RNG object carried over"
    fresh = ip.build_provider_from_config({"mode": "naive_random", "seed": 3}, _Planner("new"))
    assert fresh.rng.getstate() == reference.getstate(), "without a previous provider: seeded afresh"


def test_reload_leaves_learned_provider_to_the_marl_manager(monkeypatch):
    """mode=marl: rebuild returns the manager's provider (or None), and
    wire_intent_provider's cached provider is what the agent manager ends
    with -- unchanged contract."""
    from opencda_marl._kernel.lifecycle.pipeline_builder import PipelineBuilder

    learned = lambda context: "learned"          # noqa: E731
    marl_manager = SimpleNamespace(
        get_intent_provider=lambda training=True: learned,
        intent_provider=learned,
    )
    sm = _scenario_manager("marl", monkeypatch, marl_manager=marl_manager)
    sm.agent_manager = SimpleNamespace(intent_provider=None)
    sm.rebuild_after_world_reload(_builder("marl"))
    assert sm.agent_manager.intent_provider is learned
    # The wiring now reads the configured mode, so it needs a builder with
    # a config rather than a bare None for self.
    builder = PipelineBuilder.__new__(PipelineBuilder)
    builder.config = _builder("marl").config
    builder.wire_intent_provider(sm, marl_manager)
    assert sm.agent_manager.intent_provider is learned
    assert marl_manager.agent_manager is sm.agent_manager


def test_reload_carries_population_and_outputs_to_the_new_traffic_manager(monkeypatch):
    """The rebuild must re-wire everything the first construction wired.

    It carried only the intent config, so after a CARLA world reload a
    mixed-autonomy run silently went homogeneous (no population sampler) and
    a `record` run's destination fell back to the default root.
    """
    monkeypatch.setattr(sm_mod, "build_provider_from_config",
                        lambda *a, **k: (lambda context: "decision"))
    sm = _scenario_manager("rule_based_path", monkeypatch)
    sm.agent_manager = SimpleNamespace(intent_provider=None)

    builder = _builder("rule_based_path")
    population = {"mix": {"marl": 0.5, "rule_based": 0.5}, "seed": 7}
    builder.config["agents"]["population"] = population
    builder.config["outputs"] = {"root_dir": "runs/exp1"}

    sm.rebuild_after_world_reload(builder)

    assert sm.traffic_manager.population_cfg == population
    assert sm.traffic_manager.outputs_cfg == {"root_dir": "runs/exp1"}
