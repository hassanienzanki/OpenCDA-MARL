"""The CARLA pipeline installs the provider the profile configured.

``wire_intent_provider`` used to overwrite the agent manager's provider with
the manager's cached learned bridge whenever a learned head existed. For a
*registered* path action that is the wrong provider twice over: the bridge
carries the framework's own semantics, and it feeds its own observation into
a head the lifecycle sized for the plugin's, which fails at the first
forward pass with a shape mismatch (``mat1 and mat2 shapes cannot be
multiplied``) on CARLA, while the same profile runs normally on SUMO, whose
environment resolves the provider through the registry and never overwrites
it.

The rule now: ``mode: marl`` gets the bridge; a registered provider is
rebuilt through the registry with the manager (the scenario manager's first
copy predates the manager and has no learner); the non-learned built-ins keep
the scenario manager's copy.
"""
from types import SimpleNamespace

from opencda_marl._kernel.lifecycle.pipeline_builder import PipelineBuilder
from opencda_marl.api import register_component

BRIDGE = object()
PLANNER = object()


def _builder(path_block):
    builder = PipelineBuilder.__new__(PipelineBuilder)
    builder.config = {
        "agents": {"route_intent": {}},
        "MARL": {"action_space": {"mode": "path_speed", "path": dict(path_block)},
                 "training": {"training_mode": False}},
    }
    return builder


def _scenario(first_provider):
    return SimpleNamespace(agent_manager=SimpleNamespace(planner=PLANNER, intent_provider=first_provider))


def _manager():
    return SimpleNamespace(intent_provider=BRIDGE, agent_manager=None, intent_algorithm=object())


def test_a_registered_provider_is_rebuilt_with_the_manager_not_replaced_by_the_bridge():
    built = []

    def factory(cfg, planner, *, marl_manager=None, training=True, previous=None):
        provider = SimpleNamespace(cfg=dict(cfg), planner=planner, marl_manager=marl_manager,
                                   training=training, previous=previous)
        built.append(provider)
        return provider

    register_component("intent_provider", "wire_test_path", override=True)(factory)
    first = factory({"mode": "wire_test_path"}, PLANNER, marl_manager=None)   # the scenario manager's copy
    scenario, manager = _scenario(first), _manager()

    _builder({"provider": "wire_test_path", "n_max": 3, "custom_knob": 0.7}) \
        .wire_intent_provider(scenario, manager)

    installed = scenario.agent_manager.intent_provider
    assert installed is not BRIDGE, "the built-in bridge must not replace a registered provider"
    assert installed is not first and installed.marl_manager is manager, "rebuilt with the learner's manager"
    assert installed.previous is first, "state that should outlive the rebuild is handed over"
    assert installed.planner is PLANNER and installed.training is False
    assert installed.cfg["mode"] == "wire_test_path" and installed.cfg["n_max"] == 3, "its own block reaches it"
    assert manager.agent_manager is scenario.agent_manager, "the back-reference is still rebound"


def test_the_frameworks_own_learned_action_still_gets_the_bridge():
    scenario, manager = _scenario(None), _manager()
    _builder({"selector": "learned"}).wire_intent_provider(scenario, manager)
    assert scenario.agent_manager.intent_provider is BRIDGE
    assert manager.agent_manager is scenario.agent_manager


def test_a_non_learned_builtin_keeps_the_scenario_managers_copy():
    first = object()
    for selector in ("rule_based", "random_legal", "keep_current"):
        scenario, manager = _scenario(first), _manager()
        _builder({"selector": selector}).wire_intent_provider(scenario, manager)
        assert scenario.agent_manager.intent_provider is first, selector
        assert manager.agent_manager is scenario.agent_manager


def test_speed_only_installs_nothing():
    first = object()
    builder = PipelineBuilder.__new__(PipelineBuilder)
    builder.config = {"agents": {}, "MARL": {"action_space": {"mode": "speed_only"}}}
    scenario, manager = _scenario(first), _manager()
    builder.wire_intent_provider(scenario, manager)
    assert scenario.agent_manager.intent_provider is first
