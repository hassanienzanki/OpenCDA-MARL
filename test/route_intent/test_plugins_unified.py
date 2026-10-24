"""The unified plugin system: one door, one table, rollback, freeze.

Every kind registers through its public function but reports to
``opencda_marl.plugins``: describe covers all kinds, a plugin's API-version
mismatch rolls back registrations of every kind it made, and after freeze()
the registry rejects late registrations.
"""
import pytest

from opencda_marl import plugins
from opencda_marl.api.components import register_component, registered_components
from opencda_marl.plugins import registry as plugin_registry
from opencda_marl.api.algorithms import register_algorithm, registered_algorithm_keys
from opencda_marl.api.objectives import BUILTIN_OBJECTIVE_PROFILES, register_objective_profile
from opencda_marl.api.preferences import BUILTIN_PREFERENCE_PROFILES, register_preference_profile
from opencda_marl.api.rewards import BUILTIN_REWARD_PROFILES, register_reward_profile


@pytest.fixture
def restore_registries():
    from opencda_marl.api import components as comp
    from opencda_marl.api import algorithms as alg

    snap = (
        dict(alg._SPECS), dict(alg._SOURCES),
        {k: dict(v) for k, v in comp._COMPONENTS.items()},
        {k: dict(v) for k, v in comp._COMPONENT_SOURCES.items()},
        dict(BUILTIN_REWARD_PROFILES), dict(BUILTIN_OBJECTIVE_PROFILES),
        dict(BUILTIN_PREFERENCE_PROFILES), dict(plugin_registry._JOURNAL),
    )
    yield
    alg._SPECS.clear(); alg._SPECS.update(snap[0])
    alg._SOURCES.clear(); alg._SOURCES.update(snap[1])
    comp._COMPONENTS.clear(); comp._COMPONENTS.update({k: dict(v) for k, v in snap[2].items()})
    comp._COMPONENT_SOURCES.clear(); comp._COMPONENT_SOURCES.update({k: dict(v) for k, v in snap[3].items()})
    BUILTIN_REWARD_PROFILES.clear(); BUILTIN_REWARD_PROFILES.update(snap[4])
    BUILTIN_OBJECTIVE_PROFILES.clear(); BUILTIN_OBJECTIVE_PROFILES.update(snap[5])
    BUILTIN_PREFERENCE_PROFILES.clear(); BUILTIN_PREFERENCE_PROFILES.update(snap[6])
    plugin_registry._JOURNAL.clear(); plugin_registry._JOURNAL.update(snap[7])


def test_describe_covers_every_kind(restore_registries):
    rows = plugins.describe_plugins(load=False)
    kinds = {r["kind"] for r in rows}
    assert {"algorithm", "planner", "agent", "reward_profile",
            "objective_profile", "preference_profile"} <= kinds


def test_one_registration_per_kind_lands_in_the_table(restore_registries):
    register_algorithm("unified_algo", label="U")(lambda cfg, *, state_dim, action_dim: None)
    register_component("planner", "unified_planner")(object)
    register_reward_profile("unified_reward", {"rewards": {}})
    register_objective_profile("unified_objective", {"objective_spec": {}})
    register_preference_profile("unified_pref", {"default": [0.5, 0.5]})

    rows = plugins.describe_plugins(load=False)
    got = {(r["kind"], r["key"]) for r in rows}
    assert {("algorithm", "unified_algo"), ("planner", "unified_planner"),
            ("reward_profile", "unified_reward"),
            ("objective_profile", "unified_objective"),
            ("preference_profile", "unified_pref")} <= got


def test_rollback_removes_every_kind_a_source_registered(restore_registries):
    plugin_registry.current_source = "bad_plugin"
    try:
        register_algorithm("rb_algo", label="RB")(lambda cfg, *, state_dim, action_dim: None)
        register_component("agent", "rb_agent")(object)
        register_reward_profile("rb_reward", {"rewards": {}})
    finally:
        plugin_registry.current_source = None

    removed = plugin_registry.rollback("bad_plugin")
    assert removed == 3
    assert "rb_algo" not in registered_algorithm_keys()
    assert "rb_agent" not in registered_components("agent")
    assert "rb_reward" not in BUILTIN_REWARD_PROFILES


def test_freeze_rejects_late_registration_and_thawed_lifts_it(restore_registries):
    plugin_registry._frozen = True
    try:
        with pytest.raises(plugins.RegistryFrozenError):
            register_reward_profile("late_profile", {"rewards": {}})
        with plugins.thawed():
            register_reward_profile("late_profile", {"rewards": {}})
        assert "late_profile" in BUILTIN_REWARD_PROFILES
    finally:
        plugin_registry._frozen = False


def test_reward_calculator_is_config_selected(restore_registries):
    from opencda_marl.api.components import component_key, register_component, resolve_component

    class TracingCalculator:
        def __init__(self, reward_params, reward_spec):
            self.reward_params = reward_params
            self.reward_spec = reward_spec

    register_component("reward_calculator", "tracing")(TracingCalculator)
    key = component_key({"components": {"reward_calculator": "tracing"}}, "reward_calculator")
    calc = resolve_component("reward_calculator", key)({"a": 1}, None)
    assert isinstance(calc, TracingCalculator) and calc.reward_params == {"a": 1}
    # default stays the engine's calculator
    from opencda_marl.builtin.rewards.calculator import RewardCalculator
    default_key = component_key({}, "reward_calculator")
    assert resolve_component("reward_calculator", default_key) is RewardCalculator


def test_observation_extractor_default_resolves(restore_registries):
    from opencda_marl.api.components import resolve_component
    from opencda_marl.builtin.extractors import ObservationExtractor

    assert resolve_component("observation_extractor", "default") is ObservationExtractor


def test_intent_provider_modes_resolve_via_registry(restore_registries):
    from opencda_marl.api.components import registered_components

    assert {"tao_fixed", "naive_random", "rule_based_path"} <= set(
        registered_components("intent_provider"))
