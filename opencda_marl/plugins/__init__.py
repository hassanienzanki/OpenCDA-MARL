"""The plugin system: one door for every replaceable part.

    from opencda_marl.plugins import describe_plugins, load_plugins

Registration happens through each kind's public function — they all share
this registry's discovery, journal, rollback and freeze:

- ``register_algorithm``            (opencda_marl.api.algorithms)
- ``register_component``            (opencda_marl.api.components — planner, agent, …)
- ``register_reward_profile``       (opencda_marl.api.rewards)
- ``register_objective_profile``    (opencda_marl.api.objectives)
- ``register_preference_profile``   (opencda_marl.api.preferences)
"""
from opencda_marl.plugins.loader import (  # noqa: F401
    PLUGIN_ENTRY_POINT_GROUP,
    PLUGIN_ENV,
    POLICY_API_VERSION,
    load_plugins,
    plugin_failures,
)
from opencda_marl.plugins.registry import (  # noqa: F401
    RegistryFrozenError,
    describe_plugins,
    freeze,
    is_frozen,
    kinds,
    register_kind,
    thawed,
)

__all__ = [
    "PLUGIN_ENTRY_POINT_GROUP",
    "PLUGIN_ENV",
    "POLICY_API_VERSION",
    "RegistryFrozenError",
    "describe_plugins",
    "freeze",
    "is_frozen",
    "kinds",
    "load_plugins",
    "plugin_failures",
    "register_kind",
    "thawed",
]
