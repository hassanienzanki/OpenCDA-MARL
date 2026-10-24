"""The public surface: contracts and registration doors.

Everything an external package needs is importable from here — the data
contracts the kernel emits, the ``Policy`` contract algorithms implement,
and the registration functions for every plugin kind. External code never
imports ``opencda_marl._kernel``; the kernel never imports a concrete
plugin — both sides meet at this package and the plugin registry.

Exports resolve lazily so ``import opencda_marl.api`` stays cheap.
"""

_LAZY = {
    # contracts
    "StepEvent": ("opencda_marl.api.events", "StepEvent"),
    "RewardSpec": ("opencda_marl.api.contracts", "RewardSpec"),
    "RewardOutput": ("opencda_marl.api.contracts", "RewardOutput"),
    "NormalizedObservation": ("opencda_marl.api.contracts", "NormalizedObservation"),
    "Policy": ("opencda_marl.api.policy", "Policy"),
    "PathDecision": ("opencda_marl.api.path_action", "PathDecision"),
    "HeadSpec": ("opencda_marl.api.path_action", "HeadSpec"),
    "IntentPolicy": ("opencda_marl.api.policy", "IntentPolicy"),
    "PolicyCapabilities": ("opencda_marl.api.policy", "PolicyCapabilities"),
    # registration doors
    "register_algorithm": ("opencda_marl.api.algorithms", "register_algorithm"),
    "register_component": ("opencda_marl.api.components", "register_component"),
    "register_reward_profile": ("opencda_marl.api.rewards", "register_reward_profile"),
    "register_objective_profile": ("opencda_marl.api.objectives", "register_objective_profile"),
    "register_preference_profile": ("opencda_marl.api.preferences", "register_preference_profile"),
    # resolution helpers
    "resolve_component": ("opencda_marl.api.components", "resolve_component"),
    "component_key": ("opencda_marl.api.components", "component_key"),
    "BASELINE_AGENT_TYPES": ("opencda_marl.api.components", "BASELINE_AGENT_TYPES"),
    "resolve_algorithm_spec": ("opencda_marl.api.algorithms", "resolve_algorithm_spec"),
    "AlgorithmSpec": ("opencda_marl.api.algorithms", "AlgorithmSpec"),
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
