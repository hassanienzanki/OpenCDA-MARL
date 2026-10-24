"""Component registry: pluggable parts beyond algorithms.

The framework's promise is that research never edits the engine. Algorithms
have their own registry (:mod:`opencda_marl.api.algorithms`); everything
else that an environment composes — today the junction route **planner**
and the vehicle **agent** types, later further axes — registers here under
a *kind* and a *key* and is selected by configuration::

    from opencda_marl.api.components import register_component

    @register_component("planner", "my_planner")
    class MyPlanner:                         # same constructor as MARLPlanner
        def __init__(self, world, config): ...

    # configs/...: scenario.traffic.planner.type: my_planner

    @register_component("agent", "my_driver")
    def build_my_driver(vehicle, carla_map, config):
        return MyDriverAgent(vehicle, carla_map, config.get("my_driver", {}))

    # configs/...: agents.agent_type: my_driver

The ``agent`` kind is resolved on the CARLA backend only, one agent per
spawned vehicle. On SUMO a population key is a label: vehicles outside the
learner's population drive under SUMO's own car-following model, and a
registered agent factory never runs.

Registrations run at import time, so a component arrives exactly like an
algorithm does: through the plugin entry points, ``MARL.plugins``,
``OPENCDA_MARL_PLUGINS``, or the ``builtin/algorithms/contrib`` drop-in. The
engine's own defaults are rows in :data:`_BUILTINS` (resolved lazily, so a
slim SUMO install never imports the CARLA-side agent stack to resolve a
planner). Resolution is loud: an unknown key lists what exists and which
plugins failed to load.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

#: kind -> key -> factory registered by plugins (a class works: it is
#: called with the kind's constructor signature).
_COMPONENTS: Dict[str, Dict[str, Callable[..., Any]]] = {}
_COMPONENT_SOURCES: Dict[str, Dict[str, str]] = {}
#: kind -> key -> optional config validator (see register_component).
_COMPONENT_VALIDATORS: Dict[str, Dict[str, Callable[..., Any]]] = {}

#: Agent types that run without a learner (benchmark baselines). Kernel
#: code needs this distinction without importing the builtin factory. The
#: set is closed: a registered agent type is not treated as a baseline.
BASELINE_AGENT_TYPES = ("behavior", "vanilla", "rule_based")

#: Constructor signatures by kind, documented for registrants.
KIND_SIGNATURES = {
    "planner": "(world, config)",
    "agent": "(vehicle, carla_map, config)",
    "reward_calculator": "(reward_params, reward_spec)",
    "observation_extractor": "(config, algorithm_name, *, multi_agent)",
    "intent_provider": "(intent_cfg, planner, *, marl_manager, training, previous)",
    "context_encoder": "(input_size, hidden_size, num_layers)",
}

#: The engine's default components, as (kind, key) -> "module", "attribute".
#: Declarative rather than import-side-effect so resolution never depends on
#: which engine module happens to be imported first.
_BUILTINS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("planner", "marl"): ("opencda_marl.builtin.planners.marl_planner", "MARLPlanner"),
    ("agent", "behavior"): ("opencda_marl.builtin.agents.agent_factory", "build_behavior_agent"),
    ("agent", "vanilla"): ("opencda_marl.builtin.agents.agent_factory", "build_vanilla_agent"),
    ("agent", "rule_based"): ("opencda_marl.builtin.agents.agent_factory", "build_rule_based_agent"),
    ("agent", "marl"): ("opencda_marl.builtin.agents.agent_factory", "build_marl_agent"),
    ("reward_calculator", "default"): ("opencda_marl.builtin.rewards.calculator", "RewardCalculator"),
    ("observation_extractor", "default"): ("opencda_marl.builtin.extractors", "ObservationExtractor"),
    ("intent_provider", "tao_fixed"): ("opencda_marl.builtin.intent_providers.providers", "_build_tao_fixed"),
    ("intent_provider", "naive_random"): ("opencda_marl.builtin.intent_providers.providers", "_build_naive_random"),
    ("intent_provider", "rule_based_path"): ("opencda_marl.builtin.intent_providers.providers", "_build_rule_based_path"),
    ("context_encoder", "lstm"): ("opencda_marl.builtin.algorithms.encoders", "LSTMEncoder"),
}


def component_key(config, kind: str, default: str = "default") -> str:
    """The configured key for ``kind``: ``<block>.components.<kind>``.

    ``config`` is the block that owns the selection (the ``MARL`` block for
    loop parts). Missing or empty means the engine default."""
    section = config.get("components", {}) if hasattr(config, "get") else {}
    value = section.get(kind) if hasattr(section, "get") else None
    return str(value) if value else default


def _builtin_factory(kind: str, key: str):
    target = _BUILTINS.get((kind, key))
    if target is None:
        return None
    import importlib

    return getattr(importlib.import_module(target[0]), target[1])


def register_component(kind: str, key: str, *, override: bool = False,
                       source: str | None = None,
                       validate: Callable[[Any], None] | None = None):
    """Register a factory (or class) under ``kind``/``key``.

    ``override=True`` is required to replace an existing registration —
    built-ins included — so a typo cannot silently shadow one. Unknown
    kinds are accepted (new axes must not need an engine release); the
    documented ones and their constructor signatures are in
    :data:`KIND_SIGNATURES`.

    ``validate`` is the component's own config check: a callable taking
    the config block that selected the component and raising ``ValueError``
    on a bad configuration. It runs when the framework resolves the
    component with that block, so a misconfigured plugin fails at
    initialize with the plugin's own message instead of deep in the loop.
    """
    def _decorator(factory):
        table = _COMPONENTS.setdefault(str(kind), {})
        name = str(key)
        if (name in table or (str(kind), name) in _BUILTINS) and not override:
            raise ValueError(
                f"{kind} component {name!r} is already registered; pass "
                "override=True to replace it deliberately.")
        from opencda_marl.plugins import registry as _plugin_registry

        journal_source = _plugin_registry.guard(str(kind), name)
        table[name] = factory
        _COMPONENT_SOURCES.setdefault(str(kind), {})[name] = (
            source or journal_source
            or getattr(factory, "__module__", "unknown"))
        if validate is not None:
            _COMPONENT_VALIDATORS.setdefault(str(kind), {})[name] = validate
        return factory

    return _decorator


def resolve_component(kind: str, key: str,
                      config: Any = None) -> Callable[..., Any]:
    """Return the factory for ``kind``/``key`` or raise a loud error.

    Plugin registrations win over the engine defaults (they got there via
    ``override=True``); the defaults resolve lazily from :data:`_BUILTINS`.
    When ``config`` is given and the component registered a ``validate``
    hook, the hook runs against it before the factory is returned.
    """
    from opencda_marl.plugins.loader import load_plugins, plugin_failures

    load_plugins()
    factory = _COMPONENTS.get(str(kind), {}).get(str(key))
    if factory is None:
        factory = _builtin_factory(str(kind), str(key))
    if factory is not None:
        validator = _COMPONENT_VALIDATORS.get(str(kind), {}).get(str(key))
        if validator is not None and config is not None:
            try:
                validator(config)
            except ValueError as exc:
                raise ValueError(
                    f"{kind} component {key!r} rejected its configuration: {exc}"
                ) from exc
        return factory
    known = ", ".join(sorted(registered_components(str(kind)))) or "none registered"
    hint = ""
    failures = plugin_failures()
    if failures:
        hint = " Plugins that failed to import: " + "; ".join(
            f"{k} ({v})" for k, v in failures.items())
    raise ValueError(
        f"Unknown {kind} component {key!r}. Registered: {known}.{hint}")


def registered_components(kind: str) -> tuple[str, ...]:
    """Every key for ``kind``: engine defaults plus plugin registrations."""
    from opencda_marl.plugins.loader import load_plugins

    load_plugins()
    keys = [k for (b_kind, k) in _BUILTINS if b_kind == str(kind)]
    keys.extend(k for k in _COMPONENTS.get(str(kind), {}) if k not in keys)
    return tuple(keys)


def describe_components() -> "list[dict]":
    """One row per component, for CLIs and docs."""
    from opencda_marl.plugins.loader import load_plugins

    load_plugins()
    rows = []
    kinds = {kind for kind, _ in _BUILTINS} | set(_COMPONENTS)
    for kind in sorted(kinds):
        for key in registered_components(kind):
            source = _COMPONENT_SOURCES.get(kind, {}).get(key)
            if source is None and (kind, key) in _BUILTINS:
                source = "builtin"
            rows.append({"kind": kind, "key": key, "source": source or "unknown"})
    return rows


def _kind_rows(kind: str) -> "list[dict]":
    rows = []
    for key in registered_components(kind):
        source = _COMPONENT_SOURCES.get(kind, {}).get(key)
        if source is None and (kind, key) in _BUILTINS:
            source = "builtin"
        rows.append({"key": key, "source": source or "unknown"})
    return rows


def _kind_remove(kind: str, key: str) -> None:
    _COMPONENTS.get(kind, {}).pop(key, None)
    _COMPONENT_SOURCES.get(kind, {}).pop(key, None)
    _COMPONENT_VALIDATORS.get(kind, {}).pop(key, None)


def component_kinds() -> "tuple[str, ...]":
    """Every kind the component registry knows: builtins plus registered."""
    kinds = {kind for kind, _ in _BUILTINS} | set(_COMPONENTS)
    return tuple(sorted(kinds))


__all__ = ["register_component", "resolve_component", "registered_components",
           "describe_components", "component_key", "KIND_SIGNATURES"]
