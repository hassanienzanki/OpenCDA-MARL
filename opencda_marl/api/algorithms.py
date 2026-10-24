"""Algorithm registry: canonical names, capabilities, and construction.

Adding an algorithm means writing one module and registering it — no edits
to the manager, the factory, the strategy table, or the path-action wiring::

    from opencda_marl.api.algorithms import register_algorithm

    @register_algorithm("myalgo", label="MyAlgo", supports_intent=True)
    def build_myalgo(config, *, state_dim, action_dim):
        return MyAlgorithm(config, state_dim, action_dim)

The returned object must satisfy ``opencda_marl.api.policy.Policy``
(and ``IntentPolicy`` when ``supports_intent=True``). Framework dispatch
reads :class:`~opencda_marl.api.policy.PolicyCapabilities` from the
spec instead of testing concrete classes, so registration is the only
integration point. See ``docs/marl/custom_algorithms.md``.

Names outside the registered set are rejected loudly: a run must not
silently fall back to a differently-configured family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from opencda_marl.api.policy import PolicyCapabilities

#: A factory takes the merged algorithm config plus the resolved dimensions
#: and returns a Policy.
PolicyFactory = Callable[..., Any]


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    #: Display name for tables, logs and the monitor.
    label: str
    config_section: str
    capabilities: PolicyCapabilities = field(default_factory=PolicyCapabilities)
    factory: Optional[PolicyFactory] = None
    #: True when the framework should inject ``reward_dim`` into the config
    #: for vector-critic (objective-vector) runs.
    vector_rewards: bool = False

    def build(self, config: Dict[str, Any], *, state_dim: int, action_dim: int):
        if self.factory is None:
            raise NotImplementedError(
                f"{self.label} has no registered factory")
        return self.factory(config, state_dim=state_dim, action_dim=action_dim)


#: key -> spec. The engine imports no learners: the first-party
#: ``opencda_marl.builtin.algorithms`` registers through the same
#: :func:`register_algorithm` call and entry point any external package
#: uses, so there is exactly one way in.
_SPECS: Dict[str, AlgorithmSpec] = {}
#: key -> where the registration came from ("builtin", a plugin module name,
#: an entry-point name).
_SOURCES: Dict[str, str] = {}


# Names that are not registered learners: resolving or registering one is a
# clear error instead of a silent fallback to a different family.
def register_algorithm(
    key: str,
    *,
    label: str | None = None,
    config_section: str | None = None,
    multi_agent: bool = True,
    reward_aggregation: str = "per_agent",
    supports_intent: bool = False,
    vector_rewards: bool = False,
    override: bool = False,
    source: str | None = None,
):
    """Register a policy factory under a canonical name.

    Use as a decorator on the factory function, or call the returned
    registrar directly: ``register_algorithm("x", ...)(factory)``.
    ``label`` is the display name (default: the upper-cased key);
    ``config_section`` defaults to *key* — the ``MARL.<section>`` block passed
    to the factory. ``override=True`` is required to replace an existing
    registration, so a typo cannot silently shadow a built-in family.
    ``source`` labels the origin for :func:`describe_algorithms`; plugins
    loaded through :func:`load_plugins` are labelled automatically.
    """
    if reward_aggregation not in {"per_agent", "team_mean"}:
        raise ValueError(
            f"reward_aggregation must be 'per_agent' or 'team_mean', "
            f"got {reward_aggregation!r}")

    def _decorator(factory: PolicyFactory) -> PolicyFactory:
        name = str(key)
        if name in _SPECS and not override:
            raise ValueError(
                f"Algorithm '{name}' is already registered; pass override=True "
                "to replace it deliberately.")
        from opencda_marl.plugins import registry as _plugin_registry

        journal_source = _plugin_registry.guard("algorithm", name)
        _SPECS[name] = AlgorithmSpec(
            key=name,
            label=label or name.upper(),
            config_section=config_section or name,
            capabilities=PolicyCapabilities(
                multi_agent=multi_agent,
                reward_aggregation=reward_aggregation,
                supports_intent=supports_intent,
            ),
            factory=factory,
            vector_rewards=vector_rewards,
        )
        _SOURCES[name] = source or journal_source or _module_of(factory)
        return factory

    return _decorator


#: The loader, the entry-point group, the env var and the API version live
#: in :mod:`opencda_marl.plugins`; these names re-export them so algorithm
#: code keeps one import site.
from opencda_marl.plugins.loader import (  # noqa: E402
    PLUGIN_ENTRY_POINT_GROUP,
    PLUGIN_ENV,
    POLICY_API_VERSION,
    load_plugins,
)


def _module_of(factory) -> str:
    return getattr(factory, "__module__", None) or "unknown"


def _kind_adapter() -> dict:
    """The unified registry's view of the algorithm kind."""
    def _describe():
        return [{"key": key, "source": _SOURCES.get(key, "unknown")}
                for key in _SPECS]

    def _remove(key: str):
        _SPECS.pop(key, None)
        _SOURCES.pop(key, None)

    return {"describe": _describe, "remove": _remove}


def describe_algorithms() -> "list[dict]":
    """One row per registered algorithm, for CLIs and docs."""
    load_plugins()
    rows = []
    for key, spec in _SPECS.items():
        rows.append({
            "key": key,
            "label": spec.label,
            "config_section": spec.config_section,
            "multi_agent": spec.capabilities.multi_agent,
            "reward_aggregation": spec.capabilities.reward_aggregation,
            "supports_intent": spec.capabilities.supports_intent,
            "vector_rewards": spec.vector_rewards,
            "source": _SOURCES.get(key, "unknown"),
        })
    return rows


def registered_algorithm_keys() -> tuple[str, ...]:
    """Every currently registered name, built-in and external."""
    load_plugins()
    return tuple(_SPECS)


def resolve_algorithm_spec(name: str | None) -> AlgorithmSpec:
    """Return the registered algorithm spec or raise on an unknown name."""
    known = ", ".join(registered_algorithm_keys())
    if name is None:
        raise ValueError(f"MARL.algorithm is required and must be one of {known}")
    key = str(name)
    try:
        spec = _SPECS[key]
    except KeyError as exc:
        from opencda_marl.plugins.loader import plugin_failures

        failures = plugin_failures()
        hint = ""
        if failures:
            hint = " Plugins that failed to import: " + "; ".join(
                f"{k} ({v})" for k, v in failures.items())
        raise ValueError(
            f"Unsupported MARL.algorithm={name!r}. Use one of {known}.{hint}"
        ) from exc
    return spec


def algorithm_label(name: str) -> str:
    """Display name of a registered algorithm."""
    return resolve_algorithm_spec(name).label


def _canonical_keys() -> tuple[str, ...]:
    """Backwards-compatible view used by config validation and tests."""
    return registered_algorithm_keys()


__all__ = [
    "AlgorithmSpec",
    "PolicyCapabilities",
    "describe_algorithms",
    "load_plugins",
    "PLUGIN_ENTRY_POINT_GROUP",
    "POLICY_API_VERSION",
    "PLUGIN_ENV",
    "register_algorithm",
    "registered_algorithm_keys",
    "resolve_algorithm_spec",
    "algorithm_label",
]


def _main() -> int:
    """``python -m opencda_marl.api.algorithms``: list registered algorithms."""
    import importlib
    import os

    # Under ``-m`` this file runs as ``__main__`` while registrations land in
    # the canonical module object; read from that one.
    reg = importlib.import_module("opencda_marl.api.algorithms")
    failures = reg.load_plugins()
    rows = reg.describe_algorithms()
    print(f"{'key':12s} {'label':12s} {'agg':10s} {'intent':6s} {'vector':6s} source")
    for r in rows:
        print(f"{r['key']:12s} {r['label']:12s} {r['reward_aggregation']:10s} "
              f"{'yes' if r['supports_intent'] else '-':6s} {'yes' if r['vector_rewards'] else '-':6s} "
              f"{r['source']}")
    if failures:
        print("\nplugins that failed to import:")
        for name, err in failures.items():
            print(f"  {name}: {err}")
    print(f"\nplugin sources: entry points '{PLUGIN_ENTRY_POINT_GROUP}', "
          f"${PLUGIN_ENV}={os.environ.get(PLUGIN_ENV, '')!r}, MARL.plugins in configs")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
