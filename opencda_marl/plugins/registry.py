"""The one plugin registry: every replaceable part, one door, one table.

The framework is the stable environment; everything researchers replace —
algorithms, planners, agent types, reward calculators, observation
extractors, intent providers, reward/objective/preference profiles — is a
*plugin* under a ``kind`` and a ``key``. Kinds keep their own storage and
validation (an algorithm registration validates capabilities; a profile
registration is a data row), but they all report here, so there is exactly
one discovery mechanism, one enumeration, one API-version rollback, and
one freeze.

Stability contract (deliberately unlike hot-reload plugin hosts): plugins
load once at startup; after the shell calls :func:`freeze` the registry
rejects further registrations, so the set of behaviours in a run is fixed
and reproducible. Tests use :func:`thawed`.
"""
from __future__ import annotations

import importlib
from typing import Callable, Dict, List, Tuple

#: kind -> (module, adapter attribute). Adapters resolve lazily so a slim
#: install never imports a kind it does not use. Each adapter is a dict:
#: ``{"describe": fn() -> [row], "remove": fn(key) -> None}``.
_KIND_PROVIDERS: Dict[str, Tuple[str, str]] = {
    "algorithm": ("opencda_marl.api.algorithms", "_kind_adapter"),
    "reward_profile": ("opencda_marl.api.rewards", "_kind_adapter"),
    "objective_profile": ("opencda_marl.api.objectives", "_kind_adapter"),
    "preference_profile": ("opencda_marl.api.preferences", "_kind_adapter"),
}

#: source label -> [(kind, key)] registered while that source was loading;
#: the loader uses this to roll back a plugin whose API version mismatches.
_JOURNAL: Dict[str, List[Tuple[str, str]]] = {}

_frozen = False
_thaw_depth = 0

#: Set by the loader around a plugin import so registrations are attributed.
current_source: str | None = None


class RegistryFrozenError(RuntimeError):
    """Raised when a registration arrives after :func:`freeze`."""


def guard(kind: str, key: str) -> str | None:
    """The single hook every kind's register path calls.

    Enforces the freeze, journals the registration under the loading
    plugin's source (for rollback), and returns that source label (or None
    when the registration is not part of a plugin import).
    """
    if _frozen and _thaw_depth == 0:
        raise RegistryFrozenError(
            f"plugin registry is frozen; cannot register {kind} {key!r}. "
            "Plugins load once at startup — register at import time.")
    if current_source is not None:
        _JOURNAL.setdefault(current_source, []).append((str(kind), str(key)))
    return current_source


def _adapter(kind: str) -> dict:
    module, attr = _KIND_PROVIDERS[kind]
    return getattr(importlib.import_module(module), attr)()


def register_kind(kind: str, module: str, adapter_attr: str) -> None:
    """Declare a new kind backed by ``module.adapter_attr()`` (lazily).

    An external package may add axes the engine has never heard of; the
    kind then shows up in :func:`describe_plugins` like any built-in one.
    """
    _KIND_PROVIDERS.setdefault(str(kind), (module, adapter_attr))


def kinds() -> Tuple[str, ...]:
    return tuple(_KIND_PROVIDERS)


def describe_plugins(load: bool = True) -> List[dict]:
    """One row per registered plugin across every kind.

    Component-backed kinds (planner, agent, reward_calculator, …, plus any
    axis an external package invents) are discovered from the component
    registry itself, so a new kind needs no engine release to be listed.
    """
    if load:
        from opencda_marl.plugins.loader import load_plugins

        load_plugins()
    rows: List[dict] = []
    for kind in _KIND_PROVIDERS:
        try:
            for row in _adapter(kind)["describe"]():
                rows.append({"kind": kind, **row})
        except Exception as exc:  # noqa: BLE001 - a broken kind must not hide the others
            rows.append({"kind": kind, "key": "<kind unavailable>",
                         "source": f"error: {exc}"})
    try:
        from opencda_marl.api import components as _components

        for kind in _components.component_kinds():
            for row in _components._kind_rows(kind):
                rows.append({"kind": kind, **row})
    except Exception as exc:  # noqa: BLE001
        rows.append({"kind": "components", "key": "<kinds unavailable>",
                     "source": f"error: {exc}"})
    return rows


def rollback(source: str) -> int:
    """Remove everything ``source`` registered (all kinds). Returns count."""
    removed = 0
    for kind, key in _JOURNAL.pop(source, []):
        try:
            if kind in _KIND_PROVIDERS:
                _adapter(kind)["remove"](key)
            else:
                from opencda_marl.api import components as _components

                _components._kind_remove(kind, key)
            removed += 1
        except Exception:  # noqa: BLE001 - best effort; the entry may be gone
            pass
    return removed


def freeze() -> None:
    """Reject registrations from now on — the run's behaviour set is fixed."""
    global _frozen
    _frozen = True


def is_frozen() -> bool:
    return _frozen and _thaw_depth == 0


class thawed:
    """Context manager lifting the freeze (tests, tooling)."""

    def __enter__(self):
        global _thaw_depth
        _thaw_depth += 1
        return self

    def __exit__(self, *exc):
        global _thaw_depth
        _thaw_depth -= 1
        return False
