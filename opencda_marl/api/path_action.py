"""The path-action boundary: what the lifecycle needs from a path decision.

A *path action* is the one-shot choice a vehicle makes about how it leaves a
junction. The framework owns the lifecycle around it -- when the decision is
due, applying it to the planner or the SUMO route, recording it, and giving
the learner a masked discrete head -- and owns none of its semantics. Which
alternatives exist, what the mask means, and what makes an outcome legal are
the provider's, and a provider is a plugin of kind ``intent_provider``.

Three things cross that boundary:

- :class:`PathDecision` -- what the lifecycle reads back. A decision says
  whether it may be executed (:attr:`~PathDecision.admissible`), where the
  vehicle should now go, and how to describe itself for the trace. It does
  *not* say what an "intent" is; that word never appears in this contract.
- :class:`HeadSpec` -- how wide the learner's head must be for this
  provider. A provider that wants a different observation or a different
  number of alternatives declares it here instead of the kernel guessing.
- :func:`resolve_head_spec` -- how the lifecycle asks.

A provider declares its head shape by giving its registered factory a
``head_spec`` attribute: either a :class:`HeadSpec` or a callable taking the
provider's config block and returning one.

    from opencda_marl.api import register_component
    from opencda_marl.api.path_action import HeadSpec

    def build_my_provider(cfg, planner, *, marl_manager=None,
                          training=True, previous=None):
        ...

    build_my_provider.head_spec = lambda cfg: HeadSpec(
        observation_dim=12 * int(cfg.get("n_max", 4)),
        num_actions=int(cfg.get("n_max", 4)),
    )
    register_component("intent_provider", "my_path")(build_my_provider)

Selected with ``MARL.action_space.path.provider: my_path``; everything else
under ``path`` reaches the provider unchanged as its own config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class PathDecision(Protocol):
    """One vehicle's path decision, as the lifecycle reads it.

    Implementations carry whatever else their provider needs; only these three
    members are the framework's business.
    """

    @property
    def admissible(self) -> bool:
        """Whether the lifecycle may execute this decision.

        The provider owns the invariant. A decision that fails its own
        invariant is recorded and skipped: the vehicle keeps the route the
        demand gave it, and the run continues.
        """
        ...

    @property
    def destination(self) -> Any:
        """Where the vehicle should now be routed: a transform with a
        ``location``. ``None`` leaves the demand destination in place."""
        ...

    def as_metadata(self) -> Dict[str, Any]:
        """A JSON-safe description, for spawn metadata and the trace."""
        ...


@dataclass(frozen=True)
class HeadSpec:
    """The shape of the learner's one-shot discrete head for a provider.

    ``observation_dim`` excludes the preference vector, which the lifecycle
    appends. ``num_actions`` is how many alternatives the head spans; the
    provider's mask makes the unusable ones illegal.
    """

    observation_dim: int
    num_actions: int

    def __post_init__(self) -> None:
        if int(self.observation_dim) <= 0:
            raise ValueError("HeadSpec.observation_dim must be positive")
        if int(self.num_actions) <= 0:
            raise ValueError("HeadSpec.num_actions must be positive")


def decision_is_admissible(decision: Any) -> bool:
    """Whether ``decision`` may be executed.

    A decision that declares ``admissible`` owns the answer. One that does
    not is admissible, so a provider that never rejects anything needs to
    write nothing.
    """
    value = getattr(decision, "admissible", None)
    return True if value is None else bool(value)


def decision_destination(decision: Any) -> Any:
    """The transform to route to, or ``None`` to keep the demand route."""
    destination = getattr(decision, "destination", None)
    if destination is not None:
        return destination
    # Providers built on the framework's own candidate helpers expose the
    # chosen candidate instead; read it rather than making them adapt.
    candidate = getattr(decision, "selected_candidate", None)
    return getattr(candidate, "dest_tf", None) if candidate is not None else None


def decision_route_update(decision: Any) -> Optional[Dict[str, Any]]:
    """The route change a decision asks for, or ``None`` for no change.

    The SUMO backend re-routes by editing the spawn event, so it needs more
    than a destination: the route id it should now carry and the demand
    fields that describe it. A decision may answer directly with
    ``route_update()``; one built on the framework's candidate helpers is
    read from its chosen candidate instead.
    """
    direct = getattr(decision, "route_update", None)
    if callable(direct):
        update = direct()
        return dict(update) if update else None
    candidate = getattr(decision, "selected_candidate", None)
    if candidate is None:
        return None
    route_id = getattr(candidate, "route_id", None)
    if route_id is None:
        return None
    return {
        "route_id": int(route_id),
        "destination": getattr(candidate, "dest_tf", None),
        "route_type": getattr(candidate, "route_type", None),
        "exit_direction": getattr(candidate, "exit_direction", None),
        "exit_lane_idx": getattr(candidate, "exit_lane_idx", None),
    }


def resolve_head_spec(provider_key: str, config: Optional[Dict[str, Any]] = None
                      ) -> Optional[HeadSpec]:
    """The head shape declared by the provider registered as ``provider_key``.

    Returns ``None`` when the provider declares none, which lets the
    lifecycle keep whatever default it had.
    """
    from opencda_marl.api.components import resolve_component

    try:
        factory = resolve_component("intent_provider", str(provider_key))
    except ValueError:
        return None
    spec = getattr(factory, "head_spec", None)
    if spec is None:
        return None
    if callable(spec) and not isinstance(spec, HeadSpec):
        spec = spec(dict(config or {}))
    if spec is None:
        return None
    if not isinstance(spec, HeadSpec):
        raise TypeError(
            f"intent_provider {provider_key!r} declared head_spec of type "
            f"{type(spec).__name__}; expected {HeadSpec.__name__}")
    return spec


__all__ = ["PathDecision", "HeadSpec", "decision_is_admissible",
           "decision_destination", "decision_route_update", "resolve_head_spec"]
