"""The policy contract: what an algorithm must provide to run in this framework.

The framework owns the pipeline (observations, rewards, legality masks,
episode lifecycle); an algorithm owns only the decision. Anything satisfying
:class:`Policy` can be registered and run without editing framework
internals — see ``opencda_marl/api/algorithms.py`` for registration and
``docs/marl/custom_algorithms.md`` for a worked example.

Two protocols are defined:

``Policy``
    The speed controller. Required.

``IntentPolicy``
    Optional one-shot route-intent capability. A policy that also implements
    these methods is automatically eligible for ``action_space.mode:
    path_speed``; one that does not is rejected with a clear error instead of
    silently misbehaving. Capability is probed with :func:`supports_intent`,
    never with ``isinstance`` against a concrete class list.

Both are ``runtime_checkable``, so probes verify method *presence* only —
Python cannot check signatures at runtime. The signatures below are the
contract; getting them wrong surfaces as a normal TypeError at the call site.

Note on units: ``select_action``/``select_actions`` return a target speed in
km/h (or ``None`` to defer to the rule-based agent during warmup). Moving
that conversion into an env-side action space is future work; until then it
is part of the contract and is documented here rather than left implicit.

Single-agent policies (registered with ``multi_agent=False``) act on one
agent's own state instead: ``select_action(state, training)`` and
``store_transition(state, action, reward, next_state, done)``. Their
``select_action`` also returns the target speed, unless the policy declares
``returns_action_index = True``: then it returns an index into its
``speed_actions`` list, the framework drives that list entry, and the index
is what ``store_transition`` receives back as ``action``. The declaration is
read, never inferred: owning a ``speed_actions`` table does not make a
policy index-returning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class PolicyCapabilities:
    """Static facts about a policy family, used for framework dispatch.

    Declared once at registration instead of being re-derived from the
    algorithm's class or name at each call site (the previous design needed
    an ``isinstance`` tuple in the manager's hot path plus two separate
    name-keyed lookup tables, all of which had to be edited to add one
    algorithm).
    """

    #: Multi-agent policies receive ``(multi_agent_obs, ego_agent_id)``;
    #: single-agent ones receive a single ``state``. Drives which dispatch
    #: branch the manager takes.
    multi_agent: bool = True
    #: Reward given to each agent: its own ("per_agent") or the team mean
    #: ("team_mean", used by preference-conditioned cooperative families).
    reward_aggregation: str = "per_agent"
    #: Whether the learned one-shot route head is available for this family.
    #: Verified against the built instance at wiring time.
    supports_intent: bool = False


@runtime_checkable
class Policy(Protocol):
    """A speed controller the framework can drive.

    Attributes the framework reads:

    ``max_action``
        Upper clamp for the returned target speed, in km/h.
    ``supports_batched_actions``
        When ``True`` the manager calls :meth:`select_actions` once per
        simulation step instead of :meth:`select_action` per agent.
    """

    max_action: float
    supports_batched_actions: bool

    def select_action(
        self,
        multi_agent_obs: Dict[str, np.ndarray],
        ego_agent_id: Any,
        training: bool = True,
    ) -> Optional[float]:
        """Return the ego agent's target speed in km/h, or ``None`` to defer
        control to the rule-based agent (e.g. during warmup)."""
        ...

    def store_transition(
        self,
        multi_agent_obs: Dict[str, np.ndarray],
        ego_agent_id: Any,
        action: float,
        reward: Any,
        next_multi_agent_obs: Dict[str, np.ndarray],
        done: bool,
    ) -> None:
        """Record one transition. ``reward`` is a float, or a vector when the
        policy declares ``supports_vector_rewards``."""
        ...

    def update(self) -> Dict[str, float]:
        """Run one learning update; return metrics for logging."""
        ...

    def reset_episode(self) -> None:
        """Clear per-episode state."""
        ...

    def get_training_info(self) -> Dict[str, Any]:
        """Return diagnostics for the monitor and metrics sinks."""
        ...

    def save(self, filepath: str) -> None:
        ...

    def load(self, filepath: str) -> None:
        ...


@runtime_checkable
class IntentPolicy(Protocol):
    """Optional capability: a masked one-shot route-intent head.

    The framework supplies the legality mask; the policy only chooses among
    the legal actions and learns from the terminal outcome. Selection happens
    once per vehicle at spawn, and the reward arrives at that vehicle's
    terminal event — hence the pending/finalize split rather than a normal
    per-step transition.
    """

    def enable_route_intent(
        self,
        config: Dict[str, Any],
        *,
        state_dim: int,
        num_actions: int,
        pref_dim: int,
        reward_dim: int,
    ) -> None:
        """Construct the route head. Called once during manager setup."""
        ...

    def select_intent(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        preference: np.ndarray,
        *,
        training: bool = True,
    ):
        """Choose one legal route intent. ``action_mask`` is a boolean array;
        illegal entries must never be selected."""
        ...

    def store_pending(self, **kwargs) -> None:
        """Hold a decision until its vehicle reaches a terminal outcome."""
        ...

    def has_pending(self, actor_id: Any) -> bool:
        ...

    def finalize_terminal(self, **kwargs) -> None:
        """Convert a pending decision into a training sample."""
        ...

    def update_route_intent(self) -> Dict[str, float]:
        ...


#: Methods that make a policy intent-capable. Kept explicit so the probe
#: reports precisely which piece a partial implementation is missing.
_INTENT_METHODS = (
    "enable_route_intent",
    "select_intent",
    "store_pending",
    "has_pending",
    "finalize_terminal",
    "update_route_intent",
)


def supports_intent(policy: Any) -> bool:
    """True when *policy* implements the full :class:`IntentPolicy` capability."""
    return all(callable(getattr(policy, name, None)) for name in _INTENT_METHODS)


def missing_intent_methods(policy: Any) -> List[str]:
    """Names of the :class:`IntentPolicy` methods *policy* does not provide."""
    return [n for n in _INTENT_METHODS if not callable(getattr(policy, n, None))]


__all__ = [
    "Policy",
    "IntentPolicy",
    "PolicyCapabilities",
    "supports_intent",
    "missing_intent_methods",
]
