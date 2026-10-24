"""When the route decision is taken: at spawn, or at junction approach.

A one-shot decision made the instant a vehicle appears is convenient but
unrealistic — the vehicle commits to a path before it can see the traffic
it will interact with. The decision zone moves the same one-shot decision
to the junction approach: the vehicle drives its demand route until it is
within ``distance_m`` of the junction, decides once there on fresh
observations, and executes the outcome.

The trigger is deliberately a small, backend-independent object: both
environments own *when* they can measure the distance and *how* they
re-route, but they must agree on the policy, on one decision per vehicle,
and on what ``decision_pending`` means for an external training loop.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Set

SPAWN = "spawn"
APPROACH = "approach"
TRIGGERS = (SPAWN, APPROACH)

#: Default zone radius. Wide enough that the local planner has not yet
#: consumed the junction into its lookahead, close enough that the
#: observation describes the interaction the vehicle is about to have.
DEFAULT_DISTANCE_M = 30.0


class DecisionZone:
    """Decides which vehicles are due for their one-shot route decision."""

    def __init__(self, trigger: str = SPAWN,
                 distance_m: float = DEFAULT_DISTANCE_M):
        trigger = str(trigger or SPAWN).lower()
        if trigger not in TRIGGERS:
            raise ValueError(
                f"unknown route-intent trigger {trigger!r}; expected one of "
                f"{list(TRIGGERS)}")
        self.trigger = trigger
        self.distance_m = float(distance_m)
        #: Vehicles awaiting their decision (approach mode only).
        self._pending: Set[Any] = set()
        #: Vehicles that have already decided — the decision is one-shot.
        self._decided: Set[Any] = set()

    # ------------------------------------------------------------------ #
    @property
    def decides_at_spawn(self) -> bool:
        return self.trigger == SPAWN

    def register(self, actor_id: Any) -> None:
        """A vehicle appeared: in approach mode it now owes a decision."""
        if self.trigger == APPROACH and actor_id not in self._decided:
            self._pending.add(actor_id)

    def due(self, actor_id: Any, distance_to_junction: float | None) -> bool:
        """True when this vehicle should decide *now*."""
        if actor_id not in self._pending:
            return False
        if distance_to_junction is None:
            return False
        return float(distance_to_junction) <= self.distance_m

    def mark_decided(self, actor_id: Any) -> None:
        self._pending.discard(actor_id)
        self._decided.add(actor_id)

    def forget(self, actor_id: Any) -> None:
        """The vehicle left the scenario."""
        self._pending.discard(actor_id)
        self._decided.discard(actor_id)

    def is_pending(self, actor_id: Any) -> bool:
        return actor_id in self._pending

    @property
    def pending(self) -> Set[Any]:
        return set(self._pending)

    def reset_episode(self) -> None:
        self._pending.clear()
        self._decided.clear()

    def as_info(self) -> Dict[str, Any]:
        """What an external training loop needs to see each step."""
        return {
            "trigger": self.trigger,
            "distance_m": self.distance_m,
            "decision_pending": sorted(str(a) for a in self._pending),
        }


def build_decision_zone(intent_cfg: Mapping[str, Any] | None) -> DecisionZone:
    """Build the zone from the ``route_intent`` block (default: spawn)."""
    cfg = intent_cfg or {}
    zone_cfg = cfg.get("decision_zone") or {}
    trigger = zone_cfg.get("trigger", cfg.get("trigger", SPAWN))
    distance = zone_cfg.get("distance_m", cfg.get("decision_zone_m",
                                                  DEFAULT_DISTANCE_M))
    return DecisionZone(trigger=trigger, distance_m=float(distance))
