"""Per-episode mutable state, grouped so reset cannot drift.

Every field here lives exactly one episode. Environments hold an instance at
``self.episode`` and reset by swapping in a fresh instance — replacing the
previous pattern of hand-written ``.clear()`` calls scattered through
``reset()``, where forgetting one (or adding a field without one) silently
carried state across episodes. Tests that build environments via
``object.__new__`` get a complete, consistent state object from one line.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class SumoEpisodeState:
    """Episode-scoped registries and caches of :class:`SumoMARLEnv`."""

    # {veh_id: vehicle_info} — vehicles observed alive this episode.
    active_vehicles: Dict[str, Any] = field(default_factory=dict)
    # Vehicles that arrived at their destination.
    departed_vehicles: Set[str] = field(default_factory=set)
    # Vehicles involved in collisions.
    collided_vehicles: Set[str] = field(default_factory=set)
    # Vehicles that reached any terminal outcome (arrival/collision/removal).
    terminal_agents: Set[str] = field(default_factory=set)
    # Post-step observation snapshot reused as the next step's S_t.
    observation_cache: Dict[str, Dict] = field(default_factory=dict)
    #: vehicle id -> (population, learning), resolved once per episode: a
    #: vehicle's population is fixed by its spawn event for its whole life.
    populations: Dict[str, tuple] = field(default_factory=dict)
    # Route-intent terminal bookkeeping: ids already finalized, and ids
    # queued to finalize after this step's learning update.
    terminal_intent_notified: Set[str] = field(default_factory=set)
    route_intent_terminal_queue: List[str] = field(default_factory=list)
