"""The built-in learned path provider: the semantic turn action on a trainable head.

This is the framework's own example of the ``intent_provider`` seam. A plugin
with different path semantics registers its own provider through
``opencda_marl.api.path_action`` and never touches this module.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

from opencda_marl.api.route_intent import (
    RouteIntent,
    RouteIntentDecision,
    build_candidates_with_diagnostics,
    build_decision,
    build_mask,
    candidate_pressure_score,
    candidate_traffic_features,
    resolve_route,
)

from opencda_marl.api.intent_observation import build_spawn_observation


_FORCE_INTENT_MAP = {
    "left": RouteIntent.LEFT,
    "straight": RouteIntent.STRAIGHT,
    "right": RouteIntent.RIGHT,
    "keep-current": RouteIntent.KEEP,
    "keep_current": RouteIntent.KEEP,
    "keep": RouteIntent.KEEP,
}


def make_marl_intent_provider(
    *,
    algorithm,
    get_active_adapters: Callable[[], Iterable[Any]],
    training: bool = True,
    force_intent: Optional[str] = None,
) -> Callable[[Dict[str, Any]], Optional[RouteIntentDecision]]:
    """Return an IntentProvider callable wired to a trainable algorithm.

    Args:
        algorithm: an object implementing the route-intent interface. In the
            hybrid MOMARL path this is the same MARL algorithm object that also
            emits speed actions.
        get_active_adapters: callable returning current active vehicle adapters.
        training: pass ``False`` for eval rollouts.
        force_intent: optional override that collapses the feasibility mask to
            a single intent, for diagnostics and what-if evaluation; values:
            ``'left'``, ``'straight'``, ``'right'``, ``'keep-current'``.
            When set, the policy never samples a different intent (the masked
            categorical sees only one legal option).
    """
    forced_intent_enum: Optional[RouteIntent] = None
    if force_intent is not None:
        forced_intent_enum = _FORCE_INTENT_MAP.get(str(force_intent).lower())
        if forced_intent_enum is None:
            raise ValueError(
                f"unknown force_intent={force_intent!r}; expected one of "
                f"{sorted(_FORCE_INTENT_MAP.keys())}"
            )

    def _int_or_default(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _provider(context: Dict[str, Any]) -> Optional[RouteIntentDecision]:
        event = context["event"]
        planner = context.get("planner")
        adapter = context.get("adapter")
        if planner is None or adapter is None:
            return None

        metadata = event.metadata or {}
        entry_direction = metadata.get("entry_direction")
        entry_lane_idx = _int_or_default(
            metadata.get("entry_lane_idx", metadata.get("lane_id"))
        )
        if entry_direction is None or entry_lane_idx < 0:
            return None

        candidates, diagnostics = build_candidates_with_diagnostics(
            planner=planner,
            junction_id=int(event.junction_id),
            entry_direction=str(entry_direction),
            entry_lane_idx=entry_lane_idx,
            original_route_id=int(event.route_id),
        )
        if not candidates:
            return None
        mask = build_mask(candidates)
        if forced_intent_enum is not None:
            if mask.get(forced_intent_enum, False):
                mask = {intent: (intent is forced_intent_enum)
                        for intent in mask}
            else:
                diagnostics = dict(diagnostics)
                diagnostics["fallback_reason"] = (
                    diagnostics.get("fallback_reason")
                    or f"forced_intent_illegal:{forced_intent_enum.value}"
                )
                mask = {intent: (intent is RouteIntent.KEEP)
                        for intent in mask}
        mask_array = np.array(
            [mask[i] for i in (RouteIntent.LEFT, RouteIntent.STRAIGHT,
                               RouteIntent.RIGHT, RouteIntent.KEEP)],
            dtype=bool,
        )
        pref_dim = int(getattr(algorithm, "pref_dim", 0) or 0)
        preference = np.asarray(metadata.get("preference", []),
                                dtype=np.float32)
        if pref_dim <= 0:
            preference = np.zeros(0, dtype=np.float32)
        active_adapters = list(get_active_adapters())
        obs = build_spawn_observation(
            event,
            planner,
            active_adapters,
            preference,
            ego_adapter=adapter,
        )
        action_index, log_prob, value = algorithm.select_intent(
            obs, mask_array, preference, training=training,
        )
        intent = (RouteIntent.LEFT, RouteIntent.STRAIGHT,
                  RouteIntent.RIGHT, RouteIntent.KEEP)[int(action_index)]
        candidate = resolve_route(
            intent,
            candidates,
            mask,
            original_route_id=int(event.route_id),
            active_adapters=active_adapters,
            ego_adapter=adapter,
            preference=preference,
        )
        candidate_traffic = candidate_traffic_features(
            candidate,
            active_adapters,
            ego_adapter=adapter,
        )
        original_traffic = None
        for original_candidate in candidates:
            if original_candidate.is_original:
                original_traffic = candidate_traffic_features(
                    original_candidate,
                    active_adapters,
                    ego_adapter=adapter,
                )
                break
        candidate_score = candidate_pressure_score(
            candidate,
            candidate_traffic,
            preference=preference,
        )
        algorithm.store_pending(
            actor_id=adapter.actor_id,
            observation=obs, action_index=action_index,
            action_mask=mask_array, log_prob=log_prob,
            value=value, preference=preference,
        )
        return build_decision(
            requested_intent=intent,
            candidate=candidate,
            mask=mask,
            original_route_id=int(event.route_id),
            diagnostics=diagnostics,
            selected_candidate_score=candidate_score,
            selected_candidate_traffic=candidate_traffic.as_metadata(),
            original_candidate_traffic=(
                original_traffic.as_metadata() if original_traffic is not None else {}
            ),
            candidate_resolution="candidate_aware",
        )

    return _provider
