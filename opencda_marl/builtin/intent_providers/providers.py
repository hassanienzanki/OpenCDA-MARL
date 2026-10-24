'''
Author       : AXIBA leolihao@arizona.edu
FilePath     : /OpenCDA-MARL/opencda_marl/traffic/intent_provider.py
Description  : Decision-provider boundary for one-shot post-spawn semantic
               route-intent selection. Built-in baselines: disabled,
               tao_fixed, naive_random, rule_based_path; the MARL/MORL
               provider plugs into the same callable shape.
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
from __future__ import annotations

import random
from typing import Any, Callable, Dict, Optional, Sequence

from opencda_marl.api.route_intent import (
    ALL_INTENTS,
    RouteIntent,
    RouteIntentCandidate,
    RouteIntentDecision,
    build_candidates_with_diagnostics,
    build_decision,
    build_mask,
    candidate_pressure_score,
    candidate_traffic_features,
    resolve_route,
)


from opencda_marl.api.intent import IntentProvider  # noqa: F401 - re-export
_INTENT_RANK = {intent: idx for idx, intent in enumerate(ALL_INTENTS)}


def _int_or_default(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _decide(
    intent: RouteIntent,
    candidates: Sequence[RouteIntentCandidate],
    mask: Dict[RouteIntent, bool],
    original_route_id: int,
    diagnostics: Dict[str, Any],
) -> RouteIntentDecision:
    candidate = resolve_route(
        intent,
        candidates,
        mask,
        original_route_id=original_route_id,
    )
    return build_decision(
        requested_intent=intent,
        candidate=candidate,
        mask=mask,
        original_route_id=int(original_route_id),
        diagnostics=diagnostics,
    )


def _build_context_candidates_and_mask(
    planner: Any,
    spawn_event,
) -> Optional[tuple]:
    """Build the candidate set + feasibility mask from a SpawnEvent."""
    if planner is None:
        return None
    metadata = spawn_event.metadata or {}
    entry_direction = metadata.get("entry_direction")
    entry_lane_idx = _int_or_default(
        metadata.get("entry_lane_idx", metadata.get("lane_id"))
    )
    if entry_direction is None or entry_lane_idx < 0:
        return None

    candidates, diagnostics = build_candidates_with_diagnostics(
        planner=planner,
        junction_id=int(spawn_event.junction_id),
        entry_direction=str(entry_direction),
        entry_lane_idx=entry_lane_idx,
        original_route_id=int(spawn_event.route_id),
    )
    if not candidates:
        return None
    mask = build_mask(candidates)
    return candidates, mask, diagnostics


def make_tao_fixed_provider(planner: Any) -> IntentProvider:
    """Keep-current provider: always pick ``keep-current`` (the planner-default route).

    Equivalent in behavior to disabling the hook, but the decision is
    recorded for evaluation parity with the other providers.
    """
    def _provider(context: Dict[str, Any]) -> Optional[RouteIntentDecision]:
        event = context["event"]
        ctx = _build_context_candidates_and_mask(planner, event)
        if ctx is None:
            return None
        candidates, mask, diagnostics = ctx
        return _decide(
            RouteIntent.KEEP, candidates, mask, int(event.route_id), diagnostics
        )
    return _provider


def make_naive_random_provider(
    planner: Any, seed: int = 0, rng: Optional[random.Random] = None
) -> IntentProvider:
    """Naive baseline: uniformly sample among legal masked intents.

    Uses an internal seeded RNG so episodes are reproducible across runs.
    Does NOT use nearby-vehicle information; that would conflate the
    baseline with learned-risk reasoning. ``rng`` lets a rebuilt provider
    (new planner after a world reload) continue the previous draw sequence
    instead of restarting it from ``seed``; the RNG is exposed as
    ``provider.rng`` for that hand-over.
    """
    rng = rng if rng is not None else random.Random(int(seed))

    def _provider(context: Dict[str, Any]) -> Optional[RouteIntentDecision]:
        event = context["event"]
        ctx = _build_context_candidates_and_mask(planner, event)
        if ctx is None:
            return None
        candidates, mask, diagnostics = ctx
        legal = [intent for intent, ok in mask.items() if ok]
        if not legal:
            chosen = RouteIntent.KEEP
        else:
            chosen = rng.choice(legal)
        return _decide(chosen, candidates, mask, int(event.route_id), diagnostics)
    _provider.rng = rng
    return _provider


def _score_rule_candidate(
    candidate: RouteIntentCandidate,
    *,
    active_adapters: Sequence[Any],
    ego_adapter: Any = None,
    preference: Any,
    queue_speed_threshold_kmh: float,
    route_change_penalty: float,
) -> tuple[float, Any]:
    features = candidate_traffic_features(
        candidate,
        active_adapters,
        ego_adapter=ego_adapter,
        queue_speed_threshold_kmh=float(queue_speed_threshold_kmh),
    )
    score = candidate_pressure_score(
        candidate,
        features,
        preference=preference,
        route_change_penalty=float(route_change_penalty),
    )
    return score, features


def make_rule_based_path_provider(
    planner: Any,
    *,
    queue_speed_threshold_kmh: float = 5.0,
    route_change_penalty: float = 0.05,
) -> IntentProvider:
    """Rule-based path optimizer over the same direction-preserving mask.

    The provider makes one deterministic spawn-time decision using currently
    active traffic assignments. It never changes the requested exit direction:
    candidates come from ``build_candidates_with_diagnostics`` and therefore
    share the learned path optimizer's structural feasibility mask.
    """
    def _provider(context: Dict[str, Any]) -> Optional[RouteIntentDecision]:
        event = context["event"]
        ctx = _build_context_candidates_and_mask(planner, event)
        if ctx is None:
            return None
        candidates, mask, diagnostics = ctx
        active_adapters = list(context.get("active_adapters") or [])
        ego_adapter = context.get("adapter")
        preference = (event.metadata or {}).get("preference", [])
        legal_candidates = [c for c in candidates if mask.get(c.intent, False)]
        if not legal_candidates:
            return _decide(
                RouteIntent.KEEP, candidates, mask, int(event.route_id), diagnostics
            )

        scored = []
        for candidate in legal_candidates:
            score, traffic_features = _score_rule_candidate(
                candidate,
                active_adapters=active_adapters,
                ego_adapter=ego_adapter,
                preference=preference,
                queue_speed_threshold_kmh=float(queue_speed_threshold_kmh),
                route_change_penalty=float(route_change_penalty),
            )
            scored.append((
                score,
                candidate.intent != RouteIntent.KEEP,
                candidate.exit_lane_idx,
                candidate.route_id,
                _INTENT_RANK.get(candidate.intent, 99),
                traffic_features,
                candidate,
            ))
        scored.sort(key=lambda item: item[:5])
        chosen_score = float(scored[0][0])
        chosen_features = scored[0][-2]
        chosen = scored[0][-1]
        original_features = None
        for candidate in candidates:
            if candidate.is_original:
                original_features = candidate_traffic_features(
                    candidate,
                    active_adapters,
                    ego_adapter=ego_adapter,
                    queue_speed_threshold_kmh=float(queue_speed_threshold_kmh),
                )
                break
        diagnostics = dict(diagnostics)
        diagnostics["fallback_reason"] = diagnostics.get("fallback_reason", "")
        return build_decision(
            requested_intent=chosen.intent,
            candidate=chosen,
            mask=mask,
            original_route_id=int(event.route_id),
            diagnostics=diagnostics,
            selected_candidate_score=chosen_score,
            selected_candidate_traffic=chosen_features.as_metadata(),
            original_candidate_traffic=(
                original_features.as_metadata() if original_features is not None else {}
            ),
            candidate_resolution="rule_candidate_aware",
        )
    return _provider


def _build_tao_fixed(intent_cfg, planner, *, marl_manager=None, training=True,
                     previous=None):
    return make_tao_fixed_provider(planner)


def _build_naive_random(intent_cfg, planner, *, marl_manager=None, training=True,
                        previous=None):
    seed = int(intent_cfg.get("seed", 0))
    return make_naive_random_provider(
        planner, seed=seed, rng=getattr(previous, "rng", None)
    )


def _build_rule_based_path(intent_cfg, planner, *, marl_manager=None,
                           training=True, previous=None):
    return make_rule_based_path_provider(
        planner,
        queue_speed_threshold_kmh=float(
            intent_cfg.get("queue_speed_threshold_kmh", 5.0)
        ),
        route_change_penalty=float(intent_cfg.get("route_change_penalty", 0.05)),
    )
