"""Aggregation helpers for route-intent decisions surfaced at evaluation time."""
from __future__ import annotations

from typing import Any, Dict, Iterable

_INTENTS = ("left", "straight", "right", "keep-current")
_ROUTE_TYPES = ("left", "straight", "right", "other")
_OUTCOMES = ("success", "collision", "planner_failure", "other")


def _empty_outcome_cell() -> Dict[str, int]:
    cell = {o: 0 for o in _OUTCOMES}
    cell["total"] = 0
    return cell


def _empty_cross_tab() -> Dict[str, Dict[str, Dict[str, int]]]:
    return {intent: {rt: _empty_outcome_cell() for rt in _ROUTE_TYPES}
            for intent in _INTENTS}


def _read_mask(entry: Dict[str, Any]) -> tuple:
    """``(legal_count, width, selection_was_legal, keep_only)`` for any mask.

    The framework's own path providers write a mapping keyed by intent name;
    an external provider owns its own vocabulary and may write a plain
    sequence indexed by action, with the chosen index under ``selected_slot``.
    This summary describes decisions it does not define, so it reads both and
    never assumes either -- a provider's metadata must not be able to end an
    episode.
    """
    mask = entry.get("mask")
    if isinstance(mask, dict):
        legal = sum(1 for v in mask.values() if v)
        selected = bool(mask.get(entry.get("selected_intent"), False))
        keep_only = legal == 1 and bool(mask.get("keep-current", False))
        return legal, len(mask) or 1, selected, keep_only
    if isinstance(mask, (list, tuple)):
        legal = sum(1 for v in mask if v)
        index = entry.get("selected_slot")
        try:
            selected = bool(mask[int(index)])
        except (TypeError, ValueError, IndexError):
            selected = True          # no index to check: not an invalid action
        return legal, len(mask) or 1, selected, False
    return 0, 1, True, False


def summarize_intent_decisions(decisions: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce a stream of ``RouteIntentDecision.as_metadata()`` dicts.

    Each entry may additionally carry:
    - ``selected_route_type``: one of {left, straight, right, other};
      missing -> "other".
    - ``outcome``: one of {success, collision, planner_failure, ...};
      missing or unknown string -> "other".
    """
    distribution: Dict[str, int] = {
        "left": 0, "straight": 0, "right": 0, "keep-current": 0,
    }
    keep_count = 0
    invalid_attempts = 0
    legal_sum = 0
    total = 0
    cross_tab = _empty_cross_tab()
    direction_preserved_violations = 0
    fallback_count = 0
    keep_only_count = 0
    candidate_total_sum = 0.0
    candidate_same_sum = 0.0
    mask_sparsity_sum = 0.0
    route_change_count = 0
    selected_exit_lane_distribution: Dict[str, int] = {}
    queued_lane_avoidance_count = 0
    queued_lane_avoidance_opportunities = 0
    selected_queued_lane_sum = 0.0
    original_queued_lane_sum = 0.0

    for entry in decisions:
        intent = entry.get("selected_intent")
        if intent in distribution:
            distribution[intent] += 1
        legal_count, mask_width, selection_legal, keep_only = _read_mask(entry)
        if not selection_legal:
            invalid_attempts += 1
        legal_sum += legal_count
        if keep_only:
            keep_only_count += 1
        keep_count += 1 if entry.get("used_keep_current") else 0
        direction_preserved = entry.get(
            "direction_preserved",
            entry.get(
                "same_exit_direction_preserved",
                entry.get("same_destination_preserved", True),
            ),
        )
        direction_preserved_violations += 0 if direction_preserved else 1
        fallback_count += 1 if entry.get("fallback_reason") else 0
        route_changed = bool(
            entry.get(
                "route_changed",
                entry.get("selected_route_id") != entry.get("original_route_id"),
            )
        )
        route_change_count += 1 if route_changed else 0
        if entry.get("selected_exit_lane_idx") is not None:
            lane_key = str(entry.get("selected_exit_lane_idx"))
            selected_exit_lane_distribution[lane_key] = (
                selected_exit_lane_distribution.get(lane_key, 0) + 1
            )
        selected_traffic = entry.get("selected_candidate_traffic") or {}
        original_traffic = entry.get("original_candidate_traffic") or {}
        selected_queued = float(
            selected_traffic.get("queued_same_exit_lane_count", 0.0) or 0.0
        )
        original_queued = float(
            original_traffic.get("queued_same_exit_lane_count", 0.0) or 0.0
        )
        selected_queued_lane_sum += selected_queued
        original_queued_lane_sum += original_queued
        if original_queued > 0.0:
            queued_lane_avoidance_opportunities += 1
            if selected_queued < original_queued:
                queued_lane_avoidance_count += 1
        candidate_total_sum += float(entry.get("candidate_count_total") or 0.0)
        candidate_same_sum += float(
            entry.get(
                "candidate_count_same_exit_direction",
                entry.get("candidate_count_same_destination", 0.0),
            ) or 0.0
        )
        if "mask_sparsity" in entry:
            mask_sparsity_sum += float(entry.get("mask_sparsity") or 0.0)
        else:
            mask_sparsity_sum += 1.0 - (legal_count / float(mask_width))
        total += 1

        if intent not in _INTENTS:
            continue
        rt = entry.get("selected_route_type") or "other"
        if rt not in _ROUTE_TYPES:
            rt = "other"
        outcome = entry.get("outcome") or "other"
        if outcome not in _OUTCOMES:
            outcome = "other"
        cross_tab[intent][rt][outcome] += 1
        cross_tab[intent][rt]["total"] += 1

    denom = max(1, total)
    return {
        "intent_distribution": distribution,
        "keep_current_rate": keep_count / denom,
        "invalid_action_attempts": invalid_attempts,
        "mask_sparsity": (legal_sum / denom) / 4.0,
        "mask_legal_fraction": (legal_sum / denom) / 4.0,
        "mean_mask_sparsity": mask_sparsity_sum / denom,
        "keep_only_mask_count": keep_only_count,
        "keep_only_mask_rate": keep_only_count / denom,
        "direction_preserved_violations": direction_preserved_violations,
        "direction_preserved_violation_rate": (
            direction_preserved_violations / denom
        ),
        "same_exit_direction_violations": direction_preserved_violations,
        "same_exit_direction_violation_rate": (
            direction_preserved_violations / denom
        ),
        # Backward-compatible aliases: the former same_destination names.
        "same_destination_violations": direction_preserved_violations,
        "same_destination_violation_rate": direction_preserved_violations / denom,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / denom,
        "route_change_count": route_change_count,
        "route_change_rate": route_change_count / denom,
        "selected_exit_lane_distribution": selected_exit_lane_distribution,
        "queued_lane_avoidance_count": queued_lane_avoidance_count,
        "queued_lane_avoidance_opportunities": queued_lane_avoidance_opportunities,
        "queued_lane_avoidance_rate": (
            queued_lane_avoidance_count / max(1, queued_lane_avoidance_opportunities)
        ),
        "mean_selected_queued_same_exit_lane_count": (
            selected_queued_lane_sum / denom
        ),
        "mean_original_queued_same_exit_lane_count": (
            original_queued_lane_sum / denom
        ),
        "mean_candidate_count_total": candidate_total_sum / denom,
        "mean_candidate_count_same_exit_direction": candidate_same_sum / denom,
        "mean_candidate_count_same_destination": candidate_same_sum / denom,
        "route_type_outcome": cross_tab,
    }
