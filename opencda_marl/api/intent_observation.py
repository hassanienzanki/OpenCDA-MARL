"""Build the one-shot spawn observation used by the intent policy.

Observation layout (fixed order, contiguous slices):

    [0..4)   entry_direction_onehot          (N, S, E, W)
    [4..5)   entry_lane_idx_norm             (idx / 4, clamped)
    [5..9)   legal_mask                      (LEFT, STRAIGHT, RIGHT, KEEP)
    [9..13)  original_route_type_onehot      (left, straight, right, other)
    [13..17) outgoing_traffic_count          (per exit direction)
    [17..21) outgoing_queue_estimate         (per exit direction)
    [21..49) candidate_traffic_features      (4 intents x 7 features)
    [49..]   preference_vector               (optional, ``preference.size``)
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from opencda_marl.api.route_intent import (
    ALL_INTENTS,
    adapter_route_key,
    adapter_speed_kmh,
    build_candidates,
    build_mask,
    candidate_pressure_score,
    candidate_traffic_features,
)

_DIRECTIONS: tuple[str, ...] = ("north", "south", "east", "west")
_ROUTE_TYPES: tuple[str, ...] = ("left", "straight", "right", "other")

ENTRY_DIR_SLICE = slice(0, 4)
ENTRY_LANE_SLICE = slice(4, 5)
LEGAL_MASK_SLICE = slice(5, 9)
ROUTE_TYPE_SLICE = slice(9, 13)
TRAFFIC_COUNT_SLICE = slice(13, 17)
QUEUE_SLICE = slice(17, 21)
CANDIDATE_FEATURES_PER_INTENT = 7
CANDIDATE_TRAFFIC_SLICE = slice(
    21, 21 + len(ALL_INTENTS) * CANDIDATE_FEATURES_PER_INTENT
)
SPAWN_OBS_DIM_NO_PREF = CANDIDATE_TRAFFIC_SLICE.stop
_MAX_LANES_NORM = 4.0
_QUEUE_SPEED_THRESHOLD_KMH = 5.0


def _onehot(value: str, options: Sequence[str]) -> np.ndarray:
    out = np.zeros(len(options), dtype=np.float32)
    if value in options:
        out[options.index(value)] = 1.0
    return out


def _normalize_route_type(rt: str) -> str:
    return rt if rt in {"left", "straight", "right"} else "other"


def _adapter_exit_direction(adapter: Any) -> str | None:
    exit_dir, _, _ = adapter_route_key(adapter)
    if exit_dir in _DIRECTIONS:
        return exit_dir
    return None


def _adapter_speed_kmh(adapter: Any) -> float:
    return adapter_speed_kmh(adapter)


def _best_candidate_features_for_intent(
    intent,
    candidates,
    active_adapters,
    preference,
    *,
    ego_adapter=None,
) -> np.ndarray:
    intent_candidates = [c for c in candidates if c.intent == intent]
    if not intent_candidates:
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)

    scored = []
    for candidate in intent_candidates:
        features = candidate_traffic_features(
            candidate,
            active_adapters,
            ego_adapter=ego_adapter,
            queue_speed_threshold_kmh=_QUEUE_SPEED_THRESHOLD_KMH,
        )
        score = candidate_pressure_score(
            candidate, features, preference=preference)
        scored.append((score, candidate.exit_lane_idx,
                      candidate.route_id, features))
    scored.sort(key=lambda item: item[:3])
    score, _, _, features = scored[0]
    return np.asarray(
        features.to_feature_vector(pressure_score=float(score)),
        dtype=np.float32,
    )


def build_spawn_observation(
    event: Any,
    planner: Any,
    active_adapters: Iterable[Any],
    preference: np.ndarray,
    *,
    ego_adapter: Any | None = None,
) -> np.ndarray:
    metadata = event.metadata or {}
    entry_direction = str(metadata.get("entry_direction", ""))
    entry_lane_idx = int(metadata.get(
        "entry_lane_idx", metadata.get("lane_id", -1)))
    route_type = _normalize_route_type(str(metadata.get("route_type", "")))
    active_adapters = list(active_adapters or [])
    candidates = []

    parts: list[np.ndarray] = []
    parts.append(_onehot(entry_direction, _DIRECTIONS))
    lane_norm = max(0.0, min(1.0, (entry_lane_idx if entry_lane_idx >= 0 else 0)
                             / _MAX_LANES_NORM))
    parts.append(np.array([lane_norm], dtype=np.float32))

    mask_block = np.zeros(4, dtype=np.float32)
    if entry_lane_idx >= 0 and entry_direction:
        candidates = build_candidates(
            planner=planner,
            junction_id=int(event.junction_id),
            entry_direction=entry_direction,
            entry_lane_idx=entry_lane_idx,
            original_route_id=int(event.route_id),
        )
        if candidates:
            mask = build_mask(candidates)
            for i, intent in enumerate(ALL_INTENTS):
                mask_block[i] = 1.0 if mask.get(intent, False) else 0.0
    parts.append(mask_block)

    parts.append(_onehot(route_type, _ROUTE_TYPES))

    traffic_count = np.zeros(4, dtype=np.float32)
    queue_count = np.zeros(4, dtype=np.float32)
    for adapter in active_adapters:
        exit_dir = _adapter_exit_direction(adapter)
        if exit_dir is None:
            continue
        idx = _DIRECTIONS.index(exit_dir)
        traffic_count[idx] += 1.0
        if _adapter_speed_kmh(adapter) < _QUEUE_SPEED_THRESHOLD_KMH:
            queue_count[idx] += 1.0
    parts.append(traffic_count)
    parts.append(queue_count)

    candidate_features = []
    for intent in ALL_INTENTS:
        candidate_features.append(
            _best_candidate_features_for_intent(
                intent,
                candidates,
                active_adapters,
                preference,
                ego_adapter=ego_adapter,
            )
        )
    parts.append(np.concatenate(
        candidate_features).astype(np.float32, copy=False))

    base = np.concatenate(parts).astype(np.float32, copy=False)
    pref = np.asarray(preference, dtype=np.float32) if preference is not None \
        else np.zeros(0, dtype=np.float32)
    if pref.size == 0:
        return base
    return np.concatenate([base, pref]).astype(np.float32, copy=False)
