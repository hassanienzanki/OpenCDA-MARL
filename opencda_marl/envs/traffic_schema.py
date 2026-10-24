"""Traffic-schema spec and adapters for the standalone schema environment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class TrafficSchemaSpec:
    """Derived schema inputs for standalone schema-driven traffic scenarios."""

    approach_lambdas: dict[str, float]
    movement_priors: dict[str, np.ndarray]
    episode_horizon_s: int


DIRECTIONS = ("north", "east", "south", "west")
MOVEMENTS = ("left", "straight", "right")
DEFAULT_MOVEMENT_PRIOR = np.asarray([0.30, 0.40, 0.30], dtype=np.float32)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _normalize_probs(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    total = float(arr.sum())
    if total <= 0.0:
        return DEFAULT_MOVEMENT_PRIOR.copy()
    return arr / total


def _load_flows(traffic_config_path: Path) -> list[dict]:
    with traffic_config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    scenario = payload.get("scenario", {})
    traffic = scenario.get("traffic", {})
    flows = traffic.get("flows", [])
    if not isinstance(flows, list):
        raise ValueError("Expected scenario.traffic.flows to be a list.")
    return flows


def _derive_approach_lambdas(flows: list[dict], episode_horizon_s: int) -> dict[str, float]:
    rates = {direction: 0.0 for direction in DIRECTIONS}
    for flow in flows:
        direction = str(flow.get("direction", "")).strip().lower()
        if direction not in rates:
            continue
        rates[direction] += float(flow.get("rate_vph", 0.0))
    return {
        direction: (rate * float(episode_horizon_s) / 3600.0)
        for direction, rate in rates.items()
    }


def _derive_movement_priors(recording_path: Path | None) -> dict[str, np.ndarray]:
    priors = {direction: DEFAULT_MOVEMENT_PRIOR.copy()
              for direction in DIRECTIONS}
    if recording_path is None:
        return priors

    with recording_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload.get("events", [])

    counts = {
        direction: np.zeros(len(MOVEMENTS), dtype=np.float32)
        for direction in DIRECTIONS
    }
    for event in events:
        metadata = event.get("metadata", {}) or {}
        direction = str(metadata.get("entry_direction", "")).strip().lower()
        movement = str(metadata.get("route_type", "")).strip().lower()
        if direction not in counts or movement not in MOVEMENTS:
            continue
        counts[direction][MOVEMENTS.index(movement)] += 1.0

    for direction, direction_counts in counts.items():
        priors[direction] = _normalize_probs(direction_counts)
    return priors


def load_traffic_schema_spec(
    traffic_config_path: str,
    recording_path: str | None = None,
    episode_horizon_s: int = 120,
) -> TrafficSchemaSpec:
    """Load a standalone MOMARL traffic spec from OpenCDA schema inputs."""

    config_path = _resolve_path(traffic_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Traffic config not found: {config_path}")

    recording = None
    if recording_path is not None:
        recording = _resolve_path(recording_path)
        if not recording.exists():
            raise FileNotFoundError(f"Recording file not found: {recording}")

    flows = _load_flows(config_path)
    approach_lambdas = _derive_approach_lambdas(flows, int(episode_horizon_s))
    movement_priors = _derive_movement_priors(recording)
    return TrafficSchemaSpec(
        approach_lambdas=approach_lambdas,
        movement_priors=movement_priors,
        episode_horizon_s=int(episode_horizon_s),
    )
