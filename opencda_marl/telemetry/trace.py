"""Lightweight route-action trace writer for route visualizations."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return default


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _location_dict(location: Any) -> dict[str, float]:
    return {
        "x": float(getattr(location, "x", 0.0)),
        "y": float(getattr(location, "y", 0.0)),
        "z": float(getattr(location, "z", 0.0)),
    }


def _rotation_dict(rotation: Any) -> dict[str, float]:
    return {
        "pitch": float(getattr(rotation, "pitch", 0.0)),
        "yaw": float(getattr(rotation, "yaw", 0.0)),
        "roll": float(getattr(rotation, "roll", 0.0)),
    }


def _transform_dict(transform: Any) -> dict[str, dict[str, float]]:
    return {
        "location": _location_dict(getattr(transform, "location", None)),
        "rotation": _rotation_dict(getattr(transform, "rotation", None)),
    }


def _event_dict(event: Any) -> dict[str, Any]:
    return {
        "step": int(getattr(event, "step", -1)),
        "event_id": str(getattr(event, "event_id", "")),
        "vehicle_id": str(getattr(event, "vehicle_id", "")),
        "event_type": str(getattr(event, "event_type", "")),
        "metadata": _json_safe(getattr(event, "metadata", None) or {}),
    }


class RouteActionTraceWriter:
    """Write per-step traffic state for route-action visual evidence.

    The writer is intentionally passive: it reads observations and planner route
    metadata after each simulator step, but does not alter control decisions.
    """

    def __init__(self, config: Any, coordinator: Any):
        self.enabled = bool(_cfg_get(config, "enabled", False))
        self.sample_every = max(1, int(_cfg_get(config, "sample_every", 1)))
        self.include_routes = bool(_cfg_get(config, "include_routes", True))
        root = _cfg_get(config, "root_dir", None) or _cfg_get(config, "dir", None)
        self.root_dir = Path(root or "artifacts/traces")
        self.steps_path = self.root_dir / "steps.jsonl"
        self._fh = None
        self._last_key: tuple[int, int] | None = None

        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._fh = self.steps_path.open("a", encoding="utf-8", buffering=1)
            self.write_metadata(coordinator)
            if self.include_routes:
                self.write_routes(coordinator)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def write_metadata(self, coordinator: Any) -> None:
        scenario = _cfg_get(getattr(coordinator, "config", {}), "scenario", {}) or {}
        metadata = {
            "trace_version": 1,
            "scenario_type": _cfg_get(scenario, "type", ""),
            "map": _json_safe(_cfg_get(scenario, "map", {}) or {}),
            "sample_every": self.sample_every,
        }
        (self.root_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_routes(self, coordinator: Any) -> None:
        planner = getattr(
            getattr(getattr(coordinator, "scenario_manager", None), "traffic_manager", None),
            "planner",
            None,
        )
        if planner is None:  # SUMO: the env owns the traffic manager
            planner = getattr(getattr(getattr(coordinator, "marl_env", None),
                                      "traffic_manager", None), "planner", None)
        junctions = getattr(planner, "_junctions", {}) or {}
        records: list[dict[str, Any]] = []
        for junction_id, data in junctions.items():
            for route_id, route in (data.get("routes") or {}).items():
                records.append({
                    "junction_id": int(junction_id),
                    "route_id": int(route_id),
                    "entry_direction": str(route.get("entry_direction", "")),
                    "exit_direction": str(route.get("exit_direction", "")),
                    "entry_lane_idx": int(route.get("entry_lane_idx", -1)),
                    "exit_lane_idx": int(route.get("exit_lane_idx", -1)),
                    "route_type": str(route.get("route_type", "")),
                    "spawn_tf": _transform_dict(route.get("spawn_tf")),
                    "dest_tf": _transform_dict(route.get("dest_tf")),
                })
        (self.root_dir / "routes.json").write_text(
            json.dumps(records, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def record_step(self, coordinator: Any) -> None:
        if not self.enabled or self._fh is None:
            return

        states = getattr(coordinator, "states", {}) or {}
        episode = int(states.get("episode", 0) or 0)
        step = int(states.get("step", 0) or 0) - 1
        marl_env = getattr(coordinator, "marl_env", None)
        if getattr(coordinator, "scenario_manager", None) is None and marl_env is not None:
            # SUMO: the environment keeps the counters; the coordinator's
            # states dict does not advance on that backend.
            episode = int(getattr(marl_env, "current_episode", episode) or 0)
            step = int(getattr(marl_env, "current_step", step + 1) or 0) - 1
        step = max(0, step)
        key = (episode, step)
        if key == self._last_key:
            return
        self._last_key = key

        marl_env = getattr(coordinator, "marl_env", None)
        events = list(getattr(marl_env, "events", None) or [])
        if step % self.sample_every != 0 and not events:
            return

        # CARLA: the scenario manager owns the observation pass. SUMO: the
        # environment keeps this step's normalised observations itself.
        scenario_manager = getattr(coordinator, "scenario_manager", None)
        if scenario_manager is not None:
            observations = scenario_manager.get_observations()
        else:
            observations = (getattr(marl_env, "trace_observations", None)
                            or getattr(marl_env, "previous_observations", None) or {})
        record = {
            "type": "step",
            "episode": episode,
            "step": step,
            "events": [_event_dict(event) for event in events],
            "agents": [
                self._agent_record(agent_id, obs)
                for agent_id, obs in sorted(
                    (observations or {}).items(),
                    key=lambda item: str(item[0]),
                )
            ],
        }
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")

    def _agent_record(self, agent_id: Any, obs: Mapping[str, Any]) -> dict[str, Any]:
        route_intent = obs.get("route_intent") if isinstance(obs, Mapping) else None
        velocity = obs.get("velocity", {}) if isinstance(obs, Mapping) else {}
        preference = obs.get("preference") if isinstance(obs, Mapping) else None
        location = obs.get("location", {}) if isinstance(obs, Mapping) else {}
        location = location if isinstance(location, Mapping) else {}
        return {
            "agent_id": str(agent_id),
            "x": float(obs.get("position_x", location.get("x", 0.0)) or 0.0),
            "y": float(obs.get("position_y", location.get("y", 0.0)) or 0.0),
            "heading": float(obs.get("heading_angle", 0.0) or 0.0),
            "speed": float(obs.get("speed", 0.0) or 0.0),
            "target_speed": float(obs.get("target_speed", 0.0) or 0.0),
            "velocity": _json_safe(velocity),
            "preference": _json_safe(preference if preference is not None else []),
            "entry_direction": str(obs.get("entry_direction", "") or ""),
            "route_type": str(obs.get("route_type", "") or ""),
            "lane_id": _json_safe(obs.get("lane_id")),
            "vehicle_blueprint": str(obs.get("vehicle_blueprint", "") or ""),
            "route_intent": _json_safe(route_intent or {}),
            # Mixed autonomy: which population the vehicle belongs to and
            # whether the learner commands it, so offline metrics separate
            # CAVs from background traffic without a second lookup.
            "population": _json_safe(obs.get("population")),
            "learning": _json_safe(obs.get("learning")),
            # Safety-rule diagnostics. An agent may expose them through
            # ``shield_state()``; MARLVehicleAdapter (the CARLA vehicle
            # path) merges the returned fields into the observation and
            # these keys carry them into the trace. The bundled
            # RuleBasedAgent reports ``lane_hold``, the ``merge_priority_*``
            # fields and ``applied_speed`` (the target its rules handed the
            # controller); it has no safety filter, so ``shield_active`` and
            # ``shield_stage`` stay ``None`` for it. Each key is ``None``
            # when the agent does not report it, and always on SUMO, which
            # builds no driver agent: the learner commands its vehicles'
            # speeds through TraCI and background populations stay on
            # SUMO's own models, so nothing reports there.
            "applied_speed": _json_safe(obs.get("applied_speed")),
            "shield_active": _json_safe(obs.get("shield_active")),
            "shield_stage": _json_safe(obs.get("shield_stage")),
            "lane_hold": _json_safe(obs.get("lane_hold")),
            "merge_priority_enabled": _json_safe(obs.get("merge_priority_enabled")),
            "merge_priority_release": _json_safe(obs.get("merge_priority_release")),
            "merge_priority_override": _json_safe(obs.get("merge_priority_override")),
        }


def trace_enabled(config: Any) -> bool:
    outputs = _cfg_get(config, "outputs", {}) or {}
    trace_cfg = _cfg_get(outputs, "trace", {}) or {}
    return bool(_cfg_get(trace_cfg, "enabled", False))
