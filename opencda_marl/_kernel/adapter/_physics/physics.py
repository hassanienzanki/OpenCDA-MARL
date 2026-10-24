'''
Standalone physics helpers for MARLVehicleAdapter.

Functions read raw CARLA physics state (velocity, orientation, height)
and detect anomalous simulator states that indicate glitched actors.
'''
from typing import Any, Dict

from loguru import logger
import numpy as np

DEFAULT_CARLA_PHYSICS_GUARD = {
    "enabled": True,
    "policy": "remove_actor",
    "warn_vertical_speed_kmh": 12.0,
    "max_vertical_speed_kmh": 25.0,
    "warn_speed_3d_gap_kmh": 15.0,
    "max_speed_3d_gap_kmh": 30.0,
    "warn_pitch_deg": 10.0,
    "max_pitch_deg": 20.0,
    "warn_roll_deg": 10.0,
    "max_roll_deg": 20.0,
    "warn_height_delta_m": 0.75,
    "max_height_delta_m": 1.5,
}


def get_physics_speed_planar_kmh(vehicle, actor_id: int) -> float:
    """Get canonical physical road speed from the CARLA actor velocity vector."""
    try:
        velocity = vehicle.get_velocity()
        return 3.6 * float(np.sqrt(velocity.x**2 + velocity.y**2))
    except Exception as e:
        logger.debug(f"Could not get physics speed for vehicle {actor_id}: {e}")
        return 0.0


def get_physics_state(vehicle, carla_map, actor_id: int) -> Dict[str, float]:
    """Collect raw CARLA physics diagnostics for anomaly checks and observations."""
    try:
        velocity = vehicle.get_velocity()
        transform = vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation

        planar_speed_kmh = 3.6 * float(np.sqrt(velocity.x**2 + velocity.y**2))
        speed_3d_kmh = 3.6 * float(np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
        vertical_speed_kmh = 3.6 * float(velocity.z)

        waypoint_z = location.z
        try:
            waypoint = carla_map.get_waypoint(location)
            if waypoint:
                waypoint_z = waypoint.transform.location.z
        except Exception as e:
            logger.debug(f"Could not get waypoint z for vehicle {actor_id}: {e}")

        return {
            "planar_speed_kmh": planar_speed_kmh,
            "speed_3d_kmh": speed_3d_kmh,
            "vertical_speed_kmh": vertical_speed_kmh,
            "pitch_deg": float(rotation.pitch),
            "roll_deg": float(rotation.roll),
            "yaw_deg": float(rotation.yaw),
            "location_z": float(location.z),
            "waypoint_z": float(waypoint_z),
            "height_delta_m": float(location.z - waypoint_z),
        }
    except Exception as e:
        logger.debug(f"Could not collect physics state for vehicle {actor_id}: {e}")
        return {
            "planar_speed_kmh": 0.0,
            "speed_3d_kmh": 0.0,
            "vertical_speed_kmh": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "yaw_deg": 0.0,
            "location_z": 0.0,
            "waypoint_z": 0.0,
            "height_delta_m": 0.0,
        }


def inspect_physics_state(
    vehicle, carla_map, actor_id: int,
    target_speed: float,
    guard_config: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """Return anomaly metadata when the actor enters an invalid CARLA physics state."""
    config = DEFAULT_CARLA_PHYSICS_GUARD.copy()
    if guard_config:
        config.update(guard_config)
    if not config.get("enabled", True):
        return None

    physics_state = get_physics_state(vehicle, carla_map, actor_id)
    planar_speed = physics_state["planar_speed_kmh"]
    speed_3d = physics_state["speed_3d_kmh"]
    vertical_speed = physics_state["vertical_speed_kmh"]
    speed_3d_gap = max(0.0, speed_3d - planar_speed)
    pitch = abs(physics_state["pitch_deg"])
    roll = abs(physics_state["roll_deg"])
    height_delta = abs(physics_state["height_delta_m"])

    fatal_reasons = []
    warn_reasons = []

    def classify(value: float, warn_threshold: float, fatal_threshold: float, label: str, unit: str):
        if value > fatal_threshold:
            fatal_reasons.append(f"{label}={value:.2f}{unit} exceeds fatal threshold {fatal_threshold:.2f}{unit}")
        elif value > warn_threshold:
            warn_reasons.append(f"{label}={value:.2f}{unit} exceeds warning threshold {warn_threshold:.2f}{unit}")

    classify(abs(vertical_speed), config["warn_vertical_speed_kmh"], config["max_vertical_speed_kmh"], "vertical_speed", " km/h")
    classify(speed_3d_gap, config["warn_speed_3d_gap_kmh"], config["max_speed_3d_gap_kmh"], "speed_3d_gap", " km/h")
    classify(pitch, config["warn_pitch_deg"], config["max_pitch_deg"], "pitch", " deg")
    classify(roll, config["warn_roll_deg"], config["max_roll_deg"], "roll", " deg")
    classify(height_delta, config["warn_height_delta_m"], config["max_height_delta_m"], "height_delta", " m")

    reasons = fatal_reasons or warn_reasons
    if not reasons:
        return None

    severity = "fatal" if fatal_reasons else "warn"
    return {
        "severity": severity,
        "policy": config.get("policy", "remove_actor"),
        "reasons": reasons,
        "physics_speed_planar": round(planar_speed, 2),
        "physics_speed_3d": round(speed_3d, 2),
        "vertical_speed": round(vertical_speed, 2),
        "speed_3d_gap": round(speed_3d_gap, 2),
        "tilt_pitch": round(physics_state["pitch_deg"], 2),
        "tilt_roll": round(physics_state["roll_deg"], 2),
        "height_above_waypoint": round(physics_state["height_delta_m"], 3),
        "location_z": round(physics_state["location_z"], 3),
        "waypoint_z": round(physics_state["waypoint_z"], 3),
        "target_speed": round(float(target_speed), 2),
    }
