"""Enumerate the CARLA sensor actors owned by one vehicle's managers.

The vendored OpenCDA managers spawn sensors deep inside their constructors
(GNSS + IMU in ``LocalizationManager``, collision + IMU in ``SafetyManager``,
optional camera/LiDAR in ``PerceptionManager``). The rig walks those known
locations once, right after the ``VehicleManager`` is built, and hands the
sensor ids to the :class:`~opencda_marl._kernel.lifecycle.actor_registry.ActorRegistry`
so teardown can be verified against the server.
"""
from __future__ import annotations

from typing import Any, List, Tuple

from loguru import logger


class SensorRig:
    """The named CARLA sensor actors attached to one vehicle."""

    def __init__(self, sensors: List[Tuple[str, Any]]):
        # (name, carla sensor actor) — deduplicated by actor id.
        seen = set()
        self._sensors: List[Tuple[str, Any]] = []
        for name, sensor in sensors:
            sensor_id = getattr(sensor, "id", None)
            if sensor is None or sensor_id is None or sensor_id in seen:
                continue
            seen.add(sensor_id)
            self._sensors.append((name, sensor))

    @classmethod
    def from_vehicle_manager(cls, vm) -> "SensorRig":
        """Collect sensor actors from a built VehicleManager's sub-managers."""
        found: List[Tuple[str, Any]] = []

        def grab(name: str, wrapper: Any) -> None:
            sensor = getattr(wrapper, "sensor", None)
            if sensor is not None:
                found.append((name, sensor))

        localizer = getattr(vm, "localizer", None)
        if localizer is not None:
            grab("gnss", getattr(localizer, "gnss", None))
            grab("imu", getattr(localizer, "imu", None))

        safety = getattr(vm, "safety_manager", None)
        if safety is not None:
            grab("safety_imu", getattr(safety, "imu_sensor", None))
            for index, detector in enumerate(getattr(safety, "sensors", []) or []):
                # The safety list mixes real CARLA sensors (collision, IMU)
                # with pure-python detectors that have no ``sensor`` actor.
                grab(f"safety_{index}", detector)

        perception = getattr(vm, "perception_manager", None)
        if perception is not None:
            for index, camera in enumerate(getattr(perception, "rgb_camera", []) or []):
                grab(f"rgb_{index}", camera)
            grab("lidar", getattr(perception, "lidar", None))
            grab("semantic_lidar", getattr(perception, "semantic_lidar", None))

        return cls(found)

    # ------------------------------------------------------------------ #
    def sensor_ids(self) -> List[int]:
        return [int(sensor.id) for _, sensor in self._sensors]

    def __len__(self) -> int:
        return len(self._sensors)

    def stop_and_destroy(self) -> List[str]:
        """Best-effort ``stop() -> destroy()`` for every rig sensor.

        Each sensor is isolated in its own try block; failures are collected
        and returned (and logged) instead of aborting the loop.
        """
        failures: List[str] = []
        for name, sensor in self._sensors:
            try:
                if not getattr(sensor, "is_alive", True):
                    continue
                if hasattr(sensor, "stop"):
                    try:
                        sensor.stop()
                    except Exception as exc:
                        failures.append(f"{name}.stop: {exc}")
                sensor.destroy()
            except Exception as exc:
                failures.append(f"{name}.destroy: {exc}")
        for failure in failures:
            logger.warning(f"SensorRig teardown failure: {failure}")
        return failures
