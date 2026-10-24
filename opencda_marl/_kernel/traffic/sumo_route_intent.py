"""SUMO-side route-intent helpers.

These helpers keep SUMO route decisions aligned with the CARLA route-intent
contract without importing TraCI. The SUMO environment supplies live vehicle
state and the shared provider supplies the direction-preserving decision.
"""
from __future__ import annotations

import math

from opencda_marl._kernel.traffic.coordinate_bridge import CoordinateBridge, sumo_angle_to_heading
from types import SimpleNamespace
from typing import Any


def _location(x: float, y: float, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=float(x), y=float(y), z=float(z))


def _velocity_from_speed(speed_kmh: float, heading_rad: float) -> SimpleNamespace:
    """Velocity from a speed and a *heading* (radians, counter-clockwise from
    +x). SUMO's compass angle goes through :func:`sumo_angle_to_heading`
    first; it is not a heading."""
    speed_ms = float(speed_kmh) / 3.6
    heading = float(heading_rad)
    return SimpleNamespace(
        x=speed_ms * math.cos(heading),
        y=speed_ms * math.sin(heading),
        z=0.0,
    )


def spawn_lane_heading_rad(lane_shape, position_xy, *, bridge: Any | None = None) -> float | None:
    """The heading of a SUMO lane where it passes a spawn point.

    ``lane_shape`` is the lane's polyline in SUMO's frame (what
    ``traci.lane.getShape`` returns); ``position_xy`` is the spawn point in
    the CARLA frame (a spawn event's transform). Every vertex goes through
    ``bridge`` (a :class:`CoordinateBridge`; None is the default bridge, the
    legacy offset without a mirror), and the result is the direction of the
    bridged segment nearest the point: radians, counter-clockwise from +x,
    in the same frame as the bridged positions, so a y-flipped network
    mirrors it with them. None for a shape without two distinct points.
    """
    if bridge is None:
        bridge = CoordinateBridge()
    points = [bridge.sumo_to_carla(float(x), float(y)) for x, y in (lane_shape or ())]
    px, py = float(position_xy[0]), float(position_xy[1])
    best = None
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            continue
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
        distance = math.hypot(ax + t * dx - px, ay + t * dy - py)
        if best is None or distance < best[0]:
            best = (distance, math.atan2(dy, dx))
    return None if best is None else best[1]


def make_spawn_event_proxy(event: Any, *, actor_id: Any | None = None,
                           heading_rad: float | None = None) -> SimpleNamespace:
    """Build the ego adapter proxy used before a SUMO vehicle exists.

    The velocity follows ``heading_rad`` (radians, CARLA frame) when given,
    else the event transform's yaw. A transform recorded on CARLA carries
    the vehicle's own heading; one generated on SUMO (live, or into a
    recording) comes from a lane waypoint whose yaw is always 0, so its
    caller passes the spawn lane's heading (:func:`spawn_lane_heading_rad`)
    instead.
    """
    transform = event.transform
    loc = transform.location
    if heading_rad is not None:
        heading = float(heading_rad)
    else:
        # A transform's yaw is CARLA-style degrees about +x.
        heading = math.radians(float(getattr(transform.rotation, "yaw", 0.0) or 0.0))
    metadata = dict(event.metadata or {})
    metadata.setdefault("route_id", int(event.route_id))
    return SimpleNamespace(
        actor_id=str(actor_id if actor_id is not None else event.vehicle_id),
        spawn_metadata=metadata,
        current_speed=float(getattr(event, "target_speed", 0.0) or 0.0),
        location=_location(loc.x, loc.y, getattr(loc, "z", 0.0)),
        velocity=_velocity_from_speed(float(getattr(event, "target_speed", 0.0) or 0.0), heading),
    )


def make_active_vehicle_proxy(
    *,
    vehicle_id: Any,
    spawn_event: Any | None,
    position_xy: tuple[float, float],
    speed_kmh: float,
    angle_deg: float,
) -> SimpleNamespace:
    """Build the active adapter proxy consumed by route scoring/observations.

    ``angle_deg`` is SUMO's compass angle; the velocity is built from it in
    SUMO's own (unmirrored) frame. A caller that passes ``position_xy`` in
    another frame (the CARLA frame, through the coordinate bridge) must
    rebuild the velocity in that frame as well.
    """
    metadata = dict(getattr(spawn_event, "metadata", {}) or {})
    if spawn_event is not None:
        metadata.setdefault("route_id", int(getattr(spawn_event, "route_id", -1)))
        metadata.setdefault("lane_id", int(getattr(spawn_event, "lane_id", -1)))
    return SimpleNamespace(
        actor_id=str(vehicle_id),
        spawn_metadata=metadata,
        current_speed=float(speed_kmh),
        location=_location(position_xy[0], position_xy[1], 0.0),
        # The heading in SUMO's own frame (no flip); see the docstring.
        velocity=_velocity_from_speed(float(speed_kmh), sumo_angle_to_heading(angle_deg)),
    )


def apply_route_decision_to_event(event: Any, decision: Any) -> bool:
    """Apply a direction-preserving route decision to a SUMO SpawnEvent.

    Returns ``True`` when the concrete route changed. The route-intent metadata
    is recorded even when the invariant fails, but the event destination remains
    unchanged in that case.
    """
    metadata = dict(event.metadata or {})
    decision_metadata = decision.as_metadata()
    metadata["route_intent"] = decision_metadata

    from opencda_marl.api.path_action import decision_is_admissible, decision_route_update

    update = decision_route_update(decision)
    if not decision_is_admissible(decision) or not update:
        event.metadata = metadata
        return False

    old_route_id = int(event.route_id)
    event.route_id = int(update["route_id"])
    # The recording's destination stays when the decision keeps the demand
    # route: the planner table's point for the same route is only its stand-in
    # for a changed route, and replacing it moved every trip's end to wherever
    # the SUMO twin's table put it.
    if update.get("destination") is not None and int(update["route_id"]) != old_route_id:
        event.destination = update["destination"]
    metadata["route_id"] = int(update["route_id"])
    for key in ("route_type", "exit_direction", "exit_lane_idx"):
        value = update.get(key)
        if value is not None:
            metadata[key] = int(value) if key == "exit_lane_idx" else str(value)
    event.metadata = metadata
    return int(update["route_id"]) != old_route_id
