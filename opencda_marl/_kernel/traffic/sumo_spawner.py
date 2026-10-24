'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-11-17
FilePath     : /OpenCDA-MARL/opencda_marl/traffic/sumo_spawner.py
Description  : SUMO vehicle spawner that uses MARLTrafficManager's SpawnEvent system.
               Spawns vehicles in SUMO based on events from traffic manager.
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
from opencda_marl._kernel.traffic._sumo_lib import traci  # libsumo by default; OPENCDA_SUMO_LIB=traci to fall back
import math
from typing import Dict, List, Optional, Tuple
from loguru import logger

from .events import SpawnEvent

# Disable SUMO's junction/right-of-way safety overrides so RL speed commands
# can actually produce risky interactions at the conflict zone.
SUMO_RL_UNSHIELDED_SPEED_MODE = 96
SUMO_RL_UNSHIELDED_LANE_CHANGE_MODE = 512
SUMO_RL_SPEED_CHECKS_MODE = 31
SUMO_DEFAULT_LANE_CHANGE_MODE = 1621
SUMO_RL_SPEED_MODE = SUMO_RL_UNSHIELDED_SPEED_MODE
SUMO_RL_LANE_CHANGE_MODE = SUMO_RL_UNSHIELDED_LANE_CHANGE_MODE

SUMO_RL_RUNTIME_MODES = {
    "unshielded": (SUMO_RL_UNSHIELDED_SPEED_MODE, SUMO_RL_UNSHIELDED_LANE_CHANGE_MODE),
    "speed_checks": (SUMO_RL_SPEED_CHECKS_MODE, SUMO_RL_UNSHIELDED_LANE_CHANGE_MODE),
    "sumo_default": (SUMO_RL_SPEED_CHECKS_MODE, SUMO_DEFAULT_LANE_CHANGE_MODE),
}


def _point_to_polyline_distance(x: float, y: float, shape) -> float:
    """Shortest distance from (x, y) to the segments of *shape*."""
    if len(shape) == 1:
        return math.hypot(x - shape[0][0], y - shape[0][1])
    best = float("inf")
    for (x1, y1), (x2, y2) in zip(shape, shape[1:]):
        seg2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
        t = 0.0 if seg2 <= 1e-12 else max(0.0, min(1.0, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / seg2))
        best = min(best, math.hypot(x - (x1 + t * (x2 - x1)), y - (y1 + t * (y2 - y1))))
    return best


class SumoVehicleSpawner:
    """
    Spawns vehicles in SUMO using SpawnEvent objects from MARLTrafficManager.
    Maintains consistency with CARLA by using same route generation logic.
    """

    def __init__(
            self,
            runtime_mode: str = "unshielded",
            speed_mode: Optional[int] = None,
            lane_change_mode: Optional[int] = None,
            learning_agent_type: Optional[str] = None,
            bridge=None):
        """Initialize SUMO vehicle spawner.

        ``learning_agent_type`` is the population the learner commands;
        every other population is background traffic and is left to SUMO's
        own driver. ``bridge`` is the SUMO<->CARLA frame bridge (a
        ``CoordinateBridge``); without one the legacy constant offset of the
        bundled intersection applies.
        """
        self.spawned_vehicles = {}  # {vehicle_id: event}
        self.learning_agent_type = (
            str(learning_agent_type) if learning_agent_type else None)
        self.bridge = bridge
        self.spawn_counter = 0
        self.failed_spawns = []
        # Route definitions persist in SUMO until the connection closes, and
        # traci has no route removal — so identical edge sequences reuse one
        # route id instead of registering a fresh route per spawn (which
        # accumulated O(spawns) route objects over a long run). The cache
        # deliberately survives episode resets for the same reason.
        self._route_id_by_edges = {}  # {tuple(edges): route_id}
        self.runtime_mode = str(runtime_mode or "unshielded")
        self.speed_mode, self.lane_change_mode = self._resolve_runtime_modes(
            self.runtime_mode,
            speed_mode,
            lane_change_mode,
        )

        # SUMO network offset for coordinate transformation
        # (sumo_coord = carla_coord + offset)
        self.net_offset = self._get_network_offset()
        logger.info(f"SUMO network offset: {self.net_offset}")
        logger.info(
            "SUMO RL runtime mode: {} (speed_mode={}, lane_change_mode={})",
            self.runtime_mode,
            self.speed_mode,
            self.lane_change_mode,
        )

        # Note: We use SUMO's built-in 'DEFAULT_VEHTYPE' for all vehicles
        # No need to create custom vehicle types

    def spawn_vehicle(self, event: SpawnEvent) -> Optional[str]:
        """
        Spawn a vehicle in SUMO based on SpawnEvent.

        Args:
            event: SpawnEvent containing spawn location, destination, route type, etc.

        Returns:
            vehicle_id if successful, None otherwise
        """
        # Extract metadata
        entry_dir = event.metadata.get('entry_direction', 'unknown')
        lane_idx = event.metadata.get('lane_id', event.lane_id)

        # Keep the SUMO vehicle id equal to the traffic event id so learned
        # one-shot route decisions can be finalized by terminal SUMO events.
        vehicle_id = self.resolve_vehicle_id(event)
        self.spawn_counter += 1

        # Extract spawn and destination from event (CARLA coordinates)
        spawn_loc = event.transform.location
        dest_loc = event.destination.location

        # Transform CARLA coordinates to SUMO coordinates
        spawn_x_sumo, spawn_y_sumo = self._carla_to_sumo(spawn_loc.x, spawn_loc.y)
        dest_x_sumo, dest_y_sumo = self._carla_to_sumo(dest_loc.x, dest_loc.y)

        # Find compatible edge pair for spawn and destination
        # This considers both proximity and routing compatibility
        spawn_edge, dest_edge = self._find_compatible_edges(
            spawn_x_sumo, spawn_y_sumo,
            dest_x_sumo, dest_y_sumo,
            event.metadata.get('route_type', 'unknown')
        )

        if not spawn_edge:
            logger.warning(f"Could not find spawn edge for {vehicle_id} at CARLA ({spawn_loc.x:.1f}, {spawn_loc.y:.1f}) → SUMO ({spawn_x_sumo:.1f}, {spawn_y_sumo:.1f})")
            self.failed_spawns.append(event)
            return None

        if not dest_edge:
            logger.warning(f"Could not find destination edge for {vehicle_id} at CARLA ({dest_loc.x:.1f}, {dest_loc.y:.1f}) → SUMO ({dest_x_sumo:.1f}, {dest_y_sumo:.1f})")
            self.failed_spawns.append(event)
            return None

        # Compute route between spawn and destination
        try:
            route = traci.simulation.findRoute(spawn_edge, dest_edge)
            if not route or not route.edges:
                route_type = event.metadata.get('route_type', 'unknown')
                logger.debug(f"No route found: {spawn_edge} → {dest_edge} (route_type={route_type})")
                self.failed_spawns.append(event)
                return None
        except Exception as e:
            route_type = event.metadata.get('route_type', 'unknown')
            logger.debug(f"Route computation failed: {spawn_edge} → {dest_edge} (route_type={route_type}): {e}")
            self.failed_spawns.append(event)
            return None

        # Determine lane index from event (already extracted above)
        # Clamp to valid range for this edge
        num_lanes = traci.edge.getLaneNumber(spawn_edge)
        spawn_lane_idx = min(lane_idx, num_lanes - 1)
        arrival_lane = self._resolve_arrival_lane(
            event.metadata.get('exit_lane_idx'),
            dest_edge,
        )

        # Get vehicle type from event blueprint (convert CARLA blueprint to SUMO type)
        carla_blueprint_id = event.blueprint.id if event.blueprint else 'vehicle.audi.a2'
        vtype_id = self._carla_blueprint_to_sumo_type(carla_blueprint_id)

        try:
            route_id = self._ensure_route(route.edges)

            # Add vehicle to simulation
            traci.vehicle.add(
                vehID=vehicle_id,
                routeID=route_id,
                typeID=vtype_id,
                depart='now',
                departLane=str(spawn_lane_idx),
                departSpeed='max',
                arrivalLane=arrival_lane,
            )

            if self._is_background(event):
                # Mixed autonomy: a background vehicle is SUMO's own driver.
                # It keeps the default safety checks and the car-following
                # model (a TraCI speed would pin it at one value with the
                # checks off), and its driver parameters come from the
                # demand recording, so every method meets the same drivers.
                self._configure_background_vehicle_runtime(
                    vehicle_id, (event.metadata or {}).get("driver") or {})
            else:
                self._configure_rl_vehicle_runtime(vehicle_id)

                # Set target speed from event
                target_speed_kmh = event.target_speed
                target_speed_ms = target_speed_kmh / 3.6  # Convert km/h to m/s
                traci.vehicle.setSpeed(vehicle_id, target_speed_ms)

            # Store event for tracking
            self.spawned_vehicles[vehicle_id] = event

            logger.debug(f"Spawned {vehicle_id} on {spawn_edge} lane {spawn_lane_idx}, "
                        f"route: {event.metadata.get('route_type', 'unknown')}, "
                        f"dest: {dest_edge}, arrivalLane: {arrival_lane}")

            return vehicle_id

        except traci.exceptions.TraCIException as e:
            logger.warning(f"Failed to spawn {vehicle_id}: {e}")
            self.failed_spawns.append(event)
            return None

    def spawn_vehicles(self, events: List[SpawnEvent]) -> List[str]:
        """
        Spawn multiple vehicles from a list of events.

        Args:
            events: List of SpawnEvent objects

        Returns:
            List of successfully spawned vehicle IDs
        """
        spawned_ids = []

        for event in events:
            veh_id = self.spawn_vehicle(event)
            if veh_id:
                spawned_ids.append(veh_id)

        if spawned_ids:
            logger.info(f"Spawned {len(spawned_ids)}/{len(events)} vehicles this step")

        return spawned_ids

    def resolve_vehicle_id(self, event: SpawnEvent) -> str:
        """Return the deterministic SUMO vehicle id for a SpawnEvent."""
        if getattr(event, "vehicle_id", None):
            return str(event.vehicle_id)
        entry_dir = event.metadata.get('entry_direction', 'unknown')
        lane_idx = event.metadata.get('lane_id', event.lane_id)
        return f"{event.flow_name}_{entry_dir}_{lane_idx}_{self.spawn_counter}"

    def get_spawn_event(self, vehicle_id: str) -> Optional[SpawnEvent]:
        """Get the SpawnEvent for a vehicle."""
        return self.spawned_vehicles.get(vehicle_id)

    def retarget_vehicle(self, vehicle_id: str, event: SpawnEvent) -> bool:
        """Re-route a vehicle already under way to the event's destination.

        The decision-zone trigger decides at the junction approach, when the
        vehicle is mid-route, so the outcome is applied with SUMO's own
        re-routing (``changeTarget`` recomputes from the current position)
        rather than by editing a spawn event that has already been consumed.
        """
        try:
            spawn_loc = event.transform.location
            dest_loc = event.destination.location
            spawn_x, spawn_y = self._carla_to_sumo(spawn_loc.x, spawn_loc.y)
            dest_x, dest_y = self._carla_to_sumo(dest_loc.x, dest_loc.y)
            _, dest_edge = self._find_compatible_edges(
                spawn_x, spawn_y, dest_x, dest_y,
                (event.metadata or {}).get("route_type", "unknown"),
            )
            if not dest_edge:
                return False
            traci.vehicle.changeTarget(vehicle_id, dest_edge)
            return True
        except Exception as exc:  # noqa: BLE001 - a re-route must not end the run
            logger.warning(f"re-target failed for {vehicle_id}: {exc}")
            return False

    def remove_vehicle(self, vehicle_id: str):
        """Remove vehicle from tracking."""
        if vehicle_id in self.spawned_vehicles:
            del self.spawned_vehicles[vehicle_id]

    def get_failed_spawns(self) -> List[SpawnEvent]:
        """Get list of failed spawn attempts."""
        return self.failed_spawns.copy()

    def clear_failed_spawns(self):
        """Clear failed spawn list."""
        self.failed_spawns.clear()

    def _ensure_route(self, edges) -> str:
        """Register the edge sequence once and reuse its route id after."""
        edges_key = tuple(edges)
        route_id = self._route_id_by_edges.get(edges_key)
        if route_id is None:
            route_id = f"route_{len(self._route_id_by_edges)}"
            traci.route.add(route_id, list(edges_key))
            self._route_id_by_edges[edges_key] = route_id
        return route_id

    def reset_episode_registries(self):
        """Drop per-episode bookkeeping at an episode boundary.

        The spawn counter stays monotonic and the route-id cache survives
        (routes remain registered in SUMO until the connection closes).
        """
        self.spawned_vehicles.clear()
        self.failed_spawns.clear()

    # --------------------------------------------------------------------- #
    # Helper Methods
    # --------------------------------------------------------------------- #

    @staticmethod
    def _resolve_runtime_modes(
            runtime_mode: str,
            speed_mode: Optional[int],
            lane_change_mode: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
        if runtime_mode not in SUMO_RL_RUNTIME_MODES:
            valid = ", ".join(sorted(SUMO_RL_RUNTIME_MODES))
            raise ValueError(f"Unknown SUMO RL runtime mode {runtime_mode!r}; expected one of {valid}")
        default_speed_mode, default_lane_change_mode = SUMO_RL_RUNTIME_MODES[runtime_mode]
        return (
            default_speed_mode if speed_mode is None else int(speed_mode),
            default_lane_change_mode if lane_change_mode is None else int(lane_change_mode),
        )

    def _configure_rl_vehicle_runtime(self, vehicle_id: str):
        """
        Configure SUMO runtime behavior so RL-controlled vehicles are not
        protected by the default junction right-of-way and autonomous lane
        changing logic.
        """
        speed_mode = getattr(self, "speed_mode", SUMO_RL_SPEED_MODE)
        lane_change_mode = getattr(self, "lane_change_mode", SUMO_RL_LANE_CHANGE_MODE)
        if speed_mode is not None:
            traci.vehicle.setSpeedMode(vehicle_id, speed_mode)
        if lane_change_mode is not None:
            traci.vehicle.setLaneChangeMode(vehicle_id, lane_change_mode)

    def _is_background(self, event: SpawnEvent) -> bool:
        """Whether this vehicle belongs to a population the learner does
        not command (only meaningful once a learning population is known)."""
        if not self.learning_agent_type:
            return False
        population = str((event.metadata or {}).get("population") or self.learning_agent_type)
        return population != self.learning_agent_type

    #: Driver parameters a recording may carry per vehicle, and the TraCI
    #: setter each one maps to. Anything else in the block is ignored.
    BACKGROUND_DRIVER_SETTERS = {
        "speed_factor": "setSpeedFactor",
        "tau": "setTau",
        "min_gap": "setMinGap",
        "imperfection": "setImperfection",
        "accel": "setAccel",
        "decel": "setDecel",
    }

    def _configure_background_vehicle_runtime(self, vehicle_id: str, driver: dict):
        """Hand a background vehicle to SUMO's driver model.

        Default speed and lane-change safeguards, control released to the
        car-following model (``setSpeed(-1)``), and the per-vehicle driver
        parameters the recording states.
        """
        traci.vehicle.setSpeedMode(vehicle_id, SUMO_RL_SPEED_CHECKS_MODE)
        traci.vehicle.setLaneChangeMode(vehicle_id, SUMO_DEFAULT_LANE_CHANGE_MODE)
        for key, setter_name in self.BACKGROUND_DRIVER_SETTERS.items():
            if key not in driver or driver[key] is None:
                continue
            setter = getattr(traci.vehicle, setter_name, None)
            if setter is None:
                continue
            try:
                setter(vehicle_id, float(driver[key]))
            except Exception as exc:  # noqa: BLE001 - one bad parameter must not lose the vehicle
                logger.warning(f"driver parameter {key}={driver[key]!r} rejected for {vehicle_id}: {exc}")
        traci.vehicle.setSpeed(vehicle_id, -1)

    @staticmethod
    def _resolve_arrival_lane(exit_lane_idx, dest_edge: str) -> str:
        """Return a SUMO arrivalLane argument from route-intent metadata."""
        try:
            lane_idx = int(exit_lane_idx)
        except (TypeError, ValueError):
            return "current"
        if lane_idx < 0:
            return "current"
        try:
            lane_count = int(traci.edge.getLaneNumber(dest_edge))
        except traci.exceptions.TraCIException:
            return "current"
        if lane_count <= 0:
            return "current"
        return str(min(lane_idx, lane_count - 1))

    def _find_compatible_edges(self, spawn_x: float, spawn_y: float,
                               dest_x: float, dest_y: float,
                               route_type: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Find compatible edge pair for spawn and destination that can form a valid route.

        This method finds edges that:
        1. Are close to the given coordinates
        2. Can form a valid route (not opposite directions)
        3. Match the expected route type when possible

        Args:
            spawn_x: Spawn X coordinate (SUMO)
            spawn_y: Spawn Y coordinate (SUMO)
            dest_x: Destination X coordinate (SUMO)
            dest_y: Destination Y coordinate (SUMO)
            route_type: Expected route type (straight, left, right)

        Returns:
            Tuple of (spawn_edge, dest_edge) or (None, None) if no valid pair found
        """
        # Get candidate edges for spawn and destination
        spawn_candidates = self._get_candidate_edges(spawn_x, spawn_y, max_distance=50.0)
        dest_candidates = self._get_candidate_edges(dest_x, dest_y, max_distance=50.0)

        if not spawn_candidates or not dest_candidates:
            return None, None

        # Try to find a compatible pair
        # Prioritize pairs that avoid opposite-direction edges
        best_pair = None
        best_score = float('inf')

        for spawn_edge, spawn_dist in spawn_candidates:
            for dest_edge, dest_dist in dest_candidates:
                # Skip opposite direction edges
                if spawn_edge == f"-{dest_edge}" or dest_edge == f"-{spawn_edge}":
                    continue

                # Check if route exists (lightweight check without full computation)
                try:
                    # Quick connectivity check using SUMO's edge connections
                    from_edges = traci.edge.getIDList()
                    if spawn_edge in from_edges and dest_edge in from_edges:
                        # Score based on distance (lower is better)
                        score = spawn_dist + dest_dist

                        if score < best_score:
                            best_score = score
                            best_pair = (spawn_edge, dest_edge)
                except:
                    continue

        if best_pair:
            return best_pair

        # Fallback: return closest edges even if they might not be compatible
        # (routing check will catch issues later)
        return spawn_candidates[0][0] if spawn_candidates else None, \
               dest_candidates[0][0] if dest_candidates else None

    def _get_candidate_edges(self, x: float, y: float, max_distance: float = 50.0) -> List[Tuple[str, float]]:
        """
        Get list of candidate edges near coordinates, sorted by distance.

        Args:
            x: X coordinate (SUMO)
            y: Y coordinate (SUMO)
            max_distance: Maximum search distance

        Returns:
            List of (edge_id, distance) tuples, sorted by distance
        """
        candidates = []
        all_edges = traci.edge.getIDList()

        for edge_id in all_edges:
            # Skip internal junction edges
            if edge_id.startswith(':'):
                continue

            # Get edge shape
            num_lanes = traci.edge.getLaneNumber(edge_id)
            if num_lanes == 0:
                continue

            # Check middle lane
            lane_id = f"{edge_id}_{num_lanes // 2}"
            try:
                shape = traci.lane.getShape(lane_id)
            except:
                continue

            if not shape:
                continue

            # Distance to the lane itself, not to its vertices: a point halfway
            # along a long straight edge is half its length from either vertex
            # and was losing to any short edge with a vertex nearby.
            min_dist = _point_to_polyline_distance(x, y, shape)

            if min_dist <= max_distance:
                candidates.append((edge_id, min_dist))

        # Sort by distance
        candidates.sort(key=lambda x: x[1])
        return candidates


    def _carla_blueprint_to_sumo_type(self, carla_blueprint_id: str) -> str:
        """
        Convert CARLA vehicle blueprint ID to SUMO vehicle type ID.

        CARLA uses detailed blueprint IDs like 'vehicle.audi.a2', 'vehicle.bmw.grandtourer',
        while SUMO uses generic type IDs.

        For maximum compatibility, we use SUMO's built-in 'DEFAULT_VEHTYPE' which is
        always available in any SUMO simulation.

        Args:
            carla_blueprint_id: CARLA blueprint ID (e.g., 'vehicle.audi.a2')

        Returns:
            SUMO vehicle type ID ('DEFAULT_VEHTYPE')
        """
        # Use SUMO's built-in default vehicle type - always available
        return 'DEFAULT_VEHTYPE'

    def _get_network_offset(self) -> tuple:
        """
        Get the network offset from SUMO network.

        SUMO applies netOffset during conversion: sumo_coord = carla_coord + offset
        The offset is stored in the network file's <location> tag.

        Returns:
            (offset_x, offset_y) tuple
        """
        # For the intersection network, the offset is (99.80, 100.00)
        # This can be read from network boundary or hardcoded
        # Since CARLA junction 4 is at (99.80, 99.57) and SUMO junction 4 is also at (99.80, 99.57),
        # but SUMO uses positive coordinates (0-200 range), the offset must be applied to spawn points

        # The netOffset from intersection.net.xml is (99.80, 100.00)
        return (99.80, 100.00)

    def _carla_to_sumo(self, x: float, y: float) -> tuple:
        """
        Transform CARLA coordinates to SUMO coordinates.

        Args:
            x: CARLA X coordinate
            y: CARLA Y coordinate

        Returns:
            (sumo_x, sumo_y) tuple
        """
        if self.bridge is not None:
            return self.bridge.carla_to_sumo(x, y)
        sumo_x = x + self.net_offset[0]
        sumo_y = y + self.net_offset[1]
        return (sumo_x, sumo_y)

    def _sumo_to_carla(self, x: float, y: float) -> tuple:
        """
        Transform SUMO coordinates to CARLA coordinates.

        Args:
            x: SUMO X coordinate
            y: SUMO Y coordinate

        Returns:
            (carla_x, carla_y) tuple
        """
        if self.bridge is not None:
            return self.bridge.sumo_to_carla(x, y)
        carla_x = x - self.net_offset[0]
        carla_y = y - self.net_offset[1]
        return (carla_x, carla_y)

    def cleanup(self):
        """Cleanup spawner resources."""
        self.spawned_vehicles.clear()
        self.failed_spawns.clear()
        self.spawn_counter = 0
