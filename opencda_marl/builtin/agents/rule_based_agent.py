'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-03
FilePath     : /OpenCDA-MARL/opencda_marl/agents/rule_based_agent.py
Description  : Rule-Based Agent implementing three-stage hierarchical control

This agent implements the three-stage rule-based control system enhanced with
Junction-Aware Traffic Management from JAIR paper:
"A rule-based cooperative merging strategy" + "Junction-aware traffic management"
Using trajectory prediction and time-to-collision for better intersection handling

Three-stage hierarchy:
1. Junction Management (Highest Priority) - FCFS conflict resolution
2. Car Following - Safe distance maintenance  
3. Cruising - Default free-flow behavior

Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
import math
import random
import carla
import itertools
from typing import Dict, List, Optional, Tuple
from collections.abc import Mapping
from loguru import logger

from opencda_marl.builtin.agents.vanilla_agent import VanillaAgent
from opencda_marl.builtin.agents.merge_priority import standing_merge_precedes


class RuleBasedAgent(VanillaAgent):
    """
    Rule-based agent implementing three-stage hierarchical control for intersection management.
    
    Based on the paper's three-stage algorithm:
    1. Junction Management: Detect conflicts and apply cautious behavior
    2. Car Following: Maintain safe following distance
    3. Cruising: Default maximum speed behavior
    """

    #: Driver-profile defaults, at class level so an instance built without
    #: ``__init__`` (as the unit tests build one) still has a neutral driver:
    #: no speed scaling and no dawdling.
    desired_speed_factor = 1.0
    driver_imperfection = 0.0
    driver_imperfection_accel = 2.6
    driver_imperfection_hold_steps = 0
    driver_imperfection_scale = 1.0
    _imperfection_draw = 0.0
    _imperfection_age = 0
    #: Step length (s) the obstacle yaw-rate estimate divides by; __init__
    #: takes the world's fixed delta when the world has one.
    _step_seconds = 0.05
    #: What :meth:`shield_state` reports, at class level so the hook is safe
    #: before the first step and on an instance built without ``__init__``.
    lane_hold_active = False
    merge_priority_release = False
    merge_priority_override = False
    _applied_speed = None

    #: Keys the earlier follower law read and this one does not. A profile
    #: that still sets one is told so once per process instead of having the
    #: value ignored silently.
    RETIRED_KEYS = {
        'following_speed_buffer': 'the follower closes on a braking curve '
                                  '(closing_decel) and has no speed buffer',
    }
    _reported_retired_keys: set = set()

    @classmethod
    def _report_retired_keys(cls, config_yaml) -> List[str]:
        """Warn (once per key and process) about retired keys the profile sets."""
        reported = []
        for key, reason in cls.RETIRED_KEYS.items():
            if key in config_yaml and key not in cls._reported_retired_keys:
                cls._reported_retired_keys.add(key)
                reported.append(key)
                logger.warning(f"rule_based.{key} is no longer read: {reason}.")
        return reported

    def __init__(self, vehicle, carla_map, config_yaml):
        """
        Initialize rule-based agent with three-stage control parameters.
        
        Args:
            vehicle: CARLA vehicle actor
            carla_map: CARLA HD map
            config_yaml: Configuration containing rule_based parameters
        """
        # Initialize VanillaAgent base
        super().__init__(vehicle, carla_map, config_yaml)
        
        # Stage 1: Junction Management parameters (JAIR enhanced)
        self.junction_approach_distance = config_yaml.get('junction_approach_distance', 70.0)  # meters
        self.junction_conflict_distance = config_yaml.get('junction_conflict_distance', 50.0)   # meters
        self.cautious_speed = config_yaml.get('cautious_speed', 35.0)  # km/h
        
        # Trajectory-based conflict detection (replaces heading difference)
        self.trajectory_lookahead_time = config_yaml.get('trajectory_lookahead_time', 3.0)  # seconds
        self.ttc_threshold = config_yaml.get('ttc_threshold', 2.0)  # seconds - min TTC difference for conflict
        self.trajectory_step_size = config_yaml.get('trajectory_step_size', 1.0)  # meters
        
        # Stage 2: Car Following parameters
        self.time_headway = config_yaml.get('time_headway', 2.0)  # seconds
        self.following_gain = config_yaml.get('following_gain', 0.5)  # control gain
        self.minimum_distance_buffer = config_yaml.get('minimum_distance_buffer', 3.0)  # meters, bumper to bumper
        self.same_lane_tolerance = math.radians(config_yaml.get('same_lane_tolerance_deg', 30))  # radians
        self.front_cone_angle = math.radians(config_yaml.get('front_cone_angle_deg', 45))  # radians
        self._report_retired_keys(config_yaml)

        # Speed control parameters - Dynamic adjustment factors instead of fixed speeds
        self.junction_speed_factor = config_yaml.get('junction_speed_factor', 0.6)  # Reduce to 60% at junctions
        self.junction_min_reduction = config_yaml.get('junction_min_reduction', 15.0)  # Minimum reduction (km/h)
        # Stop-on-conflict: yield by stopping before the junction entry, not by
        # slowing to 60 %. Inside the junction the slow-down rule still applies
        # (a stop in the box would block it).
        self.junction_stop_on_conflict = bool(config_yaml.get('junction_stop_on_conflict', True))
        self.junction_stop_buffer = config_yaml.get('junction_stop_buffer', 2.0)   # metres before the entry
        self.junction_stop_decel = config_yaml.get('junction_stop_decel', 1.5)     # m/s^2: the braking curve to the stop line
        # Abreast guard: a same-direction vehicle alongside that closes in
        # laterally makes the one behind drop back until the pair separates.
        self.abreast_guard = bool(config_yaml.get('abreast_guard', True))
        self.abreast_lateral_m = config_yaml.get('abreast_lateral_m', 3.2)        # soft zone: binds only on a closing, moving neighbour
        self.abreast_hard_m = config_yaml.get('abreast_hard_m', 2.6)              # hard zone: always binds (two bodies nearly touch)
        self.abreast_closing_m = config_yaml.get('abreast_closing_m', 0.015)      # lateral shrink per step that counts as closing (~0.3 m/s)
        self.abreast_creep_kmh = config_yaml.get('abreast_creep_kmh', 5.0)        # a stationary pair in the hard zone creeps apart at this speed
        self.abreast_longitudinal_m = config_yaml.get('abreast_longitudinal_m', 6.0)  # |offset| below this counts as alongside
        self.abreast_drop_kmh = config_yaml.get('abreast_drop_kmh', 8.0)          # how much slower than the neighbour the yielder goes
        self.closing_decel = config_yaml.get('closing_decel', 1.0)  # m/s^2: the braking curve a follower closes a gap on
        # Lane hold: the route follower steers into the lane its next waypoint is
        # on without looking; while a vehicle occupies that lane alongside, the
        # ego keeps its current lane and drops back until the gap opens, the
        # way SUMO's lane-change model waits for a gap.
        self.lane_hold = bool(config_yaml.get('lane_hold', True))
        # Geometry priority releases a standing merge by position (the front
        # vehicle goes first), not by actor id. It is on with the hold: by id
        # alone, a merging vehicle with the larger id holds at a stop while the
        # standing vehicle behind it in the target lane waits on its body, and
        # neither ever moves.
        self.lane_hold_geometry_priority = bool(config_yaml.get('lane_hold_geometry_priority', True))
        self.lane_hold_behind_m = config_yaml.get('lane_hold_behind_m', 8.0)      # an occupant this far behind still blocks
        self.lane_hold_ahead_m = config_yaml.get('lane_hold_ahead_m', 12.0)       # ... or this far ahead
        self.lane_hold_drop_kmh = config_yaml.get('lane_hold_drop_kmh', 8.0)      # how far below the occupant the ego drops while holding
        self.lane_hold_lookahead_m = config_yaml.get('lane_hold_lookahead_m', 8.0)  # the current-lane point it steers to instead
        self.lane_hold_standing_kmh = config_yaml.get('lane_hold_standing_kmh', 5.0)  # below this the occupant is a standing queue
        self.lane_hold_overlap_m = config_yaml.get('lane_hold_overlap_m', 5.5)        # a standing occupant blocks only within body overlap

        # Overall speed control
        self.max_speed = config_yaml.get('max_speed', 65.0)  # Absolute speed limit - NEVER exceed
        self.cruising_speed_factor = config_yaml.get('cruising_speed_factor', 0.95)  # 95% of vanilla when cruising
        # SUMO's speedFactor scales the desired speed in every regime, not only
        # free flow; a recording that states one per vehicle lands here.
        self.desired_speed_factor = config_yaml.get('desired_speed_factor', 1.0)

        # Krauss driver imperfection (SUMO's sigma), the one recorded driver
        # parameter the CARLA side had no analogue for. SUMO draws
        # v <- max(0, v_safe - sigma * accel * U(0,1)) every step; the same
        # reduction is applied here to the speed the controller is asked for.
        # It only ever lowers the target, so no safety rule can be defeated by
        # it. The hold is in steps because the step length is fixed: 10 steps
        # = 0.5 s at the default 0.05 s delta. A hold of 0 redraws every step,
        # which is SUMO's own rule; a longer hold survives the speed
        # controller's smoothing, which SUMO has no equivalent of.
        self.driver_imperfection = float(config_yaml.get('driver_imperfection', 0.0))
        # A recording states sigma per vehicle, so the profile cannot restate
        # it; this scales what the recording gave. 0 disables dawdling while
        # the rest of the driver profile still applies; any value but 0 or 1
        # is a calibration choice.
        self.driver_imperfection_scale = float(
            config_yaml.get('driver_imperfection_scale', 1.0))
        self.driver_imperfection_accel = float(
            config_yaml.get('driver_imperfection_accel', 2.6))   # m/s^2, SUMO's default accel
        self.driver_imperfection_hold_steps = int(
            config_yaml.get('driver_imperfection_hold_steps', 0))
        self._imperfection_rng = random.Random(self._imperfection_seed(config_yaml))
        self._imperfection_draw = 0.0
        self._imperfection_age = 0

        # Debug and logging
        self.debug_rules = self._resolve_rule_debug(config_yaml)
        self.current_stage = 0  # For debugging: 1=junction, 2=following, 3=cruising
        self._junction_distance = float('inf')   # the last distance to the junction entry the junction rule saw
        self.abreast_active = False              # the abreast guard bound the target this step
        self.lane_hold_active = False            # the lane hold replaced the planner's target this step
        self.merge_priority_release = False
        self.merge_priority_override = False
        self._applied_speed = None               # the target speed this step's rules handed the controller
        self._abreast_memory: Dict[int, float] = {}   # neighbour id -> lateral distance last step
        self._heading_memory: Dict[int, Tuple[float, float]] = {}  # obstacle id -> (heading deg, yaw rate deg/s)
        try:
            fixed_delta = self._world.get_settings().fixed_delta_seconds
            if fixed_delta:
                self._step_seconds = float(fixed_delta)
        except Exception:  # noqa: BLE001 - keep the class default
            pass

        logger.debug(f"RuleBasedAgent initialized for vehicle {vehicle.id} with three-stage control")

    def _imperfection_seed(self, config_yaml) -> int:
        """A per-vehicle seed that survives a replay.

        Actor ids are reassigned every episode, so they cannot seed anything
        reproducible. This vehicle's own recorded driver draw can: the four
        knobs below carry it, and no two vehicles in a recording share one.
        """
        seed = int(config_yaml.get('driver_imperfection_seed', 0)) & 0xFFFFFFFF
        for value in (self.time_headway, self.minimum_distance_buffer,
                      self.max_speed, self.desired_speed_factor):
            seed = (seed * 1000003 + int(round(float(value) * 1e6))) & 0xFFFFFFFF
        return seed

    def _imperfection_drop_kmh(self) -> float:
        """This step's Krauss speed reduction, km/h (never negative)."""
        sigma = self.driver_imperfection * self.driver_imperfection_scale
        if sigma <= 0.0:
            return 0.0
        if self._imperfection_age <= 0:
            self._imperfection_draw = self._imperfection_rng.random()
            self._imperfection_age = max(1, self.driver_imperfection_hold_steps)
        self._imperfection_age -= 1
        return (sigma * self.driver_imperfection_accel
                * self._imperfection_draw) * 3.6

    @staticmethod
    def _resolve_rule_debug(config_yaml) -> bool:
        """
        Resolve rule-specific debug from the shared debug surface.

        `debug: false` is the master off switch.
        `debug: true` enables all rule-based debug.
        `debug.rules` can be used for explicit split control when needed.
        """
        debug_cfg = config_yaml.get('debug', False)
        legacy_debug_rules = config_yaml.get('debug_rules', None)

        if isinstance(debug_cfg, Mapping):
            if 'rules' in debug_cfg:
                return bool(debug_cfg.get('rules', False))
            if legacy_debug_rules is not None:
                return bool(legacy_debug_rules)
            return False

        return bool(debug_cfg)

    # --------------------------------------------------------------------- #
    # Main Step Function
    # --------------------------------------------------------------------- #    
    def run_step(self, target_speed=None):
        """
        Execute one step of rule-based navigation.
        Uses VanillaAgent for route following and applies three-stage rules for speed control only.
        
        Returns:
            Tuple[float, carla.Location]: (rule_based_speed, vanilla_location)
        """
        # This step's diagnostics start clear: a step that returns before the
        # rules run must not report the previous step's hold or speed.
        self._reset_step_diagnostics()
        # Check if agent is properly initialized
        if not self._ego_pos:
            if self.debug_rules:
                logger.warning("RuleBasedAgent: ego position not set")
            return 0.0, None
        self._track_obstacle_headings()

        # Check destination
        if self.is_close_to_destination():
            raise StopIteration("Destination reached - simulation complete")
        
        try:
            # Let parent VanillaAgent handle route navigation
            vanilla_speed, vanilla_location = super().run_step(target_speed)
            
            # Apply three-stage rule-based speed control using vanilla_speed as base
            rule_based_speed = self._calculate_rule_based_speed(vanilla_speed)
            # Lane hold: keep the lane while the one the planner steers into is occupied
            vanilla_location, hold_cap = self._hold_lane(vanilla_location)
            if hold_cap is not None:
                rule_based_speed = min(rule_based_speed, hold_cap)
                if self.current_stage == 3:
                    self.current_stage = 2   # the hold is a following rule: report stage 2, not cruising

            # Return rule-based speed with VanillaAgent's navigation
            if self.debug_rules:
                stage_name = {1: 'Junction', 2: 'Following', 3: 'Cruising'}.get(self.current_stage, 'Unknown')
                logger.info(f"Vehicle {self._vehicle.id}: {stage_name} - Speed: {rule_based_speed:.1f} km/h")
            
            self._applied_speed = rule_based_speed
            return rule_based_speed, vanilla_location
            
        except Exception as e:
            logger.error(f"RuleBasedAgent error for vehicle {self._vehicle.id}: {e}")
            # The fallback discards whatever the rules decided, the hold included.
            self._reset_step_diagnostics()
            # Fallback to parent behavior
            try:
                return super().run_step(target_speed)
            except Exception:
                return self.cautious_speed, self._ego_pos.location if self._ego_pos else None
    

    # --------------------------------------------------------------------- #
    # Private Functions
    # --------------------------------------------------------------------- #

    def _calculate_rule_based_speed(self, vanilla_speed: float) -> float:
        """
        Calculate speed based on three-stage rule system using vanilla_speed as base.
        
        Args:
            vanilla_speed: Base speed from VanillaAgent (km/h)
        
        Returns:
            float: Target speed in km/h based on current stage
        """
        try:
            calculated_speed = 0.0

            # SUMO's speedFactor is a property of the driver, not of a regime:
            # it scales the desired speed every stage works from.
            if self.desired_speed_factor != 1.0:
                vanilla_speed = vanilla_speed * self.desired_speed_factor

            # Stage 1: Junction Management (Highest Priority)
            if self._should_apply_junction_management():
                self.current_stage = 1
                # Dynamic adjustment: reduce speed by factor or minimum reduction, whichever is higher
                factor_reduction = vanilla_speed * self.junction_speed_factor
                fixed_reduction = max(0, vanilla_speed - self.junction_min_reduction)
                calculated_speed = max(factor_reduction, fixed_reduction)
                if self.junction_stop_on_conflict and 0.0 < self._junction_distance < float('inf'):
                    # Not yet inside: brake on a curve that reaches zero at the stop
                    # line (junction_stop_buffer before the entry) and hold there
                    # until the conflict clears. A stopped vehicle projects no
                    # trajectory for the others, so the priority vehicle proceeds.
                    stop_gap = max(0.0, self._junction_distance - self.junction_stop_buffer)
                    calculated_speed = min(calculated_speed,
                                           math.sqrt(2.0 * self.junction_stop_decel * stop_gap) * 3.6)
                # Junction priority does not remove longitudinal separation:
                # a queue may already occupy the lane while the ego yields to
                # crossing traffic. Apply both constraints, taking the safer cap.
                following_speed = self._calculate_car_following_speed(vanilla_speed)
                if following_speed is not None:
                    calculated_speed = min(calculated_speed, following_speed)
            
            # Stage 2: Car Following (Medium Priority)
            elif (following_speed := self._calculate_car_following_speed(vanilla_speed)) is not None:
                self.current_stage = 2
                calculated_speed = following_speed
            
            # Stage 3: Cruising (Default - with adjustment)
            else:
                self.current_stage = 3
                # Apply cruising factor to vanilla_speed instead of using it directly
                calculated_speed = vanilla_speed * self.cruising_speed_factor
                
            # FINAL CHECK: Enforce max_speed limit on all stages
            # Abreast guard, on top of whatever stage decided: the vehicle that
            # is behind in a closing side-by-side pair drops below the
            # neighbour's speed until the pair is no longer alongside.
            self.abreast_active = False
            if self.abreast_guard:
                cap = self._abreast_cap()
                if cap is not None and cap < calculated_speed:
                    calculated_speed = cap
                    self.abreast_active = True
                    if self.current_stage == 3:
                        self.current_stage = 2   # the guard is a following rule: report stage 2, not cruising
            final_speed = min(calculated_speed, self.max_speed)

            # Krauss imperfection, last: it lowers the target the controller
            # receives and never raises it, so every rule above still binds.
            drop = self._imperfection_drop_kmh()
            if drop:
                final_speed = max(0.0, final_speed - drop)

            if self.debug_rules and calculated_speed > self.max_speed:
                stage_name = {1: 'Junction', 2: 'Following', 3: 'Cruising'}.get(self.current_stage, 'Unknown')
                logger.info(f"Vehicle {self._vehicle.id}: {stage_name} speed limited from "
                           f"{calculated_speed:.1f} to {self.max_speed:.1f} km/h")
            
            return final_speed
            
        except Exception as e:
            logger.error(f"Error calculating rule-based speed: {e}")
            # Fallback with max_speed enforcement
            fallback_speed = max(10.0, vanilla_speed * 0.7) if vanilla_speed > 0 else 30.0
            return min(fallback_speed, self.max_speed)

    def _calculate_car_following_speed(self, vanilla_speed: float) -> Optional[float]:
        """
        Calculate car following speed if there's a leading vehicle.
        
        Args:
            vanilla_speed: Base speed from VanillaAgent (km/h)
        
        Returns:
            Optional[float]: Following speed or None if no leading vehicle
        """
        # Find leading vehicle
        leading_vehicle, gap = self._get_leading_vehicle()
        
        if leading_vehicle is None:
            return None  # No leading vehicle
        
        # Calculate safe following speed
        safe_speed = self._calculate_safe_following_speed(leading_vehicle, gap, vanilla_speed)
        
        if self.debug_rules:
            logger.info(f"Vehicle {self._vehicle.id}: Following vehicle at a {gap:.1f}m gap, "
                       f"target speed: {safe_speed:.1f} km/h")
        
        return safe_speed

    def _calculate_ttc_to_junction(self, trajectory: List[Tuple[float, float, float]], 
                                  junction_dist: float) -> float:
        """
        Calculate time-to-collision to junction entry.
        
        Args:
            trajectory: Vehicle trajectory
            junction_dist: Current distance to junction
            
        Returns:
            float: Time to reach junction in seconds
        """
        try:
            if not trajectory:
                return float('inf')
            
            # Simple approximation: use current speed and distance
            speed_ms = self._ego_speed / 3.6 if self._ego_speed > 0 else 1.0
            return junction_dist / speed_ms
            
        except Exception as e:
            logger.error(f"Error calculating TTC to junction: {e}")
            return float('inf')
    
    # --------------------------------------------------------------------- #
    # Junction Management
    # --------------------------------------------------------------------- #         
    def _should_apply_junction_management(self) -> bool:
        """
        Check if junction management should be applied.
        
        Returns:
            bool: True if should apply cautious speed for junction
        """
        # Get distance to nearest junction
        junction_dist = self._get_junction_distance()
        self._junction_distance = junction_dist
        
        # Check if approaching junction
        if junction_dist > self.junction_approach_distance:
            return False  # Not approaching junction
        
        # Detect conflicts with other vehicles
        conflicts = self._detect_junction_conflicts(junction_dist)
        
        if conflicts:
            # FCFS Logic: Check if any other vehicle has priority (arrives first)
            has_priority_conflicts = [c for c in conflicts if c['has_priority']]
            
            if has_priority_conflicts:
                # Other vehicles arrive first - yield
                if self.debug_rules:
                    # Fix: Use carla_id attribute instead of id method
                    priority_vehicles = [str(getattr(c['vehicle'], 'carla_id', 'unknown')) for c in has_priority_conflicts]
                    logger.info(f"Vehicle {self._vehicle.id}: Yielding at junction, "
                               f"distance: {junction_dist:.1f}m, yielding to vehicles: {priority_vehicles}")
                return True
            else:
                # Ego has priority - proceed but monitor
                if self.debug_rules:
                    logger.info(f"Vehicle {self._vehicle.id}: Has priority at junction, "
                               f"proceeding with conflicts: {len(conflicts)}")
        
        return False
       
    def _get_junction_distance(self) -> float:
        """
        Calculate distance to nearest junction/intersection.
        
        Returns:
            float: Distance to nearest junction in meters
        """
        if not self._ego_pos:
            return float('inf')
        
        try:
            # Get current waypoint
            current_waypoint = self._map.get_waypoint(self._ego_pos.location)
            if not current_waypoint:
                return float('inf')
            
            # Look ahead for junction waypoints
            lookahead_distance = 0.0
            waypoint = current_waypoint
            step_size = 2.0  # meters: the stop line needs a finer distance than 5 m
            
            while lookahead_distance < self.junction_approach_distance:
                # Check if current waypoint is in junction
                if waypoint.is_junction:
                    return lookahead_distance
                
                # Get next waypoint
                next_waypoints = waypoint.next(step_size)
                if not next_waypoints:
                    break
                    
                waypoint = next_waypoints[0]
                lookahead_distance += step_size
            
            return float('inf')  # No junction found within approach distance
            
        except Exception as e:
            logger.error(f"Error calculating junction distance: {e}")
            return float('inf')
    
    def _detect_junction_conflicts(self, junction_dist: float) -> List[Dict]:
        """
        Detect conflicting vehicles at junction using trajectory prediction and TTC.
        Enhanced JAIR approach for better intersection handling.
        
        Args:
            junction_dist: Distance to junction
            
        Returns:
            List[Dict]: List of conflicting vehicles with TTC information
        """
        conflicts = []
        
        if not self._ego_pos or not hasattr(self, 'obstacle_vehicles'):
            return conflicts
        
        try:
            # Get ego vehicle's predicted trajectory
            ego_trajectory = self._predict_vehicle_trajectory(
                self._ego_pos, self._ego_speed, is_ego=True
            )
            
            if not ego_trajectory:
                if self.debug_rules:
                    logger.warning(f"Vehicle {self._vehicle.id}: Could not predict ego trajectory")
                return conflicts
            
            # Calculate ego's TTC to junction
            ego_junction_ttc = self._calculate_ttc_to_junction(ego_trajectory, junction_dist)
            
            if self.debug_rules:
                logger.info(f"Vehicle {self._vehicle.id}: Ego TTC to junction: {ego_junction_ttc:.1f}s")
            
            for vehicle in self.obstacle_vehicles:
                # Get vehicle distance to junction
                vehicle_location = vehicle.get_location()
                vehicle_junction_dist = vehicle_location.distance(self._ego_pos.location)
                
                # Only consider vehicles near junction
                if vehicle_junction_dist > self.junction_conflict_distance:
                    continue
                
                # Get vehicle speed and transform
                vehicle_transform = vehicle.get_transform()
                vehicle_speed = self._get_vehicle_speed(vehicle)  # km/h
                
                # Predict other vehicle's trajectory
                other_trajectory = self._predict_vehicle_trajectory(
                    vehicle_transform, vehicle_speed, is_ego=False,
                    yaw_rate_deg=self._obstacle_yaw_rate_deg(vehicle)
                )
                
                if not other_trajectory:
                    continue
                
                # Find collision point and calculate TTC for both vehicles
                collision_data = self._find_collision_point(ego_trajectory, other_trajectory)
                
                if collision_data:
                    ego_ttc, other_ttc, collision_point = collision_data
                    
                    # Check if this is a potential conflict
                    # Conflict exists if both vehicles will reach collision point within threshold
                    ttc_diff = abs(ego_ttc - other_ttc)
                    
                    if ttc_diff < self.ttc_threshold:
                        # Use ego_junction_ttc for more accurate priority determination
                        # Compare collision TTC with junction arrival TTC
                        has_priority = other_ttc < min(ego_ttc, ego_junction_ttc)
                        
                        conflicts.append({
                            'vehicle': vehicle,
                            'distance': vehicle_junction_dist,
                            'ego_ttc': ego_ttc,
                            'ego_junction_ttc': ego_junction_ttc,  # Now using this value
                            'other_ttc': other_ttc,
                            'ttc_diff': ttc_diff,
                            'collision_point': collision_point,
                            'has_priority': has_priority  # Enhanced priority logic
                        })
                        
                        if self.debug_rules:
                            priority_str = "OTHER" if has_priority else "EGO"
                            # Fix: Use carla_id attribute instead of id method
                            vehicle_id = getattr(vehicle, 'carla_id', 'unknown')
                            logger.info(
                                f"Vehicle {self._vehicle.id}: Conflict detected with vehicle {vehicle_id}, "
                                f"TTC ego: {ego_ttc:.1f}s, other: {other_ttc:.1f}s, "
                                f"junction: {ego_junction_ttc:.1f}s, priority: {priority_str}"
                            )
                    
        except Exception as e:
            logger.error(f"Error detecting junction conflicts: {e}")
        
        return conflicts
    
    @staticmethod
    def _yaw_rate_deg(vehicle) -> float:
        """The vehicle's yaw rate in degrees per second (CARLA's angular velocity
        is in degrees per second), zero for a stand-in without one."""
        try:
            return float(vehicle.get_angular_velocity().z)
        except Exception:  # noqa: BLE001 - a test double or a dead actor
            return 0.0

    def _track_obstacle_headings(self) -> None:
        """Estimate each obstacle's yaw rate from its heading change over the
        last step.

        Perception hands over ObstacleVehicle objects, which carry a transform
        but no angular velocity, so without this the branch choice at a fork
        saw a zero yaw rate for every vehicle and predicted all of them straight
        on. Called once per step (``run_step``); a vehicle not seen on the
        previous step has no rate yet.

        The estimate needs an identity that persists between steps. A
        detection perception could not match to an actor (``carla_id`` -1)
        has none, and an id two objects carry in the same step names
        neither: subtracting their headings would read the angle between
        two vehicles as a turn. Such obstacles get no rate (straight on).
        """
        previous = self.__dict__.get('_heading_memory') or {}
        current: Dict[int, Tuple[float, float]] = {}
        shared = set()
        for vehicle in getattr(self, 'obstacle_vehicles', None) or ():
            try:
                heading = float(vehicle.get_transform().rotation.yaw)
            except Exception:  # noqa: BLE001 - a detection without a transform
                continue
            key = self._actor_id(vehicle)
            if key < 0:
                continue                    # an unmatched detection has no identity to track
            if key in current or key in shared:
                shared.add(key)             # two objects under one id: neither is tracked
                current.pop(key, None)
                continue
            rate = 0.0
            if key in previous:
                change = (heading - previous[key][0] + 180.0) % 360.0 - 180.0
                rate = change / self._step_seconds
            current[key] = (heading, rate)
        self._heading_memory = current

    def _obstacle_yaw_rate_deg(self, vehicle) -> float:
        """The vehicle's yaw rate in degrees per second: CARLA's angular
        velocity when the object has one, else this step's heading-change
        estimate (see :meth:`_track_obstacle_headings`); 0 for a detection
        without an actor id."""
        if callable(getattr(vehicle, 'get_angular_velocity', None)):
            return self._yaw_rate_deg(vehicle)
        key = self._actor_id(vehicle)
        if key < 0:
            return 0.0
        entry = (self.__dict__.get('_heading_memory') or {}).get(key)
        return entry[1] if entry else 0.0

    @staticmethod
    def _choose_branch(candidates, current_yaw_deg: float, yaw_rate_deg: float, dt: float):
        """At a fork, the branch whose heading is closest to the vehicle's heading
        extrapolated by its yaw rate over the step: a vehicle already turning
        is predicted along its turn, a straight-going one straight on. The old
        walk took the first branch and mispredicted every turning vehicle."""
        if len(candidates) == 1:
            return candidates[0]
        predicted = current_yaw_deg + yaw_rate_deg * dt

        def gap(candidate):
            diff = (candidate.transform.rotation.yaw - predicted + 180.0) % 360.0 - 180.0
            return abs(diff)

        return min(candidates, key=gap)

    def _predict_vehicle_trajectory(self, vehicle_transform: 'carla.Transform', 
                                  vehicle_speed: float, is_ego: bool = False,
                                  yaw_rate_deg: float = 0.0) -> List[Tuple[float, float, float]]:
        """
        Predict vehicle trajectory for lookahead time.
        
        Args:
            vehicle_transform: Vehicle's current transform
            vehicle_speed: Vehicle speed in km/h
            is_ego: Whether this is the ego vehicle
            
        Returns:
            List[Tuple[float, float, float]]: List of (x, y, time) points along predicted path
        """
        trajectory = []
        
        try:
            if is_ego:
                # Use ego vehicle's planned waypoints from local planner
                local_planner = self.get_local_planner()
                if local_planner and hasattr(local_planner, 'get_waypoint_buffer'):
                    waypoint_buffer = local_planner.get_waypoint_buffer()
                    if waypoint_buffer:
                        current_time = 0.0
                        speed_ms = vehicle_speed / 3.6  # Convert km/h to m/s
                        
                        for waypoint, _ in itertools.islice(waypoint_buffer, 20):  # Limit to avoid infinite loops
                            if current_time > self.trajectory_lookahead_time:
                                break
                            
                            trajectory.append((
                                waypoint.transform.location.x,
                                waypoint.transform.location.y,
                                current_time
                            ))
                            
                            # Estimate time to next waypoint (simplified)
                            current_time += self.trajectory_step_size / max(speed_ms, 1.0)
            else:
                # For other vehicles, predict based on current heading and speed
                speed_ms = vehicle_speed / 3.6
                if speed_ms < 0.1:  # Nearly stopped
                    return trajectory
                
                # Get current waypoint and follow road topology
                current_waypoint = self._map.get_waypoint(vehicle_transform.location)
                if not current_waypoint:
                    return trajectory
                
                current_time = 0.0
                distance_covered = 0.0
                waypoint = current_waypoint
                
                while (current_time < self.trajectory_lookahead_time and 
                       distance_covered < self.trajectory_lookahead_time * speed_ms):
                    
                    trajectory.append((
                        waypoint.transform.location.x,
                        waypoint.transform.location.y,
                        current_time
                    ))
                    
                    # Get next waypoint: at a fork, the branch the vehicle's yaw rate points to
                    next_waypoints = waypoint.next(self.trajectory_step_size)
                    if not next_waypoints:
                        break
                    
                    step_dt = self.trajectory_step_size / speed_ms
                    waypoint = self._choose_branch(list(next_waypoints), waypoint.transform.rotation.yaw,
                                                   yaw_rate_deg, step_dt)
                    distance_covered += self.trajectory_step_size
                    current_time += self.trajectory_step_size / speed_ms
                    
        except Exception as e:
            logger.error(f"Error predicting trajectory: {e}")
        
        return trajectory
    
    def _find_collision_point(self, ego_trajectory: List[Tuple[float, float, float]], 
                            other_trajectory: List[Tuple[float, float, float]]) -> Optional[Tuple[float, float, Tuple[float, float]]]:
        """
        Find collision point between two trajectories.
        
        Args:
            ego_trajectory: Ego vehicle trajectory [(x, y, time), ...]
            other_trajectory: Other vehicle trajectory [(x, y, time), ...]
            
        Returns:
            Optional[Tuple[float, float, Tuple[float, float]]]: (ego_ttc, other_ttc, collision_point) or None
        """
        min_distance = float('inf')
        best_collision = None
        collision_threshold = 4.0  # meters - consider collision if closer than this
        
        try:
            for ego_point in ego_trajectory:
                ego_x, ego_y, ego_time = ego_point
                
                for other_point in other_trajectory:
                    other_x, other_y, other_time = other_point
                    
                    # Calculate distance between trajectory points
                    distance = math.sqrt((ego_x - other_x)**2 + (ego_y - other_y)**2)
                    
                    if distance < collision_threshold and distance < min_distance:
                        min_distance = distance
                        collision_point = ((ego_x + other_x) / 2, (ego_y + other_y) / 2)
                        best_collision = (ego_time, other_time, collision_point)
            
            return best_collision
            
        except Exception as e:
            logger.error(f"Error finding collision point: {e}")
            return None

    def _calculate_safe_following_speed(self, leading_vehicle: 'carla.Vehicle', gap: float, vanilla_speed: float) -> float:
        """
        Calculate safe following speed using time headway principle and leading vehicle speed.
        
        Args:
            leading_vehicle: The vehicle being followed
            gap: Bumper-to-bumper gap to the leading vehicle (metres)
            vanilla_speed: Base speed from VanillaAgent (km/h)
            
        Returns:
            float: Target speed for safe following (km/h)
        """
        try:
            # Get leading vehicle speed
            leading_speed = self._get_vehicle_speed(leading_vehicle)  # km/h
            
            # Calculate safe distance: minimum buffer + time-based distance
            time_based_distance = (self._ego_speed / 3.6) * self.time_headway  # convert km/h to m/s
            safe_distance = self.minimum_distance_buffer + time_based_distance
            
            # Calculate gap error
            gap_error = gap - safe_distance
            
            # Dynamic following speed calculation using vanilla_speed as base
            # 1. The speed that closes the gap: the leader's speed plus a braking
            #    curve of closing_decel that reaches zero exactly at the safe
            #    distance, so a follower approaches a stopped queue decisively,
            #    stops at the buffer, and never gains on its leader inside the
            #    safe distance. (A fixed 5 km/h closing cap crawled toward every
            #    queue, and a linear term gave 1-2 km/h targets near the buffer
            #    that the vehicle cannot execute.)
            if gap_error > 0:
                closing = math.sqrt(2.0 * self.closing_decel * gap_error) * 3.6   # m/s -> km/h
            else:
                closing = self.following_gain * gap_error                         # inside: open the gap
            
            # 2. Around the leader's speed, not the ego's own: a follower moves
            #    off with its leader instead of waiting for a gap to open.
            gap_adjusted_speed = leading_speed + closing
            
            # 3. Never above the base speed, never below a stop.
            target_speed = max(0, min(vanilla_speed, gap_adjusted_speed))
            return target_speed
            
        except Exception as e:
            logger.error(f"Error calculating safe following speed: {e}")
            return self.cautious_speed
        
    # --------------------------------------------------------------------- #
    # Private Functions
    # --------------------------------------------------------------------- #
    def _get_vehicle_speed(self, vehicle: 'carla.Vehicle') -> float:
        """
        Get vehicle speed in km/h.
        
        Args:
            vehicle: CARLA vehicle
            
        Returns:
            float: Speed in km/h
        """
        try:
            velocity = vehicle.get_velocity()
            speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            return speed_ms * 3.6  # Convert to km/h
        except Exception as e:
            logger.error(f"Error getting vehicle speed: {e}")
            return 0.0
    
    #: Half the length of a passenger car, for a vehicle without a bounding box.
    _DEFAULT_HALF_LENGTH = 2.4

    @classmethod
    def _half_length(cls, vehicle) -> float:
        """Half the vehicle's length along its axis, from its bounding box."""
        try:
            return float(vehicle.bounding_box.extent.x)
        except Exception:  # noqa: BLE001 - a stand-in without a bounding box
            return cls._DEFAULT_HALF_LENGTH

    @staticmethod
    def _half_width(vehicle) -> float:
        """Vehicle half-width, or a passenger-car fallback for stand-ins."""
        try:
            return float(vehicle.bounding_box.extent.y)
        except Exception:  # noqa: BLE001 - a stand-in without a bounding box
            return 0.9

    def _hold_lane(self, target_location):
        """The planner's target, or a point ahead on the current lane when the
        target lies on another lane that a vehicle occupies alongside.

        Returns ``(location, cap)``: ``cap`` is ``None`` when the target passes
        through, else the speed (km/h) to hold, ``lane_hold_drop_kmh`` below the
        slowest occupant, so the ego falls behind and the gap opens. Inside a
        junction the planner's target is never replaced (turning paths cross
        lanes by design)."""
        self.lane_hold_active = False
        self.merge_priority_release = False
        self.merge_priority_override = False
        if not self.lane_hold or target_location is None or not self._ego_pos or not hasattr(self, 'obstacle_vehicles'):
            return target_location, None
        try:
            ego_location = self._ego_pos.location
            ego_wp = self._map.get_waypoint(ego_location)
            if ego_wp is None or getattr(ego_wp, 'is_junction', False):
                return target_location, None
            yaw = math.radians(ego_wp.transform.rotation.yaw)
            fx, fy = math.cos(yaw), math.sin(yaw)          # along the lane
            px, py = -fy, fx                                # across it
            centre = ego_wp.transform.location
            target_side = (target_location.x - centre.x) * px + (target_location.y - centre.y) * py
            if abs(target_side) < 1.75:
                return target_location, None                # the target is on this lane
            blocking = []                                   # (speed, along, actor id) of each occupant in the way
            geometry_priority = bool(getattr(self, 'lane_hold_geometry_priority', True))
            precedence = {}
            for vehicle in self.obstacle_vehicles:
                location = vehicle.get_location()
                side = (location.x - centre.x) * px + (location.y - centre.y) * py
                if side * target_side <= 0.0 or not (1.0 < abs(side) < 5.25):
                    continue                                # not on the target side, or not the next lane
                along = (location.x - ego_location.x) * fx + (location.y - ego_location.y) * fy
                speed = self._get_vehicle_speed(vehicle)
                if speed < self.lane_hold_standing_kmh:
                    # A standing occupant (a queue in the target lane) blocks only
                    # where the bodies would overlap: the ego may slip into a gap
                    # beside it, and must not stop next to it and freeze its own
                    # lane too, which starved the spawn points upstream.
                    blocks = abs(along) <= self.lane_hold_overlap_m
                else:
                    blocks = -self.lane_hold_behind_m <= along <= self.lane_hold_ahead_m
                if blocks:
                    other_id = self._actor_id(vehicle)
                    blocking.append((speed, along, other_id))
                    if geometry_priority:
                        ego_id = self._actor_id(self._vehicle)
                        try:
                            precedence[other_id] = standing_merge_precedes(
                                ego_id, other_id,
                                location.x - ego_location.x, location.y - ego_location.y,
                                math.radians(self._ego_pos.rotation.yaw),
                                math.radians(vehicle.get_transform().rotation.yaw),
                                self.same_lane_tolerance)
                        except (AttributeError, TypeError, ValueError):
                            precedence[other_id] = ego_id < other_id
            if not blocking:
                return target_location, None
            ahead = ego_wp.next(self.lane_hold_lookahead_m)
            if not ahead:
                return target_location, None
            self.lane_hold_active = True
            ego_id = self._actor_id(self._vehicle)
            if (self._ego_speed < self.lane_hold_standing_kmh
                    and all(s < self.lane_hold_standing_kmh and abs(a) <= self.lane_hold_overlap_m for s, a, _ in blocking)
                    and all(precedence.get(other_id, ego_id < other_id) for _, _, other_id in blocking)):
                # Clear the current lane before attempting the merge. The optional
                # geometry rule gives the front vehicle precedence; stable IDs
                # still break close ties. Other safety caps remain in force.
                self.merge_priority_release = True
                self.merge_priority_override = geometry_priority and not all(
                    ego_id < other_id for _, _, other_id in blocking)
                return ahead[0].transform.location, None
            return ahead[0].transform.location, max(0.0, min(s for s, _, _ in blocking) - self.lane_hold_drop_kmh)
        except Exception as e:  # noqa: BLE001 - a dead actor or a stand-in without a map
            logger.error(f"Error in the lane hold: {e}")
            # The planner's target passes through, so nothing was held.
            self.lane_hold_active = False
            self.merge_priority_release = False
            self.merge_priority_override = False
            return target_location, None

    def _reset_step_diagnostics(self) -> None:
        """Clear what :meth:`shield_state` reports, before this step sets it."""
        self.lane_hold_active = False
        self.merge_priority_release = False
        self.merge_priority_override = False
        self._applied_speed = None

    @staticmethod
    def _actor_id(actor) -> int:
        """The actor's CARLA id, whether it is a raw actor or a wrapper carrying carla_id."""
        for attr in ('carla_id', 'id'):
            value = getattr(actor, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return 0

    def _abreast_cap(self) -> Optional[float]:
        """The speed cap of the abreast guard, or ``None`` when no same-direction
        vehicle alongside is closing in. Inside ``abreast_hard_m`` a neighbour
        always binds; between that and ``abreast_lateral_m`` it binds only when
        its lateral distance shrank since the last step by ``abreast_closing_m``
        and it is moving, so parallel queues do not lock each other. The
        vehicle that is behind yields (ties by the larger actor id): its cap is
        the neighbour's speed less ``abreast_drop_kmh``, floored at zero."""
        if not self._ego_pos or not hasattr(self, 'obstacle_vehicles'):
            return None
        ego_heading = math.radians(self._ego_pos.rotation.yaw)
        hx, hy = math.cos(ego_heading), math.sin(ego_heading)
        ego_location = self._ego_pos.location
        # Perception hands over ObstacleVehicle objects, which carry carla_id
        # and no id: reading .id made every neighbour id 0, so the tie-break
        # always yielded and every neighbour shared one memory slot.
        ego_id = self._actor_id(self._vehicle)
        cap = None
        memory = self._abreast_memory
        seen: Dict[int, float] = {}
        try:
            for vehicle in self.obstacle_vehicles:
                heading = math.radians(vehicle.get_transform().rotation.yaw)
                diff = abs(ego_heading - heading)
                if diff > math.pi:
                    diff = 2 * math.pi - diff
                if diff > self.same_lane_tolerance:
                    continue                                 # not the same direction
                location = vehicle.get_location()
                dx, dy = location.x - ego_location.x, location.y - ego_location.y
                longitudinal = dx * hx + dy * hy
                lateral = abs(-dx * hy + dy * hx)
                if abs(longitudinal) > self.abreast_longitudinal_m or lateral >= self.abreast_lateral_m:
                    continue                                 # not alongside, or a safe lane apart
                other_id = self._actor_id(vehicle)
                seen[other_id] = lateral
                speed = self._get_vehicle_speed(vehicle)
                last = memory.get(other_id)
                closing = last is not None and lateral < last - self.abreast_closing_m
                if lateral >= self.abreast_hard_m and (not closing or speed < 1.0):
                    continue                                 # soft zone: a steady or stopped neighbour is not a threat
                behind = longitudinal > 0.5 or (abs(longitudinal) <= 0.5 and ego_id > other_id)
                if not behind:
                    continue                                 # the neighbour is the one to drop back
                if speed < 1.0 and self._ego_speed < 1.0 and not closing:
                    limit = self.abreast_creep_kmh           # a stationary pair cannot collide: creep ahead and end it
                else:
                    limit = max(0.0, speed - self.abreast_drop_kmh)
                cap = limit if cap is None else min(cap, limit)
        except Exception as e:
            logger.error(f"Error in the abreast guard: {e}")
            self._abreast_memory = seen
            return None
        self._abreast_memory = seen
        return cap

    def _get_leading_vehicle(self) -> Tuple[Optional['carla.Vehicle'], float]:
        """
        Find the leading vehicle in the same lane.
        
        Returns:
            Tuple[Optional[carla.Vehicle], float]: (vehicle, bumper-to-bumper gap) or (None, inf).
            The gap is the centre distance less both half-lengths, floored at zero,
            so a following buffer is kept between bumpers whatever the vehicle
            length; measured between centres it would let queued cars nearly touch.
        """
        if not self._ego_pos or not hasattr(self, 'obstacle_vehicles'):
            return None, float('inf')
        
        ego_heading = math.radians(self._ego_pos.rotation.yaw)
        ego_location = self._ego_pos.location
        
        closest_vehicle = None
        min_distance = float('inf')
        
        try:
            for vehicle in self.obstacle_vehicles:
                vehicle_location = vehicle.get_location()
                vehicle_heading = math.radians(vehicle.get_transform().rotation.yaw)
                
                # Heading similarity alone also accepts the adjacent lane.
                heading_diff = abs(ego_heading - vehicle_heading)
                if heading_diff > math.pi:
                    heading_diff = 2 * math.pi - heading_diff
                
                if heading_diff > self.same_lane_tolerance:
                    continue  # Not in same lane
                
                # Calculate distance and direction
                distance = ego_location.distance(vehicle_location)
                
                # Check if vehicle is in front (within front cone)
                dx = vehicle_location.x - ego_location.x
                dy = vehicle_location.y - ego_location.y
                # A leader must overlap the ego's forward driving corridor.
                # Project its footprint for a merging/turning vehicle, keeping
                # a 20 cm margin. A queue one lane over must not stop a clear
                # lane; lateral merges are separately protected by lane hold.
                lateral = abs(-math.sin(ego_heading) * dx + math.cos(ego_heading) * dy)
                projected_width = (
                    self._half_width(vehicle) * abs(math.cos(heading_diff))
                    + self._half_length(vehicle) * abs(math.sin(heading_diff)))
                if lateral > self._half_width(self._vehicle) + projected_width + 0.2:
                    continue
                angle_to_vehicle = math.atan2(dy, dx)
                
                angle_diff = abs(ego_heading - angle_to_vehicle)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                
                # Vehicle must be in front cone and closer than current minimum
                if angle_diff < self.front_cone_angle and distance < min_distance:
                    min_distance = distance
                    closest_vehicle = vehicle
                    
        except Exception as e:
            logger.error(f"Error finding leading vehicle: {e}")
        
        if closest_vehicle is None:
            return None, float('inf')
        gap = min_distance - self._half_length(self._vehicle) - self._half_length(closest_vehicle)
        return closest_vehicle, max(0.0, gap)
    
    # --------------------------------------------------------------------- #
    # Public Functions
    # --------------------------------------------------------------------- #
    def get_current_stage(self) -> int:
        """Get current active stage (for debugging/monitoring)."""
        return self.current_stage
    
    def shield_state(self) -> Dict[str, object]:
        """This step's lane-hold and merge-order state, for the step trace.

        The duck-typed hook MARLVehicleAdapter reads on CARLA: it merges the
        fields into the vehicle's observation and the trace writer records
        them. Only the fields this agent has a value for are returned:

        - ``lane_hold``: the lane hold replaced the planner's target this step;
        - ``merge_priority_enabled``: the configured standing-merge order
          (``lane_hold_geometry_priority``);
        - ``merge_priority_release`` / ``merge_priority_override``: this
          step's merge-order flags (see :meth:`_hold_lane`);
        - ``applied_speed``: the target speed (km/h) this step's rules handed
          the controller, absent when they did not run (before the first
          step, or on a fallback).

        The agent has no safety filter over a commanded speed, so
        ``shield_active`` and ``shield_stage`` are never reported.
        """
        state = {
            'lane_hold': bool(self.lane_hold_active),
            'merge_priority_enabled': bool(getattr(self, 'lane_hold_geometry_priority', True)),
            'merge_priority_release': bool(self.merge_priority_release),
            'merge_priority_override': bool(self.merge_priority_override),
        }
        if self._applied_speed is not None:
            state['applied_speed'] = round(float(self._applied_speed), 2)
        return state

    def get_debug_info(self) -> Dict:
        """Get debug information about current state."""
        return {
            'agent_type': 'rule_based_jair_enhanced',
            'current_stage': self.current_stage,
            'stage_name': {1: 'Junction', 2: 'Following', 3: 'Cruising'}.get(self.current_stage, 'Unknown'),
            'junction_distance': self._get_junction_distance(),
            'leading_vehicle_distance': self._get_leading_vehicle()[1],
            'ego_speed': self._ego_speed,
            'configuration': {
                'junction_approach_distance': self.junction_approach_distance,
                'cautious_speed': self.cautious_speed,
                'trajectory_lookahead_time': self.trajectory_lookahead_time,
                'ttc_threshold': self.ttc_threshold,
                'time_headway': self.time_headway,
                'max_speed': self.max_speed,
                'cruising_speed_factor': self.cruising_speed_factor
            }
        }
