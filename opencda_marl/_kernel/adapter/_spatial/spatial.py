'''
Standalone spatial / waypoint-based observation helpers for MARLVehicleAdapter.

Functions compute distances to intersections, lane classification,
heading angles, and destination proximity using CARLA waypoint queries.
'''
from loguru import logger
import numpy as np


def calculate_distance_to_intersection(vm, carla_map, actor_id: int) -> float:
    """Calculate distance to nearest intersection in meters."""
    try:
        if hasattr(vm, 'agent') and hasattr(vm.agent, '_ego_pos') and vm.agent._ego_pos:
            current_location = vm.agent._ego_pos.location

            current_waypoint = carla_map.get_waypoint(current_location)
            if not current_waypoint:
                return 100.0

            search_distance = 100.0
            waypoint = current_waypoint
            distance_traveled = 0.0

            while distance_traveled < search_distance:
                if waypoint.is_intersection:
                    return distance_traveled

                next_waypoints = waypoint.next(2.0)
                if not next_waypoints:
                    break

                waypoint = next_waypoints[0]
                distance_traveled += 2.0

            return search_distance
        else:
            return 100.0

    except Exception as e:
        logger.debug(
            f"Error calculating distance to intersection for vehicle {actor_id}: {e}")
        return 100.0


def calculate_distance_to_front_vehicle(vehicle, world, carla_map, actor_id: int) -> float:
    """Calculate distance to front vehicle in meters."""
    try:
        current_location = vehicle.get_location()
        all_vehicles = world.get_actors().filter('vehicle.*')

        min_distance = 100.0
        current_waypoint = carla_map.get_waypoint(current_location)

        if not current_waypoint:
            logger.debug(f"No waypoint found for vehicle {actor_id}")
            return 999.0

        vehicles_checked = 0
        vehicles_ahead = 0

        for other_vehicle in all_vehicles:
            if other_vehicle.id == vehicle.id:
                continue

            if not other_vehicle.is_alive:
                continue

            vehicles_checked += 1
            other_location = other_vehicle.get_location()

            rough_distance = current_location.distance(other_location)
            if rough_distance > min_distance:
                continue

            other_waypoint = carla_map.get_waypoint(other_location)
            if not other_waypoint:
                continue

            if (other_waypoint.lane_id == current_waypoint.lane_id and
                    other_waypoint.road_id == current_waypoint.road_id):

                vehicle_transform = vehicle.get_transform()
                forward_vector = vehicle_transform.get_forward_vector()
                to_other = other_location - current_location

                dot_product = forward_vector.x * to_other.x + forward_vector.y * to_other.y
                if dot_product > 0:
                    vehicles_ahead += 1
                    if rough_distance < min_distance:
                        min_distance = rough_distance

        if vehicles_checked == 0:
            logger.debug(
                f"Vehicle {actor_id}: No other vehicles found in world")
        elif vehicles_ahead == 0:
            logger.debug(
                f"Vehicle {actor_id}: No vehicles ahead in same lane (checked {vehicles_checked} vehicles)")
        else:
            logger.debug(
                f"Vehicle {actor_id}: Found {vehicles_ahead} vehicles ahead, closest at {min_distance:.1f}m")

        return min_distance if min_distance < 100.0 else 999.0

    except Exception as e:
        logger.warning(
            f"Error calculating distance to front vehicle for vehicle {actor_id}: {e}")
        return 999.0


def is_approaching_intersection(current_waypoint, search_distance: float = 50.0) -> bool:
    """Check if approaching an intersection by looking ahead."""
    try:
        waypoint = current_waypoint
        distance = 0.0

        while distance < search_distance:
            next_waypoints = waypoint.next(2.0)
            if not next_waypoints:
                break

            waypoint = next_waypoints[0]
            distance += 2.0

            if waypoint.is_junction:
                return True

        return False

    except Exception as e:
        logger.debug(f"Error checking if approaching intersection: {e}")
        return False


def is_leaving_intersection(current_waypoint, search_distance: float = 30.0) -> bool:
    """Check if leaving an intersection by looking behind."""
    try:
        waypoint = current_waypoint
        distance = 0.0

        while distance < search_distance:
            prev_waypoints = waypoint.previous(2.0)
            if not prev_waypoints:
                break

            waypoint = prev_waypoints[0]
            distance += 2.0

            if waypoint.is_junction:
                return True

        return False

    except Exception as e:
        logger.debug(f"Error checking if leaving intersection: {e}")
        return False


def classify_lane_position(vehicle, carla_map, actor_id: int) -> int:
    """
    Classify lane position for Q-learning:
    Entry lanes: 1 (left), 2 (center), 3 (right)
    Exit lanes: -1 (left), -2 (center), -3 (right)
    Junction/unknown: 0
    """
    try:
        current_location = vehicle.get_location()
        current_waypoint = carla_map.get_waypoint(current_location)

        if not current_waypoint:
            return 0

        if current_waypoint.is_junction or current_waypoint.get_junction() is not None:
            return 0

        lane_id = current_waypoint.lane_id
        abs_lane = abs(lane_id)
        lane_number = min(abs_lane, 3)

        is_entry = is_approaching_intersection(current_waypoint)
        is_exit = is_leaving_intersection(current_waypoint)

        if is_entry:
            return lane_number
        elif is_exit:
            return -lane_number
        else:
            return 0

    except Exception as e:
        logger.debug(
            f"Error classifying lane position for vehicle {actor_id}: {e}")
        return 0


def calculate_relative_position_to_intersection(vehicle, carla_map) -> tuple:
    """Calculate relative position to nearest intersection (x, y in meters)."""
    try:
        current_location = vehicle.get_location()
        current_waypoint = carla_map.get_waypoint(current_location)

        if not current_waypoint:
            return (0.0, 0.0)

        search_distance = 100.0
        closest_intersection = None
        min_distance = float('inf')

        # Search forward
        waypoint = current_waypoint
        distance_traveled = 0.0
        while distance_traveled < search_distance:
            if waypoint.is_intersection:
                dist = current_location.distance(waypoint.transform.location)
                if dist < min_distance:
                    min_distance = dist
                    closest_intersection = waypoint.transform.location

            next_waypoints = waypoint.next(2.0)
            if not next_waypoints:
                break
            waypoint = next_waypoints[0]
            distance_traveled += 2.0

        # Search backward
        waypoint = current_waypoint
        distance_traveled = 0.0
        while distance_traveled < search_distance:
            if waypoint.is_intersection:
                dist = current_location.distance(waypoint.transform.location)
                if dist < min_distance:
                    min_distance = dist
                    closest_intersection = waypoint.transform.location

            prev_waypoints = waypoint.previous(2.0)
            if not prev_waypoints:
                break
            waypoint = prev_waypoints[0]
            distance_traveled += 2.0

        if closest_intersection:
            rel_x = closest_intersection.x - current_location.x
            rel_y = closest_intersection.y - current_location.y
            return (rel_x, rel_y)
        else:
            return (100.0, 0.0)

    except Exception as e:
        logger.debug(f"Error calculating relative position to intersection: {e}")
        return (0.0, 0.0)


def get_vehicle_heading_angle(vehicle) -> float:
    """Get vehicle heading angle in radians (-pi to pi)."""
    try:
        vehicle_transform = vehicle.get_transform()
        yaw_degrees = vehicle_transform.rotation.yaw
        yaw_radians = np.radians(yaw_degrees)

        while yaw_radians > np.pi:
            yaw_radians -= 2 * np.pi
        while yaw_radians < -np.pi:
            yaw_radians += 2 * np.pi

        return yaw_radians

    except Exception as e:
        logger.debug(f"Error getting vehicle heading angle: {e}")
        return 0.0


def calculate_distance_to_destination(vm, vehicle, actor_id: int) -> float:
    """Calculate distance to destination in meters (999.0 if not available)."""
    try:
        if hasattr(vm, 'agent') and hasattr(vm.agent, '_local_planner'):
            planner = vm.agent._local_planner
            if hasattr(planner, '_destination') and planner._destination:
                current_loc = vehicle.get_location()
                dest_loc = planner._destination
                return current_loc.distance(dest_loc)
        return 999.0
    except Exception as e:
        logger.debug(f"Error calculating distance to destination: {e}")
        return 999.0
