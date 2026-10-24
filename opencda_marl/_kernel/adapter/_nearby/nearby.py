'''
Standalone nearby-vehicle detection helpers for MARLVehicleAdapter.

Functions find K-nearest vehicles, compute Time-to-Collision (TTC),
and build the normalised feature vector consumed by MARL policies.
'''
from loguru import logger
import numpy as np

# Default nearby vehicle detection constants (can be overridden by config)
DEFAULT_NEARBY_DETECTION_RADIUS = 50.0  # meters - detection radius for nearby vehicles
DEFAULT_MAX_NEARBY_VEHICLES = 5  # Maximum number of nearby vehicles to track (K slots)
DEFAULT_FEATURES_PER_VEHICLE = 7  # Features per vehicle: rel_x, rel_y, rel_vx, rel_vy, heading_diff, distance, ttc
DEFAULT_MAX_RELATIVE_VELOCITY = 40.0  # m/s (~144 km/h) - for normalization
DEFAULT_MAX_TTC = 10.0  # seconds - TTC values clamped to this maximum


def get_nearby_vehicles(
    vehicle, world,
    nearby_detection_radius: float,
    max_nearby_vehicles: int,
) -> list:
    """
    Get nearby vehicles within detection radius, sorted by distance.

    Returns:
        List of tuples: [(vehicle, distance), ...] sorted by distance, max K vehicles
    """
    try:
        current_location = vehicle.get_location()
        all_vehicles = world.get_actors().filter('vehicle.*')

        nearby = []
        for other in all_vehicles:
            if other.id == vehicle.id:
                continue

            if not other.is_alive:
                continue

            other_loc = other.get_location()

            # Quick bounding box check (faster than full distance calculation)
            if (abs(other_loc.x - current_location.x) > nearby_detection_radius or
                abs(other_loc.y - current_location.y) > nearby_detection_radius):
                continue

            distance = current_location.distance(other_loc)
            if distance <= nearby_detection_radius:
                nearby.append((other, distance))

        nearby.sort(key=lambda x: x[1])
        return nearby[:max_nearby_vehicles]

    except Exception as e:
        logger.debug(f"Error getting nearby vehicles for {vehicle.id}: {e}")
        return []


def calculate_ttc_to_vehicle(vehicle, other_vehicle) -> float:
    """
    Calculate Time-to-Collision (TTC) to another vehicle using constant velocity assumption.

    Returns:
        float: TTC in seconds (inf if no collision predicted within horizon)
    """
    try:
        ego_loc = vehicle.get_location()
        ego_vel = vehicle.get_velocity()
        other_loc = other_vehicle.get_location()
        other_vel = other_vehicle.get_velocity()

        rel_x = other_loc.x - ego_loc.x
        rel_y = other_loc.y - ego_loc.y
        rel_vx = other_vel.x - ego_vel.x
        rel_vy = other_vel.y - ego_vel.y

        collision_radius = 4.0  # Combined vehicle radii (meters)

        a = rel_vx**2 + rel_vy**2
        b = 2 * (rel_x * rel_vx + rel_y * rel_vy)
        c = rel_x**2 + rel_y**2 - collision_radius**2

        if abs(a) < 1e-6:
            return float('inf')

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return float('inf')

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return float('inf')

    except Exception as e:
        logger.debug(f"Error calculating TTC: {e}")
        return float('inf')


def compute_nearby_vehicle_features(
    vehicle, world,
    nearby_detection_radius: float,
    max_nearby_vehicles: int,
    features_per_vehicle: int,
    max_relative_velocity: float,
    max_ttc: float,
) -> tuple:
    """
    Compute normalized features for nearby vehicles (35D total = 5 slots x 7 features).

    Returns:
        Tuple[List[float], float]: (feature vector, minimum TTC across all vehicles)
    """
    try:
        nearby_vehicles = get_nearby_vehicles(
            vehicle, world, nearby_detection_radius, max_nearby_vehicles)

        ego_loc = vehicle.get_location()
        ego_vel = vehicle.get_velocity()
        ego_heading = np.radians(vehicle.get_transform().rotation.yaw)

        features = []
        min_ttc = float('inf')

        for i in range(max_nearby_vehicles):
            if i < len(nearby_vehicles):
                other, distance = nearby_vehicles[i]
                other_loc = other.get_location()
                other_vel = other.get_velocity()
                other_heading = np.radians(other.get_transform().rotation.yaw)

                rel_x = (other_loc.x - ego_loc.x) / nearby_detection_radius
                rel_y = (other_loc.y - ego_loc.y) / nearby_detection_radius

                rel_vx = (other_vel.x - ego_vel.x) / max_relative_velocity
                rel_vy = (other_vel.y - ego_vel.y) / max_relative_velocity

                heading_diff = other_heading - ego_heading
                while heading_diff > np.pi:
                    heading_diff -= 2 * np.pi
                while heading_diff < -np.pi:
                    heading_diff += 2 * np.pi
                heading_diff_norm = heading_diff / np.pi

                distance_norm = distance / nearby_detection_radius

                ttc = calculate_ttc_to_vehicle(vehicle, other)
                min_ttc = min(min_ttc, ttc)
                ttc_norm = min(ttc, max_ttc) / max_ttc

                features.extend([
                    np.clip(rel_x, -1.0, 1.0),
                    np.clip(rel_y, -1.0, 1.0),
                    np.clip(rel_vx, -1.0, 1.0),
                    np.clip(rel_vy, -1.0, 1.0),
                    np.clip(heading_diff_norm, -1.0, 1.0),
                    np.clip(distance_norm, 0.0, 1.0),
                    np.clip(ttc_norm, 0.0, 1.0)
                ])
            else:
                # Empty slot: zeros except distance and TTC = 1.0 (safe/far away)
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

        return features, min_ttc

    except Exception as e:
        logger.debug(f"Error computing nearby vehicle features: {e}")
        default_features = []
        for _ in range(max_nearby_vehicles):
            default_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        return default_features, float('inf')
