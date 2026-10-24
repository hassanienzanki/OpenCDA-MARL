"""Role-independent precedence for two standing vehicles beside a merge."""

import math


def standing_merge_precedes(
    ego_id: int,
    other_id: int,
    dx: float,
    dy: float,
    ego_yaw: float,
    other_yaw: float,
    heading_tolerance: float,
) -> bool:
    """Let the front vehicle clear first, using IDs for half-metre ties.

    ``dx`` and ``dy`` locate the other vehicle relative to the ego. Headings
    and their tolerance are in radians. Their shared mean heading makes the
    ordering consistent when the two vehicles are exchanged. Incompatible or
    non-finite geometry retains the legacy ID ordering.
    """
    values = (dx, dy, ego_yaw, other_yaw, heading_tolerance)
    if not all(math.isfinite(value) for value in values):
        return ego_id < other_id
    gap = abs(math.atan2(math.sin(ego_yaw - other_yaw), math.cos(ego_yaw - other_yaw)))
    if gap > heading_tolerance:
        return ego_id < other_id
    hx = math.cos(ego_yaw) + math.cos(other_yaw)
    hy = math.sin(ego_yaw) + math.sin(other_yaw)
    norm = math.hypot(hx, hy)
    if norm < 1e-9:
        return ego_id < other_id
    along = (dx * hx + dy * hy) / norm
    if along < -0.5:
        return True
    if along > 0.5:
        return False
    return ego_id < other_id
