"""2D Pareto hypervolume helper for the safety x efficiency analysis."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def hypervolume_2d(
    points: np.ndarray,
    reference_point: Tuple[float, float] = (0.0, 0.0),
) -> float:
    """Compute the 2D hypervolume dominated by ``points`` over ``reference_point``.

    Both dimensions are maximized. Points dominated by other points contribute
    nothing. A point at or below the reference in any dimension contributes
    nothing. If the (filtered) set of points is empty, raises ValueError.

    Args:
        points: ``(N, 2)`` array of objective values.
        reference_point: lower-left anchor; both dims should be reachable.

    Returns:
        Scalar hypervolume.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be shape (N, 2); got {pts.shape}")
    ref_x, ref_y = reference_point
    pts = pts[(pts[:, 0] > ref_x) & (pts[:, 1] > ref_y)]
    if pts.size == 0:
        raise ValueError("no point dominates the reference point")

    order = np.argsort(-pts[:, 0])
    pts = pts[order]
    hv = 0.0
    best_y_so_far = ref_y
    for x, y in pts:
        if y <= best_y_so_far:
            continue
        hv += (x - ref_x) * (y - best_y_so_far)
        best_y_so_far = y
    return float(hv)
