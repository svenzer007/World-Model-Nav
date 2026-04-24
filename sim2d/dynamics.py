"""Robot motion and collision helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .utils import in_bounds_rc, wrap_angle, world_to_grid


def unicycle_step(pose: Sequence[float], v: float, omega: float, dt: float) -> np.ndarray:
    """Advance a unicycle model by one time step."""
    x, y, theta = [float(value) for value in pose]
    x_next = x + float(v) * np.cos(theta) * dt
    y_next = y + float(v) * np.sin(theta) * dt
    theta_next = wrap_angle(theta + float(omega) * dt)
    return np.array([x_next, y_next, theta_next], dtype=float)


def disk_collides_with_occupancy(
    position_xy: Sequence[float],
    radius: float,
    occupancy: np.ndarray,
    resolution: float,
    origin: Sequence[float],
) -> bool:
    """Check whether a circular footprint intersects occupied cells."""
    row_center, col_center = world_to_grid(position_xy, resolution, origin)
    margin_cells = int(np.ceil(radius / resolution)) + 2
    cell_radius = np.sqrt(2.0) * resolution * 0.5

    if not in_bounds_rc(row_center, col_center, occupancy.shape):
        return True

    px = float(position_xy[0])
    py = float(position_xy[1])
    for row in range(row_center - margin_cells, row_center + margin_cells + 1):
        for col in range(col_center - margin_cells, col_center + margin_cells + 1):
            if not in_bounds_rc(row, col, occupancy.shape):
                return True
            if occupancy[row, col] == 0:
                continue
            cx = origin[0] + (col + 0.5) * resolution
            cy = origin[1] + (row + 0.5) * resolution
            if np.hypot(px - cx, py - cy) <= radius + cell_radius:
                return True
    return False


def disk_collides_with_dynamic(
    position_xy: Sequence[float],
    radius: float,
    dynamic_obstacles: Sequence[object],
) -> bool:
    """Check whether a circular footprint intersects any dynamic obstacle."""
    px = float(position_xy[0])
    py = float(position_xy[1])
    for obstacle in dynamic_obstacles:
        ox, oy = obstacle.position
        if np.hypot(px - ox, py - oy) <= radius + obstacle.radius:
            return True
    return False


def minimum_static_clearance(
    position_xy: Sequence[float],
    radius: float,
    occupied_points: np.ndarray,
    resolution: float,
) -> float:
    """Approximate clearance to the nearest occupied cell center."""
    if occupied_points.size == 0:
        return float("inf")
    point = np.asarray(position_xy, dtype=float)
    distances = np.linalg.norm(occupied_points - point[None, :], axis=1)
    cell_pad = np.sqrt(2.0) * resolution * 0.5
    return float(np.min(distances) - radius - cell_pad)


def minimum_dynamic_clearance(position_xy: Sequence[float], radius: float, dynamic_obstacles: Sequence[object]) -> float:
    """Return the minimum clearance to any dynamic obstacle."""
    if not dynamic_obstacles:
        return float("inf")
    point = np.asarray(position_xy, dtype=float)
    clearances = [
        float(np.linalg.norm(point - obstacle.position) - radius - obstacle.radius) for obstacle in dynamic_obstacles
    ]
    return float(min(clearances))


def minimum_combined_clearance(
    position_xy: Sequence[float],
    radius: float,
    occupied_points: np.ndarray,
    resolution: float,
    dynamic_obstacles: Sequence[object],
) -> float:
    """Return the minimum clearance to static and dynamic obstacles."""
    return float(
        min(
            minimum_static_clearance(position_xy, radius, occupied_points, resolution),
            minimum_dynamic_clearance(position_xy, radius, dynamic_obstacles),
        )
    )
