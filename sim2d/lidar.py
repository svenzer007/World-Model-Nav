"""2D LiDAR ray casting against static and dynamic obstacles."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import LidarConfig
from .utils import wrap_angle, world_to_grid


def beam_angles(config: LidarConfig) -> np.ndarray:
    """Return the fixed LiDAR beam angles."""
    return np.linspace(config.angle_min, config.angle_max, config.num_beams, dtype=float)


def cast_lidar_scan(
    robot_pose: Sequence[float],
    occupancy: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    lidar_config: LidarConfig,
    dynamic_obstacles: Sequence[object],
) -> np.ndarray:
    """Cast a fixed LiDAR scan."""
    pose = np.asarray(robot_pose, dtype=float)
    x0, y0, theta = pose
    ranges = np.full((lidar_config.num_beams,), lidar_config.max_range, dtype=np.float32)
    angles = beam_angles(lidar_config)
    sample_distances = np.arange(lidar_config.step_size, lidar_config.max_range + lidar_config.step_size, lidar_config.step_size)

    for beam_index, rel_angle in enumerate(angles):
        beam_theta = wrap_angle(theta + float(rel_angle))
        cos_theta = np.cos(beam_theta)
        sin_theta = np.sin(beam_theta)
        for distance in sample_distances:
            x = x0 + distance * cos_theta
            y = y0 + distance * sin_theta
            row, col = world_to_grid((x, y), resolution, origin)
            if row < 0 or row >= occupancy.shape[0] or col < 0 or col >= occupancy.shape[1]:
                ranges[beam_index] = np.float32(min(distance, lidar_config.max_range))
                break
            if occupancy[row, col] != 0:
                ranges[beam_index] = np.float32(distance)
                break
            if _point_in_dynamic_obstacle((x, y), dynamic_obstacles):
                ranges[beam_index] = np.float32(distance)
                break
    return ranges


def _point_in_dynamic_obstacle(point_xy: Sequence[float], dynamic_obstacles: Sequence[object]) -> bool:
    point = np.asarray(point_xy, dtype=float)
    for obstacle in dynamic_obstacles:
        if np.linalg.norm(point - obstacle.position) <= obstacle.radius:
            return True
    return False
