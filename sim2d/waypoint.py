"""Local waypoint selection from a global path."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .utils import cumulative_path_lengths, nearest_point_index, world_to_robot_frame


def compute_local_subgoal(
    robot_pose: Sequence[float],
    path_world: np.ndarray,
    lookahead_distance: float,
) -> dict[str, object]:
    """Compute a robot-frame lookahead subgoal from a world-frame path."""
    if len(path_world) == 0:
        raise ValueError("Cannot compute a subgoal from an empty path")

    nearest_index = nearest_point_index(path_world, robot_pose[:2])
    cum_lengths = cumulative_path_lengths(path_world)
    target_distance = cum_lengths[nearest_index] + lookahead_distance
    selected_index = int(np.searchsorted(cum_lengths, target_distance, side="left"))
    selected_index = min(selected_index, len(path_world) - 1)
    subgoal_world = path_world[selected_index]
    goal_robot = world_to_robot_frame(robot_pose, subgoal_world)
    x_rel = float(goal_robot[0])
    y_rel = float(goal_robot[1])
    distance = float(np.hypot(x_rel, y_rel))
    alpha = float(math.atan2(y_rel, x_rel))
    goal_features = np.array([x_rel, y_rel, distance, np.cos(alpha), np.sin(alpha)], dtype=np.float32)
    return {
        "goal_features": goal_features,
        "nearest_path_index": int(nearest_index),
        "selected_subgoal_world": subgoal_world.copy(),
        "selected_subgoal_index": int(selected_index),
    }
