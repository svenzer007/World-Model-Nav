"""Utility helpers for geometry, serialization, and reproducible sampling."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def ensure_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a reproducible NumPy generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
    if wrapped == -np.pi:
        return np.pi
    return wrapped


def pose_to_array(x: float, y: float, theta: float) -> np.ndarray:
    """Return a pose as a float array."""
    return np.array([float(x), float(y), float(theta)], dtype=float)


def point_to_array(x: float, y: float) -> np.ndarray:
    """Return a 2D point as a float array."""
    return np.array([float(x), float(y)], dtype=float)


def world_to_grid(point_xy: Sequence[float], resolution: float, origin: Sequence[float]) -> tuple[int, int]:
    """Convert world coordinates to integer grid row/col indices."""
    x, y = point_xy
    col = int(np.floor((x - origin[0]) / resolution))
    row = int(np.floor((y - origin[1]) / resolution))
    return row, col


def grid_to_world(row: int, col: int, resolution: float, origin: Sequence[float]) -> np.ndarray:
    """Convert a grid row/col index to the center of the corresponding world cell."""
    x = origin[0] + (col + 0.5) * resolution
    y = origin[1] + (row + 0.5) * resolution
    return np.array([x, y], dtype=float)


def in_bounds_rc(row: int, col: int, grid_shape: Sequence[int]) -> bool:
    """Return True if a grid row/col is inside the map."""
    return 0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def rotation_matrix(theta: float) -> np.ndarray:
    """Return the 2x2 rotation matrix for the given angle."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def world_to_robot_frame(robot_pose: Sequence[float], point_world: Sequence[float]) -> np.ndarray:
    """Convert a world point into the robot frame."""
    pose = np.asarray(robot_pose, dtype=float)
    point = np.asarray(point_world, dtype=float)
    rel = point - pose[:2]
    rot = rotation_matrix(-pose[2])
    return rot @ rel


def robot_to_world_frame(robot_pose: Sequence[float], point_robot: Sequence[float]) -> np.ndarray:
    """Convert a point in the robot frame into world coordinates."""
    pose = np.asarray(robot_pose, dtype=float)
    point = np.asarray(point_robot, dtype=float)
    rot = rotation_matrix(pose[2])
    return pose[:2] + rot @ point


def path_length(path_world: np.ndarray) -> float:
    """Return the total arc length of a polyline."""
    if len(path_world) < 2:
        return 0.0
    deltas = np.diff(path_world, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def cumulative_path_lengths(path_world: np.ndarray) -> np.ndarray:
    """Return cumulative arc lengths for a polyline."""
    if len(path_world) == 0:
        return np.zeros((0,), dtype=float)
    if len(path_world) == 1:
        return np.zeros((1,), dtype=float)
    deltas = np.diff(path_world, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    return np.concatenate([np.zeros((1,), dtype=float), np.cumsum(lengths)])


def distance_point_to_segment(point_xy: Sequence[float], start_xy: Sequence[float], end_xy: Sequence[float]) -> float:
    """Return distance from a point to a segment."""
    point = np.asarray(point_xy, dtype=float)
    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    segment = end - start
    denom = float(segment @ segment)
    if denom == 0.0:
        return float(np.linalg.norm(point - start))
    projection = float(((point - start) @ segment) / denom)
    projection = float(np.clip(projection, 0.0, 1.0))
    closest = start + projection * segment
    return float(np.linalg.norm(point - closest))


def nearest_point_index(path_world: np.ndarray, point_xy: Sequence[float]) -> int:
    """Return the nearest path point index to the provided point."""
    if len(path_world) == 0:
        return 0
    point = np.asarray(point_xy, dtype=float)
    distances = np.linalg.norm(path_world - point[None, :], axis=1)
    return int(np.argmin(distances))


def distance_point_to_polyline(point_xy: Sequence[float], path_world: np.ndarray) -> float:
    """Return distance from a point to a polyline."""
    if len(path_world) == 0:
        return float("inf")
    if len(path_world) == 1:
        return euclidean(point_xy, path_world[0])
    distances = [
        distance_point_to_segment(point_xy, path_world[index], path_world[index + 1])
        for index in range(len(path_world) - 1)
    ]
    return float(min(distances))


def inflate_occupancy_grid(grid: np.ndarray, inflation_radius_m: float, resolution: float) -> np.ndarray:
    """Inflate occupied cells by a disk radius in meters."""
    if inflation_radius_m <= 0.0:
        return grid.copy()
    radius_cells = int(np.ceil(inflation_radius_m / resolution))
    offsets: list[tuple[int, int]] = []
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if np.hypot(dr, dc) <= radius_cells + 1e-6:
                offsets.append((dr, dc))
    inflated = grid.copy().astype(np.uint8)
    occupied_indices = np.argwhere(grid > 0)
    for row, col in occupied_indices:
        for dr, dc in offsets:
            rr = row + dr
            cc = col + dc
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                inflated[rr, cc] = 1
    return inflated


def free_cell_centers(grid: np.ndarray, resolution: float, origin: Sequence[float]) -> np.ndarray:
    """Return world coordinates of free grid-cell centers."""
    free_indices = np.argwhere(grid == 0)
    if len(free_indices) == 0:
        return np.zeros((0, 2), dtype=float)
    points = np.array([grid_to_world(int(row), int(col), resolution, origin) for row, col in free_indices], dtype=float)
    return points


def occupied_cell_centers(grid: np.ndarray, resolution: float, origin: Sequence[float]) -> np.ndarray:
    """Return world coordinates of occupied grid-cell centers."""
    occupied_indices = np.argwhere(grid > 0)
    if len(occupied_indices) == 0:
        return np.zeros((0, 2), dtype=float)
    points = np.array(
        [grid_to_world(int(row), int(col), resolution, origin) for row, col in occupied_indices],
        dtype=float,
    )
    return points


def sample_heading_toward(start_xy: Sequence[float], goal_xy: Sequence[float], noise_std: float, rng: np.random.Generator) -> float:
    """Sample a heading that roughly faces toward a goal point."""
    delta = np.asarray(goal_xy, dtype=float) - np.asarray(start_xy, dtype=float)
    nominal = math.atan2(float(delta[1]), float(delta[0]))
    return wrap_angle(nominal + float(rng.normal(0.0, noise_std)))


def json_safe(value: object) -> object:
    """Convert nested NumPy structures into JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: Path, payload: dict) -> None:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)


def seed_sequence(base_seed: int, count: int) -> list[int]:
    """Derive deterministic per-episode seeds from a base seed."""
    rng = np.random.default_rng(base_seed)
    return [int(value) for value in rng.integers(0, 2**31 - 1, size=count)]


def rolling_pairs(items: Iterable[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return adjacent pairs from a path-like iterable."""
    items_list = list(items)
    return [(items_list[index], items_list[index + 1]) for index in range(len(items_list) - 1)]
