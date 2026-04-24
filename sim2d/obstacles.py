"""Dynamic obstacle definitions and per-episode route generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import DynamicObstacleConfig, MapConfig, RobotConfig
from .dynamics import disk_collides_with_occupancy
from .utils import ensure_rng


@dataclass
class DynamicObstacle:
    obstacle_id: str
    position: np.ndarray
    radius: float
    speed: float
    route: list[np.ndarray]
    route_index: int = 0

    def clone(self) -> "DynamicObstacle":
        """Return a deep-ish copy suitable for rollouts."""
        return DynamicObstacle(
            obstacle_id=self.obstacle_id,
            position=self.position.copy(),
            radius=float(self.radius),
            speed=float(self.speed),
            route=[point.copy() for point in self.route],
            route_index=int(self.route_index),
        )

    @property
    def current_target(self) -> np.ndarray:
        return self.route[self.route_index]

    @property
    def velocity(self) -> np.ndarray:
        """Return the current patrol velocity vector."""
        if not self.route:
            return np.zeros((2,), dtype=float)
        delta = self.current_target - self.position
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return np.zeros((2,), dtype=float)
        return (float(self.speed) / distance) * delta

    def step(self, dt: float) -> None:
        """Advance along the patrol route."""
        remaining = float(self.speed) * dt
        while remaining > 0.0 and self.route:
            target = self.current_target
            delta = target - self.position
            distance = float(np.linalg.norm(delta))
            if distance < 1e-9:
                self.route_index = (self.route_index + 1) % len(self.route)
                continue
            if remaining >= distance:
                self.position = target.copy()
                remaining -= distance
                self.route_index = (self.route_index + 1) % len(self.route)
            else:
                self.position = self.position + (remaining / distance) * delta
                remaining = 0.0

    def snapshot(self) -> dict[str, object]:
        """Return a serializable obstacle state."""
        return {
            "obstacle_id": self.obstacle_id,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "radius": float(self.radius),
            "speed": float(self.speed),
            "route_index": int(self.route_index),
            "current_target": self.current_target.copy(),
        }

    def route_metadata(self) -> dict[str, object]:
        """Return per-episode route metadata."""
        return {
            "obstacle_id": self.obstacle_id,
            "radius": float(self.radius),
            "speed": float(self.speed),
            "initial_position": self.position.copy(),
            "initial_velocity": self.velocity.copy(),
            "initial_route_index": int(self.route_index),
            "initial_current_target": self.current_target.copy(),
            "route": [point.copy() for point in self.route],
        }


def sample_dynamic_obstacles(
    occupancy: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    map_config: MapConfig,
    robot_config: RobotConfig,
    dynamic_config: DynamicObstacleConfig,
    path_world: np.ndarray,
    rng_seed: int,
    avoid_points: Sequence[np.ndarray],
) -> list[DynamicObstacle]:
    """Sample patrol-loop dynamic obstacles in free space."""
    rng = ensure_rng(rng_seed)
    obstacles: list[DynamicObstacle] = []
    if len(path_world) == 0:
        return obstacles

    count = int(dynamic_config.num_dynamic_obstacles)
    radius_min, radius_max = dynamic_config.radius_range
    speed_min, speed_max = dynamic_config.speed_range
    route_len_min, route_len_max = dynamic_config.route_length_range
    for obstacle_index in range(count):
        radius = float(rng.uniform(radius_min, radius_max))
        speed = float(rng.uniform(speed_min, speed_max))
        route_length = int(rng.integers(int(route_len_min), int(route_len_max) + 1))
        route: list[np.ndarray] = []

        anchor_idx = int(rng.integers(max(1, len(path_world) // 6), max(2, len(path_world) - len(path_world) // 6)))
        anchor = path_world[min(anchor_idx, len(path_world) - 1)]
        max_attempts = 200
        for _ in range(route_length):
            candidate = _sample_route_point(
                rng=rng,
                anchor=anchor,
                occupancy=occupancy,
                resolution=resolution,
                origin=origin,
                radius=radius,
                avoid_points=avoid_points,
                existing_routes=[point for point in route],
            )
            if candidate is not None:
                route.append(candidate)

        if len(route) < 2:
            for _ in range(max_attempts):
                point = _sample_uniform_free_point(rng, occupancy, resolution, origin, radius, avoid_points)
                if point is not None:
                    route.append(point)
                if len(route) >= 2:
                    break

        if len(route) < 2:
            continue

        start_position = route[0].copy()
        obstacle = DynamicObstacle(
            obstacle_id=f"dyn_{obstacle_index:02d}",
            position=start_position,
            radius=radius,
            speed=speed,
            route=route,
            route_index=1 % len(route),
        )
        obstacles.append(obstacle)
        avoid_points = list(avoid_points) + [start_position]
    return obstacles


def _sample_route_point(
    rng: np.random.Generator,
    anchor: np.ndarray,
    occupancy: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    radius: float,
    avoid_points: Sequence[np.ndarray],
    existing_routes: Sequence[np.ndarray],
) -> np.ndarray | None:
    world_span = min(occupancy.shape) * resolution
    offset_limit = float(np.clip(world_span * 0.18, 1.0, 1.8))
    for _ in range(100):
        offset = rng.uniform(-offset_limit, offset_limit, size=2)
        point = anchor + offset
        if _valid_obstacle_point(point, occupancy, resolution, origin, radius, avoid_points, existing_routes):
            return point.astype(float)
    return None


def _sample_uniform_free_point(
    rng: np.random.Generator,
    occupancy: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    radius: float,
    avoid_points: Sequence[np.ndarray],
) -> np.ndarray | None:
    height, width = occupancy.shape
    for _ in range(200):
        row = int(rng.integers(0, height))
        col = int(rng.integers(0, width))
        point = np.array(
            [origin[0] + (col + 0.5) * resolution, origin[1] + (row + 0.5) * resolution],
            dtype=float,
        )
        if _valid_obstacle_point(point, occupancy, resolution, origin, radius, avoid_points, []):
            return point
    return None


def _valid_obstacle_point(
    point: np.ndarray,
    occupancy: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    radius: float,
    avoid_points: Sequence[np.ndarray],
    existing_routes: Sequence[np.ndarray],
) -> bool:
    if disk_collides_with_occupancy(point, radius + 0.05, occupancy, resolution, origin):
        return False
    for other in avoid_points:
        if np.linalg.norm(point - np.asarray(other, dtype=float)) < 0.8:
            return False
    for other in existing_routes:
        if np.linalg.norm(point - np.asarray(other, dtype=float)) < 0.7:
            return False
    return True
