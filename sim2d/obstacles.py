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
    global_route_points = [np.asarray(point, dtype=float) for point in avoid_points]
    for obstacle_index in range(count):
        radius = float(rng.uniform(radius_min, radius_max))
        speed = float(rng.uniform(speed_min, speed_max))
        route_length = int(rng.integers(int(route_len_min), int(route_len_max) + 1))
        route: list[np.ndarray] = []

        anchor_idx = _spread_anchor_index(path_world, obstacle_index, count, rng)
        anchor = path_world[min(anchor_idx, len(path_world) - 1)]
        route_radius = float(
            dynamic_config.route_radius
            if dynamic_config.route_radius is not None
            else np.clip(min(occupancy.shape) * resolution * 0.18, 1.0, 1.8)
        )
        min_spacing = max(float(dynamic_config.route_point_spacing), 2.0 * radius + 0.3)
        max_attempts = 200

        if float(dynamic_config.route_crossing_probability) > 0.0 and rng.random() < float(
            dynamic_config.route_crossing_probability
        ):
            route = _sample_crossing_route(
                rng=rng,
                path_world=path_world,
                anchor_index=anchor_idx,
                anchor=anchor,
                occupancy=occupancy,
                resolution=resolution,
                origin=origin,
                radius=radius,
                route_radius=route_radius,
                min_spacing=min_spacing,
                avoid_points=global_route_points,
            )

        if len(route) < 2:
            route = []
            for _ in range(route_length):
                candidate = _sample_route_point(
                    rng=rng,
                    anchor=anchor,
                    occupancy=occupancy,
                    resolution=resolution,
                    origin=origin,
                    radius=radius,
                    route_radius=route_radius,
                    min_spacing=min_spacing,
                    avoid_points=global_route_points,
                    existing_routes=route,
                )
                if candidate is not None:
                    route.append(candidate)

        if len(route) < 2:
            route = []
            for _ in range(max_attempts):
                point = _sample_uniform_free_point(
                    rng,
                    occupancy,
                    resolution,
                    origin,
                    radius,
                    global_route_points,
                    route,
                    min_spacing=min_spacing,
                )
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
        global_route_points.extend(point.copy() for point in route)
    return obstacles


def _spread_anchor_index(
    path_world: np.ndarray,
    obstacle_index: int,
    count: int,
    rng: np.random.Generator,
) -> int:
    """Choose anchors across the usable path span instead of clustering all obstacles."""
    if len(path_world) <= 2:
        return 0
    usable_start = max(1, int(round(0.12 * len(path_world))))
    usable_stop = max(usable_start + 1, int(round(0.88 * len(path_world))))
    fraction = (float(obstacle_index) + 0.5) / max(float(count), 1.0)
    base = usable_start + fraction * float(usable_stop - usable_start)
    jitter = max(1, int(round(0.08 * len(path_world))))
    return int(np.clip(round(base + rng.integers(-jitter, jitter + 1)), 0, len(path_world) - 1))


def _sample_crossing_route(
    rng: np.random.Generator,
    path_world: np.ndarray,
    anchor_index: int,
    anchor: np.ndarray,
    occupancy: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    radius: float,
    route_radius: float,
    min_spacing: float,
    avoid_points: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Sample a long back-and-forth route crossing near the path anchor."""
    if len(path_world) < 2:
        return []
    prev_index = max(0, min(anchor_index - 2, len(path_world) - 1))
    next_index = max(0, min(anchor_index + 2, len(path_world) - 1))
    tangent = path_world[next_index] - path_world[prev_index]
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-6:
        tangent = rng.normal(size=2)
        tangent_norm = float(np.linalg.norm(tangent))
    tangent = tangent / max(tangent_norm, 1e-6)
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    if rng.random() < 0.5:
        normal = -normal
    center_shift = rng.uniform(-0.25 * route_radius, 0.25 * route_radius) * tangent

    for scale in (1.0, 0.85, 0.7):
        half = float(route_radius) * float(scale)
        endpoints = [
            np.asarray(anchor, dtype=float) + center_shift - half * normal,
            np.asarray(anchor, dtype=float) + center_shift + half * normal,
        ]
        if all(
            _valid_obstacle_point(
                point,
                occupancy,
                resolution,
                origin,
                radius,
                avoid_points,
                [],
                min_spacing=min_spacing,
            )
            for point in endpoints
        ):
            return [point.astype(float) for point in endpoints]
    return []


def _sample_route_point(
    rng: np.random.Generator,
    anchor: np.ndarray,
    occupancy: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    radius: float,
    route_radius: float,
    min_spacing: float,
    avoid_points: Sequence[np.ndarray],
    existing_routes: Sequence[np.ndarray],
) -> np.ndarray | None:
    for _ in range(100):
        angle = float(rng.uniform(-np.pi, np.pi))
        distance = float(rng.uniform(0.25 * route_radius, route_radius))
        offset = distance * np.array([np.cos(angle), np.sin(angle)], dtype=float)
        point = anchor + offset
        if _valid_obstacle_point(
            point,
            occupancy,
            resolution,
            origin,
            radius,
            avoid_points,
            existing_routes,
            min_spacing=min_spacing,
        ):
            return point.astype(float)
    return None


def _sample_uniform_free_point(
    rng: np.random.Generator,
    occupancy: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    radius: float,
    avoid_points: Sequence[np.ndarray],
    existing_routes: Sequence[np.ndarray],
    min_spacing: float,
) -> np.ndarray | None:
    height, width = occupancy.shape
    for _ in range(200):
        row = int(rng.integers(0, height))
        col = int(rng.integers(0, width))
        point = np.array(
            [origin[0] + (col + 0.5) * resolution, origin[1] + (row + 0.5) * resolution],
            dtype=float,
        )
        if _valid_obstacle_point(
            point,
            occupancy,
            resolution,
            origin,
            radius,
            avoid_points,
            existing_routes,
            min_spacing=min_spacing,
        ):
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
    min_spacing: float,
) -> bool:
    if disk_collides_with_occupancy(point, radius + 0.05, occupancy, resolution, origin):
        return False
    for other in avoid_points:
        if np.linalg.norm(point - np.asarray(other, dtype=float)) < min_spacing:
            return False
    for other in existing_routes:
        if np.linalg.norm(point - np.asarray(other, dtype=float)) < min_spacing:
            return False
    return True
