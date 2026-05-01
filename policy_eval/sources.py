"""Policy-eval source resolution and saved-episode scenario reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sim2d.maps import GeneratedMap
from sim2d.obstacles import DynamicObstacle
from sim2d.utils import build_static_obstacle_index, occupied_boundary_cell_centers


def resolve_episode_source_paths(source_path: str | Path) -> list[Path]:
    """Resolve saved episode pickles from a dataset root or episodes directory."""
    root = Path(source_path)
    episodes_dir = root if root.name == "episodes" else root / "episodes"
    paths = sorted(episodes_dir.glob("episode_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No episode pickles found under {episodes_dir}")
    return paths


def _dynamic_obstacle_from_route_metadata(route_metadata: dict[str, object]) -> DynamicObstacle:
    route = [np.asarray(point, dtype=float).copy() for point in list(route_metadata["route"])]
    return DynamicObstacle(
        obstacle_id=str(route_metadata["obstacle_id"]),
        position=np.asarray(route_metadata["initial_position"], dtype=float).copy(),
        radius=float(route_metadata["radius"]),
        speed=float(route_metadata["speed"]),
        route=route,
        route_index=int(route_metadata["initial_route_index"]),
    )


def scenario_from_saved_episode(episode: dict[str, object]) -> dict[str, Any]:
    """Reconstruct a controller-ready scenario from a saved episode pickle."""
    occupancy = np.asarray(episode["occupancy_grid"], dtype=np.uint8).copy()
    planning_occupancy = np.asarray(episode["planning_occupancy"], dtype=np.uint8).copy()
    resolution = float(episode["resolution"])
    origin = tuple(float(value) for value in episode["origin"])
    map_bundle = GeneratedMap(
        map_id=str(episode["map_id"]),
        map_type=str(episode["map_type"]),
        occupancy=occupancy,
        resolution=resolution,
        origin=origin,
    )
    return {
        "map_bundle": map_bundle,
        "planning_occupancy": planning_occupancy,
        "occupied_points": build_static_obstacle_index(occupied_boundary_cell_centers(occupancy, resolution, origin)),
        "start_pose": np.asarray(episode["start_pose"], dtype=float).copy(),
        "goal_point": np.asarray(episode["final_goal"], dtype=float).copy(),
        "path_rc": [tuple(int(value) for value in row_col) for row_col in np.asarray(episode["astar_path_grid"], dtype=int)],
        "path_world": np.asarray(episode["astar_path_world"], dtype=float).copy(),
        "dynamic_obstacles": [
            _dynamic_obstacle_from_route_metadata(dict(route_metadata))
            for route_metadata in list(episode["dynamic_obstacle_routes"])
        ],
    }
