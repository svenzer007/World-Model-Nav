"""A* search over a static occupancy grid."""

from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np

from .utils import grid_to_world, in_bounds_rc


NEIGHBORS_8 = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, np.sqrt(2.0)),
    (-1, 1, np.sqrt(2.0)),
    (1, -1, np.sqrt(2.0)),
    (1, 1, np.sqrt(2.0)),
]


def heuristic(node: tuple[int, int], goal: tuple[int, int]) -> float:
    """Octile-distance heuristic."""
    dx = abs(node[1] - goal[1])
    dy = abs(node[0] - goal[0])
    return float((dx + dy) + (np.sqrt(2.0) - 2.0) * min(dx, dy))


def reconstruct_path(came_from: dict[tuple[int, int], tuple[int, int]], current: tuple[int, int]) -> list[tuple[int, int]]:
    """Reconstruct a path from the A* parent map."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_search(
    occupancy: np.ndarray,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """Run A* over a binary occupancy grid."""
    if occupancy[start_rc] != 0 or occupancy[goal_rc] != 0:
        return None

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start_rc, goal_rc), 0.0, start_rc))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start_rc: 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_rc:
            return reconstruct_path(came_from, current)
        closed.add(current)

        for dr, dc, step_cost in NEIGHBORS_8:
            neighbor = (current[0] + dr, current[1] + dc)
            if not in_bounds_rc(neighbor[0], neighbor[1], occupancy.shape):
                continue
            if occupancy[neighbor] != 0:
                continue
            tentative_g = current_g + float(step_cost)
            if tentative_g >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            heapq.heappush(open_heap, (tentative_g + heuristic(neighbor, goal_rc), tentative_g, neighbor))
    return None


def path_to_world(
    path_rc: Iterable[tuple[int, int]],
    resolution: float,
    origin: tuple[float, float],
) -> np.ndarray:
    """Convert an A* row/col path to world coordinates."""
    return np.array(
        [grid_to_world(int(row), int(col), resolution, origin) for row, col in path_rc],
        dtype=float,
    )
