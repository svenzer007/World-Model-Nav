"""Map generators for V4 10m-scale dynamic-obstacle environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import MapConfig
from .utils import ensure_rng


@dataclass
class GeneratedMap:
    map_id: str
    map_type: str
    occupancy: np.ndarray
    resolution: float
    origin: tuple[float, float]


def _draw_rect(grid: np.ndarray, row0: int, row1: int, col0: int, col1: int, value: int = 1) -> None:
    grid[max(0, row0) : min(grid.shape[0], row1), max(0, col0) : min(grid.shape[1], col1)] = value


def _surround_walls(grid: np.ndarray, thickness: int = 2) -> None:
    _draw_rect(grid, 0, thickness, 0, grid.shape[1], 1)
    _draw_rect(grid, grid.shape[0] - thickness, grid.shape[0], 0, grid.shape[1], 1)
    _draw_rect(grid, 0, grid.shape[0], 0, thickness, 1)
    _draw_rect(grid, 0, grid.shape[0], grid.shape[1] - thickness, grid.shape[1], 1)


def _draw_vertical_wall_with_gap(
    grid: np.ndarray,
    col: int,
    row0: int,
    row1: int,
    gap_center: int,
    gap_height: int,
    thickness: int = 3,
) -> None:
    _draw_rect(grid, row0, row1, col - thickness // 2, col + (thickness + 1) // 2, 1)
    _draw_rect(
        grid,
        gap_center - gap_height // 2,
        gap_center + (gap_height + 1) // 2,
        col - thickness // 2 - 1,
        col + (thickness + 1) // 2 + 1,
        0,
    )


def _draw_horizontal_wall_with_gap(
    grid: np.ndarray,
    row: int,
    col0: int,
    col1: int,
    gap_center: int,
    gap_width: int,
    thickness: int = 3,
) -> None:
    _draw_rect(grid, row - thickness // 2, row + (thickness + 1) // 2, col0, col1, 1)
    _draw_rect(
        grid,
        row - thickness // 2 - 1,
        row + (thickness + 1) // 2 + 1,
        gap_center - gap_width // 2,
        gap_center + (gap_width + 1) // 2,
        0,
    )


def generate_local_maze_map(config: MapConfig, rng: np.random.Generator, map_index: int) -> GeneratedMap:
    """Generate offset wall partitions that force static-obstacle detours."""
    grid = np.zeros((config.height, config.width), dtype=np.uint8)
    _surround_walls(grid, thickness=2)

    margin = max(7, int(round(min(config.width, config.height) * 0.08)))
    gap = max(20, int(round(min(config.width, config.height) * 0.22)))
    thickness = 3
    vertical_cols = [int(config.width * value) for value in (0.30, 0.55, 0.76)]
    vertical_gaps = [int(config.height * value) for value in (0.72, 0.30, 0.66)]
    horizontal_rows = [int(config.height * value) for value in (0.40, 0.64)]
    horizontal_gaps = [int(config.width * value) for value in (0.28, 0.76)]

    for col, gap_center in zip(vertical_cols, vertical_gaps):
        jittered_gap = int(np.clip(gap_center + int(rng.integers(-8, 9)), margin + gap, config.height - margin - gap))
        _draw_vertical_wall_with_gap(grid, col, margin, config.height - margin, jittered_gap, gap, thickness)

    for row, gap_center in zip(horizontal_rows, horizontal_gaps):
        jittered_gap = int(np.clip(gap_center + int(rng.integers(-8, 9)), margin + gap, config.width - margin - gap))
        _draw_horizontal_wall_with_gap(grid, row, margin, config.width - margin, jittered_gap, gap, thickness)

    for _ in range(4):
        block_w = int(rng.integers(4, 8))
        block_h = int(rng.integers(4, 8))
        rr = int(rng.integers(margin, config.height - margin - block_h))
        cc = int(rng.integers(margin, config.width - margin - block_w))
        if np.mean(grid[rr - 2 : rr + block_h + 2, cc - 2 : cc + block_w + 2]) < 0.25:
            _draw_rect(grid, rr, rr + block_h, cc, cc + block_w, 1)

    return GeneratedMap(
        map_id=f"local_maze_{map_index:06d}",
        map_type="local_maze",
        occupancy=grid,
        resolution=config.resolution,
        origin=config.origin,
    )


def generate_local_corridor_map(config: MapConfig, rng: np.random.Generator, map_index: int) -> GeneratedMap:
    """Generate a compact S-corridor with bottlenecks and turn pockets."""
    grid = np.ones((config.height, config.width), dtype=np.uint8)
    _surround_walls(grid, thickness=2)

    corridor_width = max(13, int(round(min(config.width, config.height) * 0.16)))
    half_width = corridor_width // 2
    points = [
        (int(config.height * 0.18), int(config.width * 0.16)),
        (int(config.height * 0.18), int(config.width * 0.78)),
        (int(config.height * 0.42), int(config.width * 0.78)),
        (int(config.height * 0.42), int(config.width * 0.24)),
        (int(config.height * 0.68), int(config.width * 0.24)),
        (int(config.height * 0.68), int(config.width * 0.84)),
        (int(config.height * 0.84), int(config.width * 0.84)),
    ]

    for (r0, c0), (r1, c1) in zip(points[:-1], points[1:]):
        if r0 == r1:
            _draw_rect(grid, r0 - half_width, r0 + half_width + 1, min(c0, c1), max(c0, c1) + 1, 0)
        else:
            _draw_rect(grid, min(r0, r1), max(r0, r1) + 1, c0 - half_width, c0 + half_width + 1, 0)
        _draw_rect(grid, r0 - half_width, r0 + half_width + 1, c0 - half_width, c0 + half_width + 1, 0)
        _draw_rect(grid, r1 - half_width, r1 + half_width + 1, c1 - half_width, c1 + half_width + 1, 0)

    for _ in range(6):
        box_w = int(rng.integers(5, 10))
        box_h = int(rng.integers(5, 10))
        rr = int(rng.integers(8, config.height - box_h - 8))
        cc = int(rng.integers(8, config.width - box_w - 8))
        if np.mean(grid[rr : rr + box_h, cc : cc + box_w]) < 0.35:
            _draw_rect(grid, rr, rr + box_h, cc, cc + box_w, 1)

    return GeneratedMap(
        map_id=f"local_corridor_{map_index:06d}",
        map_type="local_corridor",
        occupancy=grid,
        resolution=config.resolution,
        origin=config.origin,
    )


def generate_local_rooms_map(config: MapConfig, rng: np.random.Generator, map_index: int) -> GeneratedMap:
    """Generate separated rooms with staggered doors instead of an open cross."""
    grid = np.zeros((config.height, config.width), dtype=np.uint8)
    _surround_walls(grid, thickness=2)

    margin = max(7, int(round(min(config.width, config.height) * 0.08)))
    center_row = config.height // 2
    center_col = config.width // 2
    gap = max(14, int(round(min(config.width, config.height) * 0.16)))

    _draw_vertical_wall_with_gap(grid, center_col, margin, config.height - margin, int(config.height * 0.30), gap, 3)
    _draw_vertical_wall_with_gap(grid, center_col, margin, config.height - margin, int(config.height * 0.73), gap, 3)
    _draw_horizontal_wall_with_gap(grid, center_row, margin, config.width - margin, int(config.width * 0.70), gap, 3)
    _draw_horizontal_wall_with_gap(grid, center_row, margin, config.width - margin, int(config.width * 0.28), gap, 3)

    for _ in range(8):
        pillar_w = int(rng.integers(4, 8))
        pillar_h = int(rng.integers(4, 8))
        rr = int(rng.integers(margin + 2, config.height - margin - pillar_h - 2))
        cc = int(rng.integers(margin + 2, config.width - margin - pillar_w - 2))
        near_center = abs(rr - center_row) < 7 or abs(cc - center_col) < 7
        if not near_center and np.mean(grid[rr - 2 : rr + pillar_h + 2, cc - 2 : cc + pillar_w + 2]) < 0.15:
            _draw_rect(grid, rr, rr + pillar_h, cc, cc + pillar_w, 1)

    return GeneratedMap(
        map_id=f"local_rooms_{map_index:06d}",
        map_type="local_rooms",
        occupancy=grid,
        resolution=config.resolution,
        origin=config.origin,
    )


def generate_local_clutter_map(config: MapConfig, rng: np.random.Generator, map_index: int) -> GeneratedMap:
    """Generate staggered shelves and transverse blockers with offset gaps."""
    grid = np.zeros((config.height, config.width), dtype=np.uint8)
    _surround_walls(grid, thickness=2)

    margin = max(7, int(round(min(config.width, config.height) * 0.08)))
    shelf_width = int(rng.integers(4, 7))
    row_top = margin + 5
    row_bottom = config.height - margin - 5
    columns = [int(config.width * value) for value in (0.22, 0.40, 0.58, 0.76)]

    for shelf_index, col in enumerate(columns):
        _draw_rect(grid, row_top, row_bottom, col, col + shelf_width, 1)
        gap_centers = (
            [int(config.height * 0.30), int(config.height * 0.68)]
            if shelf_index % 2 == 0
            else [int(config.height * 0.48)]
        )
        for gap_center in gap_centers:
            gap_half = int(rng.integers(7, 10))
            _draw_rect(grid, gap_center - gap_half, gap_center + gap_half + 1, col - 1, col + shelf_width + 1, 0)

    _draw_horizontal_wall_with_gap(
        grid,
        int(config.height * 0.34),
        margin,
        config.width - margin,
        int(config.width * 0.80),
        max(15, int(config.width * 0.16)),
        3,
    )
    _draw_horizontal_wall_with_gap(
        grid,
        int(config.height * 0.66),
        margin,
        config.width - margin,
        int(config.width * 0.22),
        max(15, int(config.width * 0.16)),
        3,
    )

    for _ in range(5):
        block_w = int(rng.integers(4, 8))
        block_h = int(rng.integers(4, 8))
        rr = int(rng.integers(margin, config.height - margin - block_h))
        cc = int(rng.integers(margin, config.width - margin - block_w))
        if np.mean(grid[rr - 2 : rr + block_h + 2, cc - 2 : cc + block_w + 2]) < 0.2:
            _draw_rect(grid, rr, rr + block_h, cc, cc + block_w, 1)

    return GeneratedMap(
        map_id=f"local_clutter_{map_index:06d}",
        map_type="local_clutter",
        occupancy=grid,
        resolution=config.resolution,
        origin=config.origin,
    )


MAP_GENERATORS: dict[str, Callable[[MapConfig, np.random.Generator, int], GeneratedMap]] = {
    "local_maze": generate_local_maze_map,
    "local_corridor": generate_local_corridor_map,
    "local_rooms": generate_local_rooms_map,
    "local_clutter": generate_local_clutter_map,
}


def generate_map(config: MapConfig, seed: int, map_index: int, map_type: str | None = None) -> GeneratedMap:
    """Generate a map using one of the built-in V4 local layout families."""
    rng = ensure_rng(seed)
    chosen_type = map_type or str(rng.choice(config.generator_types))
    if chosen_type not in MAP_GENERATORS:
        raise ValueError(f"Unknown map type: {chosen_type}")
    return MAP_GENERATORS[chosen_type](config, rng, map_index)
