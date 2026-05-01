"""Map generators and fixed-map loaders for V4 dynamic-obstacle environments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:  # pragma: no cover - requirements include imageio.
    imageio = None

from .config import MapConfig
from .utils import ensure_rng


@dataclass
class GeneratedMap:
    map_id: str
    map_type: str
    occupancy: np.ndarray
    resolution: float
    origin: tuple[float, float]


def load_map_yaml(map_yaml: str | Path, *, unknown_as_occupied: bool = True) -> GeneratedMap:
    """Load a ROS-style occupancy map using the Dreamer map-publisher convention."""
    yaml_path = Path(map_yaml).expanduser().resolve()
    cached = _load_map_yaml_cached(str(yaml_path), bool(unknown_as_occupied))
    return GeneratedMap(
        map_id=cached.map_id,
        map_type=cached.map_type,
        occupancy=cached.occupancy.copy(),
        resolution=cached.resolution,
        origin=cached.origin,
    )


@lru_cache(maxsize=8)
def _load_map_yaml_cached(map_yaml: str, unknown_as_occupied: bool) -> GeneratedMap:
    yaml_path = Path(map_yaml)
    with yaml_path.open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    if not isinstance(metadata, dict):
        raise TypeError(f"Map YAML must be a mapping: {yaml_path}")

    image_path = Path(str(metadata["image"]))
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path

    image = _read_image(image_path)
    gray, alpha = _split_gray_alpha(image)
    occupancy = _image_to_binary_occupancy(
        gray=gray,
        alpha=alpha,
        negate=int(metadata.get("negate", 0)),
        occupied_thresh=float(metadata.get("occupied_thresh", 0.65)),
        free_thresh=float(metadata.get("free_thresh", 0.196)),
        unknown_as_occupied=unknown_as_occupied,
    )

    origin_values = metadata.get("origin", [0.0, 0.0, 0.0])
    if len(origin_values) > 2 and abs(float(origin_values[2])) > 1e-9:
        raise ValueError(
            f"Map origin yaw is {float(origin_values[2])}, but sim2d fixed-map loading only supports yaw=0"
        )

    return GeneratedMap(
        map_id=yaml_path.stem,
        map_type="fixed_map",
        occupancy=occupancy,
        resolution=float(metadata["resolution"]),
        origin=(float(origin_values[0]), float(origin_values[1])),
    )


def _read_image(image_path: Path) -> np.ndarray:
    """Read a map image with imageio when available, falling back to Pillow/matplotlib."""
    if imageio is not None:
        return np.asarray(imageio.imread(image_path))
    try:
        from PIL import Image
    except ModuleNotFoundError:
        try:
            from matplotlib import image as mpimg
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install imageio, Pillow, or matplotlib to load fixed occupancy maps") from exc
        return np.asarray(mpimg.imread(str(image_path)))

    with Image.open(image_path) as image:
        return np.asarray(image.copy())


def _split_gray_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return grayscale and optional alpha arrays from an imageio-loaded image."""
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        return _normalize_gray(image_array), None
    if image_array.ndim != 3 or image_array.shape[2] < 3:
        raise ValueError(f"Unsupported map image shape: {image_array.shape}")

    rgb = image_array[:, :, :3].astype(float)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    alpha = image_array[:, :, 3] if image_array.shape[2] >= 4 else None
    return _normalize_gray(gray), alpha


def _normalize_gray(gray: np.ndarray) -> np.ndarray:
    gray_float = np.asarray(gray, dtype=float)
    if gray_float.size and float(np.nanmax(gray_float)) <= 1.0:
        gray_float = gray_float * 255.0
    return np.clip(gray_float, 0.0, 255.0)


def _image_to_binary_occupancy(
    *,
    gray: np.ndarray,
    alpha: np.ndarray | None,
    negate: int,
    occupied_thresh: float,
    free_thresh: float,
    unknown_as_occupied: bool,
) -> np.ndarray:
    """Convert image pixels to a row-major binary grid matching map_publisher.py."""
    gray_map_rows = gray[::-1, :]
    occupancy_prob = gray_map_rows / 255.0 if int(negate) else (255.0 - gray_map_rows) / 255.0

    occupied_mask = occupancy_prob > float(occupied_thresh)
    free_mask = occupancy_prob < float(free_thresh)
    unknown_mask = ~(occupied_mask | free_mask)
    if alpha is not None:
        alpha_map_rows = np.asarray(alpha)[::-1, :]
        unknown_mask = unknown_mask | (alpha_map_rows == 0)

    grid = np.zeros(gray.shape, dtype=np.uint8)
    grid[occupied_mask] = 1
    if unknown_as_occupied:
        grid[unknown_mask] = 1
    return grid


def _apply_loaded_map_metadata(config: MapConfig, map_bundle: GeneratedMap) -> None:
    """Keep downstream config summaries and sampling helpers aligned with fixed-map metadata."""
    config.height, config.width = int(map_bundle.occupancy.shape[0]), int(map_bundle.occupancy.shape[1])
    config.resolution = float(map_bundle.resolution)
    config.origin = tuple(float(value) for value in map_bundle.origin)


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
    if config.map_yaml:
        map_bundle = load_map_yaml(config.map_yaml)
        _apply_loaded_map_metadata(config, map_bundle)
        return GeneratedMap(
            map_id=map_bundle.map_id,
            map_type=map_bundle.map_type,
            occupancy=map_bundle.occupancy.copy(),
            resolution=map_bundle.resolution,
            origin=map_bundle.origin,
        )

    rng = ensure_rng(seed)
    chosen_type = map_type or str(rng.choice(config.generator_types))
    if chosen_type not in MAP_GENERATORS:
        raise ValueError(f"Unknown map type: {chosen_type}")
    return MAP_GENERATORS[chosen_type](config, rng, map_index)
