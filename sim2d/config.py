"""Configuration dataclasses and fixed action definitions for V4 structured dynamic data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np


ACTIONS: Dict[int, Dict[str, float | str]] = {
    0: {"name": "stop", "vx": 0.0, "vy": 0.0, "omega": 0.0},
    1: {"name": "forward", "vx": 0.8, "vy": 0.0, "omega": 0.0},
    2: {"name": "backward", "vx": -0.4, "vy": 0.0, "omega": 0.0},
    3: {"name": "strafe_left", "vx": 0.0, "vy": 0.4, "omega": 0.0},
    4: {"name": "strafe_right", "vx": 0.0, "vy": -0.4, "omega": 0.0},
    5: {"name": "forward_left", "vx": 0.6, "vy": 0.4, "omega": 0.0},
    6: {"name": "forward_right", "vx": 0.6, "vy": -0.4, "omega": 0.0},
    7: {"name": "backward_left", "vx": -0.4, "vy": 0.4, "omega": 0.0},
    8: {"name": "backward_right", "vx": -0.4, "vy": -0.4, "omega": 0.0},
    9: {"name": "rotate_left", "vx": 0.0, "vy": 0.0, "omega": 0.6},
    10: {"name": "rotate_right", "vx": 0.0, "vy": 0.0, "omega": -0.6},
    11: {"name": "forward_arc_left", "vx": 0.6, "vy": 0.0, "omega": 0.6},
    12: {"name": "forward_arc_right", "vx": 0.6, "vy": 0.0, "omega": -0.6},
}


@dataclass
class MapConfig:
    map_yaml: str | None = None
    width: int = 100
    height: int = 100
    resolution: float = 0.1
    origin: tuple[float, float] = (0.0, 0.0)
    inflation_margin: float = 0.05
    min_start_goal_clearance: float = 0.45
    min_start_goal_distance: float = 3.0
    min_path_length: float = 2.5
    min_path_directness_ratio: float = 1.2
    min_path_turn_angle_sum: float = 0.75
    generator_types: tuple[str, ...] = ("local_maze", "local_corridor", "local_rooms", "local_clutter")


@dataclass
class RobotConfig:
    radius: float = 0.19
    dt: float = 0.1
    goal_tolerance: float = 0.4


@dataclass
class RobotCommandRandomizationConfig:
    enabled: bool = False
    delay_min_steps: int = 0
    delay_max_steps: int = 0
    vx_gain_range: tuple[float, float] = (1.0, 1.0)
    vy_gain_range: tuple[float, float] = (1.0, 1.0)
    omega_gain_range: tuple[float, float] = (1.0, 1.0)
    vx_noise_std: float = 0.0
    vy_noise_std: float = 0.0
    omega_noise_std: float = 0.0
    vx_clip_range: tuple[float, float] = (-0.4, 0.8)
    vy_clip_range: tuple[float, float] = (-0.4, 0.4)
    omega_clip_range: tuple[float, float] = (-0.6, 0.6)


@dataclass
class LidarConfig:
    num_beams: int = 241
    angle_min: float = -np.pi / 2.0
    angle_max: float = np.pi / 2.0
    max_range: float = 6.0
    step_size: float = 0.05


@dataclass
class DynamicObstacleConfig:
    num_dynamic_obstacles: int = 4
    radius_range: tuple[float, float] = (0.17, 0.26)
    speed_range: tuple[float, float] = (0.18, 0.30)
    route_length_range: tuple[int, int] = (2, 4)
    route_radius: float | None = None
    route_crossing_probability: float = 0.0
    route_point_spacing: float = 0.7


@dataclass
class ExpertConfig:
    horizon_seconds: float = 1.0
    action_indices: tuple[int, ...] = tuple(ACTIONS.keys())
    goal_progress_weight: float = 2.0
    heading_weight: float = 0.8
    clearance_weight: float = 1.4
    path_adherence_weight: float = 1.1
    velocity_weight: float = 0.3
    collision_penalty: float = 1e6
    min_clearance_clip: float = 0.05

    @property
    def rollout_steps(self) -> int:
        return max(1, int(round(self.horizon_seconds / 0.1)))


@dataclass
class VisualizationConfig:
    figure_size: tuple[float, float] = (8.0, 8.0)
    dpi: int = 120
    draw_lidar_in_gif: bool = False
    gif_fps: int = 6


@dataclass
class DatasetConfig:
    num_episodes: int = 20
    output_dir: str = "./output_v4_dynamic_state"
    seed: int = 0
    max_steps: int = 250
    save_gifs: bool = False
    gif_every: int = 5
    lookahead_distance: float = 1.0
    risk_tau: float = 0.5
    dynamic_risk_tau: float = 0.5
    dynamic_risk_positive_threshold: float = 0.35
    dynamic_visibility_radius: float = 1.2
    dynamic_residual_epsilon: float = 0.03
    dynamic_visible_min_beams: int = 3
    max_robot_path_deviation: float | None = None
    max_robot_trajectory_path_ratio: float | None = None
    min_dynamic_visible_fraction: float = 0.06
    min_dynamic_risk_fraction: float = 0.04
    min_dynamic_event_steps: int = 3
    dynamic_interaction_resample_limit: int = 30
    save_static_lidar: bool = True
    save_dynamic_residual: bool = True
    episode_resample_limit: int = 200
    map_config: MapConfig = field(default_factory=MapConfig)
    robot_config: RobotConfig = field(default_factory=RobotConfig)
    robot_command_randomization_config: RobotCommandRandomizationConfig = field(default_factory=RobotCommandRandomizationConfig)
    lidar_config: LidarConfig = field(default_factory=LidarConfig)
    dynamic_obstacle_config: DynamicObstacleConfig = field(default_factory=DynamicObstacleConfig)
    expert_config: ExpertConfig = field(default_factory=ExpertConfig)
    visualization_config: VisualizationConfig = field(default_factory=VisualizationConfig)

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["action_space"] = ACTIONS
        return data


def action_space_to_list() -> List[Dict[str, object]]:
    """Return the fixed action space in a JSON-friendly order."""
    return [
        {
            "index": action_index,
            "name": action["name"],
            "vx": float(action["vx"]),
            "vy": float(action["vy"]),
            "omega": float(action["omega"]),
            "wz": float(action["omega"]),
        }
        for action_index, action in ACTIONS.items()
    ]
