"""Dataset serialization and summary helpers."""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from .config import DatasetConfig, action_space_to_list
from .utils import json_safe, save_json


def save_episode_pickle(episode: dict[str, object], output_path: Path) -> None:
    """Write an episode dictionary to a pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(episode, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_episode_pickle(path: Path) -> dict[str, object]:
    """Load an episode dictionary from a pickle file."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def summarize_dataset(episodes: list[dict[str, object]], config: DatasetConfig, episode_files: list[str]) -> dict[str, object]:
    """Build the dataset summary JSON payload."""
    done_reasons = Counter(str(episode["episode_stats"]["done_reason"]) for episode in episodes)
    average_episode_length = float(np.mean([episode["episode_stats"]["num_steps"] for episode in episodes])) if episodes else 0.0
    all_steps = [step for episode in episodes for step in episode.get("steps", [])]
    path_ratios = [
        float(episode.get("episode_stats", {}).get("path_directness_ratio", 0.0))
        for episode in episodes
        if np.isfinite(float(episode.get("episode_stats", {}).get("path_directness_ratio", 0.0)))
    ]
    path_turns = [
        float(episode.get("episode_stats", {}).get("path_turn_angle_sum", 0.0))
        for episode in episodes
        if np.isfinite(float(episode.get("episode_stats", {}).get("path_turn_angle_sum", 0.0)))
    ]
    return {
        "schema_version": "v4_dynamic_state_data_1",
        "num_episodes": int(len(episodes)),
        "average_episode_length": average_episode_length,
        "success_count": int(done_reasons.get("goal_reached", 0)),
        "static_collision_count": int(done_reasons.get("static_collision", 0)),
        "dynamic_collision_count": int(done_reasons.get("dynamic_collision", 0)),
        "timeout_count": int(done_reasons.get("timeout", 0)),
        "label_summary": _summarize_v4_labels(all_steps),
        "path_complexity": {
            "mean_directness_ratio": float(np.mean(path_ratios)) if path_ratios else 0.0,
            "min_directness_ratio": float(np.min(path_ratios)) if path_ratios else 0.0,
            "mean_turn_angle_sum": float(np.mean(path_turns)) if path_turns else 0.0,
            "min_turn_angle_sum": float(np.min(path_turns)) if path_turns else 0.0,
        },
        "lidar_config": json_safe(config.lidar_config.__dict__),
        "action_space": action_space_to_list(),
        "map_shape": [config.map_config.height, config.map_config.width],
        "world_size_m": [
            float(config.map_config.width * config.map_config.resolution),
            float(config.map_config.height * config.map_config.resolution),
        ],
        "resolution": float(config.map_config.resolution),
        "dt": float(config.robot_config.dt),
        "num_dynamic_obstacles": int(config.dynamic_obstacle_config.num_dynamic_obstacles),
        "episode_files": episode_files,
        "generation_config": json_safe(config.to_dict()),
    }


def save_dataset_index(summary: dict[str, object], output_path: Path) -> None:
    """Write the global dataset index JSON."""
    save_json(output_path, summary)


def _summarize_v4_labels(steps: list[dict[str, object]]) -> dict[str, object]:
    """Summarize dense V4 dynamic labels for quick index-level inspection."""
    total = len(steps)
    if total == 0:
        return {
            "num_steps": 0,
            "dynamic_visible_fraction": 0.0,
            "lidar_dynamic_visible_fraction": 0.0,
            "dynamic_event_step_count": 0,
            "dynamic_collision_within_3_count": 0,
            "dynamic_collision_within_5_count": 0,
            "mean_dynamic_residual_energy_t": 0.0,
            "mean_min_dynamic_clearance_t1": 0.0,
            "mean_dynamic_risk_t1": 0.0,
        }

    def numeric_values(key: str) -> np.ndarray:
        values = [float(step[key]) for step in steps if key in step and np.isfinite(float(step[key]))]
        return np.asarray(values, dtype=float)

    dynamic_clearance = numeric_values("min_dynamic_clearance_t1")
    dynamic_risk = numeric_values("dynamic_risk_t1")
    residual_energy = numeric_values("dynamic_residual_energy_t")
    hit_fraction = numeric_values("dynamic_hit_fraction_t")
    return {
        "num_steps": int(total),
        "collision_t1_count": int(sum(bool(step.get("collision_t1", False)) for step in steps)),
        "dynamic_collision_t1_count": int(sum(bool(step.get("dynamic_collision_t1", False)) for step in steps)),
        "dynamic_collision_within_3_count": int(
            sum(bool(step.get("dynamic_collision_within_3", False)) for step in steps)
        ),
        "dynamic_collision_within_5_count": int(
            sum(bool(step.get("dynamic_collision_within_5", False)) for step in steps)
        ),
        "dynamic_visible_fraction": float(np.mean([bool(step.get("dynamic_visible_t", False)) for step in steps])),
        "lidar_dynamic_visible_fraction": float(
            np.mean([bool(step.get("lidar_dynamic_visible_t", False)) for step in steps])
        ),
        "dynamic_event_step_count": int(sum(bool(step.get("dynamic_event_step_t", False)) for step in steps)),
        "mean_dynamic_hit_fraction_t": float(np.mean(hit_fraction)) if hit_fraction.size else 0.0,
        "mean_dynamic_residual_energy_t": float(np.mean(residual_energy)) if residual_energy.size else 0.0,
        "mean_min_dynamic_clearance_t1": float(np.mean(dynamic_clearance)) if dynamic_clearance.size else 0.0,
        "min_min_dynamic_clearance_t1": float(np.min(dynamic_clearance)) if dynamic_clearance.size else 0.0,
        "mean_dynamic_risk_t1": float(np.mean(dynamic_risk)) if dynamic_risk.size else 0.0,
        "max_dynamic_risk_t1": float(np.max(dynamic_risk)) if dynamic_risk.size else 0.0,
    }
