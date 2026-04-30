"""Shared closed-loop policy-eval helpers for the longer-horizons branch."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from controllers.baseline_structured_controller import BaselineStructuredController, StructuredControllerConfig
from sim2d.config import ACTIONS, DatasetConfig
from sim2d.dataset import load_episode_pickle, save_episode_pickle
from sim2d.dynamics import disk_collides_with_occupancy, minimum_dynamic_clearance, minimum_static_clearance
from sim2d.obstacles import DynamicObstacle
from sim2d.simulator import EpisodeSimulator
from sim2d.utils import euclidean, json_safe, path_length, save_json, seed_sequence
from sim2d.visualize import save_episode_gif, save_episode_preview_png, save_episode_video
from sim2d.waypoint import compute_local_subgoal
from .robustness import sample_executed_control, sample_pose_observation
from .sources import resolve_episode_source_paths, scenario_from_saved_episode


def _clone_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "map_bundle": scenario["map_bundle"].__class__(
            map_id=str(scenario["map_bundle"].map_id),
            map_type=str(scenario["map_bundle"].map_type),
            occupancy=scenario["map_bundle"].occupancy.copy(),
            resolution=float(scenario["map_bundle"].resolution),
            origin=tuple(scenario["map_bundle"].origin),
        ),
        "planning_occupancy": scenario["planning_occupancy"].copy(),
        "occupied_points": scenario["occupied_points"].copy(),
        "start_pose": scenario["start_pose"].copy(),
        "goal_point": scenario["goal_point"].copy(),
        "path_rc": list(scenario["path_rc"]),
        "path_world": scenario["path_world"].copy(),
        "dynamic_obstacles": [obstacle.clone() for obstacle in scenario["dynamic_obstacles"]],
    }


def _sample_scenario(simulator: EpisodeSimulator, episode_id: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    map_bundle, planning_occupancy, occupied_points, start_pose, goal_point, path_rc, path_world, dyn_obstacles = (
        simulator._sample_episode_context(episode_id, rng)
    )
    return {
        "map_bundle": map_bundle,
        "planning_occupancy": planning_occupancy,
        "occupied_points": occupied_points,
        "start_pose": start_pose,
        "goal_point": goal_point,
        "path_rc": path_rc,
        "path_world": path_world,
        "dynamic_obstacles": dyn_obstacles,
    }


def _snapshot_obstacles(dynamic_obstacles: list[DynamicObstacle]) -> list[dict[str, object]]:
    return [obstacle.snapshot() for obstacle in dynamic_obstacles]


def _step_action_histogram(steps: list[dict[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for step in steps:
        counts[str(step["action_name"])] += 1
    return dict(counts)


def _episode_stop_fraction(steps: list[dict[str, object]]) -> float:
    if not steps:
        return 0.0
    return float(sum(int(step["action_index"]) == 0 for step in steps) / len(steps))


def _score_components_from_debug(chosen_debug: dict[str, object]) -> dict[str, float]:
    return {
        "progress": float(chosen_debug.get("goal_progress_cost", 0.0)),
        "heading": float(chosen_debug.get("heading_cost", 0.0)),
        "path": float(chosen_debug.get("path_cost", 0.0)),
        "velocity": float(chosen_debug.get("velocity_cost", 0.0)),
        "clearance": float(chosen_debug.get("clearance_cost", 0.0)),
        "infeasibility_penalty": float(chosen_debug.get("infeasibility_penalty", 0.0)),
    }


def _safe_action_family(action_name: str) -> str:
    families = {str(action["name"]) for action in ACTIONS.values()}
    return action_name if action_name in families else "unknown"


def _scalar_error_stats(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        return {
            "num_samples": 0,
            "mean_error": float("nan"),
            "variance": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
        }
    values = np.asarray(samples, dtype=float)
    return {
        "num_samples": int(values.size),
        "mean_error": float(np.mean(values)),
        "variance": float(np.var(values)),
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
    }


def _collision_confusion_stats(predicted_flags: list[bool], true_flags: list[bool]) -> dict[str, float | int]:
    if len(predicted_flags) != len(true_flags):
        raise ValueError("Predicted and true collision flag samples must have matching lengths")
    num_samples = int(len(predicted_flags))
    if num_samples == 0:
        return {
            "num_samples": 0,
            "predicted_collision_count": 0,
            "true_collision_count": 0,
            "true_positive_count": 0,
            "true_negative_count": 0,
            "false_positive_count": 0,
            "false_negative_count": 0,
            "predicted_collision_rate": float("nan"),
            "true_collision_rate": float("nan"),
            "false_positive_rate": float("nan"),
            "false_negative_rate": float("nan"),
            "accuracy": float("nan"),
        }
    predicted = np.asarray(predicted_flags, dtype=bool)
    truth = np.asarray(true_flags, dtype=bool)
    true_positive = int(np.sum(predicted & truth))
    true_negative = int(np.sum((~predicted) & (~truth)))
    false_positive = int(np.sum(predicted & (~truth)))
    false_negative = int(np.sum((~predicted) & truth))
    predicted_collision_count = int(np.sum(predicted))
    true_collision_count = int(np.sum(truth))
    return {
        "num_samples": num_samples,
        "predicted_collision_count": predicted_collision_count,
        "true_collision_count": true_collision_count,
        "true_positive_count": true_positive,
        "true_negative_count": true_negative,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "predicted_collision_rate": float(predicted_collision_count / num_samples),
        "true_collision_rate": float(true_collision_count / num_samples),
        "false_positive_rate": float(false_positive / num_samples),
        "false_negative_rate": float(false_negative / num_samples),
        "accuracy": float((true_positive + true_negative) / num_samples),
    }


def _metric_stats_from_error_lists(metric_errors: dict[str, list[float]]) -> dict[str, object]:
    return {
        "num_samples": int(len(metric_errors["x_rel"])),
        "x_rel": _scalar_error_stats(metric_errors["x_rel"]),
        "y_rel": _scalar_error_stats(metric_errors["y_rel"]),
        "vx_rel": _scalar_error_stats(metric_errors["vx_rel"]),
        "vy_rel": _scalar_error_stats(metric_errors["vy_rel"]),
        "speed_rel": _scalar_error_stats(metric_errors["speed_rel"]),
        "radial_distance": _scalar_error_stats(metric_errors["radial_distance"]),
        "radial_closing_speed": _scalar_error_stats(metric_errors["radial_closing_speed"]),
    }


def _empty_metric_error_lists() -> dict[str, list[float]]:
    return {
        "x_rel": [],
        "y_rel": [],
        "vx_rel": [],
        "vy_rel": [],
        "speed_rel": [],
        "radial_distance": [],
        "radial_closing_speed": [],
    }


def _empty_collision_confusion_lists() -> dict[str, list[bool]]:
    return {
        "predicted": [],
        "true": [],
    }


def _radial_closing_speed(position_rel: np.ndarray, velocity_rel: np.ndarray, eps: float = 1e-6) -> float:
    radial_distance = float(np.linalg.norm(position_rel))
    return float(np.dot(position_rel, velocity_rel) / max(radial_distance, float(eps)))


def _append_rollout_error_sample(
    target: dict[str, list[float]],
    *,
    pred_pos: np.ndarray,
    true_pos: np.ndarray,
    pred_vel: np.ndarray,
    true_vel: np.ndarray,
) -> None:
    pred_speed = float(np.linalg.norm(pred_vel))
    true_speed = float(np.linalg.norm(true_vel))
    pred_radial_distance = float(np.linalg.norm(pred_pos))
    true_radial_distance = float(np.linalg.norm(true_pos))
    pred_radial_closing = _radial_closing_speed(pred_pos, pred_vel)
    true_radial_closing = _radial_closing_speed(true_pos, true_vel)

    target["x_rel"].append(float(pred_pos[0] - true_pos[0]))
    target["y_rel"].append(float(pred_pos[1] - true_pos[1]))
    target["vx_rel"].append(float(pred_vel[0] - true_vel[0]))
    target["vy_rel"].append(float(pred_vel[1] - true_vel[1]))
    target["speed_rel"].append(float(pred_speed - true_speed))
    target["radial_distance"].append(float(pred_radial_distance - true_radial_distance))
    target["radial_closing_speed"].append(float(pred_radial_closing - true_radial_closing))


def _step_key(step_index: int) -> str:
    return f"step_{int(step_index) + 1}"


def _horizon_error_growth_from_steps(step_stats: dict[str, dict[str, object]]) -> dict[str, list[float]]:
    metric_names = ["x_rel", "y_rel", "vx_rel", "vy_rel", "radial_distance", "radial_closing_speed"]
    growth = {metric: [] for metric in metric_names}
    for step_name in sorted(step_stats.keys(), key=lambda name: int(name.split("_")[1])):
        stats = step_stats[step_name]
        for metric in metric_names:
            growth[metric].append(float(dict(stats[metric]).get("mae", float("nan"))))
    return growth


def _clearance_error_growth_from_steps(step_stats: dict[str, dict[str, object]]) -> list[float]:
    return [
        float(dict(step_stats[step_name]["clearance_error"]).get("mae", float("nan")))
        for step_name in sorted(step_stats.keys(), key=lambda name: int(name.split("_")[1]))
    ]


def _false_negative_rate_growth_from_steps(step_stats: dict[str, dict[str, object]]) -> list[float]:
    return [
        float(dict(step_stats[step_name]["dynamic_collision_confusion"]).get("false_negative_rate", float("nan")))
        for step_name in sorted(step_stats.keys(), key=lambda name: int(name.split("_")[1]))
    ]


def _sorted_step_names(step_stats: dict[str, dict[str, object]]) -> list[str]:
    return sorted(step_stats.keys(), key=lambda name: int(name.split("_")[1]))


def _enrich_scalar_stats_block(stats_block: dict[str, object]) -> dict[str, object]:
    stats = dict(stats_block)
    mean_error = float(stats.get("mean_error", float("nan")))
    variance = float(stats.get("variance", float("nan")))
    if np.isfinite(mean_error) and np.isfinite(variance):
        bias_squared = float(mean_error**2)
        mse_approx = float(bias_squared + variance)
    else:
        bias_squared = float("nan")
        mse_approx = float("nan")
    stats["bias_squared"] = bias_squared
    stats["mse_approx"] = mse_approx
    return stats


def _pooled_scalar_stats_from_blocks(blocks: list[dict[str, object]]) -> dict[str, object]:
    total_samples = 0
    weighted_mean = 0.0
    weighted_mae = 0.0
    weighted_second_moment = 0.0
    weighted_mse = 0.0
    for block in blocks:
        num_samples = int(dict(block).get("num_samples", 0))
        if num_samples <= 0:
            continue
        mean_error = float(dict(block).get("mean_error", float("nan")))
        variance = float(dict(block).get("variance", float("nan")))
        mae = float(dict(block).get("mae", float("nan")))
        rmse = float(dict(block).get("rmse", float("nan")))
        if not (np.isfinite(mean_error) and np.isfinite(variance) and np.isfinite(mae) and np.isfinite(rmse)):
            continue
        total_samples += num_samples
        weighted_mean += num_samples * mean_error
        weighted_mae += num_samples * mae
        weighted_second_moment += num_samples * (variance + mean_error**2)
        weighted_mse += num_samples * (rmse**2)
    if total_samples <= 0:
        return _enrich_scalar_stats_block(_scalar_error_stats([]))
    pooled_mean = float(weighted_mean / total_samples)
    pooled_variance = float(max(0.0, (weighted_second_moment / total_samples) - pooled_mean**2))
    pooled_rmse = float(np.sqrt(max(0.0, weighted_mse / total_samples)))
    pooled_stats = {
        "num_samples": int(total_samples),
        "mean_error": pooled_mean,
        "variance": pooled_variance,
        "mae": float(weighted_mae / total_samples),
        "rmse": pooled_rmse,
    }
    return _enrich_scalar_stats_block(pooled_stats)


def _pooled_confusion_stats_from_blocks(blocks: list[dict[str, object]]) -> dict[str, float | int]:
    summed = {
        "num_samples": 0,
        "predicted_collision_count": 0,
        "true_collision_count": 0,
        "true_positive_count": 0,
        "true_negative_count": 0,
        "false_positive_count": 0,
        "false_negative_count": 0,
    }
    for block in blocks:
        for key in summed:
            summed[key] += int(dict(block).get(key, 0))
    num_samples = int(summed["num_samples"])
    if num_samples <= 0:
        return {
            **summed,
            "predicted_collision_rate": float("nan"),
            "true_collision_rate": float("nan"),
            "false_positive_rate": float("nan"),
            "false_negative_rate": float("nan"),
            "accuracy": float("nan"),
        }
    return {
        **summed,
        "predicted_collision_rate": float(summed["predicted_collision_count"] / num_samples),
        "true_collision_rate": float(summed["true_collision_count"] / num_samples),
        "false_positive_rate": float(summed["false_positive_count"] / num_samples),
        "false_negative_rate": float(summed["false_negative_count"] / num_samples),
        "accuracy": float((summed["true_positive_count"] + summed["true_negative_count"]) / num_samples),
    }


def _metric_series_from_steps(
    step_stats: dict[str, dict[str, object]],
    metric_name: str,
    field_name: str,
) -> list[float]:
    values: list[float] = []
    for step_name in _sorted_step_names(step_stats):
        metric_block = dict(dict(step_stats[step_name]).get(metric_name, {}))
        values.append(float(metric_block.get(field_name, float("nan"))))
    return values


def _fraction_of_action_horizon_cells_with_bias(
    aggregate_by_action_family: dict[str, dict[str, object]],
    metric_name: str,
    *,
    positive: bool,
    eps: float = 1e-12,
) -> float:
    matching = 0
    total = 0
    for payload in aggregate_by_action_family.values():
        for step_payload in dict(payload.get("per_horizon_step", {})).values():
            metric_block = dict(dict(step_payload).get(metric_name, {}))
            num_samples = int(metric_block.get("num_samples", 0))
            mean_error = float(metric_block.get("mean_error", float("nan")))
            if num_samples <= 0 or not np.isfinite(mean_error):
                continue
            total += 1
            if positive and mean_error > eps:
                matching += 1
            if (not positive) and mean_error < -eps:
                matching += 1
    if total <= 0:
        return float("nan")
    return float(matching / total)


def _enrich_rollout_stats_payload(
    payload: dict[str, object],
    *,
    include_episode_pooled_fields: bool,
) -> dict[str, object]:
    aggregate_per_horizon_step = dict(payload.get("aggregate_per_horizon_step", {}))
    aggregate_by_action_family = dict(payload.get("aggregate_by_action_family", {}))
    if not aggregate_per_horizon_step:
        return payload

    step_names = _sorted_step_names(aggregate_per_horizon_step)
    clearance_bias_by_horizon = _metric_series_from_steps(aggregate_per_horizon_step, "clearance_error", "mean_error")
    radial_distance_bias_by_horizon = _metric_series_from_steps(aggregate_per_horizon_step, "radial_distance", "mean_error")
    radial_closing_bias_by_horizon = _metric_series_from_steps(
        aggregate_per_horizon_step, "radial_closing_speed", "mean_error"
    )

    if include_episode_pooled_fields:
        payload["pooled_clearance_bias_by_horizon"] = clearance_bias_by_horizon
        payload["pooled_clearance_mae_by_horizon"] = _metric_series_from_steps(
            aggregate_per_horizon_step, "clearance_error", "mae"
        )
        payload["pooled_false_negative_rate_by_horizon"] = [
            float(dict(aggregate_per_horizon_step[step_name]["dynamic_collision_confusion"]).get("false_negative_rate", float("nan")))
            for step_name in step_names
        ]

    payload["mean_signed_clearance_bias_by_horizon"] = clearance_bias_by_horizon
    payload["mean_signed_radial_distance_bias_by_horizon"] = radial_distance_bias_by_horizon
    payload["mean_signed_radial_closing_speed_bias_by_horizon"] = radial_closing_bias_by_horizon
    payload["optimistic_bias_horizons"] = [
        bool(np.isfinite(value) and value > 1e-12) for value in clearance_bias_by_horizon
    ]
    payload["conservative_bias_horizons"] = [
        bool(np.isfinite(value) and value < -1e-12) for value in clearance_bias_by_horizon
    ]
    payload["fraction_of_action_horizon_cells_with_positive_clearance_bias"] = _fraction_of_action_horizon_cells_with_bias(
        aggregate_by_action_family, "clearance_error", positive=True
    )
    payload["fraction_of_action_horizon_cells_with_negative_clearance_bias"] = _fraction_of_action_horizon_cells_with_bias(
        aggregate_by_action_family, "clearance_error", positive=False
    )
    payload["fraction_of_action_horizon_cells_with_positive_radial_distance_bias"] = _fraction_of_action_horizon_cells_with_bias(
        aggregate_by_action_family, "radial_distance", positive=True
    )
    payload["fraction_of_action_horizon_cells_with_negative_radial_distance_bias"] = _fraction_of_action_horizon_cells_with_bias(
        aggregate_by_action_family, "radial_distance", positive=False
    )
    payload["fraction_of_action_horizon_cells_with_positive_radial_closing_speed_bias"] = _fraction_of_action_horizon_cells_with_bias(
        aggregate_by_action_family, "radial_closing_speed", positive=True
    )
    payload["fraction_of_action_horizon_cells_with_negative_radial_closing_speed_bias"] = _fraction_of_action_horizon_cells_with_bias(
        aggregate_by_action_family, "radial_closing_speed", positive=False
    )

    metric_names = ["clearance_error", "radial_distance", "radial_closing_speed", "x_rel", "y_rel"]
    payload["bias_variance_by_horizon"] = {
        metric_name: {
            "mean_error": _metric_series_from_steps(aggregate_per_horizon_step, metric_name, "mean_error"),
            "variance": _metric_series_from_steps(aggregate_per_horizon_step, metric_name, "variance"),
            "bias_squared": _metric_series_from_steps(
                {
                    step_name: {
                        **aggregate_per_horizon_step[step_name],
                        metric_name: _enrich_scalar_stats_block(
                            dict(dict(aggregate_per_horizon_step[step_name]).get(metric_name, {}))
                        ),
                    }
                    for step_name in step_names
                },
                metric_name,
                "bias_squared",
            ),
            "mse_approx": _metric_series_from_steps(
                {
                    step_name: {
                        **aggregate_per_horizon_step[step_name],
                        metric_name: _enrich_scalar_stats_block(
                            dict(dict(aggregate_per_horizon_step[step_name]).get(metric_name, {}))
                        ),
                    }
                    for step_name in step_names
                },
                metric_name,
                "mse_approx",
            ),
        }
        for metric_name in metric_names
    }

    pooled_metrics = {
        "clearance_error": _pooled_scalar_stats_from_blocks(
            [dict(aggregate_per_horizon_step[step_name]["clearance_error"]) for step_name in step_names]
        ),
        "radial_distance": _pooled_scalar_stats_from_blocks(
            [dict(aggregate_per_horizon_step[step_name]["radial_distance"]) for step_name in step_names]
        ),
        "radial_closing_speed": _pooled_scalar_stats_from_blocks(
            [dict(aggregate_per_horizon_step[step_name]["radial_closing_speed"]) for step_name in step_names]
        ),
        "x_rel": _pooled_scalar_stats_from_blocks(
            [dict(aggregate_per_horizon_step[step_name]["x_rel"]) for step_name in step_names]
        ),
        "y_rel": _pooled_scalar_stats_from_blocks(
            [dict(aggregate_per_horizon_step[step_name]["y_rel"]) for step_name in step_names]
        ),
        "dynamic_collision_confusion": _pooled_confusion_stats_from_blocks(
            [dict(aggregate_per_horizon_step[step_name]["dynamic_collision_confusion"]) for step_name in step_names]
        ),
    }
    payload["global_pooled_summaries"] = pooled_metrics
    payload["global_clearance_mae"] = float(dict(pooled_metrics["clearance_error"]).get("mae", float("nan")))
    payload["global_signed_clearance_bias"] = float(dict(pooled_metrics["clearance_error"]).get("mean_error", float("nan")))
    payload["global_false_negative_rate"] = float(
        dict(pooled_metrics["dynamic_collision_confusion"]).get("false_negative_rate", float("nan"))
    )
    payload["global_false_positive_rate"] = float(
        dict(pooled_metrics["dynamic_collision_confusion"]).get("false_positive_rate", float("nan"))
    )
    payload["global_radial_distance_signed_bias"] = float(
        dict(pooled_metrics["radial_distance"]).get("mean_error", float("nan"))
    )
    payload["global_radial_closing_speed_signed_bias"] = float(
        dict(pooled_metrics["radial_closing_speed"]).get("mean_error", float("nan"))
    )
    return payload


def _build_rollout_error_stats_from_episodes(episodes: list[dict[str, object]]) -> dict[str, object]:
    per_step_errors: defaultdict[int, dict[str, list[float]]] = defaultdict(_empty_metric_error_lists)
    per_step_clearance_errors: defaultdict[int, list[float]] = defaultdict(list)
    per_step_collision_confusion: defaultdict[int, dict[str, list[bool]]] = defaultdict(_empty_collision_confusion_lists)
    per_action_family_errors: defaultdict[str, defaultdict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(_empty_metric_error_lists)
    )
    per_action_family_clearance_errors: defaultdict[str, defaultdict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_action_family_collision_confusion: defaultdict[str, defaultdict[int, dict[str, list[bool]]]] = defaultdict(
        lambda: defaultdict(_empty_collision_confusion_lists)
    )
    per_action_family_timestep_counts: Counter[str] = Counter()
    horizon = 0

    for episode in episodes:
        for step in episode.get("steps", []):
            controller_debug = dict(step.get("controller_debug", {}))
            if str(controller_debug.get("mode", "")) != "learned_structured_rollout":
                continue
            chosen = dict(controller_debug.get("chosen", {}))
            pred_positions = list(chosen.get("predicted_positions_robot", []))
            pred_velocities = list(chosen.get("predicted_relative_velocities", []))
            true_positions = list(chosen.get("true_positions_robot", []))
            true_velocities = list(chosen.get("true_relative_velocities", []))
            predicted_clearances = [float(value) for value in chosen.get("dynamic_clearances", [])]
            predicted_collisions = [bool(value) for value in chosen.get("dynamic_collisions", [])]
            true_clearances = [float(value) for value in chosen.get("true_dynamic_clearances", [])]
            true_collisions = [bool(value) for value in chosen.get("true_dynamic_collisions", [])]
            action_family = _safe_action_family(str(step.get("action_name", "")))
            per_action_family_timestep_counts[action_family] += 1
            horizon = max(horizon, len(pred_positions))
            for step_index, (pred_pos_step, pred_vel_step, true_pos_step, true_vel_step) in enumerate(
                zip(pred_positions, pred_velocities, true_positions, true_velocities)
            ):
                for pred_pos, pred_vel, true_pos, true_vel in zip(pred_pos_step, pred_vel_step, true_pos_step, true_vel_step):
                    _append_rollout_error_sample(
                        per_step_errors[step_index],
                        pred_pos=np.asarray(pred_pos, dtype=float),
                        true_pos=np.asarray(true_pos, dtype=float),
                        pred_vel=np.asarray(pred_vel, dtype=float),
                        true_vel=np.asarray(true_vel, dtype=float),
                    )
                    _append_rollout_error_sample(
                        per_action_family_errors[action_family][step_index],
                        pred_pos=np.asarray(pred_pos, dtype=float),
                        true_pos=np.asarray(true_pos, dtype=float),
                        pred_vel=np.asarray(pred_vel, dtype=float),
                        true_vel=np.asarray(true_vel, dtype=float),
                    )
            for step_index, (predicted_clearance, true_clearance, predicted_collision, true_collision) in enumerate(
                zip(predicted_clearances, true_clearances, predicted_collisions, true_collisions)
            ):
                per_step_clearance_errors[step_index].append(float(predicted_clearance - true_clearance))
                per_step_collision_confusion[step_index]["predicted"].append(bool(predicted_collision))
                per_step_collision_confusion[step_index]["true"].append(bool(true_collision))
                per_action_family_clearance_errors[action_family][step_index].append(float(predicted_clearance - true_clearance))
                per_action_family_collision_confusion[action_family][step_index]["predicted"].append(bool(predicted_collision))
                per_action_family_collision_confusion[action_family][step_index]["true"].append(bool(true_collision))

    aggregate_per_horizon_step = {
        _step_key(step_index): {
            **_metric_stats_from_error_lists(errors),
            "clearance_error": _scalar_error_stats(per_step_clearance_errors[step_index]),
            "dynamic_collision_confusion": _collision_confusion_stats(
                per_step_collision_confusion[step_index]["predicted"],
                per_step_collision_confusion[step_index]["true"],
            ),
        }
        for step_index, errors in sorted(per_step_errors.items())
    }
    aggregate_by_action_family = {}
    for action_family, step_errors in sorted(per_action_family_errors.items()):
        per_horizon_step = {
            _step_key(step_index): {
                **_metric_stats_from_error_lists(errors),
                "clearance_error": _scalar_error_stats(per_action_family_clearance_errors[action_family][step_index]),
                "dynamic_collision_confusion": _collision_confusion_stats(
                    per_action_family_collision_confusion[action_family][step_index]["predicted"],
                    per_action_family_collision_confusion[action_family][step_index]["true"],
                ),
            }
            for step_index, errors in sorted(step_errors.items())
        }
        aggregate_by_action_family[action_family] = {
            "num_contributing_timesteps": int(per_action_family_timestep_counts[action_family]),
            "per_horizon_step": per_horizon_step,
        }

    summary: dict[str, object] = {
        "controller": "learned_structured",
        "num_episodes": int(len(episodes)),
        "horizon": int(horizon),
        "aggregate_per_horizon_step": aggregate_per_horizon_step,
        "aggregate_by_action_family": aggregate_by_action_family,
    }
    if aggregate_per_horizon_step:
        summary["horizon_error_growth"] = _horizon_error_growth_from_steps(aggregate_per_horizon_step)
        summary["horizon_clearance_error_mae"] = _clearance_error_growth_from_steps(aggregate_per_horizon_step)
        summary["horizon_false_negative_rate"] = _false_negative_rate_growth_from_steps(aggregate_per_horizon_step)
    if aggregate_by_action_family:
        summary["action_conditioned_horizon_error_growth"] = {
            action_family: _horizon_error_growth_from_steps(dict(payload["per_horizon_step"]))
            for action_family, payload in aggregate_by_action_family.items()
        }
        summary["action_conditioned_horizon_clearance_error_mae"] = {
            action_family: _clearance_error_growth_from_steps(dict(payload["per_horizon_step"]))
            for action_family, payload in aggregate_by_action_family.items()
        }
        summary["action_conditioned_horizon_false_negative_rate"] = {
            action_family: _false_negative_rate_growth_from_steps(dict(payload["per_horizon_step"]))
            for action_family, payload in aggregate_by_action_family.items()
        }
    return _enrich_rollout_stats_payload(summary, include_episode_pooled_fields=False)


def _build_learned_rollout_error_stats(episode: dict[str, object]) -> dict[str, object]:
    per_obstacle_per_horizon_step: defaultdict[str, defaultdict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(_empty_metric_error_lists)
    )
    aggregate_per_horizon_step_errors: defaultdict[int, dict[str, list[float]]] = defaultdict(_empty_metric_error_lists)
    aggregate_per_horizon_step_clearance_errors: defaultdict[int, list[float]] = defaultdict(list)
    aggregate_per_horizon_step_collision_confusion: defaultdict[int, dict[str, list[bool]]] = defaultdict(
        _empty_collision_confusion_lists
    )
    aggregate_by_action_family_errors: defaultdict[str, defaultdict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(_empty_metric_error_lists)
    )
    aggregate_by_action_family_clearance_errors: defaultdict[str, defaultdict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    aggregate_by_action_family_collision_confusion: defaultdict[str, defaultdict[int, dict[str, list[bool]]]] = defaultdict(
        lambda: defaultdict(_empty_collision_confusion_lists)
    )
    action_family_timestep_counts: Counter[str] = Counter()
    obstacle_timestep_counts: Counter[str] = Counter()
    episode_obstacle_ids: list[str] | None = None
    obstacle_ids_stable = True
    horizon = 0

    for step in episode.get("steps", []):
        controller_debug = dict(step.get("controller_debug", {}))
        if str(controller_debug.get("mode", "")) != "learned_structured_rollout":
            continue
        chosen = dict(controller_debug.get("chosen", {}))
        pred_positions = list(chosen.get("predicted_positions_robot", []))
        pred_velocities = list(chosen.get("predicted_relative_velocities", []))
        true_positions = list(chosen.get("true_positions_robot", []))
        true_velocities = list(chosen.get("true_relative_velocities", []))
        predicted_clearances = [float(value) for value in chosen.get("dynamic_clearances", [])]
        predicted_collisions = [bool(value) for value in chosen.get("dynamic_collisions", [])]
        true_clearances = [float(value) for value in chosen.get("true_dynamic_clearances", [])]
        true_collisions = [bool(value) for value in chosen.get("true_dynamic_collisions", [])]
        if not (pred_positions and pred_velocities and true_positions and true_velocities):
            continue

        obstacle_ids = [str(snapshot["obstacle_id"]) for snapshot in step.get("dynamic_obstacles_t", [])]
        if episode_obstacle_ids is None:
            episode_obstacle_ids = list(obstacle_ids)
        elif obstacle_ids != episode_obstacle_ids:
            obstacle_ids_stable = False

        action_family = _safe_action_family(str(step.get("action_name", "")))
        action_family_timestep_counts[action_family] += 1
        for obstacle_id in obstacle_ids:
            obstacle_timestep_counts[obstacle_id] += 1

        horizon = max(horizon, len(pred_positions))
        # Aggregate chosen-candidate rollout error per obstacle and horizon step so we can
        # tell whether failures come from long-horizon drift, turning actions, or specific obstacles.
        for step_index, (pred_pos_step, pred_vel_step, true_pos_step, true_vel_step) in enumerate(
            zip(pred_positions, pred_velocities, true_positions, true_velocities)
        ):
            for obstacle_id, pred_pos, pred_vel, true_pos, true_vel in zip(
                obstacle_ids, pred_pos_step, pred_vel_step, true_pos_step, true_vel_step
            ):
                pred_pos_arr = np.asarray(pred_pos, dtype=float)
                pred_vel_arr = np.asarray(pred_vel, dtype=float)
                true_pos_arr = np.asarray(true_pos, dtype=float)
                true_vel_arr = np.asarray(true_vel, dtype=float)
                _append_rollout_error_sample(
                    per_obstacle_per_horizon_step[obstacle_id][step_index],
                    pred_pos=pred_pos_arr,
                    true_pos=true_pos_arr,
                    pred_vel=pred_vel_arr,
                    true_vel=true_vel_arr,
                )
                _append_rollout_error_sample(
                    aggregate_per_horizon_step_errors[step_index],
                    pred_pos=pred_pos_arr,
                    true_pos=true_pos_arr,
                    pred_vel=pred_vel_arr,
                    true_vel=true_vel_arr,
                )
                _append_rollout_error_sample(
                    aggregate_by_action_family_errors[action_family][step_index],
                    pred_pos=pred_pos_arr,
                    true_pos=true_pos_arr,
                    pred_vel=pred_vel_arr,
                    true_vel=true_vel_arr,
                )
        for step_index, (predicted_clearance, true_clearance, predicted_collision, true_collision) in enumerate(
            zip(predicted_clearances, true_clearances, predicted_collisions, true_collisions)
        ):
            aggregate_per_horizon_step_clearance_errors[step_index].append(float(predicted_clearance - true_clearance))
            aggregate_per_horizon_step_collision_confusion[step_index]["predicted"].append(bool(predicted_collision))
            aggregate_per_horizon_step_collision_confusion[step_index]["true"].append(bool(true_collision))
            aggregate_by_action_family_clearance_errors[action_family][step_index].append(float(predicted_clearance - true_clearance))
            aggregate_by_action_family_collision_confusion[action_family][step_index]["predicted"].append(bool(predicted_collision))
            aggregate_by_action_family_collision_confusion[action_family][step_index]["true"].append(bool(true_collision))

    obstacle_ids_out = episode_obstacle_ids or []
    per_obstacle_stats = {
        obstacle_id: {
            _step_key(step_index): _metric_stats_from_error_lists(errors)
            for step_index, errors in sorted(step_errors.items())
        }
        for obstacle_id, step_errors in sorted(per_obstacle_per_horizon_step.items())
    }
    aggregate_per_horizon_step = {
        _step_key(step_index): {
            **_metric_stats_from_error_lists(errors),
            "clearance_error": _scalar_error_stats(aggregate_per_horizon_step_clearance_errors[step_index]),
            "dynamic_collision_confusion": _collision_confusion_stats(
                aggregate_per_horizon_step_collision_confusion[step_index]["predicted"],
                aggregate_per_horizon_step_collision_confusion[step_index]["true"],
            ),
        }
        for step_index, errors in sorted(aggregate_per_horizon_step_errors.items())
    }
    aggregate_by_action_family = {}
    for action_family, step_errors in sorted(aggregate_by_action_family_errors.items()):
        per_horizon_step = {
            _step_key(step_index): {
                **_metric_stats_from_error_lists(errors),
                "clearance_error": _scalar_error_stats(aggregate_by_action_family_clearance_errors[action_family][step_index]),
                "dynamic_collision_confusion": _collision_confusion_stats(
                    aggregate_by_action_family_collision_confusion[action_family][step_index]["predicted"],
                    aggregate_by_action_family_collision_confusion[action_family][step_index]["true"],
                ),
            }
            for step_index, errors in sorted(step_errors.items())
        }
        aggregate_by_action_family[action_family] = {
            "num_contributing_timesteps": int(action_family_timestep_counts[action_family]),
            "per_horizon_step": per_horizon_step,
        }

    stats: dict[str, object] = {
        "episode_id": int(episode["episode_id"]),
        "seed": int(episode["seed"]),
        "controller": str(episode["controller"]),
        "horizon": int(horizon),
        "num_real_steps": int(len(episode.get("steps", []))),
        "obstacle_ids": obstacle_ids_out,
        "obstacle_ids_stable": bool(obstacle_ids_stable),
        "num_contributing_timesteps_per_action_family": dict(sorted(action_family_timestep_counts.items())),
        "num_contributing_timesteps_per_obstacle": dict(sorted(obstacle_timestep_counts.items())),
        "per_obstacle_per_horizon_step": per_obstacle_stats,
        "aggregate_per_horizon_step": aggregate_per_horizon_step,
        "aggregate_by_action_family": aggregate_by_action_family,
    }
    if aggregate_per_horizon_step:
        stats["horizon_error_growth"] = _horizon_error_growth_from_steps(aggregate_per_horizon_step)
        stats["horizon_clearance_error_mae"] = _clearance_error_growth_from_steps(aggregate_per_horizon_step)
        stats["horizon_false_negative_rate"] = _false_negative_rate_growth_from_steps(aggregate_per_horizon_step)
    if aggregate_by_action_family:
        stats["action_conditioned_horizon_error_growth"] = {
            action_family: _horizon_error_growth_from_steps(dict(payload["per_horizon_step"]))
            for action_family, payload in aggregate_by_action_family.items()
        }
        stats["action_conditioned_horizon_clearance_error_mae"] = {
            action_family: _clearance_error_growth_from_steps(dict(payload["per_horizon_step"]))
            for action_family, payload in aggregate_by_action_family.items()
        }
        stats["action_conditioned_horizon_false_negative_rate"] = {
            action_family: _false_negative_rate_growth_from_steps(dict(payload["per_horizon_step"]))
            for action_family, payload in aggregate_by_action_family.items()
        }
    return _enrich_rollout_stats_payload(stats, include_episode_pooled_fields=True)


def _run_controller_episode(
    *,
    simulator: EpisodeSimulator,
    controller: Any,
    scenario: dict[str, Any],
    episode_id: int,
    seed: int,
) -> dict[str, object]:
    config = simulator.config
    map_bundle = scenario["map_bundle"]
    occupancy = map_bundle.occupancy
    planning_occupancy = scenario["planning_occupancy"]
    occupied_points = scenario["occupied_points"]
    robot_pose_true = scenario["start_pose"].copy()
    goal_point = scenario["goal_point"].copy()
    path_world = scenario["path_world"].copy()
    dyn_obstacles = [obstacle.clone() for obstacle in scenario["dynamic_obstacles"]]
    runtime_rng = np.random.default_rng(seed)

    controller.reset(initial_pose=robot_pose_true.copy(), seed=seed)
    steps: list[dict[str, object]] = []
    robot_trajectory = [robot_pose_true.copy()]
    dynamic_trajectories: dict[str, list[np.ndarray]] = {
        obstacle.obstacle_id: [obstacle.position.copy()] for obstacle in dyn_obstacles
    }
    static_clearance_history: list[float] = []
    dynamic_clearance_history: list[float] = []
    combined_clearance_history: list[float] = []
    done_reason = "timeout"
    stochastic_planning_active = bool(getattr(controller.controller_cfg, "execution_noise_enabled", False))

    for step_index in range(int(config.max_steps)):
        robot_pose_est = np.asarray(controller.current_pose_estimate(), dtype=float)
        waypoint_now_est = compute_local_subgoal(robot_pose_est, path_world, config.lookahead_distance)
        waypoint_now_true = compute_local_subgoal(robot_pose_true, path_world, config.lookahead_distance)
        decision = controller.select_action(
            robot_pose=robot_pose_est,
            current_subgoal_world=waypoint_now_est["selected_subgoal_world"],
            path_world=path_world,
            planning_occupancy=planning_occupancy,
            true_occupancy=occupancy,
            resolution=map_bundle.resolution,
            origin=map_bundle.origin,
            occupied_points=occupied_points,
            dynamic_obstacles=dyn_obstacles,
        )

        dynamic_t = _snapshot_obstacles(dyn_obstacles)
        commanded_control = np.array([float(decision["v"]), float(decision["omega"])], dtype=float)
        executed_control = sample_executed_control(
            runtime_rng,
            commanded_control,
            execution_noise_enabled=bool(getattr(controller.controller_cfg, "execution_noise_enabled", False)),
            sigma_v=float(getattr(controller.controller_cfg, "sigma_v", 0.0)),
            sigma_omega=float(getattr(controller.controller_cfg, "sigma_omega", 0.0)),
        )
        robot_pose_next_true = simulator._apply_action(robot_pose_true, float(executed_control[0]), float(executed_control[1]))
        for obstacle in dyn_obstacles:
            obstacle.step(config.robot_config.dt)
        dynamic_t1 = _snapshot_obstacles(dyn_obstacles)
        pose_observation = sample_pose_observation(
            runtime_rng,
            robot_pose_next_true,
            pose_observation_noise_enabled=bool(getattr(controller.controller_cfg, "pose_observation_noise_enabled", False)),
            sigma_obs_x=float(getattr(controller.controller_cfg, "sigma_obs_x", 0.0)),
            sigma_obs_y=float(getattr(controller.controller_cfg, "sigma_obs_y", 0.0)),
            sigma_obs_theta=float(getattr(controller.controller_cfg, "sigma_obs_theta", 0.0)),
        )

        controller.commit_step(
            robot_pose_t=robot_pose_true.copy(),
            robot_pose_t1=robot_pose_next_true.copy(),
            robot_pose_est_t=robot_pose_est.copy(),
            pose_observation=pose_observation.copy(),
            dynamic_obstacles_t=dynamic_t,
            dynamic_obstacles_t1=dynamic_t1,
            action_index=int(decision["action_index"]),
            action_cont=commanded_control.astype(np.float32),
            path_world=path_world,
        )
        robot_pose_next_est = np.asarray(controller.current_pose_estimate(), dtype=float)

        static_collision = bool(
            disk_collides_with_occupancy(
                robot_pose_next_true[:2],
                config.robot_config.radius,
                occupancy,
                map_bundle.resolution,
                map_bundle.origin,
            )
        )
        dynamic_collision = bool(
            any(
                np.linalg.norm(robot_pose_next_true[:2] - np.asarray(obstacle.position, dtype=float))
                <= config.robot_config.radius + float(obstacle.radius)
                for obstacle in dyn_obstacles
            )
        )
        goal_reached = euclidean(robot_pose_next_true[:2], goal_point) <= config.robot_config.goal_tolerance

        if goal_reached:
            done_reason = "goal_reached"
        elif static_collision:
            done_reason = "static_collision"
        elif dynamic_collision:
            done_reason = "dynamic_collision"
        elif step_index == int(config.max_steps) - 1:
            done_reason = "timeout"

        waypoint_next_true = compute_local_subgoal(robot_pose_next_true, path_world, config.lookahead_distance)
        static_clearance = float(
            minimum_static_clearance(
                robot_pose_next_true[:2],
                config.robot_config.radius,
                occupied_points,
                map_bundle.resolution,
            )
        )
        dynamic_clearance = float(minimum_dynamic_clearance(robot_pose_next_true[:2], config.robot_config.radius, dyn_obstacles))
        combined_clearance = float(min(static_clearance, dynamic_clearance))
        static_clearance_history.append(static_clearance)
        dynamic_clearance_history.append(dynamic_clearance)
        combined_clearance_history.append(combined_clearance)

        candidate_debug = decision["debug"]
        chosen_debug = dict(candidate_debug.get("chosen", {}))
        steps.append(
            {
                "t": int(step_index),
                "dt": float(config.robot_config.dt),
                "robot_pose": robot_pose_true.copy(),
                "robot_pose_t": robot_pose_true.copy(),
                "robot_pose_t1": robot_pose_next_true.copy(),
                "robot_pose_true_t": robot_pose_true.copy(),
                "robot_pose_true_t1": robot_pose_next_true.copy(),
                "robot_pose_est_t": robot_pose_est.copy(),
                "robot_pose_est_t1": robot_pose_next_est.copy(),
                "pose_observation_t1": pose_observation.copy(),
                "goal_t": waypoint_now_true["goal_features"].copy(),
                "goal_t1": waypoint_next_true["goal_features"].copy(),
                "selected_subgoal_world": waypoint_now_est["selected_subgoal_world"].copy(),
                "action_index": int(decision["action_index"]),
                "action_name": str(decision["action_name"]),
                "v": float(decision["v"]),
                "omega": float(decision["omega"]),
                "v_t": float(decision["v"]),
                "omega_t": float(decision["omega"]),
                "commanded_control": commanded_control.copy(),
                "executed_control": executed_control.copy(),
                "collision_t1": bool(static_collision or dynamic_collision),
                "static_collision_t1": bool(static_collision),
                "dynamic_collision_t1": bool(dynamic_collision),
                "done_t1": bool(goal_reached or static_collision or dynamic_collision or step_index == int(config.max_steps) - 1),
                "done_reason_t1": str(done_reason),
                "dynamic_obstacles_t": dynamic_t,
                "dynamic_obstacles_t1": dynamic_t1,
                "progress_to_subgoal": float(waypoint_now_true["goal_features"][2] - waypoint_next_true["goal_features"][2]),
                "min_static_clearance_t1": static_clearance,
                "min_dynamic_clearance_t1": dynamic_clearance,
                "min_total_clearance_t1": combined_clearance,
                "controller_debug": candidate_debug,
                "chosen_action_min_clearance": float(decision["chosen_action_min_clearance"]),
                "emergency_override_used": bool(decision["emergency_override_used"]),
                "score_components": _score_components_from_debug(chosen_debug),
            }
        )

        robot_pose_true = robot_pose_next_true
        robot_trajectory.append(robot_pose_true.copy())
        for obstacle in dyn_obstacles:
            dynamic_trajectories[obstacle.obstacle_id].append(obstacle.position.copy())
        if goal_reached or static_collision or dynamic_collision or step_index == int(config.max_steps) - 1:
            break

    trajectory_array = np.array(robot_trajectory, dtype=float)
    return {
        "episode_id": int(episode_id),
        "seed": int(seed),
        "schema_version": "v4_policy_comparison_1",
        "controller": str(controller.name),
        "map_id": map_bundle.map_id,
        "map_type": map_bundle.map_type,
        "occupancy_grid": occupancy.copy(),
        "planning_occupancy": planning_occupancy.copy(),
        "resolution": float(map_bundle.resolution),
        "origin": np.array(map_bundle.origin, dtype=float),
        "lidar_config": config.lidar_config.__dict__,
        "robot_config": config.robot_config.__dict__,
        "expert_config": config.expert_config.__dict__,
        "start_pose": scenario["start_pose"].copy(),
        "final_goal": goal_point.copy(),
        "astar_path_world": path_world.copy(),
        "astar_path_grid": np.array(scenario["path_rc"], dtype=np.int32),
        "dynamic_obstacle_routes": [obstacle.route_metadata() for obstacle in scenario["dynamic_obstacles"]],
        "steps": steps,
        "episode_stats": {
            "num_steps": int(len(steps)),
            "done_reason": done_reason,
            "success": bool(done_reason == "goal_reached"),
            "collision": bool(done_reason in {"static_collision", "dynamic_collision"}),
            "static_collision": bool(done_reason == "static_collision"),
            "dynamic_collision": bool(done_reason == "dynamic_collision"),
            "timeout": bool(done_reason == "timeout"),
            "astar_path_length": float(path_length(path_world)),
            "robot_path_length": float(path_length(trajectory_array[:, :2])),
            "final_distance_to_goal": float(euclidean(robot_pose_true[:2], goal_point)),
            "total_progress": float(sum(float(step["progress_to_subgoal"]) for step in steps)),
            "stop_fraction": _episode_stop_fraction(steps),
            "min_static_clearance": float(np.min(static_clearance_history)) if static_clearance_history else float("inf"),
            "mean_static_clearance": float(np.mean(static_clearance_history)) if static_clearance_history else float("inf"),
            "min_dynamic_clearance": float(np.min(dynamic_clearance_history)) if dynamic_clearance_history else float("inf"),
            "mean_dynamic_clearance": float(np.mean(dynamic_clearance_history)) if dynamic_clearance_history else float("inf"),
            "min_combined_clearance": float(np.min(combined_clearance_history)) if combined_clearance_history else float("inf"),
            "robot_trajectory": trajectory_array,
            "dynamic_trajectories": {obstacle_id: np.array(points, dtype=float) for obstacle_id, points in dynamic_trajectories.items()},
            "action_counts": _step_action_histogram(steps),
        },
        "stochastic_planning_active": bool(stochastic_planning_active),
    }


def _build_decision_log(episode: dict[str, object]) -> dict[str, object]:
    return {
        "episode_id": int(episode["episode_id"]),
        "seed": int(episode["seed"]),
        "controller": str(episode["controller"]),
        "done_reason": str(episode["episode_stats"]["done_reason"]),
        "num_steps": int(episode["episode_stats"]["num_steps"]),
        "steps": [
            {
                "t": int(step["t"]),
                "action_index": int(step["action_index"]),
                "action_name": str(step["action_name"]),
                "v": float(step["v"]),
                "omega": float(step["omega"]),
                "robot_pose_true_t": step.get("robot_pose_true_t", step.get("robot_pose_t")),
                "robot_pose_true_t1": step.get("robot_pose_true_t1", step.get("robot_pose_t1")),
                "robot_pose_est_t": step.get("robot_pose_est_t"),
                "robot_pose_est_t1": step.get("robot_pose_est_t1"),
                "pose_observation_t1": step.get("pose_observation_t1"),
                "commanded_control": step.get("commanded_control"),
                "executed_control": step.get("executed_control"),
                "collision_t1": bool(step["collision_t1"]),
                "static_collision_t1": bool(step["static_collision_t1"]),
                "dynamic_collision_t1": bool(step["dynamic_collision_t1"]),
                "done_t1": bool(step["done_t1"]),
                "done_reason_t1": str(step["done_reason_t1"]),
                "controller_debug": step.get("controller_debug", {}),
            }
            for step in episode["steps"]
        ],
    }


def _episode_result(episode: dict[str, object]) -> dict[str, object]:
    stats = episode["episode_stats"]
    backward_selected_count = 0
    backward_gated_count = 0
    backward_allowed_count = 0
    backward_penalties: list[float] = []
    backward_action_name = "backward"
    for step in episode["steps"]:
        controller_debug = dict(step.get("controller_debug", {}))
        backward_cfg = dict(controller_debug.get("backward_policy_config", {}))
        backward_action_name = str(backward_cfg.get("backward_action_name", backward_action_name))
        if str(step.get("action_name", "")) == backward_action_name:
            backward_selected_count += 1
        for candidate in list(controller_debug.get("candidates", [])):
            candidate_dict = dict(candidate)
            if not bool(candidate_dict.get("is_backward_action", False)):
                continue
            if bool(candidate_dict.get("backward_gated", False)):
                backward_gated_count += 1
            if bool(candidate_dict.get("backward_allowed", False)):
                backward_allowed_count += 1
            if bool(candidate_dict.get("backward_penalty_applied", False)):
                backward_penalties.append(float(candidate_dict.get("backward_penalty_value", 0.0)))
    return {
        "episode_id": int(episode["episode_id"]),
        "seed": int(episode["seed"]),
        "controller": str(episode["controller"]),
        "done_reason": str(stats["done_reason"]),
        "success": bool(stats["success"]),
        "static_collision": bool(stats["static_collision"]),
        "dynamic_collision": bool(stats["dynamic_collision"]),
        "timeout": bool(stats["timeout"]),
        "num_steps": int(stats["num_steps"]),
        "robot_path_length": float(stats["robot_path_length"]),
        "total_progress": float(stats["total_progress"]),
        "min_static_clearance": float(stats["min_static_clearance"]),
        "min_dynamic_clearance": float(stats["min_dynamic_clearance"]),
        "stop_fraction": float(stats["stop_fraction"]),
        "backward_selected_count": int(backward_selected_count),
        "backward_gated_count": int(backward_gated_count),
        "backward_allowed_count": int(backward_allowed_count),
        "average_backward_penalty_applied": _mean_finite(backward_penalties),
    }


def _mean_finite(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _aggregate_controller_episodes(episodes: list[dict[str, object]]) -> dict[str, object]:
    if not episodes:
        return {
            "num_episodes": 0,
            "success_rate": 0.0,
            "dynamic_collision_rate": 0.0,
            "static_collision_rate": 0.0,
            "timeout_rate": 0.0,
            "action_distribution": {},
            "mean_score_components": {},
        }

    n = len(episodes)
    success = sum(bool(episode["episode_stats"]["success"]) for episode in episodes)
    static_collisions = sum(bool(episode["episode_stats"]["static_collision"]) for episode in episodes)
    dynamic_collisions = sum(bool(episode["episode_stats"]["dynamic_collision"]) for episode in episodes)
    timeouts = sum(bool(episode["episode_stats"]["timeout"]) for episode in episodes)
    action_counts: Counter[str] = Counter()
    score_components: defaultdict[str, list[float]] = defaultdict(list)
    stop_fractions: list[float] = []
    path_lengths: list[float] = []
    progresses: list[float] = []
    min_dyn_clearances: list[float] = []
    min_static_clearances: list[float] = []
    backward_selected_counts: list[float] = []
    backward_gated_counts: list[float] = []
    backward_allowed_counts: list[float] = []
    backward_penalties: list[float] = []
    backward_action_name = "backward"

    for episode in episodes:
        stats = episode["episode_stats"]
        action_counts.update(stats.get("action_counts", {}))
        stop_fractions.append(float(stats["stop_fraction"]))
        path_lengths.append(float(stats["robot_path_length"]))
        progresses.append(float(stats["total_progress"]))
        min_dyn_clearances.append(float(stats["min_dynamic_clearance"]))
        min_static_clearances.append(float(stats["min_static_clearance"]))
        episode_result = _episode_result(episode)
        for step in episode["steps"]:
            controller_debug = dict(step.get("controller_debug", {}))
            backward_cfg = dict(controller_debug.get("backward_policy_config", {}))
            backward_action_name = str(backward_cfg.get("backward_action_name", backward_action_name))
            for key, value in step.get("score_components", {}).items():
                score_components[str(key)].append(float(value))
        backward_selected_counts.append(float(episode_result["backward_selected_count"]))
        backward_gated_counts.append(float(episode_result["backward_gated_count"]))
        backward_allowed_counts.append(float(episode_result["backward_allowed_count"]))
        backward_penalty_value = float(episode_result["average_backward_penalty_applied"])
        if np.isfinite(backward_penalty_value):
            backward_penalties.append(backward_penalty_value)

    total_actions = max(1, sum(action_counts.values()))
    return {
        "num_episodes": int(n),
        "success_rate": float(success / n),
        "dynamic_collision_rate": float(dynamic_collisions / n),
        "static_collision_rate": float(static_collisions / n),
        "timeout_rate": float(timeouts / n),
        "average_path_length": _mean_finite(path_lengths),
        "average_progress": _mean_finite(progresses),
        "average_min_dynamic_clearance": _mean_finite(min_dyn_clearances),
        "average_min_static_clearance": _mean_finite(min_static_clearances),
        "stop_fraction": _mean_finite(stop_fractions),
        "action_distribution": {action: float(count / total_actions) for action, count in sorted(action_counts.items())},
        "mean_score_components": {key: _mean_finite(values) for key, values in sorted(score_components.items())},
        "backward_action_fraction": float(action_counts.get(backward_action_name, 0) / total_actions),
        "backward_selected_count": float(np.sum(backward_selected_counts)),
        "backward_gated_count": float(np.sum(backward_gated_counts)),
        "backward_allowed_count": float(np.sum(backward_allowed_counts)),
        "average_backward_penalty_applied": _mean_finite(backward_penalties),
    }


def _save_episode_artifacts(
    episode: dict[str, object],
    output_dir: Path,
    vis_config: Any,
    *,
    nest_by_controller: bool,
    save_episodes: bool,
    save_previews: bool,
    save_gifs: bool,
    save_videos: bool,
) -> None:
    controller_dir = output_dir / str(episode["controller"]) if nest_by_controller else output_dir
    stem = f"episode_{int(episode['episode_id']):06d}"
    save_json(controller_dir / "decision_logs" / f"{stem}.json", _build_decision_log(episode))
    save_json(controller_dir / "episode_results" / f"{stem}.json", _episode_result(episode))
    if str(episode["controller"]) == "learned_structured" and not bool(episode.get("stochastic_planning_active", False)):
        rollout_error_stats = _build_learned_rollout_error_stats(episode)
        save_json(controller_dir / "rollout_error_stats" / f"{stem}_rollout_error_stats.json", json_safe(rollout_error_stats))
    if save_episodes:
        save_episode_pickle(episode, controller_dir / "episodes" / f"{stem}.pkl")
    if save_previews:
        save_episode_preview_png(episode, controller_dir / "preview" / f"{stem}.png", vis_config)
    if save_gifs:
        save_episode_gif(episode, controller_dir / "gif" / f"{stem}.gif", vis_config)
    if save_videos:
        save_episode_video(episode, controller_dir / "video" / f"{stem}.mp4", vis_config)


def _controller_cfg_from_args(args: argparse.Namespace) -> StructuredControllerConfig:
    return StructuredControllerConfig(
        horizon=int(args.horizon),
        w_progress=float(args.w_progress),
        w_heading=float(args.w_heading),
        w_path=float(args.w_path),
        w_velocity=float(args.w_velocity),
        w_static=float(args.w_static),
        w_dynamic=float(args.w_dynamic),
        static_collision_penalty=float(args.static_collision_penalty),
        dynamic_collision_penalty=float(args.dynamic_collision_penalty),
        dynamic_risk_tau=float(args.dynamic_risk_tau),
        min_clearance_clip=float(args.min_clearance_clip),
        backward_penalty_enabled=bool(args.backward_penalty_enabled),
        backward_action_name=str(args.backward_action_name),
        backward_penalty=float(args.backward_penalty),
        backward_gate_enabled=bool(args.backward_gate_enabled),
        dynamic_stop_clearance_threshold=float(args.dynamic_stop_clearance_threshold),
        backward_dynamic_margin=float(args.backward_dynamic_margin),
        action_indices=_effective_runtime_action_indices(args),
        execution_noise_enabled=bool(args.execution_noise_enabled),
        sigma_v=float(args.sigma_v),
        sigma_omega=float(args.sigma_omega),
        pose_observation_noise_enabled=bool(args.pose_observation_noise_enabled),
        sigma_obs_x=float(args.sigma_obs_x),
        sigma_obs_y=float(args.sigma_obs_y),
        sigma_obs_theta=float(args.sigma_obs_theta),
        num_stochastic_rollouts=int(args.num_stochastic_rollouts),
        risk_beta=float(args.risk_beta),
        collision_rate_threshold=float(args.collision_rate_threshold),
    )


def _dataset_config_from_args(args: argparse.Namespace) -> DatasetConfig:
    resolved_num_episodes = int(args.num_episodes) if int(args.num_episodes) > 0 else 1
    config = DatasetConfig(num_episodes=resolved_num_episodes, seed=int(args.seed), max_steps=int(args.max_steps))
    if args.num_dynamic_obstacles is not None:
        config.dynamic_obstacle_config.num_dynamic_obstacles = int(args.num_dynamic_obstacles)
    if args.map_width is not None:
        config.map_config.width = int(args.map_width)
    if args.map_height is not None:
        config.map_config.height = int(args.map_height)
    return config


def _effective_runtime_action_indices(args: argparse.Namespace) -> tuple[int, ...]:
    action_indices = [int(action_index) for action_index in ACTIONS]
    if bool(getattr(args, "disable_backward", False)):
        backward_name = str(getattr(args, "backward_action_name", "backward"))
        action_indices = [
            action_index
            for action_index in action_indices
            if str(ACTIONS[int(action_index)]["name"]) != backward_name
        ]
    if not action_indices:
        raise ValueError("Runtime action filtering removed every action; check --disable_backward and action settings")
    return tuple(action_indices)


def _validate_runtime_override_args(args: argparse.Namespace) -> None:
    has_map_override = args.map_width is not None or args.map_height is not None
    if getattr(args, "episode_source", None) and has_map_override:
        raise ValueError(
            "Map-size overrides are only supported for simulator-sampled policy eval; "
            "saved-episode replay already uses map geometry from the episode pickle"
        )


def _build_policy_eval_controller(
    *,
    args: argparse.Namespace,
    config: DatasetConfig,
    controller_cfg: StructuredControllerConfig,
) -> tuple[Any, str | None]:
    controller_name = str(args.controller)
    if controller_name == "baseline_structured":
        return BaselineStructuredController(config, controller_cfg), None
    if controller_name == "learned_structured":
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Structured model checkpoint not found: {checkpoint_path}")
        from controllers.learned_structured_controller import LearnedStructuredController

        return LearnedStructuredController(checkpoint_path, config, controller_cfg, device=str(args.device)), str(checkpoint_path)
    raise ValueError(f"Unsupported controller: {controller_name}")


def _iter_policy_eval_sources(
    *,
    simulator: EpisodeSimulator,
    args: argparse.Namespace,
) -> tuple[list[tuple[int, int, dict[str, Any]]], dict[str, object]]:
    episode_source_path = getattr(args, "episode_source", None)
    if not episode_source_path:
        seeds = seed_sequence(int(args.seed), int(args.num_episodes))
        items = [
            (episode_id, seed, _sample_scenario(simulator, episode_id, seed))
            for episode_id, seed in enumerate(seeds)
        ]
        details = {
            "mode": "simulator_sampled",
            "seed": int(args.seed),
            "num_episodes": int(args.num_episodes),
            "max_steps": int(args.max_steps),
            "num_dynamic_obstacles": int(simulator.config.dynamic_obstacle_config.num_dynamic_obstacles),
            "map_width": int(simulator.config.map_config.width),
            "map_height": int(simulator.config.map_config.height),
        }
        return items, details

    episode_paths = resolve_episode_source_paths(episode_source_path)
    requested = int(args.num_episodes)
    if requested > 0:
        episode_paths = episode_paths[:requested]
    items: list[tuple[int, int, dict[str, Any]]] = []
    for replay_index, episode_path in enumerate(episode_paths):
        saved_episode = load_episode_pickle(episode_path)
        episode_id = int(saved_episode.get("episode_id", replay_index))
        seed = int(saved_episode.get("seed", replay_index))
        items.append((episode_id, seed, scenario_from_saved_episode(saved_episode)))
    details = {
        "mode": "saved_episode_replay",
        "path": str(Path(episode_source_path)),
        "resolved_episode_count": int(len(items)),
        "used_episode_files": [str(path) for path in episode_paths],
        "num_requested": int(requested),
    }
    return items, details


def run_policy_eval(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    _validate_runtime_override_args(args)
    config = _dataset_config_from_args(args)
    simulator = EpisodeSimulator(config)
    controller_cfg = _controller_cfg_from_args(args)
    controller, checkpoint_path = _build_policy_eval_controller(args=args, config=config, controller_cfg=controller_cfg)
    source_items, episode_source_details = _iter_policy_eval_sources(simulator=simulator, args=args)
    episodes: list[dict[str, object]] = []

    for episode_id, seed, scenario in tqdm(source_items, desc="policy eval", dynamic_ncols=True):
        episode = _run_controller_episode(
            simulator=simulator,
            controller=controller,
            scenario=_clone_scenario(scenario),
            episode_id=episode_id,
            seed=seed,
        )
        _save_episode_artifacts(
            episode,
            output_dir,
            config.visualization_config,
            nest_by_controller=False,
            save_episodes=bool(args.save_episodes),
            save_previews=bool(args.save_previews),
            save_gifs=bool(args.save_gifs),
            save_videos=bool(args.save_videos),
        )
        episodes.append(episode)

    aggregate = _aggregate_controller_episodes(episodes)
    summary = {
        "schema_version": "v4_policy_eval_1",
        "controller": str(controller.name),
        "episode_source": episode_source_details,
        "config": {
            "controller": str(args.controller),
            "num_episodes": int(len(source_items)),
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "num_dynamic_obstacles": int(config.dynamic_obstacle_config.num_dynamic_obstacles),
            "map_width": int(config.map_config.width),
            "map_height": int(config.map_config.height),
            "horizon": int(args.horizon),
            "disable_backward": bool(args.disable_backward),
            "effective_runtime_actions": [
                {
                    "index": int(action_index),
                    "name": str(ACTIONS[int(action_index)]["name"]),
                    "v": float(ACTIONS[int(action_index)]["v"]),
                    "omega": float(ACTIONS[int(action_index)]["omega"]),
                }
                for action_index in controller_cfg.action_indices
            ],
            "weights": {
                "progress": float(args.w_progress),
                "heading": float(args.w_heading),
                "path": float(args.w_path),
                "velocity": float(args.w_velocity),
                "static": float(args.w_static),
                "dynamic": float(args.w_dynamic),
            },
            "static_collision_penalty": float(args.static_collision_penalty),
            "dynamic_collision_penalty": float(args.dynamic_collision_penalty),
            "dynamic_risk_tau": float(args.dynamic_risk_tau),
            "min_clearance_clip": float(args.min_clearance_clip),
            "backward_penalty_enabled": bool(args.backward_penalty_enabled),
            "backward_penalty": float(args.backward_penalty),
            "backward_gate_enabled": bool(args.backward_gate_enabled),
            "dynamic_stop_clearance_threshold": float(args.dynamic_stop_clearance_threshold),
            "backward_dynamic_margin": float(args.backward_dynamic_margin),
            "backward_action_name": str(args.backward_action_name),
            "execution_noise_enabled": bool(args.execution_noise_enabled),
            "sigma_v": float(args.sigma_v),
            "sigma_omega": float(args.sigma_omega),
            "pose_observation_noise_enabled": bool(args.pose_observation_noise_enabled),
            "sigma_obs_x": float(args.sigma_obs_x),
            "sigma_obs_y": float(args.sigma_obs_y),
            "sigma_obs_theta": float(args.sigma_obs_theta),
            "num_stochastic_rollouts": int(args.num_stochastic_rollouts),
            "risk_beta": float(args.risk_beta),
            "collision_rate_threshold": float(args.collision_rate_threshold),
            "stochastic_planning_active": bool(args.execution_noise_enabled),
            "rollout_error_stats_suppressed": bool(args.execution_noise_enabled and controller.name == "learned_structured"),
        },
        "aggregate": aggregate,
        "episodes": [_episode_result(episode) for episode in episodes],
    }
    if checkpoint_path is not None:
        summary["checkpoint"] = str(checkpoint_path)
    save_json(output_dir / "summary.json", json_safe(summary))
    if episodes and str(controller.name) == "learned_structured" and not bool(args.execution_noise_enabled):
        save_json(
            output_dir / "rollout_error_stats_summary.json",
            json_safe(_build_rollout_error_stats_from_episodes(episodes)),
        )
    print(json_safe(summary["aggregate"]))
    return summary
