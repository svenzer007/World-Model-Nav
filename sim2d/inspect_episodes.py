"""Inspect saved V4 episodes for structured dynamic-state consistency and quality reporting."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .config import VisualizationConfig
from .dataset import load_episode_pickle
from .utils import save_json
from .visualize import save_episode_preview_png


KNOWN_ACTIONS = (
    "stop",
    "forward",
    "forward_left",
    "forward_right",
    "slow_left",
    "slow_right",
)

TURNING_ACTIONS = {"forward_left", "forward_right", "slow_left", "slow_right"}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for episode inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=None, help="Dataset root or episodes directory")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset root or episodes directory")
    parser.add_argument("--num_show", type=int, default=5)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--report_path", type=str, default=None)
    parser.add_argument("--report_only", action="store_true")
    return parser


def main() -> None:
    """CLI entrypoint for inspection."""
    args = build_arg_parser().parse_args()
    path_arg = args.data_dir or args.path
    if path_arg is None:
        raise ValueError("Provide a dataset root or episodes directory via positional path or --data_dir")
    input_path = Path(path_arg)
    episodes_dir = input_path if input_path.name == "episodes" else input_path / "episodes"
    data_dir = episodes_dir.parent if episodes_dir.name == "episodes" else input_path
    episode_paths = sorted(episodes_dir.glob("episode_*.pkl"))
    if not episode_paths:
        raise FileNotFoundError(f"No episodes found under {episodes_dir}")

    episodes = [load_episode_pickle(path) for path in episode_paths]
    _validate_dataset(episodes)
    _print_summary(episodes)
    if args.report or args.report_path:
        report = compute_dataset_quality_report(episodes)
        _print_dataset_quality_report(report)
        report_path = Path(args.report_path) if args.report_path else data_dir / "dataset_quality_report.json"
        save_json(report_path, report)
        print(f"Saved dataset quality report to: {report_path}")

    if not args.report_only:
        inspect_dir = data_dir / "inspection_preview"
        vis_config = VisualizationConfig()
        for episode in episodes[: args.num_show]:
            try:
                save_episode_preview_png(
                    episode,
                    inspect_dir / f"episode_{int(episode['episode_id']):06d}.png",
                    vis_config,
                )
            except (KeyError, TypeError, ValueError) as exc:
                print(f"Skipped preview for episode {episode.get('episode_id', 'unknown')}: {exc}")
        print(f"Saved inspection previews to: {inspect_dir}")


def _validate_dataset(episodes: list[dict[str, object]]) -> None:
    if not episodes:
        raise ValueError("No episodes loaded")

    reference_lidar = episodes[0]["lidar_config"]
    reference_actions = episodes[0]["action_space"]
    expected_beams = int(reference_lidar["num_beams"])
    for episode in episodes:
        if episode["lidar_config"] != reference_lidar:
            raise ValueError(f"Inconsistent LiDAR config in episode {episode['episode_id']}")
        if episode["action_space"] != reference_actions:
            raise ValueError(f"Inconsistent action space in episode {episode['episode_id']}")
        for step_index, step in enumerate(episode["steps"]):
            lidar_t = np.asarray(step["lidar_t"])
            lidar_t1 = np.asarray(step["lidar_t1"])
            lidar_static_t = np.asarray(step.get("lidar_static_t"))
            lidar_static_t1 = np.asarray(step.get("lidar_static_t1"))
            residual_t = np.asarray(step.get("lidar_dynamic_residual_t"))
            residual_t1 = np.asarray(step.get("lidar_dynamic_residual_t1"))
            hit_mask_t = np.asarray(step.get("dynamic_hit_mask_t"))
            hit_mask_t1 = np.asarray(step.get("dynamic_hit_mask_t1"))
            goal_t = np.asarray(step["goal_t"])
            goal_t1 = np.asarray(step["goal_t1"])
            if lidar_t.shape != (expected_beams,) or lidar_t1.shape != (expected_beams,):
                raise ValueError(f"Unexpected LiDAR shape in episode {episode['episode_id']} step {step_index}")
            for name, array in (
                ("lidar_static_t", lidar_static_t),
                ("lidar_static_t1", lidar_static_t1),
                ("lidar_dynamic_residual_t", residual_t),
                ("lidar_dynamic_residual_t1", residual_t1),
                ("dynamic_hit_mask_t", hit_mask_t),
                ("dynamic_hit_mask_t1", hit_mask_t1),
            ):
                if array.shape != (expected_beams,):
                    raise ValueError(
                        f"Unexpected {name} shape in episode {episode['episode_id']} step {step_index}"
                    )
            if goal_t.shape != (5,) or goal_t1.shape != (5,):
                raise ValueError(f"Unexpected goal shape in episode {episode['episode_id']} step {step_index}")
            _validate_v4_step_fields(episode, step_index, step)
            _validate_v4_structured_dynamic_fields(episode, step_index, step)
            expected_residual = np.maximum(0.0, lidar_static_t.astype(float) - lidar_t.astype(float))
            if not np.allclose(residual_t.astype(float), expected_residual, atol=1e-5):
                raise ValueError(f"Dynamic residual mismatch in episode {episode['episode_id']} step {step_index}")
            label_config = episode.get("dynamic_label_config", {})
            epsilon = float(label_config.get("dynamic_residual_epsilon", 0.03))
            visible_min_beams = int(label_config.get("dynamic_visible_min_beams", 3))
            risk_threshold = float(label_config.get("dynamic_risk_positive_threshold", 0.35))
            if not np.array_equal(hit_mask_t.astype(bool), residual_t.astype(float) > epsilon):
                raise ValueError(f"Dynamic hit mask mismatch in episode {episode['episode_id']} step {step_index}")
            expected_lidar_visible = int(np.sum(hit_mask_t.astype(bool))) >= visible_min_beams
            if bool(step["lidar_dynamic_visible_t"]) != expected_lidar_visible:
                raise ValueError(f"LiDAR dynamic visibility mismatch in episode {episode['episode_id']} step {step_index}")
            visibility_radius = float(label_config.get("dynamic_visibility_radius", 1.2))
            expected_visible = float(step["min_dynamic_clearance_t"]) < visibility_radius
            if bool(step["dynamic_visible_t"]) != expected_visible:
                raise ValueError(f"Structured dynamic visibility mismatch in episode {episode['episode_id']} step {step_index}")
            expected_event = bool(
                step["dynamic_visible_t"]
                or float(step["dynamic_risk_t1"]) >= risk_threshold
                or step["dynamic_collision_within_3"]
                or step["dynamic_collision_within_5"]
            )
            if bool(step["dynamic_event_step_t"]) != expected_event:
                raise ValueError(f"Dynamic event flag mismatch in episode {episode['episode_id']} step {step_index}")
            if np.min(lidar_t) < 0.0 or np.max(lidar_t) > float(episode["lidar_config"]["max_range"]) + 1e-6:
                raise ValueError(f"LiDAR range out of bounds in episode {episode['episode_id']} step {step_index}")
            for obstacle_state in list(step.get("dynamic_obstacles_t", [])) + list(step.get("dynamic_obstacles_t1", [])):
                for key in ("obstacle_id", "position", "velocity", "radius", "speed", "route_index", "current_target"):
                    if key not in obstacle_state:
                        raise ValueError(
                            f"Missing dynamic obstacle state {key!r} in episode {episode['episode_id']} step {step_index}"
                        )
            if step_index < len(episode["steps"]) - 1:
                next_step = episode["steps"][step_index + 1]
                if not np.allclose(np.asarray(step["lidar_t1"]), np.asarray(next_step["lidar_t"]), atol=1e-5):
                    raise ValueError(f"LiDAR transition mismatch in episode {episode['episode_id']} step {step_index}")
                if not np.allclose(np.asarray(step["goal_t1"]), np.asarray(next_step["goal_t"]), atol=1e-5):
                    raise ValueError(f"Goal transition mismatch in episode {episode['episode_id']} step {step_index}")
                if not np.allclose(
                    np.asarray(step["dynamic_obstacles_world_t1"], dtype=float),
                    np.asarray(next_step["dynamic_obstacles_world_t"], dtype=float),
                    atol=1e-5,
                ):
                    raise ValueError(
                        f"Dynamic obstacle world transition mismatch in episode {episode['episode_id']} step {step_index}"
                    )
        _validate_dynamic_horizon_labels(episode)


def _validate_v4_step_fields(episode: dict[str, object], step_index: int, step: dict[str, object]) -> None:
    """Validate the required V4 per-step labels."""
    required = (
        "robot_pose_t",
        "robot_pose_t1",
        "v_t",
        "omega_t",
        "action_name",
        "subgoal_world_t",
        "subgoal_world_t1",
        "nearest_path_index_t",
        "nearest_path_index_t1",
        "selected_subgoal_index_t",
        "selected_subgoal_index_t1",
        "distance_to_subgoal_t",
        "distance_to_subgoal_t1",
        "progress_to_subgoal",
        "distance_to_path_t",
        "distance_to_path_t1",
        "heading_error_cos_t",
        "heading_error_sin_t",
        "heading_error_cos_t1",
        "heading_error_sin_t1",
        "min_static_clearance_t",
        "min_dynamic_clearance_t",
        "min_total_clearance_t",
        "min_static_clearance_t1",
        "min_dynamic_clearance_t1",
        "min_total_clearance_t1",
        "risk_t1",
        "collision_within_3",
        "collision_within_5",
        "lidar_static_t",
        "lidar_dynamic_residual_t",
        "dynamic_hit_mask_t",
        "lidar_dynamic_visible_t",
        "dynamic_hit_fraction_t",
        "dynamic_residual_energy_t",
        "lidar_static_t1",
        "lidar_dynamic_residual_t1",
        "dynamic_hit_mask_t1",
        "lidar_dynamic_visible_t1",
        "dynamic_obstacle_ids",
        "dynamic_obstacles_world_t",
        "dynamic_obstacles_world_t1",
        "dynamic_obstacle_delta_world_t",
        "dynamic_obstacles_robot_t",
        "dynamic_obstacles_robot_t1",
        "dynamic_obstacle_radii",
        "dynamic_obstacle_vel_world_t",
        "dynamic_visible_t",
        "dynamic_collision_t1",
        "dynamic_collision_within_3",
        "dynamic_collision_within_5",
        "dynamic_risk_t1",
        "dynamic_event_step_t",
    )
    for key in required:
        if key not in step:
            raise ValueError(f"Missing V4 field {key!r} in episode {episode['episode_id']} step {step_index}")
    if np.asarray(step["robot_pose_t"]).shape != (3,) or np.asarray(step["robot_pose_t1"]).shape != (3,):
        raise ValueError(f"Unexpected robot pose shape in episode {episode['episode_id']} step {step_index}")
    if np.asarray(step["subgoal_world_t"]).shape != (2,) or np.asarray(step["subgoal_world_t1"]).shape != (2,):
        raise ValueError(f"Unexpected subgoal shape in episode {episode['episode_id']} step {step_index}")


def _validate_v4_structured_dynamic_fields(
    episode: dict[str, object],
    step_index: int,
    step: dict[str, object],
) -> None:
    """Validate fixed-cardinality obstacle arrays and transform-derived fields."""
    expected_ids = list(episode.get("dynamic_obstacle_ids", []))
    actual_ids = list(step["dynamic_obstacle_ids"])
    if actual_ids != expected_ids:
        raise ValueError(f"Dynamic obstacle IDs changed in episode {episode['episode_id']} step {step_index}")

    n_dyn = int(episode.get("num_dynamic_obstacles", len(expected_ids)))
    if len(actual_ids) != n_dyn:
        raise ValueError(f"Unexpected dynamic obstacle count in episode {episode['episode_id']} step {step_index}")

    world_t = np.asarray(step["dynamic_obstacles_world_t"], dtype=float)
    world_t1 = np.asarray(step["dynamic_obstacles_world_t1"], dtype=float)
    delta = np.asarray(step["dynamic_obstacle_delta_world_t"], dtype=float)
    robot_t = np.asarray(step["dynamic_obstacles_robot_t"], dtype=float)
    robot_t1 = np.asarray(step["dynamic_obstacles_robot_t1"], dtype=float)
    radii = np.asarray(step["dynamic_obstacle_radii"], dtype=float)
    vel_world = np.asarray(step["dynamic_obstacle_vel_world_t"], dtype=float)
    vel_world_causal = np.asarray(
        step.get("dynamic_obstacle_vel_world_causal_t", np.zeros_like(vel_world)),
        dtype=float,
    )
    for name, array in (
        ("dynamic_obstacles_world_t", world_t),
        ("dynamic_obstacles_world_t1", world_t1),
        ("dynamic_obstacle_delta_world_t", delta),
        ("dynamic_obstacles_robot_t", robot_t),
        ("dynamic_obstacles_robot_t1", robot_t1),
        ("dynamic_obstacle_vel_world_t", vel_world),
    ):
        if array.shape != (n_dyn, 2):
            raise ValueError(f"Unexpected {name} shape in episode {episode['episode_id']} step {step_index}")
    if radii.shape != (n_dyn,):
        raise ValueError(f"Unexpected dynamic_obstacle_radii shape in episode {episode['episode_id']} step {step_index}")

    dt = float(step["dt"])
    pose_t = np.asarray(step["robot_pose_t"], dtype=float)
    pose_t1 = np.asarray(step["robot_pose_t1"], dtype=float)
    if not np.allclose(delta, world_t1 - world_t, atol=1e-5):
        raise ValueError(f"Dynamic obstacle delta mismatch in episode {episode['episode_id']} step {step_index}")
    if not np.allclose(vel_world, delta / dt, atol=1e-5):
        raise ValueError(f"Dynamic obstacle world velocity mismatch in episode {episode['episode_id']} step {step_index}")
    if "dynamic_obstacle_vel_world_causal_t" in step:
        if step_index == 0:
            expected_vel_world_causal = np.asarray(
                [np.asarray(snapshot["velocity"], dtype=float) for snapshot in step.get("dynamic_obstacles_t", [])],
                dtype=float,
            )
        else:
            previous_world_t = np.asarray(episode["steps"][step_index - 1]["dynamic_obstacles_world_t"], dtype=float)
            expected_vel_world_causal = (world_t - previous_world_t) / dt
        if not np.allclose(vel_world_causal, expected_vel_world_causal, atol=1e-5):
            raise ValueError(f"Causal world velocity mismatch in episode {episode['episode_id']} step {step_index}")
    expected_robot_t = _world_points_to_robot_frame(world_t, pose_t)
    expected_robot_t1 = _world_points_to_robot_frame(world_t1, pose_t1)
    if not np.allclose(robot_t, expected_robot_t, atol=1e-5):
        raise ValueError(f"Robot-frame obstacle position mismatch in episode {episode['episode_id']} step {step_index}")
    if not np.allclose(robot_t1, expected_robot_t1, atol=1e-5):
        raise ValueError(f"Robot-frame t1 obstacle position mismatch in episode {episode['episode_id']} step {step_index}")
    expected_vel_robot = _world_vectors_to_robot_frame(vel_world, float(pose_t[2]))
    robot_vel_world = (pose_t1[:2] - pose_t[:2]) / dt
    expected_rel_vel_robot = _world_vectors_to_robot_frame(vel_world - robot_vel_world[None, :], float(pose_t[2]))
    expected_rel_vel_robot_fd = (robot_t1 - robot_t) / dt
    if "dynamic_obstacle_delta_robot_t" in step:
        delta_robot = np.asarray(step["dynamic_obstacle_delta_robot_t"], dtype=float)
        if delta_robot.shape != (n_dyn, 2):
            raise ValueError(f"Unexpected dynamic_obstacle_delta_robot_t shape in episode {episode['episode_id']} step {step_index}")
        if not np.allclose(delta_robot, robot_t1 - robot_t, atol=1e-5):
            raise ValueError(f"Robot-frame obstacle delta mismatch in episode {episode['episode_id']} step {step_index}")
    if "dynamic_obstacle_vel_robot_t" in step:
        vel_robot = np.asarray(step["dynamic_obstacle_vel_robot_t"], dtype=float)
        if vel_robot.shape != (n_dyn, 2):
            raise ValueError(f"Unexpected dynamic_obstacle_vel_robot_t shape in episode {episode['episode_id']} step {step_index}")
        if not np.allclose(vel_robot, expected_vel_robot, atol=1e-5):
            raise ValueError(f"Robot-axis obstacle velocity mismatch in episode {episode['episode_id']} step {step_index}")
    if "dynamic_obstacle_rel_vel_robot_t" in step:
        rel_vel_robot = np.asarray(step["dynamic_obstacle_rel_vel_robot_t"], dtype=float)
        if rel_vel_robot.shape != (n_dyn, 2):
            raise ValueError(f"Unexpected dynamic_obstacle_rel_vel_robot_t shape in episode {episode['episode_id']} step {step_index}")
        if not np.allclose(rel_vel_robot, expected_rel_vel_robot, atol=1e-5):
            raise ValueError(f"Relative obstacle velocity mismatch in episode {episode['episode_id']} step {step_index}")
    if "dynamic_obstacle_rel_vel_robot_fd_t" in step:
        rel_vel_robot_fd = np.asarray(step["dynamic_obstacle_rel_vel_robot_fd_t"], dtype=float)
        if rel_vel_robot_fd.shape != (n_dyn, 2):
            raise ValueError(
                f"Unexpected dynamic_obstacle_rel_vel_robot_fd_t shape in episode {episode['episode_id']} step {step_index}"
            )
        if not np.allclose(rel_vel_robot_fd, expected_rel_vel_robot_fd, atol=1e-5):
            raise ValueError(f"Finite-difference relative obstacle velocity mismatch in episode {episode['episode_id']} step {step_index}")
    if "dynamic_obstacle_rel_vel_robot_fd_t1" in step:
        rel_vel_robot_fd_t1 = np.asarray(step["dynamic_obstacle_rel_vel_robot_fd_t1"], dtype=float)
        if rel_vel_robot_fd_t1.shape != (n_dyn, 2):
            raise ValueError(
                f"Unexpected dynamic_obstacle_rel_vel_robot_fd_t1 shape in episode {episode['episode_id']} step {step_index}"
            )
        if not np.allclose(rel_vel_robot_fd_t1, expected_rel_vel_robot_fd, atol=1e-5):
            raise ValueError(
                f"Per-step next finite-difference relative velocity mismatch in episode {episode['episode_id']} step {step_index}"
            )


def _validate_dynamic_horizon_labels(episode: dict[str, object]) -> None:
    """Validate that horizon labels cover dynamic collision events from t+1 through t+H."""
    steps = list(episode.get("steps", []))
    for index, step in enumerate(steps):
        expected_collision = float(step["min_dynamic_clearance_t1"]) <= 0.0
        if bool(step["dynamic_collision_t1"]) != expected_collision:
            raise ValueError(f"Dynamic collision label mismatch in episode {episode['episode_id']} step {index}")
        for horizon in (3, 5):
            end = min(len(steps), index + horizon)
            expected_horizon = any(bool(future.get("dynamic_collision_t1", False)) for future in steps[index:end])
            key = f"dynamic_collision_within_{horizon}"
            if bool(step[key]) != expected_horizon:
                raise ValueError(f"{key} mismatch in episode {episode['episode_id']} step {index}")


def _print_summary(episodes: list[dict[str, object]]) -> None:
    terminations = Counter(str(episode["episode_stats"]["done_reason"]) for episode in episodes)
    action_usage = Counter()
    lengths = []
    stop_count = 0
    total_steps = 0
    all_non_stop_infeasible_steps = 0
    feasible_counts: list[int] = []
    for episode in episodes:
        lengths.append(int(episode["episode_stats"]["num_steps"]))
        for step in episode["steps"]:
            total_steps += 1
            action_usage[str(step["action_name"])] += 1
            if str(step["action_name"]) == "stop":
                stop_count += 1
            candidates = step.get("expert_debug", {}).get("candidates", [])
            if candidates:
                feasible_actions = [candidate for candidate in candidates if bool(candidate.get("feasible", False))]
                feasible_counts.append(len(feasible_actions))
                non_stop = [candidate for candidate in candidates if int(candidate.get("action_index", -1)) != 0]
                if non_stop and not any(bool(candidate.get("feasible", False)) for candidate in non_stop):
                    all_non_stop_infeasible_steps += 1

    print(f"Loaded {len(episodes)} episodes")
    print(f"Average episode length: {float(np.mean(lengths)):.2f} steps")
    if total_steps:
        print(f"Chosen stop fraction: {stop_count / total_steps:.3f}")
        print(f"All non-stop actions infeasible steps: {all_non_stop_infeasible_steps}")
        all_steps = [step for episode in episodes for step in episode["steps"]]
        print(
            "Dynamic visible/event fractions: "
            f"{float(np.mean([bool(step.get('dynamic_visible_t', False)) for step in all_steps])):.3f} / "
            f"{float(np.mean([bool(step.get('dynamic_event_step_t', False)) for step in all_steps])):.3f}"
        )
    if feasible_counts:
        print(f"Average feasible actions per step: {float(np.mean(feasible_counts)):.2f}")
    print("Termination counts:")
    for key in sorted(terminations):
        print(f"  {key}: {terminations[key]}")
    print("Expert action usage:")
    for action_name, count in sorted(action_usage.items()):
        print(f"  {action_name}: {count}")


def compute_dataset_quality_report(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Compute a dataset-level quality report from saved episode pickles."""
    basic, termination = _compute_basic_and_termination_stats(episodes)
    actions = _compute_action_stats(episodes)
    stop_persistence = _compute_stop_persistence(episodes)
    maneuver_diversity = _compute_maneuver_diversity(episodes)
    clearance = _compute_clearance_stats(episodes)
    episode_lengths = _compute_episode_length_stats(episodes)
    trajectory_quality = _compute_trajectory_stats(episodes)
    path_complexity = _compute_path_complexity_stats(episodes)
    feasibility = _compute_feasibility_stats(episodes)
    goal_features = _compute_goal_feature_stats(episodes)
    map_distribution = _compute_map_distribution(episodes)
    v4_dynamic_labels = _compute_v4_dynamic_label_stats(episodes)
    v4_structured_state = _compute_v4_structured_state_stats(episodes)
    v2_base_labels = _compute_v2_label_stats(episodes)

    report = {
        "basic": basic,
        "termination": termination,
        "actions": actions,
        "stop_persistence": stop_persistence,
        "maneuver_diversity": maneuver_diversity,
        "clearance": clearance,
        "episode_lengths": episode_lengths,
        "trajectory_quality": trajectory_quality,
        "path_complexity": path_complexity,
        "feasibility": feasibility,
        "goal_features": goal_features,
        "map_distribution": map_distribution,
        "v4_dynamic_labels": v4_dynamic_labels,
        "v4_structured_state": v4_structured_state,
        "v2_base_labels": v2_base_labels,
    }
    report["warnings"] = _compute_quality_warnings(report)
    return report


def _compute_basic_and_termination_stats(episodes: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    """Compute basic dataset size stats and termination breakdown."""
    num_episodes = len(episodes)
    total_steps = sum(len(episode.get("steps", [])) for episode in episodes)
    avg_steps_per_episode = float(total_steps / num_episodes) if num_episodes else 0.0

    termination_counts = Counter(_get_episode_done_reason(episode) for episode in episodes)
    termination = {}
    for reason in ("goal_reached", "static_collision", "dynamic_collision", "timeout"):
        count = int(termination_counts.get(reason, 0))
        termination[reason] = {
            "count": count,
            "percentage": _fraction(count, num_episodes),
        }
    if termination_counts.get("unknown", 0):
        termination["unknown"] = {
            "count": int(termination_counts["unknown"]),
            "percentage": _fraction(int(termination_counts["unknown"]), num_episodes),
        }

    basic = {
        "num_episodes": int(num_episodes),
        "total_steps": int(total_steps),
        "avg_steps_per_episode": avg_steps_per_episode,
    }
    return basic, termination


def _compute_action_stats(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Compute action counts and action percentages across all steps."""
    action_counts = Counter()
    total_steps = 0
    for episode in episodes:
        for step in episode.get("steps", []):
            total_steps += 1
            action_name = step.get("action_name")
            if isinstance(action_name, str) and action_name in KNOWN_ACTIONS:
                action_counts[action_name] += 1
            else:
                action_counts["unknown"] += 1

    actions = {
        action_name: {
            "count": int(action_counts.get(action_name, 0)),
            "percentage": _fraction(int(action_counts.get(action_name, 0)), total_steps),
        }
        for action_name in KNOWN_ACTIONS
    }
    actions["stop_fraction"] = _fraction(int(action_counts.get("stop", 0)), total_steps)
    actions["unknown_action_count"] = int(action_counts.get("unknown", 0))
    return actions


def _compute_stop_persistence(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Compute longest per-episode stop runs."""
    longest_runs = [_longest_stop_run(episode.get("steps", [])) for episode in episodes]
    num_episodes = len(episodes)
    return {
        "avg_longest_stop_run": float(np.mean(longest_runs)) if longest_runs else 0.0,
        "max_longest_stop_run": int(max(longest_runs)) if longest_runs else 0,
        "episodes_with_stop_run_gt_10": int(sum(run > 10 for run in longest_runs)),
        "episodes_with_stop_run_gt_20": int(sum(run > 20 for run in longest_runs)),
        "num_episodes": int(num_episodes),
    }


def _compute_maneuver_diversity(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Compute turning fraction and per-episode action diversity."""
    total_steps = 0
    turning_steps = 0
    unique_actions_per_episode: list[int] = []
    for episode in episodes:
        unique_actions = set()
        for step in episode.get("steps", []):
            total_steps += 1
            action_name = step.get("action_name")
            if isinstance(action_name, str):
                unique_actions.add(action_name)
                if action_name in TURNING_ACTIONS:
                    turning_steps += 1
        unique_actions_per_episode.append(len(unique_actions))

    return {
        "turning_action_fraction": _fraction(turning_steps, total_steps),
        "avg_unique_actions_per_episode": float(np.mean(unique_actions_per_episode)) if unique_actions_per_episode else 0.0,
    }


def _compute_clearance_stats(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Compute chosen-action clearance statistics."""
    clearances: list[float] = []
    for episode in episodes:
        for step in episode.get("steps", []):
            value = step.get("chosen_action_min_clearance")
            if _is_number(value):
                clearances.append(float(value))

    if not clearances:
        return {
            "mean_clearance": 0.0,
            "histogram_counts": {"lt_0_2": 0, "0_2_to_0_3": 0, "0_3_to_0_5": 0, "gt_0_5": 0},
            "fraction_clearance_lt_0_3": 0.0,
            "num_clearance_samples": 0,
        }

    clearance_array = np.asarray(clearances, dtype=float)
    return {
        "mean_clearance": float(np.mean(clearance_array)),
        "histogram_counts": {
            "lt_0_2": int(np.sum(clearance_array < 0.2)),
            "0_2_to_0_3": int(np.sum((clearance_array >= 0.2) & (clearance_array < 0.3))),
            "0_3_to_0_5": int(np.sum((clearance_array >= 0.3) & (clearance_array <= 0.5))),
            "gt_0_5": int(np.sum(clearance_array > 0.5)),
        },
        "fraction_clearance_lt_0_3": _fraction(int(np.sum(clearance_array < 0.3)), len(clearance_array)),
        "num_clearance_samples": int(len(clearance_array)),
    }


def _compute_episode_length_stats(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Compute episode length summary statistics."""
    lengths = np.asarray([len(episode.get("steps", [])) for episode in episodes], dtype=float)
    if lengths.size == 0:
        return {
            "min_episode_length": 0,
            "max_episode_length": 0,
            "mean_episode_length": 0.0,
            "median_episode_length": 0.0,
        }
    return {
        "min_episode_length": int(np.min(lengths)),
        "max_episode_length": int(np.max(lengths)),
        "mean_episode_length": float(np.mean(lengths)),
        "median_episode_length": float(np.median(lengths)),
    }


def _compute_trajectory_stats(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Compute path length and displacement from step-level robot poses."""
    path_lengths: list[float] = []
    displacements: list[float] = []
    short_path_episodes = 0

    for episode in episodes:
        poses = []
        for step in episode.get("steps", []):
            pose = step.get("robot_pose")
            pose_array = np.asarray(pose, dtype=float) if pose is not None else np.asarray([])
            if pose_array.shape[0] >= 2:
                poses.append(pose_array[:2])

        if len(poses) < 2:
            path_length = 0.0
            displacement = 0.0
        else:
            positions = np.asarray(poses, dtype=float)
            path_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
            displacement = float(np.linalg.norm(positions[-1] - positions[0]))

        path_lengths.append(path_length)
        displacements.append(displacement)
        if path_length < 1.0:
            short_path_episodes += 1

    return {
        "avg_path_length": float(np.mean(path_lengths)) if path_lengths else 0.0,
        "avg_displacement": float(np.mean(displacements)) if displacements else 0.0,
        "episodes_with_path_length_lt_1_0": int(short_path_episodes),
    }


def _compute_path_complexity_stats(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Compute accepted A* path complexity stats."""
    directness_ratios: list[float] = []
    turn_angle_sums: list[float] = []
    for episode in episodes:
        stats = episode.get("episode_stats", {})
        ratio = stats.get("path_directness_ratio")
        turn_sum = stats.get("path_turn_angle_sum")
        if _is_number(ratio):
            directness_ratios.append(float(ratio))
        if _is_number(turn_sum):
            turn_angle_sums.append(float(turn_sum))
    return {
        "directness_ratio": _numeric_summary(np.asarray(directness_ratios, dtype=float)),
        "turn_angle_sum": _numeric_summary(np.asarray(turn_angle_sums, dtype=float)),
    }


def _compute_feasibility_stats(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Compute feasible-action richness from expert candidate logs."""
    feasible_counts: list[int] = []
    feasible_non_stop_counts: list[int] = []
    only_stop_feasible = 0
    zero_non_stop_feasible = 0

    for episode in episodes:
        for step in episode.get("steps", []):
            candidates = step.get("expert_debug", {}).get("candidates", [])
            if not isinstance(candidates, list) or not candidates:
                continue
            feasible = [candidate for candidate in candidates if bool(candidate.get("feasible", False))]
            feasible_non_stop = [candidate for candidate in feasible if candidate.get("action_name") != "stop"]
            feasible_counts.append(len(feasible))
            feasible_non_stop_counts.append(len(feasible_non_stop))
            if len(feasible) == 1 and feasible[0].get("action_name") == "stop":
                only_stop_feasible += 1
            if len(feasible_non_stop) == 0:
                zero_non_stop_feasible += 1

    num_samples = len(feasible_counts)
    return {
        "avg_feasible_actions_per_step": float(np.mean(feasible_counts)) if feasible_counts else 0.0,
        "avg_feasible_non_stop_actions_per_step": float(np.mean(feasible_non_stop_counts)) if feasible_non_stop_counts else 0.0,
        "fraction_only_stop_feasible": _fraction(only_stop_feasible, num_samples),
        "fraction_zero_non_stop_actions_feasible": _fraction(zero_non_stop_feasible, num_samples),
        "num_feasibility_samples": int(num_samples),
    }


def _compute_goal_feature_stats(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Compute distance and angle diversity from goal features."""
    distances: list[float] = []
    alphas: list[float] = []
    for episode in episodes:
        for step in episode.get("steps", []):
            goal_t = np.asarray(step.get("goal_t"), dtype=float)
            if goal_t.shape != (5,):
                continue
            distances.append(float(goal_t[2]))
            alphas.append(float(math.atan2(goal_t[4], goal_t[3])))

    if not distances:
        return {
            "distance": {"mean": 0.0, "min": 0.0, "max": 0.0},
            "angle_bins": {"front": 0, "moderate": 0, "large": 0},
            "num_goal_samples": 0,
        }

    distances_arr = np.asarray(distances, dtype=float)
    alpha_abs_deg = np.degrees(np.abs(np.asarray(alphas, dtype=float)))
    return {
        "distance": {
            "mean": float(np.mean(distances_arr)),
            "min": float(np.min(distances_arr)),
            "max": float(np.max(distances_arr)),
        },
        "angle_bins": {
            "front": int(np.sum(alpha_abs_deg < 30.0)),
            "moderate": int(np.sum((alpha_abs_deg >= 30.0) & (alpha_abs_deg <= 90.0))),
            "large": int(np.sum(alpha_abs_deg > 90.0)),
        },
        "num_goal_samples": int(len(distances_arr)),
    }


def _compute_map_distribution(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Compute map generator type counts."""
    counts = Counter(str(episode.get("map_type", "unknown")) for episode in episodes)
    total = len(episodes)
    return {
        map_type: {"count": int(count), "fraction": _fraction(int(count), total)}
        for map_type, count in sorted(counts.items())
    }


def _compute_v4_dynamic_label_stats(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Compute V4 dynamic state and hazard-label distributions."""
    steps = [step for episode in episodes for step in episode.get("steps", [])]
    total_steps = len(steps)
    if total_steps == 0:
        return {
            "num_steps": 0,
            "dynamic_visible_t": {"count": 0, "fraction": 0.0},
            "lidar_dynamic_visible_t": {"count": 0, "fraction": 0.0},
            "dynamic_event_step_t": {"count": 0, "fraction": 0.0},
            "dynamic_residual_nonzero_beam_fraction": 0.0,
            "dynamic_residual_energy_t": {},
            "dynamic_collision_t1": {"count": 0, "fraction": 0.0},
            "dynamic_collision_within_3": {"count": 0, "fraction": 0.0},
            "dynamic_collision_within_5": {"count": 0, "fraction": 0.0},
            "min_dynamic_clearance_t1": {},
            "dynamic_risk_t1": {},
        }

    def values(key: str) -> np.ndarray:
        collected: list[float] = []
        for step in steps:
            value = step.get(key)
            if _is_number(value):
                collected.append(float(value))
        return np.asarray(collected, dtype=float)

    total_beams = 0
    nonzero_beams = 0
    for step in steps:
        mask = np.asarray(step.get("dynamic_hit_mask_t"), dtype=bool)
        total_beams += int(mask.size)
        nonzero_beams += int(np.sum(mask))

    visible_count = int(sum(bool(step.get("dynamic_visible_t", False)) for step in steps))
    lidar_visible_count = int(sum(bool(step.get("lidar_dynamic_visible_t", False)) for step in steps))
    event_count = int(sum(bool(step.get("dynamic_event_step_t", False)) for step in steps))
    dynamic_collision_count = int(sum(bool(step.get("dynamic_collision_t1", False)) for step in steps))
    dynamic_h3_count = int(sum(bool(step.get("dynamic_collision_within_3", False)) for step in steps))
    dynamic_h5_count = int(sum(bool(step.get("dynamic_collision_within_5", False)) for step in steps))

    return {
        "num_steps": int(total_steps),
        "dynamic_visible_t": {"count": visible_count, "fraction": _fraction(visible_count, total_steps)},
        "lidar_dynamic_visible_t": {
            "count": lidar_visible_count,
            "fraction": _fraction(lidar_visible_count, total_steps),
        },
        "dynamic_event_step_t": {"count": event_count, "fraction": _fraction(event_count, total_steps)},
        "dynamic_residual_nonzero_beam_fraction": _fraction(nonzero_beams, total_beams),
        "dynamic_hit_fraction_t": _numeric_summary(values("dynamic_hit_fraction_t")),
        "dynamic_residual_energy_t": _numeric_summary(values("dynamic_residual_energy_t")),
        "dynamic_collision_t1": {
            "count": dynamic_collision_count,
            "fraction": _fraction(dynamic_collision_count, total_steps),
        },
        "dynamic_collision_within_3": {"count": dynamic_h3_count, "fraction": _fraction(dynamic_h3_count, total_steps)},
        "dynamic_collision_within_5": {"count": dynamic_h5_count, "fraction": _fraction(dynamic_h5_count, total_steps)},
        "min_dynamic_clearance_t1": _numeric_summary(values("min_dynamic_clearance_t1")),
        "dynamic_risk_t1": _numeric_summary(values("dynamic_risk_t1")),
    }


def _compute_v4_structured_state_stats(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Summarize fixed-ID structured obstacle-state arrays."""
    shape_counts: Counter[str] = Counter()
    ids_consistent = True
    transition_consistent = True
    deltas: list[float] = []
    distances: list[float] = []
    clearances: list[float] = []
    n_dyn_values: list[int] = []

    for episode in episodes:
        expected_ids = list(episode.get("dynamic_obstacle_ids", []))
        if expected_ids:
            n_dyn_values.append(len(expected_ids))
        steps = list(episode.get("steps", []))
        for index, step in enumerate(steps):
            if list(step.get("dynamic_obstacle_ids", [])) != expected_ids:
                ids_consistent = False
            world_t = np.asarray(step.get("dynamic_obstacles_world_t"), dtype=float)
            world_t1 = np.asarray(step.get("dynamic_obstacles_world_t1"), dtype=float)
            delta = np.asarray(step.get("dynamic_obstacle_delta_world_t"), dtype=float)
            robot_t = np.asarray(step.get("dynamic_obstacles_robot_t"), dtype=float)
            radii = np.asarray(step.get("dynamic_obstacle_radii"), dtype=float)
            shape_counts[f"world_t{tuple(world_t.shape)}"] += 1
            shape_counts[f"robot_t{tuple(robot_t.shape)}"] += 1
            shape_counts[f"radii{tuple(radii.shape)}"] += 1
            if world_t.ndim == 2 and world_t.shape[1] == 2:
                distances.extend(np.linalg.norm(robot_t, axis=1).astype(float).tolist())
                if radii.shape == (world_t.shape[0],):
                    clearances.extend(
                        (
                            np.linalg.norm(robot_t, axis=1)
                            - float(episode.get("robot_config", {}).get("radius", 0.19))
                            - radii
                        )
                        .astype(float)
                        .tolist()
                    )
            if world_t.shape == world_t1.shape and world_t.shape == delta.shape:
                deltas.extend(np.linalg.norm(delta, axis=1).astype(float).tolist())
            if index < len(steps) - 1:
                next_world = np.asarray(steps[index + 1].get("dynamic_obstacles_world_t"), dtype=float)
                if world_t1.shape != next_world.shape or not np.allclose(world_t1, next_world, atol=1e-5):
                    transition_consistent = False

    return {
        "num_dynamic_obstacles_values": sorted(set(int(value) for value in n_dyn_values)),
        "ids_consistent": bool(ids_consistent),
        "world_t1_to_next_world_t_consistent": bool(transition_consistent),
        "shape_counts": dict(sorted(shape_counts.items())),
        "obstacle_distance_robot_frame_t": _numeric_summary(np.asarray(distances, dtype=float)),
        "obstacle_clearance_robot_frame_t": _numeric_summary(np.asarray(clearances, dtype=float)),
        "delta_world_norm_t": _numeric_summary(np.asarray(deltas, dtype=float)),
    }


def _compute_v2_label_stats(episodes: list[dict[str, object]]) -> dict[str, object]:
    """Compute V2 supervision-label distributions."""
    steps = [step for episode in episodes for step in episode.get("steps", [])]
    total_steps = len(steps)
    if total_steps == 0:
        return {
            "num_steps": 0,
            "collision_within_3": {"count": 0, "fraction": 0.0},
            "collision_within_5": {"count": 0, "fraction": 0.0},
            "clearance": {},
            "progress_to_subgoal": {},
            "distance_to_path": {},
            "heading_error_abs": {},
            "risk_t1": {},
        }

    def values(key: str) -> np.ndarray:
        collected: list[float] = []
        for step in steps:
            value = step.get(key)
            if _is_number(value):
                collected.append(float(value))
        return np.asarray(collected, dtype=float)

    collision_3 = int(sum(bool(step.get("collision_within_3", False)) for step in steps))
    collision_5 = int(sum(bool(step.get("collision_within_5", False)) for step in steps))
    heading_abs = np.abs(values("heading_error_t"))
    return {
        "num_steps": int(total_steps),
        "collision_t1": {
            "count": int(sum(bool(step.get("collision_t1", False)) for step in steps)),
            "fraction": _fraction(int(sum(bool(step.get("collision_t1", False)) for step in steps)), total_steps),
        },
        "collision_within_3": {
            "count": collision_3,
            "fraction": _fraction(collision_3, total_steps),
        },
        "collision_within_5": {
            "count": collision_5,
            "fraction": _fraction(collision_5, total_steps),
        },
        "clearance": {
            "static_t1": _numeric_summary(values("min_static_clearance_t1")),
            "dynamic_t1": _numeric_summary(values("min_dynamic_clearance_t1")),
            "total_t1": _numeric_summary(values("min_total_clearance_t1")),
        },
        "progress_to_subgoal": _numeric_summary(values("progress_to_subgoal")),
        "distance_to_path": {
            "t": _numeric_summary(values("distance_to_path_t")),
            "t1": _numeric_summary(values("distance_to_path_t1")),
        },
        "heading_error_abs": _numeric_summary(heading_abs),
        "risk_t1": _numeric_summary(values("risk_t1")),
    }


def _compute_quality_warnings(report: dict[str, object]) -> list[str]:
    """Generate simple warnings for likely dataset issues."""
    warnings: list[str] = []
    num_episodes = int(report["basic"]["num_episodes"])
    stop_fraction = float(report["actions"]["stop_fraction"])
    avg_feasible_non_stop = float(report["feasibility"]["avg_feasible_non_stop_actions_per_step"])
    stop_runs_gt_10 = int(report["stop_persistence"]["episodes_with_stop_run_gt_10"])
    clearance_lt_0_3 = float(report["clearance"]["fraction_clearance_lt_0_3"])
    short_paths = int(report["trajectory_quality"]["episodes_with_path_length_lt_1_0"])
    v4 = report.get("v4_dynamic_labels", {})
    dynamic_visible_fraction = float(v4.get("dynamic_visible_t", {}).get("fraction", 0.0)) if v4 else 0.0
    dynamic_event_fraction = float(v4.get("dynamic_event_step_t", {}).get("fraction", 0.0)) if v4 else 0.0
    dynamic_residual_beam_fraction = float(v4.get("dynamic_residual_nonzero_beam_fraction", 0.0)) if v4 else 0.0
    structured = report.get("v4_structured_state", {})

    termination_percentages = [
        float(section["percentage"])
        for section in report["termination"].values()
        if isinstance(section, dict) and "percentage" in section
    ]

    if stop_fraction > 0.4:
        warnings.append("High stop fraction; policy may be over-conservative.")
    if avg_feasible_non_stop < 1.0:
        warnings.append("Few feasible non-stop actions on average; navigation may be getting stuck.")
    if num_episodes and (stop_runs_gt_10 / num_episodes) > 0.2:
        warnings.append("Many episodes have long stop runs; persistent freezing detected.")
    if clearance_lt_0_3 < 0.05:
        warnings.append("Near-obstacle coverage is rare; clearance diversity may be too weak.")
    if num_episodes and (short_paths / num_episodes) > 0.1:
        warnings.append("Many episodes have very short path length; movement quality may be poor.")
    if termination_percentages and max(termination_percentages) > 0.8:
        warnings.append("Termination diversity is low; outcomes are dominated by one mode.")
    if dynamic_visible_fraction < 0.05:
        warnings.append("Dynamic visibility is rare; residual supervision may be sparse.")
    if dynamic_event_fraction < 0.05:
        warnings.append("Dynamic event steps are rare; dynamic hazard labels may be weak.")
    if dynamic_residual_beam_fraction < 0.005:
        warnings.append("Dynamic residual beam coverage is very low.")
    if structured and not bool(structured.get("ids_consistent", False)):
        warnings.append("Dynamic obstacle IDs are not consistent across steps.")
    if structured and not bool(structured.get("world_t1_to_next_world_t_consistent", False)):
        warnings.append("Dynamic obstacle world_t1 does not consistently match next world_t.")
    return warnings


def _print_dataset_quality_report(report: dict[str, object]) -> None:
    """Print a structured dataset quality report."""
    print("\n==== DATASET QUALITY REPORT ====\n")

    basic = report["basic"]
    print(f"Episodes: {basic['num_episodes']}")
    print(f"Total steps: {basic['total_steps']}")
    print(f"Avg steps/episode: {basic['avg_steps_per_episode']:.2f}")

    print("\nTermination:")
    for reason, stats in report["termination"].items():
        print(f"{reason}: {stats['count']} ({stats['percentage'] * 100.0:.1f}%)")

    print("\nAction distribution:")
    for action_name in KNOWN_ACTIONS:
        stats = report["actions"][action_name]
        print(f"{action_name}: {stats['count']} ({stats['percentage'] * 100.0:.1f}%)")
    print(f"stop_fraction: {report['actions']['stop_fraction'] * 100.0:.1f}%")

    stop_stats = report["stop_persistence"]
    print("\nStop behavior:")
    print(f"avg longest run: {stop_stats['avg_longest_stop_run']:.2f}")
    print(f"max run: {stop_stats['max_longest_stop_run']}")
    print(f"episodes with run >10: {stop_stats['episodes_with_stop_run_gt_10']}")
    print(f"episodes with run >20: {stop_stats['episodes_with_stop_run_gt_20']}")

    maneuver = report["maneuver_diversity"]
    print("\nTurning / maneuver diversity:")
    print(f"turning action fraction: {maneuver['turning_action_fraction'] * 100.0:.1f}%")
    print(f"avg unique actions/episode: {maneuver['avg_unique_actions_per_episode']:.2f}")

    clearance = report["clearance"]
    print("\nClearance:")
    print(f"mean clearance: {clearance['mean_clearance']:.3f}")
    print(f"clearance <0.3 fraction: {clearance['fraction_clearance_lt_0_3'] * 100.0:.1f}%")
    print(
        "histogram: "
        f"<0.2={clearance['histogram_counts']['lt_0_2']}, "
        f"0.2-0.3={clearance['histogram_counts']['0_2_to_0_3']}, "
        f"0.3-0.5={clearance['histogram_counts']['0_3_to_0_5']}, "
        f">0.5={clearance['histogram_counts']['gt_0_5']}"
    )

    length_stats = report["episode_lengths"]
    print("\nEpisode lengths:")
    print(f"min: {length_stats['min_episode_length']}")
    print(f"max: {length_stats['max_episode_length']}")
    print(f"mean: {length_stats['mean_episode_length']:.2f}")
    print(f"median: {length_stats['median_episode_length']:.2f}")

    traj = report["trajectory_quality"]
    print("\nTrajectory quality:")
    print(f"avg path length: {traj['avg_path_length']:.3f}")
    print(f"avg displacement: {traj['avg_displacement']:.3f}")
    print(f"episodes with path length <1.0 m: {traj['episodes_with_path_length_lt_1_0']}")

    path_complexity = report.get("path_complexity", {})
    if path_complexity:
        directness = path_complexity["directness_ratio"]
        turn_sum = path_complexity["turn_angle_sum"]
        print("\nA* path complexity:")
        print(
            f"directness ratio mean/min/max: "
            f"{directness['mean']:.3f} / {directness['min']:.3f} / {directness['max']:.3f}"
        )
        print(
            f"turn angle sum mean/min/max: "
            f"{turn_sum['mean']:.3f} / {turn_sum['min']:.3f} / {turn_sum['max']:.3f}"
        )

    feasibility = report["feasibility"]
    print("\nFeasibility:")
    print(f"avg feasible actions: {feasibility['avg_feasible_actions_per_step']:.2f}")
    print(f"avg feasible non-stop actions: {feasibility['avg_feasible_non_stop_actions_per_step']:.2f}")
    print(f"fraction only stop feasible: {feasibility['fraction_only_stop_feasible'] * 100.0:.1f}%")
    print(
        f"fraction zero non-stop actions feasible: "
        f"{feasibility['fraction_zero_non_stop_actions_feasible'] * 100.0:.1f}%"
    )

    goal_features = report["goal_features"]
    print("\nGoal feature diversity:")
    print(
        f"distance mean/min/max: "
        f"{goal_features['distance']['mean']:.3f} / "
        f"{goal_features['distance']['min']:.3f} / "
        f"{goal_features['distance']['max']:.3f}"
    )
    print(
        "angle bins: "
        f"front={goal_features['angle_bins']['front']}, "
        f"moderate={goal_features['angle_bins']['moderate']}, "
        f"large={goal_features['angle_bins']['large']}"
    )

    maps = report.get("map_distribution", {})
    if maps:
        print("\nMap distribution:")
        for map_type, stats in maps.items():
            print(f"{map_type}: {stats['count']} ({stats['fraction'] * 100.0:.1f}%)")

    v4 = report.get("v4_dynamic_labels", {})
    if v4:
        print("\nV4 dynamic labels:")
        print(
            f"visible/event fraction: "
            f"{v4['dynamic_visible_t']['fraction'] * 100.0:.1f}% / "
            f"{v4['dynamic_event_step_t']['fraction'] * 100.0:.1f}%"
        )
        print(f"LiDAR-visible dynamic fraction: {v4['lidar_dynamic_visible_t']['fraction'] * 100.0:.1f}%")
        print(f"nonzero residual beam fraction: {v4['dynamic_residual_nonzero_beam_fraction'] * 100.0:.2f}%")
        print(
            f"dynamic collision t1/within 3/within 5: "
            f"{v4['dynamic_collision_t1']['count']} / "
            f"{v4['dynamic_collision_within_3']['count']} / "
            f"{v4['dynamic_collision_within_5']['count']}"
        )
        residual = v4["dynamic_residual_energy_t"]
        clearance = v4["min_dynamic_clearance_t1"]
        risk = v4["dynamic_risk_t1"]
        print(
            f"residual energy mean/p95/max: "
            f"{residual['mean']:.4f} / {residual['p95']:.4f} / {residual['max']:.4f}"
        )
        print(
            f"dynamic clearance t1 p05/p50/p95: "
            f"{clearance['p05']:.3f} / {clearance['p50']:.3f} / {clearance['p95']:.3f}"
        )
        print(
            f"dynamic risk t1 p05/p50/p95/max: "
            f"{risk['p05']:.3f} / {risk['p50']:.3f} / {risk['p95']:.3f} / {risk['max']:.3f}"
        )

    structured = report.get("v4_structured_state", {})
    if structured:
        distance = structured["obstacle_distance_robot_frame_t"]
        delta = structured["delta_world_norm_t"]
        print("\nV4 structured obstacle states:")
        print(f"N_dyn values: {structured['num_dynamic_obstacles_values']}")
        print(f"IDs consistent: {structured['ids_consistent']}")
        print(f"world_t1 -> next world_t consistent: {structured['world_t1_to_next_world_t_consistent']}")
        print(f"distance p05/p50/p95: {distance['p05']:.3f} / {distance['p50']:.3f} / {distance['p95']:.3f}")
        print(f"delta norm mean/p95: {delta['mean']:.4f} / {delta['p95']:.4f}")

    v2 = report.get("v2_base_labels", {})
    if v2:
        print("\nV2-compatible base labels:")
        print(
            f"collision within 3/5: "
            f"{v2['collision_within_3']['count']} / {v2['collision_within_5']['count']}"
        )
        total_clearance = v2["clearance"]["total_t1"]
        progress = v2["progress_to_subgoal"]
        distance_to_path = v2["distance_to_path"]["t"]
        risk = v2["risk_t1"]
        print(
            f"total clearance t1 mean/min: "
            f"{total_clearance['mean']:.3f} / {total_clearance['min']:.3f}"
        )
        print(f"progress mean/min/max: {progress['mean']:.3f} / {progress['min']:.3f} / {progress['max']:.3f}")
        print(f"path distance mean/max: {distance_to_path['mean']:.3f} / {distance_to_path['max']:.3f}")
        print(f"risk_t1 mean/max: {risk['mean']:.3f} / {risk['max']:.3f}")

    warnings = report["warnings"]
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")


def _get_episode_done_reason(episode: dict[str, object]) -> str:
    """Return the best available episode termination reason."""
    episode_stats = episode.get("episode_stats", {})
    done_reason = episode_stats.get("done_reason")
    if isinstance(done_reason, str) and done_reason:
        return done_reason
    steps = episode.get("steps", [])
    if steps:
        final_reason = steps[-1].get("done_reason_t1")
        if isinstance(final_reason, str) and final_reason:
            return final_reason
    return "unknown"


def _longest_stop_run(steps: list[dict[str, object]]) -> int:
    """Return the longest consecutive run of stop actions in one episode."""
    longest = 0
    current = 0
    for step in steps:
        if step.get("action_name") == "stop":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _fraction(numerator: int, denominator: int) -> float:
    """Return a safe fraction."""
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _is_number(value: object) -> bool:
    """Return True for finite scalar numeric values."""
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _numeric_summary(values: np.ndarray) -> dict[str, object]:
    """Return stable summary stats for a numeric vector."""
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _world_points_to_robot_frame(points_world: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=float)
    translated = points - np.asarray(robot_pose[:2], dtype=float)[None, :]
    return _world_vectors_to_robot_frame(translated, float(robot_pose[2]))


def _world_vectors_to_robot_frame(vectors_world: np.ndarray, robot_theta: float) -> np.ndarray:
    vectors = np.asarray(vectors_world, dtype=float)
    cos_theta = float(np.cos(robot_theta))
    sin_theta = float(np.sin(robot_theta))
    rotation = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]], dtype=float)
    return vectors @ rotation.T


if __name__ == "__main__":
    main()
