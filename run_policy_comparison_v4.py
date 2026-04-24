"""Compare baseline and learned V4 structured controllers in closed-loop simulation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from controllers.baseline_structured_controller import BaselineStructuredController, StructuredControllerConfig
from controllers.learned_structured_controller import LearnedStructuredController
from sim2d.config import DatasetConfig
from sim2d.dataset import save_episode_pickle
from sim2d.dynamics import disk_collides_with_occupancy, minimum_dynamic_clearance, minimum_static_clearance
from sim2d.maps import GeneratedMap
from sim2d.obstacles import DynamicObstacle
from sim2d.simulator import EpisodeSimulator
from sim2d.utils import euclidean, json_safe, path_length, save_json, seed_sequence
from sim2d.visualize import save_episode_gif, save_episode_preview_png, save_episode_video
from sim2d.waypoint import compute_local_subgoal


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="artifacts/structured_world_model_rollout_from_epoch_10/best.pt")
    parser.add_argument("--output_dir", type=str, default="artifacts/policy_comparison_v4_h10")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--num_dynamic_obstacles", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--w_progress", type=float, default=2.0)
    parser.add_argument("--w_heading", type=float, default=0.8)
    parser.add_argument("--w_path", type=float, default=1.1)
    parser.add_argument("--w_velocity", type=float, default=0.3)
    parser.add_argument("--w_static", type=float, default=0.5)
    parser.add_argument("--w_dynamic", type=float, default=2.0)
    parser.add_argument("--static_collision_penalty", type=float, default=1e6)
    parser.add_argument("--dynamic_collision_penalty", type=float, default=1e6)
    parser.add_argument("--dynamic_risk_tau", type=float, default=0.5)
    parser.add_argument("--min_clearance_clip", type=float, default=0.05)
    parser.add_argument("--save_episodes", action="store_true")
    parser.add_argument("--save_previews", action="store_true")
    parser.add_argument("--save_gifs", action="store_true")
    parser.add_argument("--save_videos", action="store_true")
    return parser


def _clone_map(map_bundle: GeneratedMap) -> GeneratedMap:
    return GeneratedMap(
        map_id=str(map_bundle.map_id),
        map_type=str(map_bundle.map_type),
        occupancy=map_bundle.occupancy.copy(),
        resolution=float(map_bundle.resolution),
        origin=tuple(map_bundle.origin),
    )


def _clone_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "map_bundle": _clone_map(scenario["map_bundle"]),
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
    robot_pose = scenario["start_pose"].copy()
    goal_point = scenario["goal_point"].copy()
    path_world = scenario["path_world"].copy()
    dyn_obstacles = [obstacle.clone() for obstacle in scenario["dynamic_obstacles"]]

    controller.reset()
    steps: list[dict[str, object]] = []
    robot_trajectory = [robot_pose.copy()]
    dynamic_trajectories: dict[str, list[np.ndarray]] = {
        obstacle.obstacle_id: [obstacle.position.copy()] for obstacle in dyn_obstacles
    }
    static_clearance_history: list[float] = []
    dynamic_clearance_history: list[float] = []
    combined_clearance_history: list[float] = []
    done_reason = "timeout"

    for step_index in range(int(config.max_steps)):
        waypoint_now = compute_local_subgoal(robot_pose, path_world, config.lookahead_distance)
        decision = controller.select_action(
            robot_pose=robot_pose,
            current_subgoal_world=waypoint_now["selected_subgoal_world"],
            path_world=path_world,
            planning_occupancy=planning_occupancy,
            true_occupancy=occupancy,
            resolution=map_bundle.resolution,
            origin=map_bundle.origin,
            occupied_points=occupied_points,
            dynamic_obstacles=dyn_obstacles,
        )

        dynamic_t = _snapshot_obstacles(dyn_obstacles)
        robot_pose_next = simulator._apply_action(robot_pose, float(decision["v"]), float(decision["omega"]))
        for obstacle in dyn_obstacles:
            obstacle.step(config.robot_config.dt)
        dynamic_t1 = _snapshot_obstacles(dyn_obstacles)

        controller.commit_step(
            robot_pose_t=robot_pose.copy(),
            robot_pose_t1=robot_pose_next.copy(),
            dynamic_obstacles_t=dynamic_t,
            dynamic_obstacles_t1=dynamic_t1,
            action_index=int(decision["action_index"]),
            action_cont=np.array([float(decision["v"]), float(decision["omega"])], dtype=np.float32),
            path_world=path_world,
        )

        static_collision = bool(
            disk_collides_with_occupancy(
                robot_pose_next[:2],
                config.robot_config.radius,
                occupancy,
                map_bundle.resolution,
                map_bundle.origin,
            )
        )
        dynamic_collision = bool(
            any(
                np.linalg.norm(robot_pose_next[:2] - np.asarray(obstacle.position, dtype=float))
                <= config.robot_config.radius + float(obstacle.radius)
                for obstacle in dyn_obstacles
            )
        )
        goal_reached = euclidean(robot_pose_next[:2], goal_point) <= config.robot_config.goal_tolerance

        if goal_reached:
            done_reason = "goal_reached"
        elif static_collision:
            done_reason = "static_collision"
        elif dynamic_collision:
            done_reason = "dynamic_collision"
        elif step_index == int(config.max_steps) - 1:
            done_reason = "timeout"

        waypoint_next = compute_local_subgoal(robot_pose_next, path_world, config.lookahead_distance)
        static_clearance = float(
            minimum_static_clearance(
                robot_pose_next[:2],
                config.robot_config.radius,
                occupied_points,
                map_bundle.resolution,
            )
        )
        dynamic_clearance = float(minimum_dynamic_clearance(robot_pose_next[:2], config.robot_config.radius, dyn_obstacles))
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
                "robot_pose": robot_pose.copy(),
                "robot_pose_t": robot_pose.copy(),
                "robot_pose_t1": robot_pose_next.copy(),
                "goal_t": waypoint_now["goal_features"].copy(),
                "goal_t1": waypoint_next["goal_features"].copy(),
                "selected_subgoal_world": waypoint_now["selected_subgoal_world"].copy(),
                "action_index": int(decision["action_index"]),
                "action_name": str(decision["action_name"]),
                "v": float(decision["v"]),
                "omega": float(decision["omega"]),
                "v_t": float(decision["v"]),
                "omega_t": float(decision["omega"]),
                "collision_t1": bool(static_collision or dynamic_collision),
                "static_collision_t1": bool(static_collision),
                "dynamic_collision_t1": bool(dynamic_collision),
                "done_t1": bool(goal_reached or static_collision or dynamic_collision or step_index == int(config.max_steps) - 1),
                "done_reason_t1": str(done_reason),
                "dynamic_obstacles_t": dynamic_t,
                "dynamic_obstacles_t1": dynamic_t1,
                "progress_to_subgoal": float(waypoint_now["goal_features"][2] - waypoint_next["goal_features"][2]),
                "min_static_clearance_t1": static_clearance,
                "min_dynamic_clearance_t1": dynamic_clearance,
                "min_total_clearance_t1": combined_clearance,
                "controller_debug": candidate_debug,
                "chosen_action_min_clearance": float(decision["chosen_action_min_clearance"]),
                "emergency_override_used": bool(decision["emergency_override_used"]),
                "score_components": _score_components_from_debug(chosen_debug),
            }
        )

        robot_pose = robot_pose_next
        robot_trajectory.append(robot_pose.copy())
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
            "final_distance_to_goal": float(euclidean(robot_pose[:2], goal_point)),
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

    for episode in episodes:
        stats = episode["episode_stats"]
        action_counts.update(stats.get("action_counts", {}))
        stop_fractions.append(float(stats["stop_fraction"]))
        path_lengths.append(float(stats["robot_path_length"]))
        progresses.append(float(stats["total_progress"]))
        min_dyn_clearances.append(float(stats["min_dynamic_clearance"]))
        min_static_clearances.append(float(stats["min_static_clearance"]))
        for step in episode["steps"]:
            for key, value in step.get("score_components", {}).items():
                score_components[str(key)].append(float(value))

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
    }


def _save_episode_artifacts(
    episode: dict[str, object],
    output_dir: Path,
    vis_config: Any,
    *,
    save_episodes: bool,
    save_previews: bool,
    save_gifs: bool,
    save_videos: bool,
) -> None:
    controller_dir = output_dir / str(episode["controller"])
    stem = f"episode_{int(episode['episode_id']):06d}"
    save_json(controller_dir / "decision_logs" / f"{stem}.json", _build_decision_log(episode))
    save_json(controller_dir / "episode_results" / f"{stem}.json", _episode_result(episode))
    if save_episodes:
        save_episode_pickle(episode, controller_dir / "episodes" / f"{stem}.pkl")
    if save_previews:
        save_episode_preview_png(episode, controller_dir / "preview" / f"{stem}.png", vis_config)
    if save_gifs:
        save_episode_gif(episode, controller_dir / "gif" / f"{stem}.gif", vis_config)
    if save_videos:
        save_episode_video(episode, controller_dir / "video" / f"{stem}.mp4", vis_config)


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Structured model checkpoint not found: {checkpoint_path}")

    config = DatasetConfig(num_episodes=int(args.num_episodes), seed=int(args.seed), max_steps=int(args.max_steps))
    if args.num_dynamic_obstacles is not None:
        config.dynamic_obstacle_config.num_dynamic_obstacles = int(args.num_dynamic_obstacles)

    simulator = EpisodeSimulator(config)
    controller_cfg = StructuredControllerConfig(
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
    )
    controllers = [
        BaselineStructuredController(config, controller_cfg),
        LearnedStructuredController(checkpoint_path, config, controller_cfg, device=str(args.device)),
    ]

    seeds = seed_sequence(int(args.seed), int(args.num_episodes))
    all_episodes: dict[str, list[dict[str, object]]] = {controller.name: [] for controller in controllers}

    for episode_id, seed in enumerate(tqdm(seeds, desc="policy comparison v4", dynamic_ncols=True)):
        scenario = _sample_scenario(simulator, episode_id, seed)
        for controller in controllers:
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
                save_episodes=bool(args.save_episodes),
                save_previews=bool(args.save_previews),
                save_gifs=bool(args.save_gifs),
                save_videos=bool(args.save_videos),
            )
            all_episodes[controller.name].append(episode)

    summary = {
        "schema_version": "v4_policy_comparison_1",
        "checkpoint": str(checkpoint_path),
        "config": {
            "num_episodes": int(args.num_episodes),
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "num_dynamic_obstacles": int(config.dynamic_obstacle_config.num_dynamic_obstacles),
            "horizon": int(args.horizon),
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
        },
        "controllers": {name: _aggregate_controller_episodes(episodes) for name, episodes in all_episodes.items()},
        "episodes": {name: [_episode_result(episode) for episode in episodes] for name, episodes in all_episodes.items()},
    }
    save_json(output_dir / "summary.json", json_safe(summary))
    print(json_safe(summary["controllers"]))


if __name__ == "__main__":
    main()
