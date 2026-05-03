"""Generate a self-contained V4 structured dynamic-obstacle dataset.

Example:
    python -m sim2d.generate_data --num_episodes 20 --output_dir ./output_v4_dynamic_state --seed 0 --max_steps 250 --save_gifs --gif_every 5

This module generates occupancy-grid episodes, DWA-inspired expert actions,
structured fixed-ID dynamic obstacle states, auxiliary front-sector LiDAR,
per-episode pickles, previews, and reports.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

from tqdm import tqdm

from .config import ACTIONS, DatasetConfig
from .dataset import load_episode_pickle, save_dataset_index, save_episode_pickle, summarize_dataset
from .inspect_episodes import compute_dataset_quality_report
from .simulator import EpisodeSimulator
from .utils import seed_sequence
from .visualize import save_episode_gif, save_episode_preview_png, save_timestep_debug_artifacts


_WORKER_SIMULATOR: EpisodeSimulator | None = None


def _init_worker(config: DatasetConfig) -> None:
    global _WORKER_SIMULATOR
    _WORKER_SIMULATOR = EpisodeSimulator(config)


def _generate_episode_worker(job: dict[str, object]) -> int:
    if _WORKER_SIMULATOR is None:
        raise RuntimeError("Dataset generation worker was not initialized")
    episode_id = int(job["episode_id"])
    episode = _WORKER_SIMULATOR.generate_episode(episode_id=episode_id, seed=int(job["seed"]))
    output_dir = Path(str(job["output_dir"]))
    episodes_dir = output_dir / "episodes"
    preview_dir = output_dir / "preview"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    save_episode_pickle(episode, episodes_dir / f"episode_{episode_id:06d}.pkl")
    preview_every = int(job["preview_every"])
    if preview_every > 0 and (episode_id % max(1, preview_every) == 0):
        save_episode_preview_png(episode, preview_dir / f"episode_{episode_id:06d}.png", _WORKER_SIMULATOR.config.visualization_config)
    if bool(job["save_gifs"]) and (episode_id % max(1, int(job["gif_every"])) == 0):
        save_episode_gif(episode, preview_dir / f"episode_{episode_id:06d}.gif", _WORKER_SIMULATOR.config.visualization_config)
    debug_episode_id = job.get("debug_episode_id")
    debug_timestep = job.get("debug_timestep")
    if debug_episode_id is not None and debug_timestep is not None and episode_id == int(debug_episode_id):
        debug_dir = Path(str(job["debug_output_dir"])) if job.get("debug_output_dir") else output_dir / "debug_timestep"
        save_timestep_debug_artifacts(
            episode=episode,
            timestep=int(debug_timestep),
            output_dir=debug_dir,
            vis_config=_WORKER_SIMULATOR.config.visualization_config,
        )
    return episode_id


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./output_v4_dynamic_state")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--save_gifs", action="store_true")
    parser.add_argument("--gif_every", type=int, default=5)
    parser.add_argument("--preview_every", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--map_yaml", type=str, default=None)
    parser.add_argument("--robot_radius", type=float, default=None)
    parser.add_argument("--num_dynamic_obstacles", type=int, default=None)
    parser.add_argument("--dynamic_obstacle_radius", type=float, default=None)
    parser.add_argument("--dynamic_obstacle_speed", type=float, default=None)
    parser.add_argument("--dynamic_obstacle_route_radius", type=float, default=None)
    parser.add_argument("--dynamic_obstacle_route_length_min", type=int, default=None)
    parser.add_argument("--dynamic_obstacle_route_length_max", type=int, default=None)
    parser.add_argument("--dynamic_obstacle_route_spacing", type=float, default=None)
    parser.add_argument("--dynamic_obstacle_crossing_probability", type=float, default=None)
    parser.add_argument("--max_robot_path_deviation", type=float, default=None)
    parser.add_argument("--max_robot_trajectory_path_ratio", type=float, default=None)
    parser.add_argument("--dynamic_interaction_resample_limit", type=int, default=None)
    parser.add_argument("--robot_command_randomization", action="store_true")
    parser.add_argument("--robot_command_delay_min_steps", type=int, default=None)
    parser.add_argument("--robot_command_delay_max_steps", type=int, default=None)
    parser.add_argument("--robot_command_v_gain_min", type=float, default=None)
    parser.add_argument("--robot_command_v_gain_max", type=float, default=None)
    parser.add_argument("--robot_command_omega_gain_min", type=float, default=None)
    parser.add_argument("--robot_command_omega_gain_max", type=float, default=None)
    parser.add_argument("--robot_command_v_noise_std", type=float, default=None)
    parser.add_argument("--robot_command_omega_noise_std", type=float, default=None)
    parser.add_argument("--robot_command_v_clip_min", type=float, default=None)
    parser.add_argument("--robot_command_v_clip_max", type=float, default=None)
    parser.add_argument("--robot_command_omega_clip_min", type=float, default=None)
    parser.add_argument("--robot_command_omega_clip_max", type=float, default=None)
    parser.add_argument("--draw_lidar_in_gif", action="store_true")
    parser.add_argument("--disable_backward", action="store_true")
    lidar_group = parser.add_mutually_exclusive_group()
    lidar_group.add_argument("--generate_lidar", dest="generate_lidar", action="store_true")
    lidar_group.add_argument("--no_lidar", dest="generate_lidar", action="store_false")
    parser.set_defaults(generate_lidar=None)
    parser.add_argument("--expert_horizon_seconds", type=float, default=None)
    parser.add_argument("--debug_episode_id", type=int, default=None)
    parser.add_argument("--debug_timestep", type=int, default=None)
    parser.add_argument("--debug_output_dir", type=str, default=None)
    parser.add_argument("--skip_quality_report", action="store_true")
    return parser


def main() -> None:
    """CLI entrypoint for dataset generation."""
    args = build_arg_parser().parse_args()
    config = DatasetConfig(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        save_gifs=args.save_gifs,
        gif_every=args.gif_every,
    )
    if args.num_dynamic_obstacles is not None:
        config.dynamic_obstacle_config.num_dynamic_obstacles = int(args.num_dynamic_obstacles)
    if args.map_yaml is not None:
        config.map_config.map_yaml = str(args.map_yaml)
        config.map_config.min_start_goal_distance = 5.0
        config.map_config.min_path_length = 6.0
        config.map_config.min_path_directness_ratio = 1.05
        config.map_config.min_path_turn_angle_sum = 0.2
        config.dynamic_obstacle_config.route_length_range = (3, 6)
        config.dynamic_obstacle_config.route_radius = 3.0
        config.dynamic_obstacle_config.route_crossing_probability = 0.75
        config.dynamic_obstacle_config.route_point_spacing = 2.2
        config.expert_config.goal_progress_weight = 2.4
        config.expert_config.clearance_weight = 1.0
        config.expert_config.path_adherence_weight = 2.8
        config.max_robot_path_deviation = 2.5
        config.max_robot_trajectory_path_ratio = 1.9
        config.dynamic_interaction_resample_limit = 10
        config.save_static_lidar = False
        config.save_dynamic_residual = False
    if args.disable_backward:
        config.expert_config.action_indices = tuple(
            int(action_index)
            for action_index, action in ACTIONS.items()
            if str(action["name"]) != "backward"
        )
    if args.generate_lidar is not None:
        config.save_static_lidar = bool(args.generate_lidar)
        config.save_dynamic_residual = bool(args.generate_lidar)
    if args.robot_radius is not None:
        config.robot_config.radius = float(args.robot_radius)
    if args.dynamic_obstacle_radius is not None:
        radius = float(args.dynamic_obstacle_radius)
        config.dynamic_obstacle_config.radius_range = (radius, radius)
    if args.dynamic_obstacle_speed is not None:
        speed = float(args.dynamic_obstacle_speed)
        config.dynamic_obstacle_config.speed_range = (speed, speed)
    if args.dynamic_obstacle_route_radius is not None:
        config.dynamic_obstacle_config.route_radius = float(args.dynamic_obstacle_route_radius)
    if args.dynamic_obstacle_route_length_min is not None or args.dynamic_obstacle_route_length_max is not None:
        current_min, current_max = config.dynamic_obstacle_config.route_length_range
        route_min = int(args.dynamic_obstacle_route_length_min) if args.dynamic_obstacle_route_length_min is not None else int(current_min)
        route_max = int(args.dynamic_obstacle_route_length_max) if args.dynamic_obstacle_route_length_max is not None else int(current_max)
        if route_min < 2 or route_max < route_min:
            raise ValueError("Dynamic obstacle route length must satisfy 2 <= min <= max")
        config.dynamic_obstacle_config.route_length_range = (route_min, route_max)
    if args.dynamic_obstacle_route_spacing is not None:
        config.dynamic_obstacle_config.route_point_spacing = float(args.dynamic_obstacle_route_spacing)
    if args.dynamic_obstacle_crossing_probability is not None:
        config.dynamic_obstacle_config.route_crossing_probability = float(args.dynamic_obstacle_crossing_probability)
    if args.max_robot_path_deviation is not None:
        config.max_robot_path_deviation = float(args.max_robot_path_deviation)
    if args.max_robot_trajectory_path_ratio is not None:
        config.max_robot_trajectory_path_ratio = float(args.max_robot_trajectory_path_ratio)
    config.visualization_config.draw_lidar_in_gif = bool(args.draw_lidar_in_gif)
    command_cfg = config.robot_command_randomization_config
    if args.robot_command_randomization:
        command_cfg.enabled = True
    if args.robot_command_delay_min_steps is not None:
        command_cfg.delay_min_steps = int(args.robot_command_delay_min_steps)
    if args.robot_command_delay_max_steps is not None:
        command_cfg.delay_max_steps = int(args.robot_command_delay_max_steps)
    if args.robot_command_v_gain_min is not None or args.robot_command_v_gain_max is not None:
        current_min, current_max = command_cfg.v_gain_range
        gain_min = float(args.robot_command_v_gain_min) if args.robot_command_v_gain_min is not None else float(current_min)
        gain_max = float(args.robot_command_v_gain_max) if args.robot_command_v_gain_max is not None else float(current_max)
        command_cfg.v_gain_range = (gain_min, gain_max)
    if args.robot_command_omega_gain_min is not None or args.robot_command_omega_gain_max is not None:
        current_min, current_max = command_cfg.omega_gain_range
        gain_min = (
            float(args.robot_command_omega_gain_min)
            if args.robot_command_omega_gain_min is not None
            else float(current_min)
        )
        gain_max = (
            float(args.robot_command_omega_gain_max)
            if args.robot_command_omega_gain_max is not None
            else float(current_max)
        )
        command_cfg.omega_gain_range = (gain_min, gain_max)
    if args.robot_command_v_noise_std is not None:
        command_cfg.v_noise_std = float(args.robot_command_v_noise_std)
    if args.robot_command_omega_noise_std is not None:
        command_cfg.omega_noise_std = float(args.robot_command_omega_noise_std)
    if args.robot_command_v_clip_min is not None or args.robot_command_v_clip_max is not None:
        current_min, current_max = command_cfg.v_clip_range
        clip_min = float(args.robot_command_v_clip_min) if args.robot_command_v_clip_min is not None else float(current_min)
        clip_max = float(args.robot_command_v_clip_max) if args.robot_command_v_clip_max is not None else float(current_max)
        command_cfg.v_clip_range = (clip_min, clip_max)
    if args.robot_command_omega_clip_min is not None or args.robot_command_omega_clip_max is not None:
        current_min, current_max = command_cfg.omega_clip_range
        clip_min = (
            float(args.robot_command_omega_clip_min)
            if args.robot_command_omega_clip_min is not None
            else float(current_min)
        )
        clip_max = (
            float(args.robot_command_omega_clip_max)
            if args.robot_command_omega_clip_max is not None
            else float(current_max)
        )
        command_cfg.omega_clip_range = (clip_min, clip_max)
    if args.expert_horizon_seconds is not None:
        config.expert_config.horizon_seconds = float(args.expert_horizon_seconds)
    if args.dynamic_interaction_resample_limit is not None:
        config.dynamic_interaction_resample_limit = int(args.dynamic_interaction_resample_limit)

    output_dir = Path(config.output_dir)
    episodes_dir = output_dir / "episodes"
    preview_dir = output_dir / "preview"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    existing_episode_paths = sorted(episodes_dir.glob("episode_*.pkl"))
    start_episode_id = len(existing_episode_paths)
    target_episode_count = config.num_episodes
    all_target_seeds = seed_sequence(config.seed, target_episode_count)
    seeds = all_target_seeds[start_episode_id:]

    jobs = [
        {
            "episode_id": start_episode_id + offset,
            "seed": episode_seed,
            "output_dir": str(output_dir),
            "preview_every": int(args.preview_every),
            "save_gifs": bool(config.save_gifs),
            "gif_every": int(config.gif_every),
            "debug_episode_id": args.debug_episode_id,
            "debug_timestep": args.debug_timestep,
            "debug_output_dir": args.debug_output_dir,
        }
        for offset, episode_seed in enumerate(seeds)
    ]

    num_workers = max(0, int(args.num_workers))
    if num_workers > 1 and jobs:
        with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(config,)) as pool:
            for _ in tqdm(
                pool.imap_unordered(_generate_episode_worker, jobs),
                total=len(jobs),
                desc=f"Generating episodes ({num_workers} workers)",
            ):
                pass
    else:
        _init_worker(config)
        for job in tqdm(jobs, desc="Generating episodes"):
            _generate_episode_worker(job)
    all_episode_paths = sorted(episodes_dir.glob("episode_*.pkl"))
    episodes = [load_episode_pickle(path) for path in all_episode_paths]
    episode_files = [str(path.relative_to(output_dir)) for path in all_episode_paths]
    summary = summarize_dataset(episodes, config, episode_files)
    save_dataset_index(summary, output_dir / "dataset_index.json")
    if not args.skip_quality_report:
        report = compute_dataset_quality_report(episodes)
        from .utils import save_json

        save_json(output_dir / "dataset_quality_report.json", report)


if __name__ == "__main__":
    main()
