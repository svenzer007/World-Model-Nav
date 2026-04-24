"""Generate a self-contained V4 structured dynamic-obstacle dataset.

Example:
    python -m sim2d.generate_data --num_episodes 20 --output_dir ./output_v4_dynamic_state --seed 0 --max_steps 250 --save_gifs --gif_every 5

This module generates occupancy-grid episodes, DWA-inspired expert actions,
structured fixed-ID dynamic obstacle states, auxiliary front-sector LiDAR,
per-episode pickles, previews, and reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .config import DatasetConfig
from .dataset import load_episode_pickle, save_dataset_index, save_episode_pickle, summarize_dataset
from .inspect_episodes import compute_dataset_quality_report
from .simulator import EpisodeSimulator
from .utils import seed_sequence
from .visualize import save_episode_gif, save_episode_preview_png, save_timestep_debug_artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./output_v4_dynamic_state")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--save_gifs", action="store_true")
    parser.add_argument("--gif_every", type=int, default=5)
    parser.add_argument("--num_dynamic_obstacles", type=int, default=None)
    parser.add_argument("--dynamic_interaction_resample_limit", type=int, default=None)
    parser.add_argument("--draw_lidar_in_gif", action="store_true")
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
    config.visualization_config.draw_lidar_in_gif = bool(args.draw_lidar_in_gif)
    if args.expert_horizon_seconds is not None:
        config.expert_config.horizon_seconds = float(args.expert_horizon_seconds)
    if args.dynamic_interaction_resample_limit is not None:
        config.dynamic_interaction_resample_limit = int(args.dynamic_interaction_resample_limit)

    simulator = EpisodeSimulator(config)
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

    for offset, episode_seed in enumerate(tqdm(seeds, desc="Generating episodes")):
        episode_id = start_episode_id + offset
        episode = simulator.generate_episode(episode_id=episode_id, seed=episode_seed)
        episode_path = episodes_dir / f"episode_{episode_id:06d}.pkl"
        png_path = preview_dir / f"episode_{episode_id:06d}.png"
        save_episode_pickle(episode, episode_path)
        save_episode_preview_png(episode, png_path, config.visualization_config)
        if config.save_gifs and (episode_id % max(1, config.gif_every) == 0):
            gif_path = preview_dir / f"episode_{episode_id:06d}.gif"
            save_episode_gif(episode, gif_path, config.visualization_config)
        if args.debug_episode_id is not None and args.debug_timestep is not None and episode_id == args.debug_episode_id:
            debug_dir = Path(args.debug_output_dir) if args.debug_output_dir else output_dir / "debug_timestep"
            save_timestep_debug_artifacts(
                episode=episode,
                timestep=args.debug_timestep,
                output_dir=debug_dir,
                vis_config=config.visualization_config,
            )
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
