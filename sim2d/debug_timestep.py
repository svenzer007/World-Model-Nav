"""Focused timestep debug CLI for expert-policy diagnosis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import VisualizationConfig
from .dataset import load_episode_pickle
from .visualize import build_timestep_debug_summary, save_timestep_debug_artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the debug CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--episode_id", type=int, required=True)
    parser.add_argument("--timestep", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser


def main() -> None:
    """CLI entrypoint for focused timestep debugging."""
    args = build_arg_parser().parse_args()
    data_dir = Path(args.data_dir)
    episode_path = data_dir / "episodes" / f"episode_{args.episode_id:06d}.pkl"
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_path}")

    episode = load_episode_pickle(episode_path)
    if args.timestep < 0 or args.timestep >= len(episode["steps"]):
        raise IndexError(f"Timestep {args.timestep} out of range for episode {args.episode_id}")

    output_dir = Path(args.output_dir)
    artifacts = save_timestep_debug_artifacts(
        episode=episode,
        timestep=args.timestep,
        output_dir=output_dir,
        vis_config=VisualizationConfig(),
    )
    summary = build_timestep_debug_summary(episode, args.timestep)
    _print_terminal_summary(summary)
    print("Saved artifacts:")
    for key, value in artifacts.items():
        print(f"  {key}: {value}")


def _print_terminal_summary(summary: dict[str, object]) -> None:
    chosen = summary["chosen_action"]
    candidates = summary["candidates"]
    feasible_names = [str(candidate["action_name"]) for candidate in candidates if bool(candidate.get("feasible", False))]
    static_failures = [
        str(candidate["action_name"])
        for candidate in candidates
        if str(candidate.get("collision_reason", "")) == "static_collision"
    ]

    print(f"Chosen action: {chosen['action_name']} (index={chosen['action_index']}, feasible={chosen['feasible']})")
    print(f"Feasible actions: {feasible_names if feasible_names else '[]'}")
    print(f"Static-collision failures: {static_failures if static_failures else '[]'}")
    print("Candidate details:")
    for candidate in candidates:
        one_step_true_feasible = not bool(candidate.get("true_map_one_step_collision", False))
        print(
            "  "
            f"{candidate['action_name']}: feasible={candidate.get('feasible', False)} "
            f"collision_reason={candidate.get('collision_reason', '')} "
            f"collision_step={candidate.get('collision_step_index', None)} "
            f"true_map_one_step_feasible={one_step_true_feasible}"
        )


if __name__ == "__main__":
    main()
