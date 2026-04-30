"""Run one closed-loop policy evaluation."""

from __future__ import annotations

import argparse


def _add_bool_flag(
    parser: argparse.ArgumentParser,
    *,
    name: str,
    default: bool,
    enable_help: str,
    disable_help: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name, action="store_true", help=enable_help)
    group.add_argument(f"--no_{name}", dest=name, action="store_false", help=disable_help)
    parser.set_defaults(**{name: default})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        type=str,
        choices=("learned_structured", "baseline_structured"),
        default="learned_structured",
    )
    parser.add_argument("--checkpoint", type=str, default="artifacts/structured_world_model_factorized_world_v3_stage2/best.pt")
    parser.add_argument("--output_dir", type=str, default="artifacts/policy_eval")
    parser.add_argument("--episode_source", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--num_dynamic_obstacles", type=int, default=None)
    parser.add_argument("--map_width", type=int, default=None)
    parser.add_argument("--map_height", type=int, default=None)
    parser.add_argument("--disable_backward", action="store_true")
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
    _add_bool_flag(
        parser,
        name="backward_penalty_enabled",
        default=True,
        enable_help="Enable the extra backward-action selection penalty",
        disable_help="Disable the extra backward-action selection penalty",
    )
    parser.add_argument("--backward_penalty", type=float, default=10.0)
    _add_bool_flag(
        parser,
        name="backward_gate_enabled",
        default=True,
        enable_help="Enable escape-only gating for the backward action",
        disable_help="Disable escape-only gating for the backward action",
    )
    parser.add_argument("--dynamic_stop_clearance_threshold", type=float, default=0.20)
    parser.add_argument("--backward_dynamic_margin", type=float, default=0.10)
    parser.add_argument("--backward_action_name", type=str, default="backward")
    _add_bool_flag(
        parser,
        name="execution_noise_enabled",
        default=False,
        enable_help="Enable Gaussian execution noise and stochastic candidate rollouts during policy eval",
        disable_help="Disable execution noise and keep deterministic candidate rollouts",
    )
    parser.add_argument("--sigma_v", type=float, default=0.05)
    parser.add_argument("--sigma_omega", type=float, default=0.1)
    _add_bool_flag(
        parser,
        name="pose_observation_noise_enabled",
        default=False,
        enable_help="Enable Gaussian noisy pose observations for EKF correction",
        disable_help="Disable pose-observation noise and use exact pose observations",
    )
    parser.add_argument("--sigma_obs_x", type=float, default=0.03)
    parser.add_argument("--sigma_obs_y", type=float, default=0.03)
    parser.add_argument("--sigma_obs_theta", type=float, default=0.05)
    parser.add_argument("--num_stochastic_rollouts", type=int, default=8)
    parser.add_argument("--risk_beta", type=float, default=1.0)
    parser.add_argument("--collision_rate_threshold", type=float, default=0.15)
    parser.add_argument("--save_episodes", action="store_true")
    parser.add_argument("--save_previews", action="store_true")
    parser.add_argument("--save_gifs", action="store_true")
    parser.add_argument("--save_videos", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    from policy_eval.common import run_policy_eval

    run_policy_eval(args)


if __name__ == "__main__":
    main()
