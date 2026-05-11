"""Run one closed-loop policy evaluation."""

from __future__ import annotations

import argparse

from controllers.cbf_qp_controller import infer_cbf_command_bounds


CBF_V_MIN_DEFAULT, CBF_V_MAX_DEFAULT, CBF_OMEGA_MIN_DEFAULT, CBF_OMEGA_MAX_DEFAULT = infer_cbf_command_bounds()


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
        choices=("learned_structured", "baseline_structured", "cbf_qp", "mpc_static"),
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
    parser.add_argument("--cbf_alpha", type=float, default=1.0)
    parser.add_argument("--cbf_safety_margin", type=float, default=0.08)
    parser.add_argument("--cbf_k_v", type=float, default=0.4)
    parser.add_argument("--cbf_k_omega", type=float, default=1.2)
    parser.add_argument("--cbf_w_v", type=float, default=1.0)
    parser.add_argument("--cbf_w_omega", type=float, default=0.2)
    _add_bool_flag(
        parser,
        name="cbf_omega_scan_enabled",
        default=True,
        enable_help="Enable omega-candidate scan for cbf_qp",
        disable_help="Disable omega-candidate scan and use omega_nom only for cbf_qp",
    )
    parser.add_argument("--cbf_omega_candidates", type=str, default="-1.0,-0.5,0.0,0.5,1.0")
    parser.add_argument("--cbf_static_rollout_steps", type=int, default=10)
    _add_bool_flag(
        parser,
        name="cbf_heading_lookahead_enabled",
        default=False,
        enable_help="Enable heading-lookahead time for omega-candidate CBF evaluation",
        disable_help="Disable heading-lookahead time and use one simulator dt for omega-candidate CBF evaluation",
    )
    parser.add_argument("--cbf_heading_lookahead_time", type=float, default=1.0)
    _add_bool_flag(
        parser,
        name="cbf_margin_priority_use_margin_trigger",
        default=False,
        enable_help="Enable margin-based priority selection trigger for cbf_qp",
        disable_help="Disable margin-based priority selection trigger for cbf_qp",
    )
    parser.add_argument("--cbf_margin_priority_threshold", type=float, default=0.17)
    parser.add_argument("--cbf_margin_tie_positive_eps", type=float, default=1e-4)
    parser.add_argument("--cbf_v_min", type=float, default=CBF_V_MIN_DEFAULT)
    parser.add_argument("--cbf_v_max", type=float, default=CBF_V_MAX_DEFAULT)
    parser.add_argument("--cbf_omega_min", type=float, default=CBF_OMEGA_MIN_DEFAULT)
    parser.add_argument("--cbf_omega_max", type=float, default=CBF_OMEGA_MAX_DEFAULT)
    parser.add_argument("--cbf_constraint_eps", type=float, default=1e-8)
    _add_bool_flag(
        parser,
        name="cbf_use_reverse_fallback",
        default=False,
        enable_help="Enable reverse-speed fallback when the CBF projection is infeasible",
        disable_help="Disable reverse-speed fallback when the CBF projection is infeasible",
    )
    parser.add_argument("--cbf_reverse_fallback_speed", type=float, default=0.1)
    parser.add_argument("--cbf_debug", action="store_true")
    parser.add_argument("--mpc_horizon", type=int, default=10)
    parser.add_argument("--mpc_obstacle_model", type=str, choices=("static",), default="static")
    parser.add_argument("--mpc_debug", action="store_true")
    parser.add_argument("--save_episodes", action="store_true")
    parser.add_argument("--save_previews", action="store_true")
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        help="Deprecated alias for --save_videos; saves MP4, not GIF.",
    )
    parser.add_argument("--save_videos", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    from policy_eval.common import run_policy_eval

    run_policy_eval(args)


if __name__ == "__main__":
    main()
