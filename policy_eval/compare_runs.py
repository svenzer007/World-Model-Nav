"""Compare two completed policy-eval output directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim2d.utils import json_safe, save_json


SELECTED_AGGREGATE_METRICS = [
    "success_rate",
    "dynamic_collision_rate",
    "static_collision_rate",
    "timeout_rate",
    "average_path_length",
    "average_progress",
    "average_min_dynamic_clearance",
    "average_min_static_clearance",
    "stop_fraction",
]

ROLLOUT_METRICS = [
    "global_clearance_mae",
    "global_signed_clearance_bias",
    "global_false_negative_rate",
    "global_false_positive_rate",
    "global_radial_distance_signed_bias",
    "global_radial_closing_speed_signed_bias",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_a", type=str, required=True)
    parser.add_argument("--run_b", type=str, required=True)
    parser.add_argument("--label_a", type=str, default="run_a")
    parser.add_argument("--label_b", type=str, default="run_b")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json under {run_dir}")
    return _load_json(summary_path)


def _load_rollout_summary(run_dir: Path) -> dict[str, Any] | None:
    candidates = [
        run_dir / "rollout_error_stats_summary.json",
        run_dir / "learned_structured" / "rollout_error_stats_summary.json",
    ]
    for path in candidates:
        if path.exists():
            return _load_json(path)
    return None


def _load_episode_results(run_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        run_dir / "episode_results",
        run_dir / "learned_structured" / "episode_results",
    ]
    for directory in candidates:
        if directory.exists():
            return [_load_json(path) for path in sorted(directory.glob("episode_*.json"))]
    return []


def _summary_aggregate(summary: dict[str, Any]) -> dict[str, Any]:
    if "aggregate" in summary:
        return dict(summary["aggregate"])
    controllers = dict(summary.get("controllers", {}))
    if len(controllers) == 1:
        return dict(next(iter(controllers.values())))
    raise ValueError("Could not resolve aggregate metrics from summary payload")


def _compare_scalar_block(
    block_a: dict[str, Any],
    block_b: dict[str, Any],
    keys: list[str],
) -> dict[str, dict[str, float]]:
    compared: dict[str, dict[str, float]] = {}
    for key in keys:
        value_a = float(block_a.get(key, float("nan")))
        value_b = float(block_b.get(key, float("nan")))
        compared[key] = {
            "run_a": value_a,
            "run_b": value_b,
            "delta_b_minus_a": value_b - value_a,
        }
    return compared


def _compare_action_distribution(
    block_a: dict[str, Any],
    block_b: dict[str, Any],
) -> dict[str, dict[str, float]]:
    keys = sorted(set(dict(block_a.get("action_distribution", {})).keys()) | set(dict(block_b.get("action_distribution", {})).keys()))
    compared: dict[str, dict[str, float]] = {}
    for key in keys:
        value_a = float(dict(block_a.get("action_distribution", {})).get(key, 0.0))
        value_b = float(dict(block_b.get("action_distribution", {})).get(key, 0.0))
        compared[key] = {
            "run_a": value_a,
            "run_b": value_b,
            "delta_b_minus_a": value_b - value_a,
        }
    return compared


def _compare_score_components(
    block_a: dict[str, Any],
    block_b: dict[str, Any],
) -> dict[str, dict[str, float]]:
    keys = sorted(set(dict(block_a.get("mean_score_components", {})).keys()) | set(dict(block_b.get("mean_score_components", {})).keys()))
    compared: dict[str, dict[str, float]] = {}
    for key in keys:
        value_a = float(dict(block_a.get("mean_score_components", {})).get(key, float("nan")))
        value_b = float(dict(block_b.get("mean_score_components", {})).get(key, float("nan")))
        compared[key] = {
            "run_a": value_a,
            "run_b": value_b,
            "delta_b_minus_a": value_b - value_a,
        }
    return compared


def build_comparison_payload(
    *,
    run_a: Path,
    run_b: Path,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    summary_a = _load_summary(run_a)
    summary_b = _load_summary(run_b)
    aggregate_a = _summary_aggregate(summary_a)
    aggregate_b = _summary_aggregate(summary_b)
    rollout_a = _load_rollout_summary(run_a)
    rollout_b = _load_rollout_summary(run_b)
    episodes_a = _load_episode_results(run_a)
    episodes_b = _load_episode_results(run_b)

    payload: dict[str, Any] = {
        "schema_version": "v4_policy_eval_comparison_1",
        "run_a": {
            "label": label_a,
            "path": str(run_a),
            "controller": str(summary_a.get("controller", "")),
            "checkpoint": summary_a.get("checkpoint"),
            "aggregate": aggregate_a,
        },
        "run_b": {
            "label": label_b,
            "path": str(run_b),
            "controller": str(summary_b.get("controller", "")),
            "checkpoint": summary_b.get("checkpoint"),
            "aggregate": aggregate_b,
        },
        "aggregate_metrics": _compare_scalar_block(aggregate_a, aggregate_b, SELECTED_AGGREGATE_METRICS),
        "action_distribution": _compare_action_distribution(aggregate_a, aggregate_b),
        "mean_score_components": _compare_score_components(aggregate_a, aggregate_b),
        "episode_result_counts": {
            "run_a": int(len(episodes_a)),
            "run_b": int(len(episodes_b)),
        },
    }
    if rollout_a is not None and rollout_b is not None:
        payload["rollout_error_metrics"] = _compare_scalar_block(rollout_a, rollout_b, ROLLOUT_METRICS)
    if episodes_a and episodes_b:
        by_episode_a = {int(item["episode_id"]): item for item in episodes_a}
        by_episode_b = {int(item["episode_id"]): item for item in episodes_b}
        shared_ids = sorted(set(by_episode_a.keys()) & set(by_episode_b.keys()))
        payload["per_episode_deltas"] = [
            {
                "episode_id": episode_id,
                "seed_a": int(by_episode_a[episode_id]["seed"]),
                "seed_b": int(by_episode_b[episode_id]["seed"]),
                "success_delta": int(bool(by_episode_b[episode_id]["success"])) - int(bool(by_episode_a[episode_id]["success"])),
                "dynamic_collision_delta": int(bool(by_episode_b[episode_id]["dynamic_collision"])) - int(bool(by_episode_a[episode_id]["dynamic_collision"])),
                "static_collision_delta": int(bool(by_episode_b[episode_id]["static_collision"])) - int(bool(by_episode_a[episode_id]["static_collision"])),
                "timeout_delta": int(bool(by_episode_b[episode_id]["timeout"])) - int(bool(by_episode_a[episode_id]["timeout"])),
                "path_length_delta": float(by_episode_b[episode_id]["robot_path_length"]) - float(by_episode_a[episode_id]["robot_path_length"]),
                "progress_delta": float(by_episode_b[episode_id]["total_progress"]) - float(by_episode_a[episode_id]["total_progress"]),
                "min_dynamic_clearance_delta": float(by_episode_b[episode_id]["min_dynamic_clearance"]) - float(by_episode_a[episode_id]["min_dynamic_clearance"]),
                "min_static_clearance_delta": float(by_episode_b[episode_id]["min_static_clearance"]) - float(by_episode_a[episode_id]["min_static_clearance"]),
                "stop_fraction_delta": float(by_episode_b[episode_id]["stop_fraction"]) - float(by_episode_a[episode_id]["stop_fraction"]),
            }
            for episode_id in shared_ids
        ]
    return payload


def _print_terminal_report(payload: dict[str, Any]) -> None:
    run_a = dict(payload["run_a"])
    run_b = dict(payload["run_b"])
    print(f"{run_a['label']} vs {run_b['label']}")
    for key in SELECTED_AGGREGATE_METRICS:
        block = dict(payload["aggregate_metrics"][key])
        print(
            f"{key}: "
            f"{run_a['label']}={block['run_a']:.6f} "
            f"{run_b['label']}={block['run_b']:.6f} "
            f"delta={block['delta_b_minus_a']:.6f}"
        )


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = build_comparison_payload(
        run_a=Path(args.run_a),
        run_b=Path(args.run_b),
        label_a=str(args.label_a),
        label_b=str(args.label_b),
    )
    output_dir = Path(args.output_dir)
    save_json(output_dir / "comparison_summary.json", json_safe(payload))
    _print_terminal_report(payload)


if __name__ == "__main__":
    main()
