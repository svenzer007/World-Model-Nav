"""Evaluate a trained V4 structured dynamics model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.sequence_dataset import StructuredSequenceDataset, list_episode_paths, split_episode_paths
from models import StructuredDynamicsConfig, StructuredDynamicsModel
from train_structured_world_model import load_yaml_config, move_batch_to_device, resolve_device, save_json


def binary_counts(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, threshold: float) -> dict[str, float]:
    pred = (torch.sigmoid(logits) >= threshold).to(torch.bool)
    truth = (target >= 0.5).to(torch.bool)
    valid = mask.to(torch.bool)
    return {
        "tp": float(torch.sum(pred & truth & valid).detach().cpu()),
        "fp": float(torch.sum(pred & (~truth) & valid).detach().cpu()),
        "tn": float(torch.sum((~pred) & (~truth) & valid).detach().cpu()),
        "fn": float(torch.sum((~pred) & truth & valid).detach().cpu()),
    }


def metrics_from_counts(counts: Mapping[str, float], prefix: str) -> dict[str, float]:
    tp = float(counts.get("tp", 0.0))
    fp = float(counts.get("fp", 0.0))
    fn = float(counts.get("fn", 0.0))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_tp": tp,
        f"{prefix}_fp": fp,
        f"{prefix}_fn": fn,
    }


def accumulate_count_totals(total: dict[str, dict[str, float]], name: str, counts: dict[str, float]) -> None:
    bucket = total.setdefault(name, {})
    for key, value in counts.items():
        bucket[key] = bucket.get(key, 0.0) + float(value)


def compute_action_sensitivity(
    model: StructuredDynamicsModel,
    batch: Mapping[str, torch.Tensor],
    action_vocab_size: int,
    action_cont_values: list[list[float]] | None = None,
) -> torch.Tensor:
    """Return normalized delta variance across all discrete actions with eps=1e-6."""
    batch_size, seq_len = batch["action_index"].shape
    preds = []
    for action_index in range(action_vocab_size):
        actions = torch.full_like(batch["action_index"], int(action_index))
        if action_cont_values is None:
            action_cont = batch["action_cont"].clone()
        else:
            value = torch.tensor(action_cont_values[action_index], dtype=batch["action_cont"].dtype, device=batch["action_cont"].device)
            action_cont = value[None, None, :].expand(batch_size, seq_len, -1)
        outputs = model.forward_sequence(
            obstacle_pos=batch["obstacle_pos"],
            obstacle_vel=batch["obstacle_vel"],
            radii=batch["radii"],
            action_index=actions,
            action_cont=action_cont,
            goal=batch.get("goal"),
            lidar=batch.get("lidar"),
        )
        preds.append(outputs["pred_delta_rel"])
    stacked = torch.stack(preds, dim=0)
    variance = torch.var(stacked, dim=0, unbiased=False).mean(dim=(-1, -2))
    mean_norm = torch.linalg.norm(stacked.mean(dim=0), dim=-1).mean(dim=-1)
    eps = 1e-6
    normalized = variance / (mean_norm + eps)
    valid = batch["valid_mask"].to(normalized.dtype)
    return torch.sum(normalized * valid) / torch.clamp(torch.sum(valid), min=1.0)


def compute_rollout_delta_metrics(
    model: StructuredDynamicsModel,
    batch: Mapping[str, torch.Tensor],
    horizon: int,
) -> dict[str, float]:
    max_steps = min(int(horizon), batch["obstacle_pos"].shape[1] - 1)
    if max_steps <= 0:
        return {"rollout_delta_mae": 0.0, "rollout_delta_rmse": 0.0}
    x_prev = batch["obstacle_pos"][:, 0]
    v_prev = batch["obstacle_vel"][:, 0]
    hidden = None
    abs_sum = 0.0
    sq_sum = 0.0
    count = 0.0
    for step_index in range(max_steps):
        goal = batch.get("goal")
        lidar = batch.get("lidar")
        outputs, hidden = model.forward_step(
            obstacle_pos=x_prev,
            obstacle_vel=v_prev,
            radii=batch["radii"],
            action_index=batch["action_index"][:, step_index],
            action_cont=batch["action_cont"][:, step_index],
            goal=goal[:, step_index] if goal is not None else None,
            lidar=lidar[:, step_index] if lidar is not None else None,
            hidden=hidden,
        )
        pred_delta = outputs["pred_delta_rel"]
        x_new = x_prev + pred_delta
        v_pred = (x_new - x_prev) / (batch["dt"][:, None, None] + 1e-6)
        error = pred_delta - batch["delta_rel"][:, step_index]
        mask = batch["valid_mask"][:, step_index].to(error.dtype)[:, None, None]
        abs_sum += float(torch.sum(torch.abs(error) * mask).detach().cpu())
        sq_sum += float(torch.sum((error**2) * mask).detach().cpu())
        count += float(torch.sum(mask).detach().cpu()) * error.shape[-2] * error.shape[-1]
        x_prev = x_new.detach()
        v_prev = v_pred.detach()
        if hidden is not None:
            hidden = hidden.detach()
    return {
        "rollout_delta_mae": abs_sum / max(count, 1.0),
        "rollout_delta_rmse": (sq_sum / max(count, 1.0)) ** 0.5,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/structured_world_model.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("config", config)["model"]
    data_cfg = config["data"]
    training_cfg = config["training"]
    eval_cfg = config.get("evaluation", {})
    threshold = float(args.threshold if args.threshold is not None else eval_cfg.get("threshold", 0.5))

    paths = list_episode_paths(data_cfg["data_dir"])
    _, val_paths = split_episode_paths(paths, float(data_cfg.get("train_fraction", 0.9)))
    dataset = StructuredSequenceDataset(
        val_paths,
        sequence_length=int(data_cfg.get("sequence_length", 20)),
        stride=int(data_cfg.get("stride", 1)),
        use_goal=bool(model_config.get("use_goal", True)),
        use_lidar=bool(model_config.get("use_lidar", False)),
    )
    if len(dataset) == 0:
        raise ValueError("Validation dataset has no sequences")
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )

    device = resolve_device(str(training_cfg.get("device", "auto")))
    model = StructuredDynamicsModel(StructuredDynamicsConfig(**model_config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    delta_abs = 0.0
    delta_sq = 0.0
    delta_count = 0.0
    risk_abs = 0.0
    risk_sq = 0.0
    risk_count = 0.0
    count_totals: dict[str, dict[str, float]] = {}
    sensitivity_sum = 0.0
    sensitivity_batches = 0
    rollout_totals: dict[str, float] = {}
    rollout_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False, dynamic_ncols=True):
            batch = move_batch_to_device(batch, device)
            outputs = model.forward_sequence(
                obstacle_pos=batch["obstacle_pos"],
                obstacle_vel=batch["obstacle_vel"],
                radii=batch["radii"],
                action_index=batch["action_index"],
                action_cont=batch["action_cont"],
                goal=batch.get("goal"),
                lidar=batch.get("lidar"),
            )
            delta_error = outputs["pred_delta_rel"] - batch["delta_rel"]
            delta_mask = batch["valid_mask"][:, :, None, None].to(delta_error.dtype)
            delta_abs += float(torch.sum(torch.abs(delta_error) * delta_mask).detach().cpu())
            delta_sq += float(torch.sum((delta_error**2) * delta_mask).detach().cpu())
            delta_count += float(torch.sum(delta_mask).detach().cpu()) * delta_error.shape[-2] * delta_error.shape[-1]

            risk_error = outputs["pred_risk"] - batch["risk"]
            risk_mask = batch["valid_mask"].to(risk_error.dtype)
            risk_abs += float(torch.sum(torch.abs(risk_error) * risk_mask).detach().cpu())
            risk_sq += float(torch.sum((risk_error**2) * risk_mask).detach().cpu())
            risk_count += float(torch.sum(risk_mask).detach().cpu())

            accumulate_count_totals(
                count_totals,
                "collision_next",
                binary_counts(outputs["collision_next_logits"], batch["collision_next"], batch["valid_mask"], threshold),
            )
            accumulate_count_totals(
                count_totals,
                "collision_within_3",
                binary_counts(
                    outputs["collision_within_3_logits"], batch["collision_within_3"], batch["valid_mask"], threshold
                ),
            )
            accumulate_count_totals(
                count_totals,
                "collision_within_5",
                binary_counts(
                    outputs["collision_within_5_logits"], batch["collision_within_5"], batch["valid_mask"], threshold
                ),
            )

            sensitivity_sum += float(
                compute_action_sensitivity(
                    model,
                    batch,
                    int(model_config.get("action_vocab_size", 6)),
                    eval_cfg.get("action_cont_values"),
                )
                .detach()
                .cpu()
            )
            sensitivity_batches += 1

            if bool(config.get("rollout", {}).get("eval_enabled", True)):
                rollout = compute_rollout_delta_metrics(model, batch, int(config.get("rollout", {}).get("horizon", 3)))
                for key, value in rollout.items():
                    rollout_totals[key] = rollout_totals.get(key, 0.0) + float(value)
                rollout_batches += 1

    metrics: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "num_val_sequences": len(dataset),
        "threshold": threshold,
        "delta_mae": delta_abs / max(delta_count, 1.0),
        "delta_rmse": (delta_sq / max(delta_count, 1.0)) ** 0.5,
        "risk_mae": risk_abs / max(risk_count, 1.0),
        "risk_mse": risk_sq / max(risk_count, 1.0),
        "normalized_action_sensitivity": sensitivity_sum / max(sensitivity_batches, 1),
        "action_sensitivity_eps": 1e-6,
    }
    for name, counts in count_totals.items():
        metrics.update(metrics_from_counts(counts, name))
    if rollout_batches:
        for key, value in rollout_totals.items():
            metrics[key] = value / rollout_batches

    output_path = Path(args.output) if args.output else Path(eval_cfg.get("output_path", "./artifacts/structured_world_model/eval_metrics.json"))
    save_json(output_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
