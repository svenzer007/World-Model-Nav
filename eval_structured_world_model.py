"""Evaluate trained structured world models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.sequence_dataset import StructuredSequenceDataset, list_episode_paths, split_episode_paths
from models import BASELINE_MODEL_TYPE, VERSION_C_MODEL_TYPE, build_model, resolve_model_type
from train_structured_world_model import (
    detach_recurrent_state,
    is_version_c_world_delta,
    load_yaml_config,
    move_batch_to_device,
    resolve_device,
    resolve_model_version,
    save_json,
    version_c_prev_context,
)


def compute_action_sensitivity(
    model: nn.Module,
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


def compute_rollout_delta_metrics_baseline(
    model: nn.Module,
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
        hidden = detach_recurrent_state(hidden)
    return {
        "rollout_delta_mae": abs_sum / max(count, 1.0),
        "rollout_delta_rmse": (sq_sum / max(count, 1.0)) ** 0.5,
    }


def compute_rollout_delta_metrics_version_c(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    horizon: int,
) -> dict[str, float]:
    model_version = str(batch.get("model_version", ""))
    world_delta_only = is_version_c_world_delta(VERSION_C_MODEL_TYPE, model_version)
    max_rollout_index = min(int(horizon), batch["obstacle_pos"].shape[1] - 1)
    if max_rollout_index <= 0:
        return {"rollout_delta_mae": 0.0, "rollout_delta_rmse": 0.0}
    goal = batch.get("goal")
    _, rollout_state = model.forward_step(
        obstacle_pos=batch["obstacle_pos"][:, 0],
        obstacle_vel=batch["obstacle_vel"][:, 0],
        radii=batch["radii"],
        action_index=batch["action_index"][:, 0],
        action_cont=batch["action_cont"][:, 0],
        goal=goal[:, 0] if goal is not None else None,
        lidar=None,
        hidden=None,
        use_posterior=True,
    )
    prev_context = version_c_prev_context(model, batch, 0)
    abs_sum = 0.0
    sq_sum = 0.0
    vel_abs_sum = 0.0
    vel_sq_sum = 0.0
    count = 0.0
    for step_index in range(1, max_rollout_index + 1):
        outputs, rollout_state = model.forward_step(
            obstacle_pos=None,
            obstacle_vel=None,
            radii=batch["radii"],
            action_index=batch["action_index"][:, step_index],
            action_cont=batch["action_cont"][:, step_index],
            goal=goal[:, step_index] if goal is not None else None,
            lidar=None,
            hidden=rollout_state,
            prev_context=prev_context,
            use_posterior=False,
        )
        delta_output_key = "pred_delta_world" if world_delta_only and "pred_delta_world" in outputs else "pred_delta_rel"
        delta_target_key = "delta_world" if world_delta_only else "delta_rel"
        error = outputs[delta_output_key] - batch[delta_target_key][:, step_index]
        mask = batch["valid_mask"][:, step_index].to(error.dtype)[:, None, None]
        abs_sum += float(torch.sum(torch.abs(error) * mask).detach().cpu())
        sq_sum += float(torch.sum((error**2) * mask).detach().cpu())
        count += float(torch.sum(mask).detach().cpu()) * error.shape[-2] * error.shape[-1]
        if not world_delta_only:
            vel_error = outputs["pred_next_rel_vel"] - batch["target_next_obstacle_vel"][:, step_index]
            vel_abs_sum += float(torch.sum(torch.abs(vel_error) * mask).detach().cpu())
            vel_sq_sum += float(torch.sum((vel_error**2) * mask).detach().cpu())
        prev_context = version_c_prev_context(model, batch, step_index)
        rollout_state = detach_recurrent_state(rollout_state)
    metrics = {
        "rollout_delta_mae": abs_sum / max(count, 1.0),
        "rollout_delta_rmse": (sq_sum / max(count, 1.0)) ** 0.5,
    }
    if not world_delta_only:
        metrics["rollout_vel_mae"] = vel_abs_sum / max(count, 1.0)
        metrics["rollout_vel_rmse"] = (vel_sq_sum / max(count, 1.0)) ** 0.5
    return metrics


def compute_rollout_delta_metrics(
    model_type: str,
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    horizon: int,
) -> dict[str, float]:
    if model_type == VERSION_C_MODEL_TYPE:
        return compute_rollout_delta_metrics_version_c(model, batch, horizon)
    return compute_rollout_delta_metrics_baseline(model, batch, horizon)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/structured_world_model_version_c_stage2.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{args.checkpoint} does not look like a full training checkpoint")
    model_config = dict(checkpoint.get("config", config).get("model", {}))
    model_type = str(checkpoint.get("model_type") or resolve_model_type(model_config))
    if resolve_model_type(model_config) != model_type:
        model_config["model_type"] = model_type
    data_cfg = config["data"]
    training_cfg = config["training"]
    eval_cfg = config.get("evaluation", {})
    threshold = float(args.threshold if args.threshold is not None else eval_cfg.get("threshold", 0.5))
    rollout_horizon = int(checkpoint.get("active_rollout_horizon", config.get("rollout", {}).get("horizon", 3)))

    paths = list_episode_paths(data_cfg["data_dir"])
    _, val_paths = split_episode_paths(paths, float(data_cfg.get("train_fraction", 0.9)))
    dataset = StructuredSequenceDataset(
        val_paths,
        sequence_length=int(data_cfg.get("sequence_length", 20)),
        stride=int(data_cfg.get("stride", 1)),
        use_goal=bool(model_config.get("use_goal", True)),
        use_lidar=bool(model_config.get("use_lidar", False)),
        model_type=model_type,
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
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    delta_abs = 0.0
    delta_sq = 0.0
    delta_count = 0.0
    vel_abs = 0.0
    vel_sq = 0.0
    vel_count = 0.0
    sensitivity_sum = 0.0
    sensitivity_batches = 0
    rollout_totals: dict[str, float] = {}
    rollout_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False, dynamic_ncols=True):
            batch = move_batch_to_device(batch, device)
            batch["model_version"] = resolve_model_version(model_type, model_config)
            outputs = model.forward_sequence(
                obstacle_pos=batch["obstacle_pos"],
                obstacle_vel=batch["obstacle_vel"],
                radii=batch["radii"],
                action_index=batch["action_index"],
                action_cont=batch["action_cont"],
                goal=batch.get("goal"),
                lidar=batch.get("lidar"),
            )
            world_delta_only = is_version_c_world_delta(model_type, str(batch["model_version"]))
            delta_output_key = "pred_delta_world" if world_delta_only and "pred_delta_world" in outputs else "pred_delta_rel"
            delta_target_key = "delta_world" if world_delta_only else "delta_rel"
            delta_error = outputs[delta_output_key] - batch[delta_target_key]
            delta_mask = batch["valid_mask"][:, :, None, None].to(delta_error.dtype)
            delta_abs += float(torch.sum(torch.abs(delta_error) * delta_mask).detach().cpu())
            delta_sq += float(torch.sum((delta_error**2) * delta_mask).detach().cpu())
            delta_count += float(torch.sum(delta_mask).detach().cpu()) * delta_error.shape[-2] * delta_error.shape[-1]
            if not world_delta_only:
                vel_error = outputs["pred_next_rel_vel"] - batch["target_next_obstacle_vel"]
                vel_mask = batch["valid_mask"][:, :, None, None].to(vel_error.dtype)
                vel_abs += float(torch.sum(torch.abs(vel_error) * vel_mask).detach().cpu())
                vel_sq += float(torch.sum((vel_error**2) * vel_mask).detach().cpu())
                vel_count += float(torch.sum(vel_mask).detach().cpu()) * vel_error.shape[-2] * vel_error.shape[-1]

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
                rollout = compute_rollout_delta_metrics(model_type, model, batch, rollout_horizon)
                for key, value in rollout.items():
                    rollout_totals[key] = rollout_totals.get(key, 0.0) + float(value)
                rollout_batches += 1

    metrics: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "model_type": model_type,
        "model_version": checkpoint.get("model_version", resolve_model_version(model_type, model_config)),
        "active_rollout_horizon": rollout_horizon,
        "num_val_sequences": len(dataset),
        "threshold": threshold,
        "delta_mae": delta_abs / max(delta_count, 1.0),
        "delta_rmse": (delta_sq / max(delta_count, 1.0)) ** 0.5,
        "normalized_action_sensitivity": sensitivity_sum / max(sensitivity_batches, 1),
        "action_sensitivity_eps": 1e-6,
    }
    if vel_count > 0.0:
        metrics["vel_mae"] = vel_abs / max(vel_count, 1.0)
        metrics["vel_rmse"] = (vel_sq / max(vel_count, 1.0)) ** 0.5
    if rollout_batches:
        for key, value in rollout_totals.items():
            metrics[key] = value / rollout_batches

    output_path = Path(args.output) if args.output else Path(eval_cfg.get("output_path", "./artifacts/structured_world_model/eval_metrics.json"))
    save_json(output_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
