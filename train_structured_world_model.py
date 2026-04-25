"""Train the V4 structured recurrent dynamic-obstacle model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.sequence_dataset import StructuredSequenceDataset, _load_episode_cached, list_episode_paths, split_episode_paths
from data.weighted_sampler import build_weighted_sampler
from models import StructuredDynamicsConfig, StructuredDynamicsModel


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load structured world-model configs") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Config must be a YAML mapping")
    return payload


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=values.dtype)
    return torch.sum(values * mask) / torch.clamp(torch.sum(mask) * values.shape[-1] if values.ndim > mask.ndim else torch.sum(mask), min=1.0)


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.huber_loss(pred, target, reduction="none")
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=loss.dtype)
    return torch.sum(loss * mask) / torch.clamp(torch.sum(mask) * (loss.numel() / max(mask.numel(), 1)), min=1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (pred - target) ** 2
    return torch.sum(loss * mask.to(loss.dtype)) / torch.clamp(torch.sum(mask), min=1.0)


def masked_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    return torch.sum(loss * mask.to(loss.dtype)) / torch.clamp(torch.sum(mask), min=1.0)


def estimate_pos_weights(dataset: StructuredSequenceDataset, caps: Mapping[str, float]) -> dict[str, float]:
    counts = {
        "collision_next": [0.0, 0.0],
        "collision_within_3": [0.0, 0.0],
        "collision_within_5": [0.0, 0.0],
    }
    step_keys = {
        "collision_next": "dynamic_collision_t1",
        "collision_within_3": "dynamic_collision_within_3",
        "collision_within_5": "dynamic_collision_within_5",
    }
    for window in dataset.windows:
        episode = _load_episode_cached(str(window.episode_path))
        steps = episode["steps"][window.start_step : window.start_step + window.sequence_length]
        for key in counts:
            values = [float(step[step_keys[key]]) for step in steps]
            counts[key][0] += float(sum(values))
            counts[key][1] += float(len(values))
    weights = {}
    for key, (pos, total) in counts.items():
        neg = max(total - pos, 0.0)
        value = neg / max(pos, 1.0)
        cap = float(caps.get(f"{key}_pos_weight_cap", 100.0))
        weights[key] = float(min(max(value, 1.0), cap))
    return weights


def configured_pos_weights(config: Mapping[str, Any], train_dataset: StructuredSequenceDataset) -> dict[str, float]:
    loss_cfg = config.get("loss_weights", {})
    if not bool(loss_cfg.get("auto_pos_weight", True)):
        return {
            "collision_next": float(loss_cfg.get("collision_next_pos_weight", 1.0)),
            "collision_within_3": float(loss_cfg.get("collision_within_3_pos_weight", 1.0)),
            "collision_within_5": float(loss_cfg.get("collision_within_5_pos_weight", 1.0)),
        }
    return estimate_pos_weights(train_dataset, loss_cfg)


def compute_one_step_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    pos_weights: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_cfg = config["loss_weights"]
    delta_scale = float(config["model"].get("delta_scale", 0.05))
    mask = batch["valid_mask"]
    target_delta_scaled = batch["delta_rel"] / delta_scale

    loss_delta = masked_huber(outputs["pred_delta_rel_scaled"], target_delta_scaled, mask)
    dt = batch["dt"][:, None, None, None] + 1e-6
    pred_vel = outputs["pred_delta_rel"] / dt
    target_vel = batch["delta_rel"] / dt
    loss_vel = masked_huber(pred_vel, target_vel, mask)
    loss_risk = masked_mse(outputs["pred_risk"], batch["risk"], mask)
    loss_collision_next = masked_bce(
        outputs["collision_next_logits"], batch["collision_next"], mask, pos_weights.get("collision_next")
    )
    loss_collision_within_3 = masked_bce(
        outputs["collision_within_3_logits"],
        batch["collision_within_3"],
        mask,
        pos_weights.get("collision_within_3"),
    )
    loss_collision_within_5 = masked_bce(
        outputs["collision_within_5_logits"],
        batch["collision_within_5"],
        mask,
        pos_weights.get("collision_within_5"),
    )

    total = (
        float(loss_cfg.get("lambda_delta", 1.0)) * loss_delta
        + float(loss_cfg.get("lambda_vel", 0.1)) * loss_vel
        + float(loss_cfg.get("lambda_risk", 0.5)) * loss_risk
        + float(loss_cfg.get("lambda_collision_next", 0.1)) * loss_collision_next
        + float(loss_cfg.get("lambda_collision_within_3", 0.1)) * loss_collision_within_3
        + float(loss_cfg.get("lambda_collision_within_5", 0.05)) * loss_collision_within_5
    )
    return total, {
        "loss_delta": loss_delta.detach(),
        "loss_vel": loss_vel.detach(),
        "loss_risk": loss_risk.detach(),
        "loss_collision_next": loss_collision_next.detach(),
        "loss_collision_within_3": loss_collision_within_3.detach(),
        "loss_collision_within_5": loss_collision_within_5.detach(),
        "loss_total_one_step": total.detach(),
    }


def compute_rollout_loss(
    model: StructuredDynamicsModel,
    batch: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    pos_weights: Mapping[str, torch.Tensor],
    *,
    active_rollout_horizon: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rollout_cfg = config.get("rollout", {})
    horizon = int(active_rollout_horizon if active_rollout_horizon is not None else rollout_cfg.get("horizon", 3))
    if horizon <= 0 or batch["obstacle_pos"].shape[1] <= 1:
        zero = batch["obstacle_pos"].sum() * 0.0
        return zero, {"loss_rollout": zero.detach()}

    delta_scale = float(config["model"].get("delta_scale", 0.05))
    loss_cfg = config["loss_weights"]
    max_steps = min(horizon, batch["obstacle_pos"].shape[1] - 1)
    x_prev = batch["obstacle_pos"][:, 0]
    v_prev = batch["obstacle_vel"][:, 0]
    hidden = None
    losses: list[torch.Tensor] = []

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

        # Row-local targets are t -> t1 supervision. In contiguous sequence windows,
        # row i's t1 state corresponds to row i+1's t state, so rollout can
        # recursively feed predicted states while supervising against row-local targets.
        target_index = step_index
        mask = batch["valid_mask"][:, target_index]
        target_delta_scaled = batch["delta_rel"][:, target_index] / delta_scale
        step_loss = float(loss_cfg.get("lambda_delta", 1.0)) * masked_huber(
            outputs["pred_delta_rel_scaled"][:, None], target_delta_scaled[:, None], mask[:, None]
        )
        # Rollout velocity targets must use the same finite-difference robot-frame
        # convention as the predicted rollout velocity, especially during turning.
        target_vel = (batch["obstacle_pos"][:, target_index + 1] - batch["obstacle_pos"][:, target_index]) / (
            batch["dt"][:, None, None] + 1e-6
        )
        step_loss = step_loss + float(loss_cfg.get("lambda_vel", 0.1)) * masked_huber(
            v_pred[:, None], target_vel[:, None], mask[:, None]
        )
        step_loss = step_loss + float(loss_cfg.get("lambda_risk", 0.5)) * masked_mse(
            outputs["pred_risk"], batch["risk"][:, target_index], mask
        )
        step_loss = step_loss + float(loss_cfg.get("lambda_collision_next", 0.1)) * masked_bce(
            outputs["collision_next_logits"],
            batch["collision_next"][:, target_index],
            mask,
            pos_weights.get("collision_next"),
        )
        losses.append(step_loss)

        x_prev = x_new.detach()
        v_prev = v_pred.detach()
        if hidden is not None:
            hidden = hidden.detach()

    if not losses:
        zero = batch["obstacle_pos"].sum() * 0.0
        return zero, {"loss_rollout": zero.detach()}
    loss_rollout = torch.stack(losses).mean()
    return loss_rollout, {"loss_rollout": loss_rollout.detach()}


def compute_metrics(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mask = batch["valid_mask"]
    delta_error = outputs["pred_delta_rel"] - batch["delta_rel"]
    while mask.ndim < delta_error.ndim:
        mask = mask.unsqueeze(-1)
    mask_float = mask.to(delta_error.dtype)
    denom = torch.clamp(torch.sum(mask_float) * delta_error.shape[-2] * delta_error.shape[-1], min=1.0)
    delta_mae = torch.sum(torch.abs(delta_error) * mask_float) / denom
    delta_rmse = torch.sqrt(torch.sum((delta_error**2) * mask_float) / denom)
    risk_error = outputs["pred_risk"] - batch["risk"]
    risk_mask = batch["valid_mask"].to(risk_error.dtype)
    risk_mae = torch.sum(torch.abs(risk_error) * risk_mask) / torch.clamp(torch.sum(risk_mask), min=1.0)
    risk_mse = torch.sum((risk_error**2) * risk_mask) / torch.clamp(torch.sum(risk_mask), min=1.0)
    return {
        "delta_mae": delta_mae.detach(),
        "delta_rmse": delta_rmse.detach(),
        "risk_mae": risk_mae.detach(),
        "risk_mse": risk_mse.detach(),
    }


def run_epoch(
    model: StructuredDynamicsModel,
    loader: DataLoader,
    device: torch.device,
    config: Mapping[str, Any],
    pos_weights: Mapping[str, torch.Tensor],
    *,
    active_rollout_horizon: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    progress_label: str | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    total_batches = 0
    iterator = tqdm(loader, desc=progress_label, leave=False, dynamic_ncols=True) if progress_label else loader
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(training):
            outputs = model.forward_sequence(
                obstacle_pos=batch["obstacle_pos"],
                obstacle_vel=batch["obstacle_vel"],
                radii=batch["radii"],
                action_index=batch["action_index"],
                action_cont=batch["action_cont"],
                goal=batch.get("goal"),
                lidar=batch.get("lidar"),
            )
            loss, components = compute_one_step_loss(outputs, batch, config, pos_weights)
            if bool(config.get("rollout", {}).get("enabled", False)):
                rollout_loss, rollout_components = compute_rollout_loss(
                    model,
                    batch,
                    config,
                    pos_weights,
                    active_rollout_horizon=active_rollout_horizon,
                )
                loss = loss + float(config["loss_weights"].get("lambda_rollout", 0.5)) * rollout_loss
                components.update(rollout_components)
            metrics = compute_metrics(outputs, batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip = config["training"].get("grad_clip_norm")
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()

        total_batches += 1
        all_values = {"loss_total": loss.detach(), **components, **metrics}
        for key, value in all_values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        if progress_label:
            iterator.set_postfix(loss=f"{totals['loss_total'] / total_batches:.4f}")

    if total_batches == 0:
        raise ValueError("DataLoader produced no batches")
    return {key: value / total_batches for key, value in sorted(totals.items())}


def save_checkpoint(
    path: str | Path,
    model: StructuredDynamicsModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Mapping[str, Any],
    metrics: Mapping[str, float],
    best_val_loss: float,
    active_rollout_horizon: int,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    weights_output = output.with_name(f"{output.stem}_weights{output.suffix}")
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dict(config),
            "metrics": dict(metrics),
            "best_val_loss": float(best_val_loss),
            "active_rollout_horizon": int(active_rollout_horizon),
        },
        output,
    )
    torch.save(model.state_dict(), weights_output)


def resolve_rollout_curriculum(config: Mapping[str, Any]) -> tuple[list[dict[str, int]], int]:
    rollout_cfg = dict(config.get("rollout", {}))
    curriculum_cfg = rollout_cfg.get("curriculum")
    if not curriculum_cfg:
        horizon = int(rollout_cfg.get("horizon", 0))
        epochs = int(config.get("training", {}).get("epochs", 0))
        return ([{"horizon": horizon, "epochs": epochs}] if horizon > 0 and epochs > 0 else []), epochs
    if not isinstance(curriculum_cfg, list):
        raise TypeError("rollout.curriculum must be a list of {horizon, epochs} mappings")
    phases: list[dict[str, int]] = []
    total_epochs = 0
    for phase in curriculum_cfg:
        if not isinstance(phase, Mapping):
            raise TypeError("Each rollout curriculum phase must be a mapping")
        horizon = int(phase.get("horizon", 0))
        epochs = int(phase.get("epochs", 0))
        if horizon <= 0 or epochs <= 0:
            raise ValueError("Each rollout curriculum phase must have positive horizon and epochs")
        phases.append({"horizon": horizon, "epochs": epochs})
        total_epochs += epochs
    return phases, total_epochs


def active_rollout_horizon_for_epoch(curriculum: list[dict[str, int]], epoch: int, fallback_horizon: int) -> int:
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if not curriculum:
        return int(fallback_horizon)
    running_total = 0
    for phase in curriculum:
        running_total += int(phase["epochs"])
        if epoch <= running_total:
            return int(phase["horizon"])
    return int(curriculum[-1]["horizon"])


def load_checkpoint(
    path: str | Path,
    model: StructuredDynamicsModel,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    load_optimizer: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{path} does not look like a full training checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"])
    if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def load_history(path: str | Path) -> list[dict[str, Any]]:
    history_path = Path(path)
    if not history_path.exists():
        return []
    payload = json.loads(history_path.read_text())
    history = payload.get("history", [])
    if not isinstance(history, list):
        raise TypeError(f"Expected history list in {history_path}")
    return history


def save_init_source(
    path: str | Path,
    init_checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    optimizer_state_loaded: bool,
    new_output_dir: str | Path,
) -> None:
    save_json(
        path,
        {
            "init_checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
            "init_checkpoint_epoch": checkpoint.get("epoch"),
            "init_checkpoint_metrics": checkpoint.get("metrics"),
            "init_checkpoint_path": str(Path(init_checkpoint_path)),
            "new_output_dir": str(Path(new_output_dir)),
            "optimizer_state_loaded": bool(optimizer_state_loaded),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/structured_world_model.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint or --checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for resume")
    parser.add_argument(
        "--fresh_optimizer",
        action="store_true",
        help="When resuming, load model weights but start with a fresh optimizer state",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.config)
    set_seed(int(config.get("seed", 0)))

    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    rollout_cfg = dict(config.get("rollout", {}))
    rollout_enabled = bool(rollout_cfg.get("enabled", False))
    fallback_rollout_horizon = int(rollout_cfg.get("horizon", 0)) if rollout_enabled else 0
    rollout_curriculum, curriculum_total_epochs = resolve_rollout_curriculum(config) if rollout_enabled else ([], 0)
    total_epochs = int(curriculum_total_epochs if rollout_curriculum else training_cfg.get("epochs", 40))
    active_rollout_horizon = active_rollout_horizon_for_epoch(
        rollout_curriculum,
        1,
        fallback_rollout_horizon,
    ) if rollout_enabled else 0

    episode_paths = list_episode_paths(data_cfg["data_dir"])
    train_paths, val_paths = split_episode_paths(episode_paths, float(data_cfg.get("train_fraction", 0.9)))
    train_dataset = StructuredSequenceDataset(
        train_paths,
        sequence_length=int(data_cfg.get("sequence_length", 20)),
        stride=int(data_cfg.get("stride", 1)),
        use_goal=bool(model_cfg.get("use_goal", True)),
        use_lidar=bool(model_cfg.get("use_lidar", False)),
        rollout_horizon=active_rollout_horizon,
    )
    val_dataset = StructuredSequenceDataset(
        val_paths,
        sequence_length=int(data_cfg.get("sequence_length", 20)),
        stride=int(data_cfg.get("stride", 1)),
        use_goal=bool(model_cfg.get("use_goal", True)),
        use_lidar=bool(model_cfg.get("use_lidar", False)),
        rollout_horizon=active_rollout_horizon,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Train/val datasets must both have at least one sequence")

    sampler = None
    shuffle = True
    if bool(config.get("sampler", {}).get("enabled", True)):
        sampler = build_weighted_sampler(train_dataset, config.get("sampler", {}))
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 64)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )

    device = resolve_device(str(training_cfg.get("device", "auto")))
    model = StructuredDynamicsModel(StructuredDynamicsConfig(**model_cfg)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg.get("learning_rate", 3e-4)))
    pos_weights_float = configured_pos_weights(config, train_dataset)
    pos_weights = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in pos_weights_float.items()}

    output_dir = Path(training_cfg.get("output_dir", "./artifacts/structured_world_model"))
    init_checkpoint_value = training_cfg.get("init_checkpoint")
    init_checkpoint_path = Path(init_checkpoint_value) if init_checkpoint_value else None
    init_load_optimizer_state = bool(training_cfg.get("init_load_optimizer_state", False))
    if args.resume and init_checkpoint_path is not None:
        raise ValueError(
            "Cannot use --resume with training.init_checkpoint: --resume continues the same run, "
            "while training.init_checkpoint starts a new run from pretrained weights."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "config_snapshot.json",
        {
            "config": config,
            "resolved_rollout_curriculum": rollout_curriculum,
            "num_train_episodes": len(train_paths),
            "num_val_episodes": len(val_paths),
            "num_train_sequences": len(train_dataset),
            "num_val_sequences": len(val_dataset),
            "collision_pos_weights": pos_weights_float,
        },
    )

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "latest.pt"
    history_path = output_dir / "history.json"
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    start_epoch = 1
    if args.resume:
        history = load_history(history_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {checkpoint_path}")
        checkpoint = load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
            load_optimizer=not bool(args.fresh_optimizer),
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("metrics", {}).get("loss_total", float("inf"))))
        resumed_horizon = active_rollout_horizon_for_epoch(
            rollout_curriculum,
            start_epoch,
            fallback_rollout_horizon,
        ) if rollout_enabled else 0
        print(
            f"Resuming from {checkpoint_path} at epoch {start_epoch} "
            f"({'fresh optimizer' if args.fresh_optimizer else 'optimizer restored'}); "
            f"active rollout horizon={resumed_horizon}.",
            flush=True,
        )
    elif init_checkpoint_path is not None:
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {init_checkpoint_path}")
        checkpoint = load_checkpoint(
            init_checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
            load_optimizer=init_load_optimizer_state,
        )
        save_init_source(
            output_dir / "init_source.json",
            init_checkpoint_path=init_checkpoint_path,
            checkpoint=checkpoint,
            optimizer_state_loaded=init_load_optimizer_state,
            new_output_dir=output_dir,
        )
        print(
            f"Initialized new run from {init_checkpoint_path} at local epoch 1 "
            f"({'optimizer restored' if init_load_optimizer_state else 'fresh optimizer'}); "
            f"starting rollout horizon={active_rollout_horizon}.",
            flush=True,
        )

    previous_rollout_horizon: int | None = (
        active_rollout_horizon_for_epoch(
            rollout_curriculum,
            start_epoch - 1,
            fallback_rollout_horizon,
        )
        if rollout_enabled and start_epoch > 1
        else None
    )
    if start_epoch > total_epochs:
        print(
            f"Checkpoint is already at or beyond configured epochs={total_epochs}; nothing to train.",
            flush=True,
        )
<<<<<<< Updated upstream
        return

    for epoch in range(start_epoch, total_epochs + 1):
=======
        print("TRAINING_COMPLETE", flush=True)
        sys.exit(0)

    end_epoch = total_epochs
    if args.epochs_per_run is not None:
        if int(args.epochs_per_run) <= 0:
            raise ValueError("--epochs-per-run must be positive")
        end_epoch = min(total_epochs, start_epoch + int(args.epochs_per_run) - 1)

    print(
        f"Training epochs {start_epoch} through {end_epoch} of {total_epochs}.",
        flush=True,
    )

    for epoch in range(start_epoch, end_epoch + 1):
>>>>>>> Stashed changes
        active_rollout_horizon = active_rollout_horizon_for_epoch(
            rollout_curriculum,
            epoch,
            fallback_rollout_horizon,
        ) if rollout_enabled else 0
        if previous_rollout_horizon is not None and active_rollout_horizon != previous_rollout_horizon:
            print(
                f"Switching rollout horizon from {previous_rollout_horizon} to {active_rollout_horizon} at epoch {epoch}",
                flush=True,
            )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            config,
            pos_weights,
            active_rollout_horizon=active_rollout_horizon,
            optimizer=optimizer,
            progress_label=f"train {epoch} h={active_rollout_horizon}",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                config,
                pos_weights,
                active_rollout_horizon=active_rollout_horizon,
                optimizer=None,
                progress_label=f"val {epoch} h={active_rollout_horizon}",
            )
        summary = {
            "epoch": epoch,
            "active_rollout_horizon": int(active_rollout_horizon),
            "learning_rate": learning_rate,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(summary)
        print(json.dumps(summary, sort_keys=True))
        save_json(history_path, {"history": history})

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = float(val_metrics["loss_total"])
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, config, val_metrics, best_val_loss, active_rollout_horizon)
        save_checkpoint(
            output_dir / f"epoch_{epoch:03d}.pt",
            model,
            optimizer,
            epoch,
            config,
            val_metrics,
            best_val_loss,
            active_rollout_horizon,
        )
        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, config, val_metrics, best_val_loss, active_rollout_horizon)
        previous_rollout_horizon = active_rollout_horizon

<<<<<<< Updated upstream

=======
    if end_epoch < total_epochs:
        print("CHUNK_COMPLETE_MORE_REMAINING", flush=True)
        sys.exit(10)

    print("TRAINING_COMPLETE", flush=True)
    sys.exit(0)
>>>>>>> Stashed changes
if __name__ == "__main__":
    main()
