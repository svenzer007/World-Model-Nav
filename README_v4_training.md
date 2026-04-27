# World-Model Version-C Training

This branch trains the Version-C / RSSM obstacle world model. It is still the same latent prior/posterior family, but the practical learned-model path is now motion-only:

- obstacle inputs: robot-relative position, robot-relative finite-difference velocity, radius
- obstacle outputs: delta relative position, next-step robot-relative finite-difference velocity
- no risk head
- no collision auxiliary heads

See [README_v4_dynamic_state_data.md](/home/stavan/s4/rnl/world_model/README_v4_dynamic_state_data.md) for the detailed field and frame semantics.

## Model path

The active Version-C model is `structured_obstacle_rssm_vc` with:

- shared per-obstacle encoder over `[x_rel, y_rel, vx_rel_fd, vy_rel_fd, radius]`
- action and optional goal context encoder
- deterministic RSSM-style history update with prior/posterior latent state
- shared per-obstacle decoder heads for:
  - `pred_delta_rel`
  - `pred_next_rel_vel`

The practical cleaned-up config version is:

```text
model_version: version_c_motion_v2
```

Old checkpoints with the previous output definition are intentionally incompatible with this version and are rejected explicitly.

## Losses

Stage 1 and Stage 2 both keep one-step supervision active:

- `loss_delta`
- `loss_next_vel`
- `loss_consistency`
- `loss_latent`

where:

```text
loss_consistency: pred_next_rel_vel ~= pred_delta_rel / dt
```

Stage 2 additionally adds rollout loss on the prior chain:

- rollout delta supervision
- rollout next-velocity supervision
- rollout latent consistency

Risk and collision auxiliaries were removed from the Version-C learned-model path.

## Training stages

Stage 1 is one-step RSSM training only:

```bash
cd /home/stavan/s4/rnl/world_model
python train_structured_world_model.py \
  --config configs/structured_world_model_version_c_stage1.yaml
```

Stage 2 resumes from a Stage-1 checkpoint through `training.init_checkpoint` and adds rollout curriculum:

```bash
cd /home/stavan/s4/rnl/world_model
python train_structured_world_model.py \
  --config configs/structured_world_model_version_c_stage2.yaml
```

Resume the current run in-place:

```bash
cd /home/stavan/s4/rnl/world_model
python train_structured_world_model.py \
  --config configs/structured_world_model_version_c_stage2.yaml \
  --resume
```

The default CLI config is now the stage-1 Version-C config. The default eval config is the stage-2 Version-C config.

## Evaluation

Evaluation now focuses on motion metrics:

- `delta_mae`, `delta_rmse`
- `vel_mae`, `vel_rmse`
- rollout delta/velocity metrics when rollout eval is enabled
- normalized action sensitivity

Example:

```bash
cd /home/stavan/s4/rnl/world_model
python eval_structured_world_model.py \
  --config configs/structured_world_model_version_c_stage2.yaml \
  --checkpoint artifacts/structured_world_model_version_c_stage2/best.pt
```

## Notes

- Existing older datasets remain trainable because the loader derives the new finite-difference velocity fields from stored robot-relative positions.
- If you regenerate data with the updated simulator, the dataset schema becomes `v4_dynamic_state_data_2` and stores the practical motion fields explicitly.
