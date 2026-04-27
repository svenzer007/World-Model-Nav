# V4 Structured World-Model Training

This README covers the practical longer-horizon training path under `/home/stavan/s4/rnl/longer_horizons`.

For the structured dynamic-state semantics and frame definitions, see [README_v4_dynamic_state_data.md](./README_v4_dynamic_state_data.md).

## Current Recommended Branch

The practical redesigned branch is driven by:

- `train_structured_world_model.py`
- `configs/structured_world_model.yaml`

The default config now targets the explicit factorized practical model type:

```yaml
model:
  model_type: factorized_v2
```

This is intentionally checkpoint-incompatible with older joint practical checkpoints.

## Architecture Change

### Old practical architecture: `joint_v1`

The older practical model used:

- per-obstacle feature encoding
- flattening all obstacle embeddings into one joint slot-entangled vector
- one GRU over the joint vector plus action and optional goal context
- one joint output head that predicted all obstacle deltas together
- auxiliary risk and collision heads

This made the dynamic-obstacle block more entangled than needed for the practical branch.

### New practical architecture: `factorized_v2`

The redesigned branch uses:

- a shared per-obstacle encoder applied independently to each obstacle slot
- a small shared context input from action and optional goal
- a lightweight recurrent temporal state over pooled obstacle/context features
- a shared per-obstacle prediction head conditioned on obstacle embedding plus recurrent context

Per-obstacle input:

- `x_rel`
- `y_rel`
- `vx_rel_fd`
- `vy_rel_fd`
- `radius`

Per-obstacle outputs:

- `pred_delta_rel`
- `pred_next_rel_vel_fd`

Removed from the new practical model type:

- risk auxiliary head
- collision-next auxiliary head
- collision-within-3 auxiliary head
- collision-within-5 auxiliary head

The redesign is meant to improve structure and target semantics without materially increasing capacity.

## Losses

### One-step losses

The redesigned practical branch trains on:

- delta position loss
- next-step finite-difference velocity loss
- a soft velocity-position consistency loss

For `model_type: factorized_v2`, the practical branch does not use:

- risk loss
- collision-next loss
- collision-within-3 loss
- collision-within-5 loss
- collision class positive-weight estimation

The consistency loss is lightweight:

```text
pred_next_rel_vel_fd  ~  pred_delta_rel / dt
```

It is only a soft regularizer, not a hard constraint.

### Rollout loss

Rollout training now propagates:

- `x_next = x_prev + pred_delta_rel`
- `v_next = pred_next_rel_vel_fd`

Rollout supervision uses:

- rollout delta error
- rollout next-velocity error
- the same small consistency regularization term

## Config

The default config is:

```bash
/home/stavan/s4/rnl/longer_horizons/configs/structured_world_model.yaml
```

Important defaults:

- `data.data_dir: ./dataset`
- `data.sequence_length: 20`
- `model.model_type: factorized_v2`
- `model.num_dynamic_obstacles: 4`
- `model.use_goal: true`
- `model.use_lidar: false`
- `rollout.enabled: true`
- `rollout.eval_enabled: true`
- `training.output_dir: ./artifacts/structured_world_model_factorized_v2`

The checked-in rollout curriculum remains:

- horizon `3` for `5` epochs
- horizon `6` for `5` epochs
- horizon `8` for `3` epochs
- horizon `10` for `4` epochs

## Dataset Expectations

The redesigned practical path expects the loader to provide:

- `obstacle_pos`
- `obstacle_vel`
- `radii`
- `delta_rel`
- `target_next_obstacle_vel`
- `action_index`
- `action_cont`
- `goal` when enabled

The loader prefers the new finite-difference practical fields from [README_v4_dynamic_state_data.md](./README_v4_dynamic_state_data.md) and derives them on the fly for older datasets when needed.

## Train

Compile first:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python -m compileall .
```

Run training with the default config:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python train_structured_world_model.py --config configs/structured_world_model.yaml
```

Default output directory:

```yaml
training:
  output_dir: ./artifacts/structured_world_model_factorized_v2
```

## Resume

Resume from the default latest checkpoint in the configured output directory:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python train_structured_world_model.py \
  --config configs/structured_world_model.yaml \
  --resume
```

Resume from a specific full checkpoint:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python train_structured_world_model.py \
  --config configs/structured_world_model.yaml \
  --resume \
  --checkpoint /home/stavan/s4/rnl/longer_horizons/artifacts/structured_world_model_factorized_v2/latest.pt
```

## Compatibility

The branch now uses explicit model-type checking.

That means:

- old `joint_v1` checkpoints do not silently load into `factorized_v2`
- new `factorized_v2` checkpoints do not pretend to be old joint checkpoints

Older checkpoints remain recoverable through the legacy model path, but they are intentionally separated from the redesigned practical branch.

## Outputs

Training writes artifacts under `training.output_dir`.

Main files include:

- `epoch_001.pt`, `epoch_002.pt`, ...
- `epoch_001_weights.pt`, `epoch_002_weights.pt`, ...
- `latest.pt`
- `latest_weights.pt`
- `best.pt`
- `best_weights.pt`
- `history.json`
- `config_snapshot.json`
- `init_source.json` when `training.init_checkpoint` is used

Each full checkpoint includes:

- `model_state_dict`
- `optimizer_state_dict`
- `epoch`
- `config`
- `metrics`
- `best_val_loss`
- `active_rollout_horizon`

## Notes

- This README describes the practical factorized redesign, not the separate Version C / RSSM-style branch elsewhere in the repo.
- The first redesign version intentionally prioritizes motion prediction simplicity over keeping low-value auxiliary heads.
