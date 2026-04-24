# V4 Structured Dynamics Training

This training code learns a structured recurrent dynamics model for dynamic obstacles. It is not policy learning, not controller code, and not a LiDAR-first model.

The primary inputs are:

```text
dynamic_obstacles_robot_t
dynamic_obstacle_rel_vel_robot_t
dynamic_obstacle_radii
action_index, v_t, omega_t
goal_t when enabled
```

LiDAR can be enabled for auxiliary context, but `use_lidar` defaults to `false`.

## Target

The primary target is robot-relative delta:

```text
delta_rel_t = dynamic_obstacles_robot_t1 - dynamic_obstacles_robot_t
```

World-frame obstacle deltas are not used as the primary training target.

Radii are constant per episode. The dataset returns `radii` as `(N_dyn,)` per sample and the model broadcasts them across time.

## Model

`models/structured_dynamics_model.py` implements a simple GRU model:

- per-obstacle MLP over `[x_rel, y_rel, vx_rel, vy_rel, radius]`
- fixed-N flattening across obstacles
- action embedding plus continuous action `[v_t, omega_t]`
- optional goal and optional LiDAR branches
- GRU temporal core with hidden state reset per sequence/batch
- heads for obstacle delta, dynamic risk, and three auxiliary collision logits

There are no transformers or attention layers.

## Delta Scaling

Obstacle deltas are small, so training uses scaled deltas:

```text
target_delta_scaled = target_delta / delta_scale
loss_delta = Huber(pred_delta_rel_scaled, target_delta_scaled)
pred_delta_rel = pred_delta_rel_scaled * delta_scale
```

Metrics and evaluation report unscaled delta MAE/RMSE.

Default `delta_scale` is `0.05`.

## One-Step and Rollout Supervision

One-step training uses each row's local `*_t1` fields directly because every row already stores the post-action target for its `t` state. For contiguous valid timesteps, row `i`'s `t1` state should correspond to row `i+1`'s `t` state.

Rollout training uses future row supervision while recursively feeding predicted states. In other words, the rollout state is predicted forward, but the loss at each rollout step is still compared against the matching row-local `t -> t1` target.

## Losses

All losses are multiplied by `valid_mask`.

Defaults:

```text
lambda_delta = 1.0
lambda_risk = 0.5
lambda_collision_next = 0.1
lambda_collision_within_3 = 0.1
lambda_collision_within_5 = 0.05
lambda_rollout = 0.5
lambda_vel = 0.1
```

Risk is predicted with a sigmoid, so `pred_risk` is always in `[0, 1]`.

Collision heads are auxiliary BCE-with-logits losses. Positive class weights can be auto-estimated from the training split.

## Rollout Training

One-step training is the default and should work first.

Optional rollout training uses predicted state after the seed. It does not use ground-truth velocity after the first step. The update order is intentionally explicit:

```text
x_new = x_prev + pred_delta
v_pred = (x_new - x_prev) / (dt + 1e-6)
compute masked rollout loss
x_prev = x_new.detach()
v_prev = v_pred.detach()
```

Detaching between rollout steps limits graph length and helps avoid exploding gradients.

## Weighted Sampler

The training sampler biases sequence selection without dropping data:

- downweights high stop-fraction windows
- upweights dynamic events, high risk, low clearance, and collision labels
- normalizes exactly with `weights = weights / weights.sum()` before creating the `WeightedRandomSampler`

## Action Sensitivity

Evaluation includes a normalized action sensitivity diagnostic:

```text
variance = Var(pred_delta_rel across actions)
eps = 1e-6
normalized_action_sensitivity = variance / (mean(||pred_delta_rel||) + eps)
```

This helps verify whether the model responds to action input without letting large-motion samples dominate the metric.
The evaluation config provides the matching continuous action values for each discrete action, so this diagnostic varies both the action embedding and `[v_t, omega_t]`.

## Commands

Compile:

```bash
cd /home/stavan/s4/rnl/4proj
python -m compileall .
```

Train:

```bash
cd /home/stavan/s4/rnl/4proj
python train_structured_world_model.py --config configs/structured_world_model.yaml
```

Resume from the default latest checkpoint:

```bash
cd /home/stavan/s4/rnl/4proj
python train_structured_world_model.py \
  --config configs/structured_world_model.yaml \
  --resume
```

Resume from a specific checkpoint:

```bash
cd /home/stavan/s4/rnl/4proj
python train_structured_world_model.py \
  --config configs/structured_world_model.yaml \
  --resume \
  --checkpoint artifacts/structured_world_model/best.pt
```

Use `--fresh_optimizer` with `--resume` to load model weights from the full checkpoint while restarting the optimizer state.

## Resume vs Init Checkpoint

`Resume` continues the same run in the same output directory. It keeps epoch numbering, history, best validation loss, and can restore optimizer state from `latest.pt` or a `--checkpoint` override.

`Init checkpoint` starts a brand-new run in a new output directory from pretrained weights. It resets local epoch numbering to `epoch_001`, starts with fresh history and `best_val_loss = inf`, and writes `init_source.json` so Stage-2 provenance stays with the new run. Stage-1 files remain untouched.

Example Stage-2 rollout finetuning:

```yaml
training:
  output_dir: ./artifacts/structured_world_model_rollout_from_epoch007
  init_checkpoint: ./artifacts/structured_world_model/epoch_007.pt
  init_load_optimizer_state: false
  learning_rate: 0.0001

rollout:
  enabled: true
  eval_enabled: true
  horizon: 3
```

In this setup:

- Stage 1 output dir: `./artifacts/structured_world_model`
- Stage 2 output dir: `./artifacts/structured_world_model_rollout_from_epoch007`
- Stage 2 init checkpoint: `./artifacts/structured_world_model/epoch_007.pt`

Evaluate:

```bash
cd /home/stavan/s4/rnl/4proj
python eval_structured_world_model.py \
  --config configs/structured_world_model.yaml \
  --checkpoint artifacts/structured_world_model/best.pt
```

Outputs are written under:

```text
artifacts/structured_world_model/
```

Expected files:

```text
best.pt
best_weights.pt
epoch_001.pt
epoch_001_weights.pt
latest.pt
latest_weights.pt
history.json
config_snapshot.json
eval_metrics.json
```

`best.pt` and `latest.pt` are full training checkpoints with model state, optimizer state, config, metrics, epoch, and best validation loss. `best_weights.pt` and `latest_weights.pt` are weights-only `state_dict` files saved alongside the corresponding full checkpoint.

## Dependencies

Training and evaluation require PyTorch and PyYAML. This shell may not have them installed yet, so `compileall` can pass while train/eval still need dependency installation.
