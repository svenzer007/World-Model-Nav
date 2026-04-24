# V4 Policy Comparison Simulator

This runner compares two closed-loop controllers on identical sampled V4 scenarios:

- `baseline_structured`: analytic robot rollout plus true cloned dynamic-obstacle rollout
- `learned_structured`: analytic robot rollout plus V4 GRU-predicted robot-relative obstacle rollout

Both controllers use the same repeated-action candidate set:

- `stop`
- `forward`
- `forward_left`
- `forward_right`
- `slow_left`
- `slow_right`

Each step evaluates every action repeated for `H` rollout steps, executes only the first action, and replans on the next simulator step.

Policy comparison now mirrors the dataset expert's feasibility-first selection rule:

- prefer minimum-cost feasible non-stop action
- else minimum-cost feasible action
- else least-bad fallback over all actions

Feasibility uses the inflated planning map together with static and dynamic collision checks over the rollout horizon.

## Score

Both controllers now use the same expert-aligned rollout score:

```text
total_cost =
  w_progress * goal_progress_cost
  + w_heading * heading_cost
  + w_path * path_cost
  + w_velocity * velocity_cost
  + clearance_cost
  + static_collision_penalty if not feasible else 0
```

Where:

- `goal_progress_cost`: rollout endpoint distance to `current_subgoal_world`
- `heading_cost`: rollout endpoint heading alignment to that same `current_subgoal_world`
- `path_cost`: path distance plus path tangent penalty
- `velocity_cost`: expert-style low-speed penalty `max(0, 0.4 - v)`
- `clearance_cost`: `1 / max(min_combined_clearance, min_clearance_clip)`
- `min_combined_clearance`: minimum over rollout of `min(static_clearance, dynamic_clearance)`

Separate static and dynamic clearances are still logged for analysis, but action scoring now uses the single expert-style combined-clearance term. A non-feasible candidate gets exactly one infeasibility penalty. For backward-compatible CLI behavior that penalty uses `static_collision_penalty`; `dynamic_collision_penalty` is still accepted and logged, but it is not used in the expert-aligned total-cost formula.

The default comparison horizon is `10`, which better matches the dataset expert's effective rollout horizon.

## Frame Rules

- Robot rollout: world frame
- Static collision/clearance: world frame
- Learned obstacle rollout: robot-relative frame
- Dynamic collision/clearance: robot-relative frame

The learned controller never mixes world-frame obstacle positions into learned dynamic collision checks.

## Learned Rollout

The learned controller loads a full structured-dynamics checkpoint, rejects `use_lidar=true`, and allows `use_goal=true`.

Per rollout step:

```text
pos_rel_next = pos_rel_current + pred_delta_rel
vel_rel_next = (pos_rel_next - pos_rel_current) / dt
```

Predicted relative positions are interpreted relative to the rolled robot pose at that rollout step. Dynamic collision checks use those same predicted robot-frame positions directly.

Hidden-state rules:

- `h_real` is carried only across real executed simulator steps
- each candidate gets a safe clone of `h_real`
- rejected candidates never mutate `h_real`
- `commit_step(...)` updates `h_real` only after execution using the true observed transition

Hidden-state cloning helper supports:

- `None`
- tensor hidden state
- tuple/list of tensors, including `(h_tensor,)`

The learned controller's initial relative-velocity construction now explicitly matches the training convention:
obstacle world velocity minus zero robot translational velocity, then rotated into robot frame.

## Obstacle Ordering

The learned controller fixes obstacle ordering by obstacle ID at episode start and reorders later observations to that canonical order. Missing, duplicated, or inconsistent IDs raise a clear error instead of silently corrupting per-obstacle dynamics.

## Logging

Both controllers log per step:

- chosen action
- candidate scores
- score breakdown
- robot rollout poses
- static collision flags
- dynamic collision flags
- static clearance sequences
- dynamic clearance sequences

The learned controller also logs:

- predicted obstacle positions over horizon
- predicted relative velocities
- predicted dynamic clearances
- true horizon obstacle positions for analysis
- true dynamic clearances for analysis
- drift `||pos_rel_pred - pos_rel_true||`
- `pred_risk`
- collision-head probabilities

The learned heads are logging-only and do not drive decisions.

The only intended difference between controllers is future dynamic-obstacle rollout source:

- `baseline_structured`: true cloned dynamic-obstacle rollout
- `learned_structured`: GRU-predicted future robot-relative obstacle rollout

The learned controller still maintains real hidden state only through true executed transitions in `commit_step(...)`.

## Smoke Run

```bash
cd /home/stavan/s4/rnl/4proj
python run_policy_comparison_v4.py \
  --num_episodes 2 \
  --max_steps 80 \
  --horizon 10 \
  --checkpoint artifacts/structured_world_model_rollout_from_epoch_10/best.pt \
  --output_dir artifacts/policy_comparison_v4_smoke \
  --save_previews
```

If your local checkpoint directory name differs, point `--checkpoint` at the available `best.pt`.

## Debug Guidance

Two useful checks:

1. Plot predicted vs true obstacle trajectories in robot frame over the rollout horizon.
2. Plot mean `||pred - true||` drift versus horizon step.

Interpretation:

- drift exploding with horizon suggests model rollout weakness
- small drift with poor avoidance suggests controller/scoring weakness
