# V4 Policy Comparison Simulator

This runner compares two closed-loop controllers on identical sampled V4 scenarios:

- `baseline_structured`: analytic robot rollout plus true cloned dynamic-obstacle rollout
- `learned_structured`: analytic robot rollout plus Version-C world-model-predicted obstacle world rollout, transformed into the rolled-out robot frame for scoring

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
- Learned obstacle rollout state: world frame
- Dynamic collision/clearance: robot-relative frame after transforming predicted obstacle world positions into the rolled-out robot frame

The active `version_c_world_delta_v3` controller keeps obstacle prediction in the
world frame and derives robot-relative geometry analytically for dynamic scoring.

## Learned Rollout

The learned controller loads a full structured-dynamics checkpoint, rejects `use_lidar=true`, and allows `use_goal=true`.

Per rollout step:

```text
pos_world_next = pos_world_current + pred_delta_world
vel_world_next = (pos_world_next - pos_world_current) / dt
pos_rel_next = transform(pos_world_next, rolled_robot_pose)
```

Predicted obstacle world positions are rolled forward by the learned model.
At each candidate rollout step they are transformed into the rolled robot frame,
and dynamic collision checks use that derived robot-relative geometry.

Hidden-state rules:

- `h_real` is carried only across real executed simulator steps
- each candidate gets a safe clone of `h_real`
- rejected candidates never mutate `h_real`
- `commit_step(...)` updates `h_real` only after execution using the true observed transition

Hidden-state cloning helper supports:

- `None`
- tensor hidden state
- tuple/list of tensors, including `(h_tensor,)`

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

- predicted obstacle positions over horizon in robot-frame scoring coordinates
- predicted relative velocities in robot-frame scoring coordinates
- predicted dynamic clearances
- true horizon obstacle positions for analysis
- true dynamic clearances for analysis
- drift `||pos_rel_pred - pos_rel_true||`

The only intended difference between controllers is future dynamic-obstacle rollout source:

- `baseline_structured`: true cloned dynamic-obstacle rollout
- `learned_structured`: Version-C world-model-predicted future obstacle world rollout, transformed into robot-relative geometry for scoring

The learned controller still maintains real hidden state only through true executed transitions in `commit_step(...)`.

## Smoke Run

```bash
cd /home/stavan/s4/rnl/world_model
python run_policy_comparison_v4.py \
  --num_episodes 2 \
  --max_steps 80 \
  --horizon 10 \
  --checkpoint artifacts/structured_world_model_version_c_stage2/best.pt \
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
