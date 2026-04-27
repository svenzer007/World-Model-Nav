# World-Model Dynamic State Data

This branch keeps the Version-C / RSSM model family, but the practical dynamic-obstacle state is now defined around robot-relative positions and robot-relative finite-difference velocities. This note documents both the pre-cleanup audit and the cleaned-up practical representation used by the learned-model path.

See [README_v4_training.md](/home/stavan/s4/rnl/world_model/README_v4_training.md) for the stage-1 / stage-2 RSSM training path.

## Current practical fields

Per step, the simulator stores:

```text
dynamic_obstacle_ids
dynamic_obstacles_world_t
dynamic_obstacles_world_t1
dynamic_obstacle_delta_world_t
dynamic_obstacles_robot_t
dynamic_obstacles_robot_t1
dynamic_obstacle_delta_robot_t
dynamic_obstacle_radii
dynamic_obstacle_vel_world_t
dynamic_obstacle_vel_robot_t
dynamic_obstacle_rel_vel_robot_t
dynamic_obstacle_rel_vel_robot_fd_t
dynamic_obstacle_rel_vel_robot_fd_t1
```

The key semantics are:

- `dynamic_obstacles_robot_t`: obstacle position relative to the robot origin, expressed in the robot frame at time `t`.
- `dynamic_obstacles_robot_t1`: obstacle position relative to the robot origin, expressed in the robot frame at time `t+1`.
- `dynamic_obstacle_delta_robot_t = dynamic_obstacles_robot_t1 - dynamic_obstacles_robot_t`.
- `dynamic_obstacle_rel_vel_robot_fd_t = dynamic_obstacle_delta_robot_t / dt`.
- `dynamic_obstacle_rel_vel_robot_fd_t1`: next-step finite-difference velocity target used by the practical model path. Newly generated datasets store it directly; older datasets synthesize it in the loader.

Legacy fields are still present for compatibility and inspection:

- `dynamic_obstacle_vel_world_t = (dynamic_obstacles_world_t1 - dynamic_obstacles_world_t) / dt`
- `dynamic_obstacle_vel_robot_t = R_t dynamic_obstacle_vel_world_t`
- `dynamic_obstacle_rel_vel_robot_t = R_t (v_obs_world - v_robot_world)`

Those legacy velocity fields are no longer the practical learned-model inputs.

## Velocity audit

The important distinction is:

```text
dynamic_obstacle_rel_vel_robot_t = R_t (v_obs_world - v_robot_world)
```

is not the same as:

```text
(dynamic_obstacles_robot_t1 - dynamic_obstacles_robot_t) / dt
```

Why they differ:

- `R_t (v_obs_world - v_robot_world)` subtracts robot translation in world coordinates and rotates the result into the robot axes at time `t`.
- `(dynamic_obstacles_robot_t1 - dynamic_obstacles_robot_t) / dt` differentiates a quantity that is already robot-relative and already expressed in the robot frame, so it also captures robot-frame rotation between `t` and `t+1`.

During turning, those are not equivalent. If the robot rotates in place near a stationary obstacle, the obstacle's robot-relative coordinates change even when `v_obs_world - v_robot_world` is near zero. That turning-induced change appears in the finite-difference robot-frame velocity and is missed by the older translational-relative convention.

This branch previously had the same practical mismatch seen elsewhere:

- stored obstacle positions were robot-relative
- RSSM/controller inputs used `dynamic_obstacle_rel_vel_robot_t`
- rollout-style propagation naturally implied finite-difference robot-relative velocity

The cleaned-up branch now treats finite-difference robot-frame relative velocity as canonical for the learned-model path.

## Model inputs and targets

The practical Version-C / RSSM path now uses:

- input obstacle state per slot: `[x_rel, y_rel, vx_rel_fd, vy_rel_fd, radius]`
- target obstacle delta: `dynamic_obstacle_delta_robot_t`
- target next velocity: `dynamic_obstacle_rel_vel_robot_fd_t1`

The model predicts:

- `pred_delta_rel`
- `pred_next_rel_vel`

This keeps target semantics aligned with rollout propagation:

```text
pos_rel_next = pos_rel_current + pred_delta_rel
vel_rel_next = pred_next_rel_vel
```

and lightly regularizes them with:

```text
pred_next_rel_vel ~= pred_delta_rel / dt
```

## Data loader behavior

New datasets can provide the practical fields directly.

Older `v4_dynamic_state_data_1` datasets remain readable. The loader derives:

```text
dynamic_obstacle_delta_robot_t = dynamic_obstacles_robot_t1 - dynamic_obstacles_robot_t
dynamic_obstacle_rel_vel_robot_fd_t = dynamic_obstacle_delta_robot_t / dt
```

and synthesizes `dynamic_obstacle_rel_vel_robot_fd_t1` from the next row when available, with a row-local finite-difference fallback at the episode end.

So retraining does not require regeneration, although regeneration is cleaner if you want the serialized dataset to carry the new fields explicitly.

## Why `delta_world / dt` is still valid

`dynamic_obstacle_vel_world_t = (world_t1 - world_t) / dt` is still a valid realized world-frame step velocity for this simulator branch. It is the actual finite-difference velocity realized over the executed step, even if it differs from a separate instantaneous velocity snapshot convention.

That does not make it the right practical learned-state velocity here. The issue was not that `delta_world / dt` was invalid, but that the learned branch mixed world/translational-relative notions with robot-relative position targets.

## Edge cases and assumptions

- Identity/order stability:
  - obstacle IDs are fixed and checked per step
  - slot ordering is stable by construction
- Fixed obstacle count:
  - this branch still assumes a fixed `N_dyn`
  - no padding redesign was added
- Episode boundaries:
  - each row stores a valid local `t -> t+1` transition
  - sequence windows stay within an episode
  - synthesized `*_fd_t1` falls back row-locally on terminal rows
- `dt` usage:
  - data generation uses `robot_config.dt`
  - loader/training use stored row `dt` with episode fallback
- Compatibility:
  - newly generated data is marked `v4_dynamic_state_data_2`
  - old datasets remain usable through loader-side derivation
