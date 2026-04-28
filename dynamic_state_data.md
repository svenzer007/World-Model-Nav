# Longer-Horizons Dynamic State Data

This document is the authoritative design note for the practical learned-model branch under `/home/stavan/s4/rnl/longer_horizons`.

For training commands and stage-1 / stage-2 workflow, see [README_v4_training.md](/home/stavan/s4/rnl/longer_horizons/README_v4_training.md).

## Current branch audit

Before this redesign, the practical factorized branch used:

- `dynamic_obstacles_robot_t`
- `dynamic_obstacles_robot_t1`
- `dynamic_obstacle_delta_robot_t`
- `dynamic_obstacle_rel_vel_robot_fd_t`
- `dynamic_obstacle_rel_vel_robot_fd_t1`
- `dynamic_obstacle_radii`

The branch also stored world-frame fields:

- `dynamic_obstacles_world_t`
- `dynamic_obstacles_world_t1`
- `dynamic_obstacle_delta_world_t`
- `dynamic_obstacle_vel_world_t`

and legacy robot-frame velocity variants:

- `dynamic_obstacle_vel_robot_t`
- `dynamic_obstacle_rel_vel_robot_t`

The practical factorized model then used robot-frame position and robot-frame finite-difference velocity as its main input state. Rollout training propagated that robot-frame state, and the learned controller also rolled out obstacle state directly in the robot frame.

That made robot-frame semantics central even though obstacle dynamics themselves happen in world coordinates.

## Why move to world-frame prediction

Obstacle dynamics are mostly uncoupled from the robot action. The robot does not physically drive obstacle motion, but the old robot-frame formulation forced the model to learn:

- obstacle world motion
- robot translation
- robot rotation
- the interaction of those terms inside a moving coordinate frame

This is especially awkward in this branch because the action set includes large angular velocity:

- `omega = +1.0`
- `omega = -1.0`

So even when obstacle world velocity is smooth, robot-frame relative velocity can change sharply. That makes robot-frame motion a poor main state variable for a practical learned predictor.

The cleaner decomposition is:

- learned model predicts obstacle motion in world frame
- controller rolls out robot pose analytically under candidate actions
- controller transforms predicted obstacle world positions into the rolled-out robot frame for scoring

This keeps obstacle dynamics in a smoother frame and moves ego-motion geometry back to the analytic controller side.

## Causal velocity-state requirement

The world-frame practical model still uses obstacle velocity as part of its input state, but that velocity must be causal.

At time `t`, the model input velocity is:

```text
v_world_t =
- simulator-provided current obstacle world velocity, for the first step t = 0
- (p_world_t - p_world_t-1) / dt, for t >= 1
```

This is the correct practical state definition because:

- it matches what is available at evaluation time
- it avoids future leakage
- it gives a grounded first-order motion state

Not acceptable as learned input state:

```text
(p_world_t+1 - p_world_t) / dt
```

That uses future information. It can still exist as an offline diagnostic or realized-step label, but it is not the causal model state.

The branch now serializes:

- `dynamic_obstacle_vel_world_causal_t`

with:

- `t = 0`: copied from `dynamic_obstacles_t[*]["velocity"]`
- `t >= 1`: filled after episode assembly as `(p_world_t - p_world_t-1) / dt`

## New practical branch representation

Per obstacle learned input:

- `x_world`
- `y_world`
- `vx_world_causal`
- `vy_world_causal`
- `radius`

Per obstacle learned target:

- world-frame delta position for `t -> t+1`

```text
dynamic_obstacle_delta_world_t = dynamic_obstacles_world_t1 - dynamic_obstacles_world_t
```

The preferred practical formulation is intentionally simple:

- predict world-frame delta only
- do not keep an explicit learned next-velocity head
- if rollout needs velocity for the next recurrent input, derive it causally from consecutive predicted world positions

## What is not a model training target

Future robot-relative obstacle position is not the main learned target in this redesign.

It is now a controller-side derived quantity:

1. the learned model predicts obstacle world displacement
2. the controller rolls out robot pose analytically
3. predicted obstacle world positions are transformed into the rolled-out robot frame
4. dynamic clearance and collision scoring happen there

## Controller decomposition

Learned model:

- predicts obstacle world motion only

Controller:

- stores current obstacle world position and causal world velocity
- rolls out robot pose under each candidate action
- advances obstacle world position with predicted world delta
- derives next world velocity causally from predicted positions
- transforms predicted world positions into robot-relative geometry for scoring

This keeps the controller scoring philosophy unchanged apart from the world-to-robot transform step.

## World-frame vs robot-frame fields

Primary practical learned-state fields:

- `dynamic_obstacles_world_t`
- `dynamic_obstacles_world_t1`
- `dynamic_obstacle_delta_world_t`
- `dynamic_obstacle_vel_world_causal_t`
- `dynamic_obstacle_radii`

Still retained for compatibility/debugging but deprecated as primary learned-state inputs:

- `dynamic_obstacles_robot_t`
- `dynamic_obstacles_robot_t1`
- `dynamic_obstacle_delta_robot_t`
- `dynamic_obstacle_rel_vel_robot_fd_t`
- `dynamic_obstacle_rel_vel_robot_fd_t1`
- `dynamic_obstacle_vel_robot_t`
- `dynamic_obstacle_rel_vel_robot_t`

`dynamic_obstacle_vel_world_t = (p_world_t+1 - p_world_t) / dt` is still useful as an offline realized-step quantity, but it is not the general causal world-velocity input state.

## Slot identity and fixed-cardinality assumptions

- Dynamic obstacle count is fixed by generation.
- Obstacle IDs are stable and checked within each step.
- Slot order is stable across the episode and enforced again by the learned controller at runtime.
- No padding path is used in the practical branch.

## Dataset windows and episode boundaries

- Training windows remain contiguous and do not cross episode boundaries.
- Loader-side causal world velocity synthesis can use the previous row in the episode even if that previous row is outside the sampled window.
- On the first row of an episode, causal velocity is bootstrapped from the stored snapshot velocity.

## `dt` usage

- Generation uses `robot_config.dt`.
- Loader uses row `dt` first, then episode-level fallback.
- The branch assumes `dt` is effectively constant within an episode, but keeps row-level handling for robustness.

## Compatibility implications

- Existing datasets already contain the world positions and obstacle snapshot velocities needed for the redesign.
- Regeneration is optional, not mandatory, because the loader can synthesize `dynamic_obstacle_vel_world_causal_t` for older datasets.
- New datasets are marked with schema version `v4_dynamic_state_data_3`.

## Why this avoids the old robot-frame semantic issue

Previously, the learned model had to predict obstacle motion inside a frame that was itself moving and rotating with the robot. That mixed obstacle dynamics with ego-motion-induced geometry.

The world-frame redesign avoids that by:

- learning obstacle motion in the world frame
- keeping the input velocity causal
- deriving robot-relative future geometry analytically in the controller

So the learned model now focuses on obstacle dynamics, while the controller handles frame transformation explicitly.
